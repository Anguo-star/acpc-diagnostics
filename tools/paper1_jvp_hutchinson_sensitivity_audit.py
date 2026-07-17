#!/usr/bin/env python3
"""Exact-JVP/Hutchinson Gaussian sensitivity decomposition for Paper 1.

This analysis estimates Frobenius-trace sensitivities with exact autograd JVPs
through the differentiable encoder and predictor maps. Hutchinson directions are
sampled over the matched history-pixel input and over the encoded history state.
The output decomposes local map sensitivity, but it does
not recover the full Jacobian matrix, an SVD, or a closed-loop guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev, stdev
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_acpc_metrics import uniform_horizon_weights, weighted_stacked_rollout
from tools.paper1_gaussian_sensitivity_audit import _checkpoint_plan, _full_sweep_index, _fmt_rho
from tools.paper1_margin_flip_curve import MANIFEST_DIR, SEEDS, TASKS, _success

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_SWEEP = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_JSON = ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_audit_v2.json"
DEFAULT_CSV = ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_audit_v2.csv"
DEFAULT_SUMMARY = ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_summary_v2.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_jvp_hutchinson_sensitivity_audit.tex"
KAPPA_EPS = 1e-12
SCHEMA_VERSION = "paper1-jvp-hutchinson-sensitivity-0.2"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return _jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _f(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(values: Sequence[Any]) -> list[float]:
    out = []
    for value in values:
        x = _f(value)
        if math.isfinite(x):
            out.append(x)
    return out


def _mean(values: Sequence[Any]) -> float:
    xs = _finite(values)
    return mean(xs) if xs else math.nan


def _median(values: Sequence[Any]) -> float:
    xs = _finite(values)
    return median(xs) if xs else math.nan


def _pstdev(values: Sequence[Any]) -> float:
    xs = _finite(values)
    if not xs:
        return math.nan
    return pstdev(xs) if len(xs) > 1 else 0.0


def _normal_mean_ci95(values: Sequence[Any]) -> list[float] | None:
    """Return an unclipped finite-probe normal interval for the sample mean."""
    xs = _finite(values)
    if not xs:
        return None
    center = mean(xs)
    if len(xs) == 1:
        return [center, center]
    half_width = 1.96 * stdev(xs) / math.sqrt(len(xs))
    return [center - half_width, center + half_width]


def _ratio(num: float, den: float) -> float:
    return num / den if math.isfinite(num) and math.isfinite(den) and abs(den) > 1e-12 else math.nan


def kappa_submultiplicative(
    *,
    composed_frobenius_sq: float,
    rollout_frobenius_sq: float,
    encoder_frobenius_sq: float,
    eps: float = KAPPA_EPS,
) -> float:
    """Return ||J_G J_E||_F^2 / (||J_G||_F^2 ||J_E||_F^2 + eps).

    The helper intentionally does not clip finite-probe estimates. Exact
    synthetic Jacobians obey the submultiplicative identity, while independent
    Hutchinson estimates can cross it because of estimator error.
    """
    composed = float(composed_frobenius_sq)
    rollout = float(rollout_frobenius_sq)
    encoder = float(encoder_frobenius_sq)
    epsilon = float(eps)
    if epsilon < 0:
        raise ValueError(f"eps must be non-negative, got {epsilon}")
    for name, value in (
        ("composed_frobenius_sq", composed),
        ("rollout_frobenius_sq", rollout),
        ("encoder_frobenius_sq", encoder),
    ):
        if math.isfinite(value) and value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    denominator = rollout * encoder + epsilon
    if not math.isfinite(composed) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return composed / denominator


def kappa_relative_isotropic(kappa_sub: float, *, latent_input_dim: int) -> float:
    """Return the relative isotropic alignment gain d_z * kappa_sub.

    ``latent_input_dim`` is the complete intermediate dimension of the map
    whose Jacobian norms enter ``kappa_sub``. For the batched audit this is
    batch_size * history_size * latent_feature_dim, not merely the feature
    width of one latent token.
    """
    if isinstance(latent_input_dim, bool) or int(latent_input_dim) != latent_input_dim:
        raise ValueError(f"latent_input_dim must be a positive integer, got {latent_input_dim!r}")
    latent_dim = int(latent_input_dim)
    if latent_dim <= 0:
        raise ValueError(f"latent_input_dim must be positive, got {latent_dim}")
    return latent_dim * float(kappa_sub)


def _manifest_std_key(value: Any) -> str:
    x = float(value)
    return "0.0" if abs(x) < 1e-12 else f"{x:.2f}"


def _resolve(entry: Mapping[str, Any], model_roots: Sequence[Path]) -> tuple[Path | None, list[str]]:
    return phase0.resolve_model_file(str(entry.get("path", "")), str(entry.get("subdir", "")), model_roots)


def _disable_forward_ad_incompatible_sdp() -> None:
    if not torch.cuda.is_available():
        return
    # The efficient SDPA kernel has no forward-AD rule in this PyTorch build.
    # Math/eager attention keeps the directional derivative exact for the same
    # differentiable computation, while avoiding unsupported fused kernels.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _force_eager_attention(module: Any) -> None:
    for obj in (getattr(module, "encoder", None), getattr(getattr(module, "encoder", None), "config", None)):
        if obj is None:
            continue
        if hasattr(obj, "_attn_implementation"):
            obj._attn_implementation = "eager"
        if hasattr(obj, "attn_implementation"):
            obj.attn_implementation = "eager"


def _cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _autoregressive_rollout_grad(
    model: Any,
    init_emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    n_steps: int,
) -> torch.Tensor:
    chain = init_emb
    for t in range(n_steps):
        a_win = act_emb[:, t : t + history_size]
        if a_win.size(1) < history_size:
            break
        pred = model.predict(chain[:, -history_size:], a_win)[:, -1:]
        chain = torch.cat([chain, pred], dim=1)
    return chain


def _rademacher_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    # Rademacher Hutchinson directions have E[v v^T] = I like unit-variance
    # Gaussian perturbations, with lower variance for trace estimation.
    vals = torch.randint(0, 2, x.shape, device=x.device, generator=generator, dtype=torch.int8)
    return vals.to(dtype=x.dtype).mul_(2).sub_(1)


def _estimate_checkpoint(
    *,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    history_size: int,
    rollout_horizon: int,
    embedding_space: str,
    hutchinson_probes: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pixels_full = batch["pixels"].detach()
    action = batch["action"].detach()
    pixels_hist = pixels_full[:, :history_size].detach()
    device = pixels_hist.device
    generator = torch.Generator(device=device).manual_seed(int(seed))

    with torch.no_grad():
        clean_info = model.encode({"pixels": pixels_full, "action": action})
        clean_emb = phase0.get_embedding_space(clean_info, embedding_space).detach()
        clean_act_emb = clean_info["act_emb"].detach()
    max_steps = min(rollout_horizon, max(0, clean_act_emb.size(1) - history_size + 1))
    if max_steps < 1:
        raise ValueError("rollout map requires at least one available future step")
    init_clean = clean_emb[:, :history_size].detach()
    horizon_weights = uniform_horizon_weights(
        max_steps,
        dtype=init_clean.dtype,
        device=init_clean.device,
    )

    def encoder_and_composed(pix_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = torch.cat([pix_hist, pixels_full[:, history_size:]], dim=1)
        info = model.encode({"pixels": pixels, "action": action})
        emb = phase0.get_embedding_space(info, embedding_space)
        act_emb = info["act_emb"]
        init = emb[:, :history_size]
        chain = _autoregressive_rollout_grad(model, init, act_emb, history_size, max_steps)
        pred = chain[:, history_size : history_size + max_steps]
        return (
            init.reshape(-1),
            weighted_stacked_rollout(pred, weights=horizon_weights).reshape(-1),
        )

    def rollout_from_latent(init: torch.Tensor) -> torch.Tensor:
        chain = _autoregressive_rollout_grad(model, init, clean_act_emb, history_size, max_steps)
        pred = chain[:, history_size : history_size + max_steps]
        return weighted_stacked_rollout(pred, weights=horizon_weights).reshape(-1)

    probe_rows: list[dict[str, Any]] = []
    encoder_vals: list[float] = []
    composed_vals: list[float] = []
    rollout_vals: list[float] = []
    for probe in range(int(hutchinson_probes)):
        pixel_v = _rademacher_like(pixels_hist, generator)
        latent_v = _rademacher_like(init_clean, generator)
        try:
            (_enc_y, _comp_y), (enc_jv, comp_jv) = torch.func.jvp(
                encoder_and_composed,
                (pixels_hist,),
                (pixel_v,),
            )
            _roll_y, roll_jv = torch.func.jvp(
                rollout_from_latent,
                (init_clean,),
                (latent_v,),
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                "Exact JVP failed. This usually means an unsupported fused kernel is still active; "
                "the script disables flash/mem-efficient SDPA and requests eager attention."
            ) from exc
        enc_sq = float(enc_jv.detach().float().square().sum().cpu().item())
        comp_sq = float(comp_jv.detach().float().square().sum().cpu().item())
        roll_sq = float(roll_jv.detach().float().square().sum().cpu().item())
        encoder_vals.append(enc_sq)
        composed_vals.append(comp_sq)
        rollout_vals.append(roll_sq)
        probe_rows.append({
            "probe_index": probe,
            "encoder_trace_probe": enc_sq,
            "composed_trace_probe": comp_sq,
            "rollout_trace_probe": roll_sq,
        })

    pixel_dim = int(pixels_hist.numel())
    latent_dim = int(init_clean.numel())
    rollout_dim = int(max(1, batch["pixels"].size(0) * max_steps * init_clean.size(-1)))
    latent_feature_dim = int(init_clean.size(-1))
    latent_sequence_dim = int(init_clean[0].numel())
    encoder_trace = _mean(encoder_vals)
    composed_trace = _mean(composed_vals)
    rollout_trace = _mean(rollout_vals)
    rollout_trace_per_latent_dim = rollout_trace / latent_dim if latent_dim else math.nan
    kappa_sub = kappa_submultiplicative(
        composed_frobenius_sq=composed_trace,
        rollout_frobenius_sq=rollout_trace,
        encoder_frobenius_sq=encoder_trace,
        eps=KAPPA_EPS,
    )
    kappa_relative = kappa_relative_isotropic(kappa_sub, latent_input_dim=latent_dim)
    kappa_sub_probe_values = [
        kappa_submultiplicative(
            composed_frobenius_sq=composed,
            rollout_frobenius_sq=rollout,
            encoder_frobenius_sq=encoder,
            eps=KAPPA_EPS,
        )
        for composed, rollout, encoder in zip(
            composed_vals,
            rollout_vals,
            encoder_vals,
        )
    ]
    kappa_relative_probe_values = [
        kappa_relative_isotropic(value, latent_input_dim=latent_dim)
        for value in kappa_sub_probe_values
    ]
    summary = {
        "status": "ok",
        "history_size": int(history_size),
        "n_sequences": int(batch["pixels"].size(0)),
        "rollout_horizon": int(rollout_horizon),
        "rollout_horizon_actual": int(max_steps),
        "horizon_weights": horizon_weights.detach().cpu().tolist(),
        "horizon_weighting": "uniform alpha_k=1/H applied as sqrt(alpha_k)",
        "rollout_vectorization": "batch-major canonical weighted stack of (H, D) into (H*D)",
        "rollout_projection": "identity in the selected embedding_space",
        "rollout_normalization": "none before per-anchor radius normalization",
        "embedding_space": embedding_space,
        "hutchinson_probes": int(hutchinson_probes),
        "pixel_input_dim": pixel_dim,
        "latent_input_dim": latent_dim,
        "latent_sequence_dim": latent_sequence_dim,
        "latent_feature_dim": latent_feature_dim,
        "kappa_latent_dimension_semantics": "batch_size * history_size * latent_feature_dim",
        "rollout_output_dim": rollout_dim,
        "encoder_trace": encoder_trace,
        "encoder_trace_std": _pstdev(encoder_vals),
        "encoder_trace_mean_ci95_unclipped": _normal_mean_ci95(encoder_vals),
        "encoder_trace_per_pixel_dim": encoder_trace / pixel_dim if pixel_dim else math.nan,
        "rollout_trace": rollout_trace,
        "rollout_trace_std": _pstdev(rollout_vals),
        "rollout_trace_mean_ci95_unclipped": _normal_mean_ci95(rollout_vals),
        "rollout_trace_per_latent_dim": rollout_trace_per_latent_dim,
        "composed_trace": composed_trace,
        "composed_trace_std": _pstdev(composed_vals),
        "composed_trace_mean_ci95_unclipped": _normal_mean_ci95(composed_vals),
        "composed_trace_per_pixel_dim": composed_trace / pixel_dim if pixel_dim else math.nan,
        "kappa_epsilon": KAPPA_EPS,
        "kappa_submultiplicative": kappa_sub,
        "kappa_relative_isotropic": kappa_relative,
        "kappa_submultiplicative_per_probe": kappa_sub_probe_values,
        "kappa_submultiplicative_probe_mean": _mean(
            kappa_sub_probe_values
        ),
        "kappa_submultiplicative_probe_ci95_unclipped": (
            _normal_mean_ci95(kappa_sub_probe_values)
        ),
        "kappa_relative_isotropic_per_probe": (
            kappa_relative_probe_values
        ),
        "kappa_relative_isotropic_probe_mean": _mean(
            kappa_relative_probe_values
        ),
        "kappa_relative_isotropic_probe_ci95_unclipped": (
            _normal_mean_ci95(kappa_relative_probe_values)
        ),
        "finite_probe_interval_method": (
            "unclipped normal mean interval using sample standard error; "
            "descriptive Monte Carlo uncertainty, not a formal coverage guarantee"
        ),
        "alignment_coefficient": kappa_relative,
        "alignment_coefficient_deprecated_alias_of": "kappa_relative_isotropic",
        "notes": "exact autograd JVP with unclipped Hutchinson trace estimates and finite-probe intervals; not a full Jacobian matrix or closed-loop guarantee",
    }
    return summary, probe_rows


def run_checkpoint(
    *,
    task: str,
    seed: int,
    checkpoint_type: str,
    std_key: str,
    entry: Mapping[str, Any],
    full_sweep_row: Mapping[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wall_started = time.perf_counter()
    runtime_setup_started = time.perf_counter()
    phase0._ensure_runtime_deps()
    runtime_dependency_setup_time = time.perf_counter() - runtime_setup_started
    _disable_forward_ad_incompatible_sdp()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    uses_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if uses_cuda:
        _cuda_sync(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    model_roots = [Path(p).expanduser() for p in args.model_root]
    model_file, tried = _resolve(entry, model_roots)
    base = {
        "model_family": "LeWM",
        "task": task,
        "training_seed": int(seed),
        "checkpoint_type": checkpoint_type,
        "std_key": std_key,
        "subdir": entry.get("subdir"),
        "run_path": entry.get("path"),
        "model_file": str(model_file) if model_file else None,
        "model_search_dirs": tried,
        "clean_success": _success(entry, "clean"),
        "pixels_std0.08_success": _success(entry, "pixels_std0.08"),
        "atr_q90": _f(full_sweep_row.get("atr_q90")),
        "atr_normalized_q90": _f(full_sweep_row.get("atr_normalized_q90")),
        "smpr_delta0": _f(full_sweep_row.get("smpr_delta0")),
        "recovery_label": full_sweep_row.get("recovery_label", ""),
    }
    if model_file is None:
        return {
            **base,
            "status": "skipped_missing_model",
            "runtime_dependency_setup_time": runtime_dependency_setup_time,
            "model_load_time": None,
            "data_io_time": None,
            "jvp_time": None,
            "fixed_pool_time": None,
            "wall_time_per_row": time.perf_counter() - wall_started,
            "timing_unit": "seconds",
            "peak_gpu_memory": None,
            "peak_gpu_memory_unit": "bytes",
        }, []

    model_load_started = time.perf_counter()
    model = phase0.load_model(str(model_file), device)
    _force_eager_attention(model)
    _cuda_sync(device)
    model_load_time = time.perf_counter() - model_load_started

    data_io_started = time.perf_counter()
    history_size = phase0.infer_history_size(model)
    future_steps = max(args.future_steps, args.rollout_horizon + 1)
    batch = phase0.load_dataset_samples(
        dataset_name=phase0.TASK_DATASETS[task],
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=seed,
        device=device,
    )
    _cuda_sync(device)
    data_io_time = time.perf_counter() - data_io_started

    spaces = phase0.get_model_spaces(model)
    embedding_space = args.embedding_space or spaces["inference_cost_space"]
    jvp_started = time.perf_counter()
    summary, probe_rows = _estimate_checkpoint(
        model=model,
        batch=batch,
        history_size=history_size,
        rollout_horizon=args.rollout_horizon,
        embedding_space=embedding_space,
        hutchinson_probes=args.hutchinson_probes,
        seed=seed + int(round(float(std_key) * 10000)) + 7919,
    )
    _cuda_sync(device)
    jvp_time = time.perf_counter() - jvp_started
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if uses_cuda
        else None
    )
    profiling = {
        "runtime_dependency_setup_time": runtime_dependency_setup_time,
        "model_load_time": model_load_time,
        "data_io_time": data_io_time,
        "jvp_time": jvp_time,
        "fixed_pool_time": None,
        "wall_time_per_row": time.perf_counter() - wall_started,
        "timing_unit": "seconds",
        "peak_gpu_memory": peak_gpu_memory,
        "peak_gpu_memory_unit": "bytes",
    }
    for row in probe_rows:
        row.update(base)
    return {**base, **summary, **profiling}, probe_rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        groups[(row["task"], row["checkpoint_type"])].append(row)
    summary: list[dict[str, Any]] = []
    for task in TASKS:
        base_rows = groups.get((task, "base"), [])
        base_encoder = _median([r["encoder_trace_per_pixel_dim"] for r in base_rows])
        base_rollout = _median([r["rollout_trace_per_latent_dim"] for r in base_rows])
        base_composed = _median([r["composed_trace_per_pixel_dim"] for r in base_rows])
        base_kappa_sub = _median([r.get("kappa_submultiplicative") for r in base_rows])
        base_kappa_relative = _median(
            [r.get("kappa_relative_isotropic", r.get("alignment_coefficient")) for r in base_rows]
        )
        for checkpoint_type in ("base", "onset", "endpoint"):
            rs = groups.get((task, checkpoint_type), [])
            if not rs:
                continue
            encoder = _median([r["encoder_trace_per_pixel_dim"] for r in rs])
            rollout = _median([r["rollout_trace_per_latent_dim"] for r in rs])
            composed = _median([r["composed_trace_per_pixel_dim"] for r in rs])
            kappa_sub = _median([r.get("kappa_submultiplicative") for r in rs])
            kappa_relative = _median(
                [r.get("kappa_relative_isotropic", r.get("alignment_coefficient")) for r in rs]
            )
            summary.append({
                "task": task,
                "checkpoint_type": checkpoint_type,
                "n_training_seeds": len(rs),
                "n_sequences": int(_median([r.get("n_sequences") for r in rs])),
                "hutchinson_probes": int(_median([r.get("hutchinson_probes") for r in rs])),
                "encoder_trace_per_pixel_dim_median": encoder,
                "encoder_trace_per_pixel_dim_vs_base": _ratio(encoder, base_encoder),
                "rollout_trace_per_latent_dim_median": rollout,
                "rollout_trace_per_latent_dim_vs_base": _ratio(rollout, base_rollout),
                "composed_trace_per_pixel_dim_median": composed,
                "composed_trace_per_pixel_dim_vs_base": _ratio(composed, base_composed),
                "kappa_submultiplicative_median": kappa_sub,
                "kappa_submultiplicative_vs_base": _ratio(kappa_sub, base_kappa_sub),
                "kappa_relative_isotropic_median": kappa_relative,
                "kappa_relative_isotropic_vs_base": _ratio(kappa_relative, base_kappa_relative),
                "alignment_coefficient_median": kappa_relative,
                "alignment_coefficient_vs_base": _ratio(kappa_relative, base_kappa_relative),
            })
    return summary


def _fmt(x: Any, digits: int = 3) -> str:
    y = _f(x)
    if not math.isfinite(y):
        return "--"
    return f"{y:.{digits}f}"


def _fmt_count_set(values: Sequence[Any]) -> str:
    counts = sorted({int(_f(value)) for value in values if math.isfinite(_f(value))})
    if not counts:
        return "the reported"
    return str(counts[0]) if len(counts) == 1 else "/".join(str(v) for v in counts)


def _kappa_relative_isotropic_vs_base(row: Mapping[str, Any]) -> Any:
    """Read the canonical summary key, with the legacy alias as fallback."""
    value = row.get("kappa_relative_isotropic_vs_base")
    if value is None or value == "":
        return row.get("alignment_coefficient_vs_base")
    return value


def write_table(path: Path, summary_rows: list[dict[str, Any]], table_label: str) -> None:
    idx = {(row["task"], row["checkpoint_type"]): row for row in summary_rows}
    n_desc = _fmt_count_set(row.get("n_sequences") for row in summary_rows)
    probe_desc = _fmt_count_set(row.get("hutchinson_probes") for row in summary_rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Exact-JVP/Hutchinson local sensitivity decomposition. Every entry is an endpoint/base ratio, using {n_desc} sampled sequences and {probe_desc} Rademacher probes per checkpoint, then taking medians over training seeds when multiple seeds are present. The first three columns are ratios of the raw encoder, rollout, and composed Frobenius-trace estimates; the constant per-input-dimension normalizers cancel within each endpoint/base comparison. The final column reports \texttt{{kappa\_relative\_isotropic}}, the relative isotropic alignment gain $\kappa_{{\mathrm{{rel}}}}=d_z\lVert J_GJ_E\rVert_F^2/(\lVert J_G\rVert_F^2\lVert J_E\rVert_F^2+\varepsilon)$. This diagnostic can exceed 1 and is not an angle, cosine, or certificate. Exact autograd JVPs with math/eager attention estimate local traces, not a full Jacobian matrix or a closed-loop guarantee.}}",
        rf"\label{{{table_label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Task & \shortstack{raw encoder\\trace} & \shortstack{raw rollout\\trace} & \shortstack{raw composed\\trace} & \shortstack{relative isotropic\\gain} \\",
        r"\midrule",
    ]
    for task in TASKS:
        row = idx.get((task, "endpoint"), {})
        lines.append(
            f"{task} & {_fmt(row.get('encoder_trace_per_pixel_dim_vs_base'))} & "
            f"{_fmt(row.get('rollout_trace_per_latent_dim_vs_base'))} & "
            f"{_fmt(row.get('composed_trace_per_pixel_dim_vs_base'))} & "
            f"{_fmt(_kappa_relative_isotropic_vs_base(row))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--eval-manifest-dir", type=Path, default=MANIFEST_DIR)
    p.add_argument("--full-sweep", type=Path, default=DEFAULT_FULL_SWEEP)
    p.add_argument("--model-root", action="append", default=[])
    p.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--table-label", default="tab:jvp-hutchinson-sensitivity-audit")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-sequences", type=int, default=16)
    p.add_argument("--hutchinson-probes", type=int, default=8)
    p.add_argument("--future-steps", type=int, default=9)
    p.add_argument("--rollout-horizon", type=int, default=8)
    p.add_argument("--state-key", default=None)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embedding-space", default=None)
    p.add_argument("--device", default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    full_sweep = _full_sweep_index(args.full_sweep)
    manifest_by_seed = {seed: _load_json(args.eval_manifest_dir / f"lewm_seed{seed}_evals.json") for seed in args.seeds}
    specs: list[tuple[str, int, str, str, Mapping[str, Any], Mapping[str, str]]] = []
    for task, seed, checkpoint_type, rho in _checkpoint_plan(full_sweep, args.tasks, args.seeds):
        entry = manifest_by_seed[seed].get(task, {}).get(_manifest_std_key(rho), None)
        if entry is None:
            entry = manifest_by_seed[seed].get(task, {}).get(rho)
        row = full_sweep.get((task, seed, _fmt_rho(rho)), {})
        if entry is None:
            specs.append((task, seed, checkpoint_type, rho, {}, row))
        else:
            specs.append((task, seed, checkpoint_type, rho, entry, row))
    if args.limit is not None:
        specs = specs[: args.limit]

    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for idx, (task, seed, checkpoint_type, rho, entry, full_sweep_row) in enumerate(specs, start=1):
        print(f"[{idx}/{len(specs)}] {task} seed{seed} {checkpoint_type} std{rho}", flush=True)
        if not entry:
            rows.append({"task": task, "training_seed": seed, "checkpoint_type": checkpoint_type, "std_key": rho, "status": "skipped_missing_manifest"})
            continue
        try:
            row, probes = run_checkpoint(
                task=task,
                seed=seed,
                checkpoint_type=checkpoint_type,
                std_key=rho,
                entry=entry,
                full_sweep_row=full_sweep_row,
                args=args,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failures in the artifact.
            rows.append(
                {
                    "model_family": "LeWM",
                    "task": task,
                    "training_seed": int(seed),
                    "checkpoint_type": checkpoint_type,
                    "std_key": rho,
                    "subdir": entry.get("subdir"),
                    "run_path": entry.get("path"),
                    "status": "error",
                    "error": repr(exc),
                }
            )
            continue
        rows.append(row)
        probe_rows.extend(probes)

    summary = _summarize(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest_paths = {
        f"lewm_seed{seed}_evals": args.eval_manifest_dir / f"lewm_seed{seed}_evals.json"
        for seed in args.seeds
    }
    source_paths = {"full_sweep": args.full_sweep, **manifest_paths}
    source_hashes = {
        name: _sha256(path)
        for name, path in source_paths.items()
        if path.is_file()
    }
    script_path = Path(__file__).resolve()
    metric_path = ROOT / "tools" / "paper1_acpc_metrics.py"
    created_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "metric_implementation_path": str(metric_path.relative_to(ROOT)),
            "metric_implementation_sha256": _sha256(metric_path),
            "source_paths": {
                name: str(path)
                for name, path in source_paths.items()
            },
            "source_hashes": source_hashes,
            "protocol_hash": None,
            "protocol_hash_status": "not_frozen_phase1_correctness_audit",
            "model_family": "LeWM",
            "training_seed_semantics": (
                "independently trained checkpoint seed; never an evaluation seed"
            ),
            "evaluation_seed_semantics": (
                "not applicable; this is a checkpoint-local anchor/JVP audit"
            ),
            "anchor_sampling_seed_rule": "training_seed",
            "jvp_probe_seed_rule": "training_seed+round(std*10000)+7919",
            "status": (
                "complete"
                if status_counts and set(status_counts) == {"ok"}
                else "partial"
            ),
            "status_counts": status_counts,
            "missing_rows": [
                {
                    "task": row.get("task"),
                    "training_seed": row.get("training_seed"),
                    "checkpoint_type": row.get("checkpoint_type"),
                    "std_key": row.get("std_key"),
                    "status": row.get("status"),
                }
                for row in rows
                if str(row.get("status", "")).startswith("skipped_")
            ],
            "errors": [
                {
                    "task": row.get("task"),
                    "training_seed": row.get("training_seed"),
                    "checkpoint_type": row.get("checkpoint_type"),
                    "std_key": row.get("std_key"),
                    "error": row.get("error"),
                }
                for row in rows
                if row.get("status") == "error"
            ],
            "map_contract": {
                "radius_metric": "horizon_weighted_stacked_l2_v2",
                "horizon_weights": "uniform alpha_k=1/H applied as sqrt(alpha_k)",
                "rollout_projection": "identity in the selected embedding_space",
                "rollout_vectorization": (
                    "batch-major canonical weighted stack of (H,D) into (H*D)"
                ),
                "normalization": (
                    "none inside JVP; canonical per-anchor transition scale is "
                    "applied after this map"
                ),
            },
        },
        "generated_at": created_utc,
        "args": vars(args),
        "rows": rows,
        "probe_rows": probe_rows,
        "summary": summary,
        "notes": [
            "Exact autograd JVPs are used after switching ViT attention to math/eager kernels because efficient SDPA lacks forward-AD support in this PyTorch build.",
            "Hutchinson directions estimate local Frobenius traces; the analysis does not materialize the full Jacobian matrix.",
            "Kappa metrics are computed from the reported trace estimates without clipping.",
            "kappa_relative_isotropic is the relative isotropic alignment gain; alignment_coefficient is its deprecated compatibility alias.",
        ],
    }

    _write_csv(args.out_csv, rows)
    _write_csv(args.summary_csv, summary)
    write_table(args.table, summary, args.table_label)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.table}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
