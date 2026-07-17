#!/usr/bin/env python3
"""Matched-map linearization calibration and horizon/quantile audit.

The runner evaluates one or more task/seed blocks serially.  For every
base/onset/endpoint checkpoint it uses the same fixed anchors and canonical
weighted rollout map as the Paper-1 JVP audit, while making the input tangent
covariance match the repository's *pixel-space* Gaussian corruption.

This is a local, descriptive diagnostic.  It is neither a robustness
certificate nor a replacement for closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_acpc_metrics import (
    compute_acpc_horizon_metrics,
    horizon_weighted_stacked_l2,
    per_anchor_clean_transition_scale,
    uniform_horizon_weights,
    weighted_stacked_rollout,
)
from tools.paper1_gaussian_sensitivity_audit import (
    _checkpoint_plan,
    _full_sweep_index,
    _fmt_rho,
)
from tools.paper1_jvp_hutchinson_sensitivity_audit import (
    _disable_forward_ad_incompatible_sdp,
    _force_eager_attention,
    _git_commit,
    _jsonable,
    _manifest_std_key,
    _normal_mean_ci95,
    _resolve,
    _rademacher_like,
    _success,
    _write_csv,
)
from tools.paper1_margin_flip_curve import MANIFEST_DIR, SEEDS, TASKS
from utils import AddNormalizedGaussianNoise


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_SWEEP = ROOT / "paper1/results/full_sweep_diagnostics.csv"
DEFAULT_OUT = ROOT / "paper1/results/linearization_horizon_sensitivity_v1.json"
SCHEMA_VERSION = "paper1-linearization-horizon-audit-0.1"
SMALL_SIGMAS = (0.0025, 0.005, 0.01, 0.02)
HORIZONS = (1, 2, 4, 8)
QUANTILES = (0.80, 0.90, 0.95)
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected finite value, got {value!r}")
    return result


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else math.nan


def _pstdev(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    if len(finite) == 1:
        return 0.0
    center = mean(finite)
    return math.sqrt(mean((value - center) ** 2 for value in finite))


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values.detach().float().cpu(), q).item())


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def calibration_row_metrics(
    *, measured_mean_r2: float, sigma: float, jvp_trace_per_sequence: float
) -> dict[str, float]:
    """Return the explicit measured/predicted calibration quantities."""

    measured = _finite(measured_mean_r2, name="measured_mean_r2")
    sigma_value = _finite(sigma, name="sigma")
    trace = _finite(jvp_trace_per_sequence, name="jvp_trace_per_sequence")
    if measured < 0 or sigma_value <= 0 or trace < 0:
        raise ValueError("measured R2 and trace must be non-negative and sigma positive")
    empirical_trace = measured / sigma_value**2
    predicted_mean_r2 = sigma_value**2 * trace
    signed_remainder = measured - predicted_mean_r2
    absolute_remainder = abs(signed_remainder)
    return {
        "empirical_mean_R2_over_sigma2": empirical_trace,
        "jvp_gaussian_trace_per_sequence": trace,
        "measured_to_jvp_ratio": _ratio(empirical_trace, trace),
        "relative_error": _ratio(abs(empirical_trace - trace), trace),
        "predicted_mean_R2": predicted_mean_r2,
        "signed_remainder": signed_remainder,
        "absolute_remainder": absolute_remainder,
        "absolute_remainder_over_sigma3": absolute_remainder / sigma_value**3,
    }


def log_remainder_order(rows: Sequence[Mapping[str, Any]]) -> float:
    """Descriptive log-log slope of |measured - linear prediction| vs sigma."""

    points: list[tuple[float, float]] = []
    for row in rows:
        sigma = float(row["sigma"])
        remainder = float(row["absolute_remainder"])
        if sigma > 0 and remainder > 0 and math.isfinite(sigma) and math.isfinite(remainder):
            points.append((math.log(sigma), math.log(remainder)))
    if len(points) < 2:
        return math.nan
    x_bar = mean(x for x, _ in points)
    y_bar = mean(y for _, y in points)
    denom = sum((x - x_bar) ** 2 for x, _ in points)
    if denom <= 0:
        return math.nan
    return sum((x - x_bar) * (y - y_bar) for x, y in points) / denom


def _pixel_coordinate_scale(reference: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """Map unit pixel-space perturbations into ImageNet-normalized coordinates."""

    transform = AddNormalizedGaussianNoise(1.0, 1.0)
    channel_std = transform.channel_std.to(device=reference.device, dtype=reference.dtype)
    if reference.ndim < 3 or reference.shape[-3] != channel_std.numel():
        scale = torch.ones_like(reference)
        return scale, [1.0]
    shape = (1,) * (reference.ndim - 3) + (channel_std.numel(), 1, 1)
    scale = (1.0 / channel_std).reshape(shape)
    return scale, [float(value) for value in channel_std.detach().cpu().tolist()]


def _autoregressive_predictions(
    model: Any,
    initial_embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    *,
    history_size: int,
    horizon: int,
) -> torch.Tensor:
    chain = phase0._autoregressive_rollout(
        model,
        initial_embedding,
        action_embedding,
        history_size,
        horizon,
    )
    return chain[:, history_size : history_size + horizon]


def _jvp_gaussian_trace(
    *,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    history_size: int,
    horizon: int,
    embedding_space: str,
    probes: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pixels_full = batch["pixels"].detach()
    action = batch["action"].detach()
    pixels_hist = pixels_full[:, :history_size].detach()
    weights = uniform_horizon_weights(
        horizon, dtype=pixels_hist.dtype, device=pixels_hist.device
    )
    coordinate_scale, channel_std = _pixel_coordinate_scale(pixels_hist)
    generator = torch.Generator(device=pixels_hist.device).manual_seed(int(seed))

    def composed_map(history_pixels: torch.Tensor) -> torch.Tensor:
        pixels = torch.cat([history_pixels, pixels_full[:, history_size:]], dim=1)
        info = model.encode({"pixels": pixels, "action": action})
        embedding = phase0.get_embedding_space(info, embedding_space)
        predicted = _autoregressive_predictions(
            model,
            embedding[:, :history_size],
            info["act_emb"],
            history_size=history_size,
            horizon=horizon,
        )
        return weighted_stacked_rollout(predicted, weights=weights).reshape(-1)

    values: list[float] = []
    probe_rows: list[dict[str, Any]] = []
    for index in range(int(probes)):
        tangent = _rademacher_like(pixels_hist, generator) * coordinate_scale
        _, derivative = torch.func.jvp(composed_map, (pixels_hist,), (tangent,))
        squared_norm = float(derivative.detach().float().square().sum().cpu().item())
        values.append(squared_norm)
        probe_rows.append(
            {
                "probe_index": index,
                "gaussian_pixel_composed_trace_probe": squared_norm,
            }
        )
    total = _mean(values)
    return (
        {
            "jvp_gaussian_trace": total,
            "jvp_gaussian_trace_per_sequence": total / int(pixels_hist.shape[0]),
            "jvp_gaussian_trace_probe_pstdev": _pstdev(values),
            "jvp_gaussian_trace_mean_ci95_unclipped": _normal_mean_ci95(values),
            "hutchinson_probes": int(probes),
            "input_coordinate_system": "raw pixel-space Gaussian sigma",
            "model_input_coordinate_system": "ImageNet-normalized tensor",
            "pixel_to_model_tangent_scale": "per-channel reciprocal ImageNet std",
            "imagenet_channel_std": channel_std,
        },
        probe_rows,
    )


def _encode_rollout(
    *,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    history_size: int,
    horizon: int,
    embedding_space: str,
    clean_action_embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    embedding = phase0.get_embedding_space(outputs, embedding_space).detach()
    action_embedding = (
        outputs["act_emb"].detach()
        if clean_action_embedding is None
        else clean_action_embedding
    )
    predicted = _autoregressive_predictions(
        model,
        embedding[:, :history_size],
        action_embedding,
        history_size=history_size,
        horizon=horizon,
    )
    return embedding, action_embedding, predicted


def _noisy_rollouts(
    *,
    model: Any,
    batch: Mapping[str, torch.Tensor],
    clean_action_embedding: torch.Tensor,
    history_size: int,
    horizon: int,
    embedding_space: str,
    sigma: float,
    draws: int,
    seed: int,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for draw in range(int(draws)):
        noisy = phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=float(sigma),
            seed=int(seed) + 104729 * draw,
            corruption_type="gaussian_noise",
            corrupt_goal=False,
        )
        _, _, predicted = _encode_rollout(
            model=model,
            batch=noisy,
            history_size=history_size,
            horizon=horizon,
            embedding_space=embedding_space,
            clean_action_embedding=clean_action_embedding,
        )
        predictions.append(predicted)
    return torch.stack(predictions, dim=1)


def _horizon_rows(
    *,
    clean_embedding: torch.Tensor,
    clean_prediction: torch.Tensor,
    noisy_prediction: torch.Tensor,
    history_size: int,
    horizons: Sequence[int],
    quantiles: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        clean_h = clean_prediction[:, :horizon]
        noisy_h = noisy_prediction[:, :, :horizon]
        observed_future = clean_embedding[:, history_size : history_size + horizon]
        scale = per_anchor_clean_transition_scale(
            observed_future,
            initial_clean_state=clean_embedding[:, history_size - 1],
            transition_quantile=0.50,
        )
        raw_per_draw = horizon_weighted_stacked_l2(
            clean_h.unsqueeze(1).expand_as(noisy_h), noisy_h
        )
        raw_per_anchor = raw_per_draw.mean(dim=1)
        for quantile in quantiles:
            metric = compute_acpc_horizon_metrics(
                clean_h,
                noisy_h,
                clean_transition_scale=scale,
                noise_draw_dim=1,
                noise_draw_aggregation="mean",
                atr_quantile=float(quantile),
                transition_quantile=0.50,
            )
            rows.append(
                {
                    "horizon": int(horizon),
                    "atr_quantile": float(quantile),
                    "atr_horizon_v2": float(metric["atr"].detach().cpu().item()),
                    "raw_horizon_radius_quantile": _quantile(
                        raw_per_anchor, float(quantile)
                    ),
                    "clean_transition_scale_q50": _quantile(scale, 0.50),
                    "radius_metric": str(metric["radius_metric"]),
                    "noise_draw_aggregation": str(metric["noise_draw_aggregation"]),
                }
            )
    return rows


def run_checkpoint(
    *,
    task: str,
    training_seed: int,
    checkpoint_type: str,
    std_key: str,
    entry: Mapping[str, Any],
    full_sweep_row: Mapping[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    phase0._ensure_runtime_deps()
    _disable_forward_ad_incompatible_sdp()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_file, tried = _resolve(entry, [Path(path).expanduser() for path in args.model_root])
    base = {
        "model_family": "LeWM",
        "task": task,
        "training_seed": int(training_seed),
        "checkpoint_type": checkpoint_type,
        "std_key": std_key,
        "subdir": entry.get("subdir"),
        "run_path": entry.get("path"),
        "model_file": str(model_file) if model_file else None,
        "model_search_dirs": tried,
        "clean_success": _success(entry, "clean"),
        "pixels_std0.08_success": _success(entry, "pixels_std0.08"),
        "atr_q90_source": float(full_sweep_row.get("atr_q90", "nan")),
        "recovery_label": full_sweep_row.get("recovery_label", ""),
    }
    if model_file is None:
        return {**base, "status": "skipped_missing_model"}, [], [], []

    load_started = time.perf_counter()
    model = phase0.load_model(str(model_file), device)
    _force_eager_attention(model)
    model_load_time = time.perf_counter() - load_started
    history_size = phase0.infer_history_size(model)
    max_horizon = max(int(value) for value in args.horizons)
    batch = phase0.load_dataset_samples(
        dataset_name=phase0.TASK_DATASETS[task],
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=max(args.future_steps, max_horizon + 1),
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=training_seed,
        device=device,
    )
    spaces = phase0.get_model_spaces(model)
    embedding_space = args.embedding_space or spaces["inference_cost_space"]
    with torch.no_grad():
        clean_embedding, clean_action_embedding, clean_prediction = _encode_rollout(
            model=model,
            batch=batch,
            history_size=history_size,
            horizon=max_horizon,
            embedding_space=embedding_space,
        )

    jvp, probe_rows = _jvp_gaussian_trace(
        model=model,
        batch=batch,
        history_size=history_size,
        horizon=max_horizon,
        embedding_space=embedding_space,
        probes=args.hutchinson_probes,
        seed=training_seed + int(round(float(std_key) * 10000)) + 15485863,
    )
    calibration_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        weights = uniform_horizon_weights(
            max_horizon,
            dtype=clean_prediction.dtype,
            device=clean_prediction.device,
        )
        for sigma in args.small_sigmas:
            noisy_prediction = _noisy_rollouts(
                model=model,
                batch=batch,
                clean_action_embedding=clean_action_embedding,
                history_size=history_size,
                horizon=max_horizon,
                embedding_space=embedding_space,
                sigma=float(sigma),
                draws=args.num_noise_draws,
                seed=training_seed + 1009,
            )
            raw_radius = horizon_weighted_stacked_l2(
                clean_prediction.unsqueeze(1).expand_as(noisy_prediction),
                noisy_prediction,
                weights=weights,
            )
            radius_squared = raw_radius.square()
            measured_mean_r2 = float(radius_squared.mean().detach().cpu().item())
            metrics = calibration_row_metrics(
                measured_mean_r2=measured_mean_r2,
                sigma=float(sigma),
                jvp_trace_per_sequence=float(jvp["jvp_gaussian_trace_per_sequence"]),
            )
            calibration_rows.append(
                {
                    **base,
                    "status": "ok",
                    "sigma": float(sigma),
                    "n_sequences": int(args.n_sequences),
                    "num_noise_draws": int(args.num_noise_draws),
                    "measured_mean_R2": measured_mean_r2,
                    "measured_R2_standard_error": (
                        float(radius_squared.detach().float().std(unbiased=True).cpu().item())
                        / math.sqrt(radius_squared.numel())
                        if radius_squared.numel() > 1
                        else 0.0
                    ),
                    **metrics,
                }
            )

        stress_prediction = _noisy_rollouts(
            model=model,
            batch=batch,
            clean_action_embedding=clean_action_embedding,
            history_size=history_size,
            horizon=max_horizon,
            embedding_space=embedding_space,
            sigma=args.horizon_stress_sigma,
            draws=args.horizon_noise_draws,
            seed=training_seed + 2003,
        )
        horizon_rows = _horizon_rows(
            clean_embedding=clean_embedding,
            clean_prediction=clean_prediction,
            noisy_prediction=stress_prediction,
            history_size=history_size,
            horizons=args.horizons,
            quantiles=args.quantiles,
        )

    for row in probe_rows:
        row.update(base)
    for row in horizon_rows:
        row.update(
            {
                **base,
                "status": "ok",
                "stress_sigma": float(args.horizon_stress_sigma),
                "n_sequences": int(args.n_sequences),
                "num_noise_draws": int(args.horizon_noise_draws),
            }
        )
    checkpoint_row = {
        **base,
        "status": "ok",
        "history_size": int(history_size),
        "n_sequences": int(args.n_sequences),
        "embedding_space": embedding_space,
        "rollout_horizon": max_horizon,
        "horizon_weights": "uniform alpha_k=1/H applied as sqrt(alpha_k)",
        "rollout_projection": "identity in selected embedding_space",
        "rollout_normalization": "none for linearization; canonical per-anchor scale for ATR grid",
        **jvp,
        "calibration_error_mean_relative": _mean(
            [float(row["relative_error"]) for row in calibration_rows]
        ),
        "calibration_error_max_relative": max(
            float(row["relative_error"]) for row in calibration_rows
        ),
        "remainder_loglog_order_descriptive": log_remainder_order(calibration_rows),
        "model_load_time": model_load_time,
        "wall_time": time.perf_counter() - started,
        "timing_unit": "seconds",
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(device)))
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    del model, batch, clean_embedding, clean_action_embedding, clean_prediction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint_row, calibration_rows, horizon_rows, probe_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--full-sweep", type=Path, default=DEFAULT_FULL_SWEEP)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-csv", type=Path, default=None)
    parser.add_argument("--calibration-csv", type=Path, default=None)
    parser.add_argument("--horizon-csv", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--small-sigmas", type=float, nargs="+", default=list(SMALL_SIGMAS))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--quantiles", type=float, nargs="+", default=list(QUANTILES))
    parser.add_argument("--n-sequences", type=int, default=16)
    parser.add_argument("--num-noise-draws", type=int, default=8)
    parser.add_argument("--hutchinson-probes", type=int, default=8)
    parser.add_argument("--horizon-stress-sigma", type=float, default=0.08)
    parser.add_argument("--horizon-noise-draws", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", default=None)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if tuple(sorted(set(args.small_sigmas))) != SMALL_SIGMAS:
        raise ValueError(f"small-sigma grid must be exactly {SMALL_SIGMAS}")
    if tuple(sorted(set(args.horizons))) != HORIZONS:
        raise ValueError(f"horizon grid must be exactly {HORIZONS}")
    if tuple(sorted(set(args.quantiles))) != QUANTILES:
        raise ValueError(f"quantile grid must be exactly {QUANTILES}")
    if args.n_sequences < 1 or args.num_noise_draws < 1 or args.hutchinson_probes < 1:
        raise ValueError("sample/draw/probe counts must be positive")

    full_sweep = _full_sweep_index(args.full_sweep)
    plan = _checkpoint_plan(full_sweep, args.tasks, args.seeds)
    if args.limit is not None:
        plan = plan[: args.limit]
    manifests = {
        seed: json.loads(
            (args.manifest_dir / f"lewm_seed{seed}_evals.json").read_text(encoding="utf-8")
        )
        for seed in args.seeds
    }
    checkpoints: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for index, (task, seed, checkpoint_type, std_key) in enumerate(plan, start=1):
        print(
            f"[{index}/{len(plan)}] {task} seed{seed} {checkpoint_type} std{std_key}",
            flush=True,
        )
        entry = manifests[seed].get(task, {}).get(_manifest_std_key(std_key), {})
        if not entry:
            checkpoints.append(
                {
                    "task": task,
                    "training_seed": seed,
                    "checkpoint_type": checkpoint_type,
                    "std_key": std_key,
                    "status": "skipped_missing_manifest",
                }
            )
            continue
        try:
            checkpoint, calibration, horizons, probes = run_checkpoint(
                task=task,
                training_seed=seed,
                checkpoint_type=checkpoint_type,
                std_key=std_key,
                entry=entry,
                full_sweep_row=full_sweep.get((task, seed, _fmt_rho(std_key)), {}),
                args=args,
            )
        except Exception as exc:  # noqa: BLE001 - preserve audit failures in JSON.
            checkpoints.append(
                {
                    "task": task,
                    "training_seed": seed,
                    "checkpoint_type": checkpoint_type,
                    "std_key": std_key,
                    "status": "error",
                    "error": repr(exc),
                }
            )
            continue
        checkpoints.append(checkpoint)
        calibration_rows.extend(calibration)
        horizon_rows.extend(horizons)
        probe_rows.extend(probes)

    counts: dict[str, int] = {}
    for row in checkpoints:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "code_commit": _git_commit(),
            "status": "complete" if counts and set(counts) == {"ok"} else "partial",
            "status_counts": counts,
            "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "protocol_role": "read-only analysis grid; no threshold or gate retuning",
            "map_contract": {
                "radius_metric": "horizon_weighted_stacked_l2_v2",
                "rollout_horizon": 8,
                "horizon_weights": "uniform alpha_k=1/H applied as sqrt(alpha_k)",
                "projection": "identity in selected embedding_space",
                "normalization": "none for JVP/finite differences; per-anchor clean transition q50 for ATR grid",
                "input_covariance": "pixel-space iid Gaussian mapped by reciprocal ImageNet channel std",
            },
            "limitations": [
                "finite-probe and finite-noise-draw estimates are descriptive Monte Carlo estimates",
                "local linearization is not a global or closed-loop robustness guarantee",
                "horizon/quantile rows use fixed sigma=0.08 and do not tune the frozen gate",
            ],
        },
        "args": vars(args),
        "checkpoint_rows": checkpoints,
        "calibration_rows": calibration_rows,
        "horizon_rows": horizon_rows,
        "probe_rows": probe_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.checkpoint_csv:
        _write_csv(args.checkpoint_csv, checkpoints)
    if args.calibration_csv:
        _write_csv(args.calibration_csv, calibration_rows)
    if args.horizon_csv:
        _write_csv(args.horizon_csv, horizon_rows)
    print(
        f"wrote {args.out_json}: checkpoints={len(checkpoints)} "
        f"calibration={len(calibration_rows)} horizon={len(horizon_rows)} probes={len(probe_rows)}",
        flush=True,
    )
    return 0 if counts and set(counts) == {"ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
