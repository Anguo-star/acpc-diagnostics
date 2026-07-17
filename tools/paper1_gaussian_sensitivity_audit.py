#!/usr/bin/env python3
"""Finite-difference Gaussian sensitivity analysis for Paper 1.

The audit supports the local sensitivity interpretation only. It estimates
E[R_sigma^2] / sigma^2 for small image-noise sigma values on fixed checkpoints
and does not claim a global robustness certificate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_margin_flip_curve import MANIFEST_DIR, SEEDS, TASKS, _success

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_SWEEP = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_JSON = ROOT / "paper1" / "results" / "gaussian_sensitivity_audit.json"
DEFAULT_CSV = ROOT / "paper1" / "results" / "gaussian_sensitivity_audit.csv"
DEFAULT_SUMMARY = ROOT / "paper1" / "results" / "gaussian_sensitivity_summary.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_gaussian_sensitivity_audit.tex"
SMALL_SIGMAS = (0.0025, 0.005, 0.01, 0.02)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
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
        return obj.detach().cpu().tolist()
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


def _fmt_rho(value: Any) -> str:
    return f"{float(value):.2f}"


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


def _q(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values.detach().float().cpu(), q).item())


def _full_sweep_index(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    return {
        (row["task"], int(row["training_seed"]), _fmt_rho(row["rho"])): row
        for row in _read_csv(path)
    }


def _checkpoint_plan(full_sweep: dict[tuple[str, int, str], dict[str, str]], tasks: Sequence[str], seeds: Sequence[int]) -> list[tuple[str, int, str, str]]:
    plan: list[tuple[str, int, str, str]] = []
    for task in tasks:
        for seed in seeds:
            rows = [row for (t, s, _rho), row in full_sweep.items() if t == task and s == seed]
            rows = sorted(rows, key=lambda r: _f(r["rho"]))
            recovered = [row for row in rows if str(row.get("recovery_label", "")).lower() == "true"]
            onset = _fmt_rho(recovered[0]["rho"]) if recovered else "0.08"
            seen = set()
            for kind, rho in (("base", "0.00"), ("onset", onset), ("endpoint", "0.08")):
                key = (kind, rho)
                if key in seen:
                    continue
                seen.add(key)
                plan.append((task, seed, kind, rho))
    return plan


def _resolve(entry: Mapping[str, Any], model_roots: Sequence[Path]) -> tuple[Path | None, list[str]]:
    return phase0.resolve_model_file(str(entry.get("path", "")), str(entry.get("subdir", "")), model_roots)


def _rollout_radius(
    model,
    batch: Mapping[str, torch.Tensor],
    noisy_batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    rollout_horizon: int,
    embedding_space: str,
) -> torch.Tensor:
    clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
    noisy_outputs = phase0.encode_sequences(model, phase0._clone_batch(noisy_batch))
    clean_emb = phase0.get_embedding_space(clean_outputs, embedding_space).detach()
    noisy_emb = phase0.get_embedding_space(noisy_outputs, embedding_space).detach()
    act_emb = clean_outputs["act_emb"].detach()
    max_steps = min(rollout_horizon, max(0, act_emb.size(1) - history_size + 1))
    clean_chain = phase0._autoregressive_rollout(model, clean_emb[:, :history_size], act_emb, history_size, max_steps)
    noisy_chain = phase0._autoregressive_rollout(model, noisy_emb[:, :history_size], act_emb, history_size, max_steps)
    clean_pred = clean_chain[:, history_size : history_size + max_steps]
    noisy_pred = noisy_chain[:, history_size : history_size + max_steps]
    return torch.linalg.vector_norm((clean_pred - noisy_pred).reshape(clean_pred.size(0), -1), dim=-1)


def run_checkpoint(
    *,
    task: str,
    seed: int,
    checkpoint_type: str,
    std_key: str,
    entry: Mapping[str, Any],
    full_sweep_row: Mapping[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    phase0._ensure_runtime_deps()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_roots = [Path(p).expanduser() for p in args.model_root]
    model_file, tried = _resolve(entry, model_roots)
    base = {
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
        return [{**base, "status": "skipped_missing_model"}]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        model = phase0.load_model(str(model_file), device)
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
        spaces = phase0.get_model_spaces(model)
        embedding_space = args.embedding_space or spaces["inference_cost_space"]
        for sigma in args.small_sigmas:
            draw_radii = []
            for draw in range(args.num_noise_draws):
                noisy = phase0.make_paired_noisy_batch(
                    batch,
                    history_size=history_size,
                    noise_std=float(sigma),
                    seed=seed + 1009 + 7919 * draw,
                    corruption_type=args.corruption_type,
                    corrupt_goal=False,
                )
                draw_radii.append(
                    _rollout_radius(
                        model,
                        batch,
                        noisy,
                        history_size=history_size,
                        rollout_horizon=args.rollout_horizon,
                        embedding_space=embedding_space,
                    )
                )
            radius = torch.cat(draw_radii, dim=0)
            r2 = radius.square()
            rows.append({
                **base,
                "status": "ok",
                "finite_difference_only": "true",
                "history_size": int(history_size),
                "n_sequences": int(args.n_sequences),
                "num_noise_draws": int(args.num_noise_draws),
                "rollout_horizon": int(args.rollout_horizon),
                "embedding_space": embedding_space,
                "sigma_small": float(sigma),
                "mean_radius": float(radius.mean().detach().cpu().item()),
                "radius_q50": _q(radius, 0.50),
                "radius_q90": _q(radius, 0.90),
                "mean_R2": float(r2.mean().detach().cpu().item()),
                "mean_R2_over_sigma2": float(r2.mean().detach().cpu().item()) / (float(sigma) ** 2),
                "radius_q90_over_sigma": _q(radius, 0.90) / float(sigma),
                "notes": "finite-difference local Gaussian sensitivity proxy; not a global robustness certificate",
            })
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed_type: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            by_seed_type[(row["task"], int(row["training_seed"]), row["checkpoint_type"])].append(row)
    collapsed = []
    for (task, seed, kind), block in by_seed_type.items():
        collapsed.append({
            "task": task,
            "training_seed": seed,
            "checkpoint_type": kind,
            "std_key": block[0]["std_key"],
            "n_sequences": block[0]["n_sequences"],
            "num_noise_draws": block[0]["num_noise_draws"],
            "sensitivity_slope_median": _median([r["mean_R2_over_sigma2"] for r in block]),
            "sensitivity_slope_pstdev_over_sigmas": _pstdev([r["mean_R2_over_sigma2"] for r in block]),
            "radius_q90_over_sigma_median": _median([r["radius_q90_over_sigma"] for r in block]),
            "atr_q90": block[0]["atr_q90"],
            "atr_normalized_q90": block[0]["atr_normalized_q90"],
        })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in collapsed:
        grouped[(row["task"], row["checkpoint_type"])].append(row)
    out = []
    for task in TASKS:
        base = grouped.get((task, "base"), [])
        endpoint = grouped.get((task, "endpoint"), [])
        base_slope = _mean([r["sensitivity_slope_median"] for r in base])
        end_slope = _mean([r["sensitivity_slope_median"] for r in endpoint])
        for kind in ("base", "onset", "endpoint"):
            block = grouped.get((task, kind), [])
            if not block:
                continue
            slope = _mean([r["sensitivity_slope_median"] for r in block])
            out.append({
                "task": task,
                "checkpoint_type": kind,
                "n_training_seeds": len(block),
                "n_sequences": int(_median([r["n_sequences"] for r in block])),
                "num_noise_draws": int(_median([r["num_noise_draws"] for r in block])),
                "std_key_median": _median([r["std_key"] for r in block]),
                "sensitivity_slope_mean": slope,
                "sensitivity_slope_pstdev": _pstdev([r["sensitivity_slope_median"] for r in block]),
                "sensitivity_slope_vs_base": slope / base_slope if math.isfinite(base_slope) and base_slope > 0 else math.nan,
                "endpoint_slope_vs_base": end_slope / base_slope if math.isfinite(base_slope) and base_slope > 0 else math.nan,
                "radius_q90_over_sigma_mean": _mean([r["radius_q90_over_sigma_median"] for r in block]),
                "atr_q90_mean": _mean([r["atr_q90"] for r in block]),
                "atr_normalized_q90_mean": _mean([r["atr_normalized_q90"] for r in block]),
                "notes": "median over small sigmas per seed, then mean over training seeds",
            })
    return out


def _sci(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.2e}"


def _num(x: float, digits: int = 2) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def write_table(path: Path, summary: list[dict[str, Any]]) -> None:
    by = {(row["task"], row["checkpoint_type"]): row for row in summary}
    n_sequences = sorted({int(_f(row.get("n_sequences"))) for row in summary if math.isfinite(_f(row.get("n_sequences")))})
    noise_draws = sorted({int(_f(row.get("num_noise_draws"))) for row in summary if math.isfinite(_f(row.get("num_noise_draws")))})
    seq_desc = str(n_sequences[0]) if len(n_sequences) == 1 else "/".join(str(x) for x in n_sequences)
    draw_desc = str(noise_draws[0]) if len(noise_draws) == 1 else "/".join(str(x) for x in noise_draws)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Finite-difference Gaussian sensitivity analysis. The slope is the mean small-noise rollout radius squared divided by $\sigma^2$, using {seq_desc} sampled sequences and {draw_desc} noise draws per small $\sigma$ and checkpoint, then summarized over the reported small-noise probes and training seeds. Lower endpoint/base ratios indicate reduced local composed encoder--predictor sensitivity. This is a local finite-difference proxy, not a global robustness guarantee.}}",
        r"\label{tab:gaussian-sensitivity-audit}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Task & slope base $\to$ endpoint & endpoint/base & ATR base $\to$ endpoint \\",
        r"\midrule",
    ]
    for task in TASKS:
        base = by.get((task, "base"), {})
        endpoint = by.get((task, "endpoint"), {})
        lines.append(
            f"{task} & "
            f"{_sci(_f(base.get('sensitivity_slope_mean')))} $\\to$ {_sci(_f(endpoint.get('sensitivity_slope_mean')))} & "
            f"{_num(_f(endpoint.get('sensitivity_slope_vs_base')), 3)} & "
            f"{_num(_f(base.get('atr_q90_mean')))} $\\to$ {_num(_f(endpoint.get('atr_q90_mean')))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--full-sweep", type=Path, default=DEFAULT_FULL_SWEEP)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--small-sigmas", type=float, nargs="+", default=list(SMALL_SIGMAS))
    parser.add_argument("--num-noise-draws", type=int, default=2)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", default=None)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    full_sweep = _full_sweep_index(args.full_sweep)
    plan = _checkpoint_plan(full_sweep, args.tasks, args.seeds)
    if args.limit is not None:
        plan = plan[: args.limit]
    rows: list[dict[str, Any]] = []
    manifests = {seed: _load_json(args.manifest_dir / f"lewm_seed{seed}_evals.json") for seed in args.seeds}
    for idx, (task, seed, kind, std_key) in enumerate(plan, start=1):
        print(f"[{idx}/{len(plan)}] {task} seed{seed} {kind} std{std_key}", flush=True)
        entry = manifests[seed][task][str(float(std_key))]
        try:
            rows.extend(
                run_checkpoint(
                    task=task,
                    seed=seed,
                    checkpoint_type=kind,
                    std_key=std_key,
                    entry=entry,
                    full_sweep_row=full_sweep.get((task, seed, std_key), {}),
                    args=args,
                )
            )
        except Exception as exc:  # noqa: BLE001 - audit records row-level failures.
            rows.append({"status": "error", "task": task, "training_seed": seed, "checkpoint_type": kind, "std_key": std_key, "error": repr(exc)})
    summary = build_summary(rows)
    payload = {
        "metadata": {
            "schema_version": "paper1-gaussian-sensitivity-audit-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": list(args.tasks),
            "seeds": list(args.seeds),
            "small_sigmas": list(args.small_sigmas),
            "num_noise_draws": int(args.num_noise_draws),
            "n_sequences": int(args.n_sequences),
            "note": "Finite-difference local Gaussian sensitivity proxy; no global guarantee is claimed.",
        },
        "rows": rows,
        "summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _write_csv(args.out_csv, rows)
    _write_csv(args.summary, summary)
    write_table(args.table, summary)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.table}")
    print("status counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
