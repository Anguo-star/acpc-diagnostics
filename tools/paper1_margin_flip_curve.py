"""Build Paper 1 margin-conditioned action-flip diagnostics.

This no-retraining audit links the sampled-pool ACPC statement to measured
candidate behavior: for each fixed checkpoint, sample a shared candidate set,
measure the clean top-1/top-2 cost margin, and record whether the clean and
noisy branches choose different top-1 candidates. The released summary reports
flip rates after filtering to high-clean-margin samples within each checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
MANIFEST_DIR = DATA_DIR / "training_seed_eval_manifests"
DEFAULT_OUT = DATA_DIR / "margin_flip_curve_lewm_three_seed.json"
SEEDS = (3072, 3073, 3074)
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
STD_KEYS = ("0.0", "0.08")
THRESHOLD_QUANTILES = (0.0, 0.5, 0.75, 0.9)


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def _safe_pstdev(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan")
    mu = _safe_mean(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / len(clean))


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().float().cpu(), float(q)))


def _success(entry: Mapping[str, Any], metric: str) -> float:
    return float(entry.get("metrics", {}).get(metric, {}).get("mean", float("nan")))


def _costs_for_branch(model, batch: Mapping[str, torch.Tensor], candidates: torch.Tensor, *, method: str, history_size: int) -> torch.Tensor:
    if method == "PLDM":
        return phase0._manual_candidate_costs(model, batch, candidates, history_size=history_size)
    return model.get_cost(phase0._cost_info(batch, history_size), candidates)


def run_checkpoint(*, seed: int, task: str, std_key: str, entry: Mapping[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase0._ensure_runtime_deps()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_roots = [Path(p).expanduser() for p in args.model_root]
    model_file, tried = phase0.resolve_model_file(str(entry.get("path", "")), str(entry.get("subdir", "")), model_roots)
    base = {
        "method": "LeWM",
        "training_seed": int(seed),
        "task": task,
        "std_key": std_key,
        "subdir": entry.get("subdir"),
        "run_path": entry.get("path"),
        "model_file": str(model_file) if model_file else None,
        "model_search_dirs": tried,
        "clean_success": _success(entry, "clean"),
        "pixels_std0.08_success": _success(entry, "pixels_std0.08"),
    }
    if model_file is None:
        return [{**base, "status": "skipped_missing_model"}], []

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
        noisy_batch = phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=seed + 1009,
            corruption_type=args.corruption_type,
            corrupt_goal=False,
        )
        candidates = phase0.build_action_candidates(
            batch["action"],
            history_size=history_size,
            future_steps=args.future_steps,
            random_action_trials=args.random_action_trials,
            seed=seed + 2027,
        )
        clean_costs = _costs_for_branch(model, batch, candidates, method="LeWM", history_size=history_size)
        noisy_costs = _costs_for_branch(model, noisy_batch, candidates, method="LeWM", history_size=history_size)

    clean_cpu = clean_costs.detach().float().cpu()
    noisy_cpu = noisy_costs.detach().float().cpu()
    if clean_cpu.size(1) < 2:
        return [{**base, "status": "error", "error": "candidate_count_lt_2"}], []

    clean_sorted = torch.sort(clean_cpu, dim=1).values
    margins = clean_sorted[:, 1] - clean_sorted[:, 0]
    clean_best = torch.argmin(clean_cpu, dim=1)
    noisy_best = torch.argmin(noisy_cpu, dim=1)
    flips = clean_best != noisy_best

    rows: list[dict[str, Any]] = []
    for q in args.threshold_quantiles:
        threshold = 0.0 if float(q) == 0.0 else _safe_quantile(margins, float(q))
        eligible = margins >= float(threshold)
        eligible_count = int(eligible.sum().item())
        if eligible_count:
            flip_rate = float(flips[eligible].float().mean().item())
            eligible_margin_mean = float(margins[eligible].mean().item())
        else:
            flip_rate = float("nan")
            eligible_margin_mean = float("nan")
        rows.append(
            {
                **base,
                "status": "ok",
                "n_sequences": int(args.n_sequences),
                "candidate_count": int(clean_cpu.size(1)),
                "noise_std": float(args.noise_std),
                "threshold_quantile": float(q),
                "threshold_value": float(threshold),
                "eligible_count": eligible_count,
                "eligible_fraction": eligible_count / float(max(1, margins.numel())),
                "margin_clean_q50": _safe_quantile(margins, 0.5),
                "margin_clean_q75": _safe_quantile(margins, 0.75),
                "margin_clean_q90": _safe_quantile(margins, 0.9),
                "eligible_margin_mean": eligible_margin_mean,
                "flip_rate": flip_rate,
            }
        )

    samples: list[dict[str, Any]] = []
    if args.include_samples:
        for i in range(margins.numel()):
            samples.append(
                {
                    **base,
                    "status": "ok",
                    "sample_index": int(i),
                    "candidate_count": int(clean_cpu.size(1)),
                    "clean_margin": float(margins[i].item()),
                    "clean_best": int(clean_best[i].item()),
                    "noisy_best": int(noisy_best[i].item()),
                    "flip": bool(flips[i].item()),
                }
            )
    return rows, samples


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["task"]), str(row["std_key"]), float(row["threshold_quantile"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (task, std_key, q), group in sorted(groups.items()):
        flips = [float(r["flip_rate"]) for r in group]
        elig = [float(r["eligible_fraction"]) for r in group]
        margins = [float(r["eligible_margin_mean"]) for r in group]
        out.append(
            {
                "task": task,
                "std_key": std_key,
                "threshold_quantile": q,
                "seed_count": len(group),
                "flip_rate_mean": _safe_mean(flips),
                "flip_rate_std": _safe_pstdev(flips),
                "eligible_fraction_mean": _safe_mean(elig),
                "eligible_margin_mean": _safe_mean(margins),
                "clean_success_mean": _safe_mean([float(r["clean_success"]) for r in group]),
                "pixels_std0.08_success_mean": _safe_mean([float(r["pixels_std0.08_success"]) for r in group]),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Paper 1 margin-conditioned flip diagnostics.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    p.add_argument("--eval-manifest-dir", type=Path, default=MANIFEST_DIR)
    p.add_argument("--model-root", action="append", default=[])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--include-samples", action="store_true")
    p.add_argument("--n-sequences", type=int, default=100)
    p.add_argument("--future-steps", type=int, default=9)
    p.add_argument("--rollout-horizon", type=int, default=8)
    p.add_argument("--random-action-trials", type=int, default=64)
    p.add_argument("--threshold-quantiles", type=float, nargs="+", default=list(THRESHOLD_QUANTILES))
    p.add_argument("--noise-std", type=float, default=0.08)
    p.add_argument("--corruption-type", default="gaussian_noise")
    p.add_argument("--state-key", default=None)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for seed in args.seeds:
        manifest = _load(args.eval_manifest_dir / f"lewm_seed{seed}_evals.json")
        for task in args.tasks:
            task_block = manifest.get(task, {})
            for std_key in args.std_keys:
                entry = task_block.get(std_key)
                if entry is None:
                    rows.append({"status": "skipped_missing_manifest", "training_seed": seed, "task": task, "std_key": std_key})
                    continue
                try:
                    new_rows, new_samples = run_checkpoint(seed=seed, task=task, std_key=std_key, entry=entry, args=args)
                    rows.extend(new_rows)
                    samples.extend(new_samples)
                except Exception as exc:  # noqa: BLE001 - release artifact records per-row failures.
                    rows.append({"status": "error", "training_seed": seed, "task": task, "std_key": std_key, "error": repr(exc)})
    payload = {
        "metadata": {
            "schema_version": "paper1-margin-flip-curve-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": list(args.seeds),
            "tasks": list(args.tasks),
            "std_keys": list(args.std_keys),
            "threshold_quantiles": [float(q) for q in args.threshold_quantiles],
            "n_sequences": int(args.n_sequences),
            "candidate_count": int(args.random_action_trials) + 1,
            "note": "No-retraining sample-level clean-margin/top-1 flip audit for fixed LeWM checkpoints.",
        },
        "rows": rows,
        "summary_rows": summarize(rows),
    }
    if args.include_samples:
        payload["sample_rows"] = samples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(payload), indent=2))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[paper1_margin_flip_curve] wrote {args.out}")
    print("[paper1_margin_flip_curve] status counts:", counts)


if __name__ == "__main__":
    main()
