#!/usr/bin/env python3
"""Run a no-retraining offline CEM trace audit for Paper 1.

The fixed-pool ACPC diagnostics evaluate shared candidate sets. This script
uses the actual stable-worldmodel CEMSolver update loop on existing LeWM
checkpoints and compares clean/noisy replanning traces from the same logged
states and solver random seed. It is an offline planner-side diagnostic, not a
closed-loop environment evaluation and not a full-budget replacement for the
released eval sweeps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.callbacks.common import Callback

from tools import paper1_phase0_acpc as phase0

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_MANIFEST_DIR = DATA_DIR / "training_seed_eval_manifests"
DEFAULT_OUT = DATA_DIR / "cem_trace_audit_20260704.json"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
STD_KEYS = ("0.0", "0.08")


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


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def _pstdev(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    mu = _mean(finite)
    return math.sqrt(sum((v - mu) ** 2 for v in finite) / len(finite))


def _quantile(values: Sequence[float], q: float) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    pos = q * (len(finite) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)


def _success(entry: Mapping[str, Any], metric: str) -> float:
    return float(entry.get("metrics", {}).get(metric, {}).get("mean", float("nan")))


class CEMTraceRecorder(Callback):
    """Record per-step CEM rank and distribution summaries without tensors."""

    name = "CEMTraceRecorder"

    def compute(self, **state: Any) -> dict[str, Any]:
        costs: torch.Tensor = state["costs"].detach().float().cpu()
        topk_vals: torch.Tensor = state["topk_vals"].detach().float().cpu()
        topk_inds: torch.Tensor = state["topk_inds"].detach().cpu()
        var: torch.Tensor = state["var"].detach().float().cpu()
        mean: torch.Tensor = state["mean"].detach().float().cpu()
        return {
            "best_idx": torch.argmin(costs, dim=1).tolist(),
            "topk_inds": topk_inds.tolist(),
            "best_cost": costs.min(dim=1).values.tolist(),
            "elite_cost_mean": topk_vals.mean(dim=1).tolist(),
            "elite_cost_min": topk_vals.min(dim=1).values.tolist(),
            "var_mean": var.flatten(1).mean(dim=1).tolist(),
            "mean_norm": mean.flatten(1).norm(dim=1).tolist(),
        }


def _load_manifest(manifest_dir: Path, seed: int) -> dict[str, Any]:
    return json.loads((manifest_dir / f"lewm_seed{seed}_evals.json").read_text(encoding="utf-8"))


def _solver_info(batch: Mapping[str, torch.Tensor], history_size: int) -> dict[str, torch.Tensor]:
    return {
        "pixels": batch["pixels"][:, :history_size],
        "action": batch["action"][:, :history_size],
        "goal": batch["pixels"][:, -1:],
    }


def _make_solver(model: Any, *, args: argparse.Namespace, n_envs: int, raw_action_dim: int, device: str) -> CEMSolver:
    action_block = int(args.action_block)
    if raw_action_dim % action_block != 0:
        raise ValueError(
            f"raw action dim {raw_action_dim} is not divisible by action_block={action_block}"
        )
    base_action_dim = raw_action_dim // action_block
    solver = CEMSolver(
        model=model,
        batch_size=int(args.batch_size),
        num_samples=int(args.num_samples),
        var_scale=float(args.var_scale),
        n_steps=int(args.n_steps),
        topk=int(args.topk),
        device=device,
        seed=int(args.cem_seed),
        callbacks=[CEMTraceRecorder()],
    )
    action_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(n_envs, base_action_dim),
        dtype=np.float32,
    )
    solver.configure(
        action_space=action_space,
        n_envs=n_envs,
        config=PlanConfig(
            horizon=int(args.plan_horizon),
            receding_horizon=int(args.plan_horizon),
            history_len=int(args.history_size_for_plan),
            action_block=action_block,
            warm_start=False,
        ),
    )
    return solver


def _flatten_trace(history: list[list[dict[str, Any]]], n_steps: int) -> list[dict[str, list[Any]]]:
    steps: list[dict[str, list[Any]]] = [
        {
            "best_idx": [],
            "topk_inds": [],
            "best_cost": [],
            "elite_cost_mean": [],
            "elite_cost_min": [],
            "var_mean": [],
            "mean_norm": [],
        }
        for _ in range(n_steps)
    ]
    for batch in history:
        for step_idx, rec in enumerate(batch):
            if step_idx >= n_steps:
                continue
            for key in steps[step_idx]:
                steps[step_idx][key].extend(rec.get(key, []))
    return steps


def _topk_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    denom = len(sa | sb)
    return len(sa & sb) / denom if denom else float("nan")


def _trace_compare(clean_steps: list[dict[str, list[Any]]], noisy_steps: list[dict[str, list[Any]]], n_steps: int) -> dict[str, Any]:
    step_rows = []
    for idx in range(n_steps):
        c = clean_steps[idx]
        n = noisy_steps[idx]
        count = min(len(c["best_idx"]), len(n["best_idx"]))
        if count == 0:
            continue
        top1_flips = [int(c["best_idx"][i]) != int(n["best_idx"][i]) for i in range(count)]
        topk_j = [_topk_jaccard(c["topk_inds"][i], n["topk_inds"][i]) for i in range(count)]
        best_delta = [abs(float(c["best_cost"][i]) - float(n["best_cost"][i])) for i in range(count)]
        elite_delta = [abs(float(c["elite_cost_mean"][i]) - float(n["elite_cost_mean"][i])) for i in range(count)]
        step_rows.append(
            {
                "step": idx,
                "n_states": count,
                "seeded_top1_flip_rate": _mean([float(v) for v in top1_flips]),
                "seeded_topk_jaccard_mean": _mean(topk_j),
                "best_cost_abs_delta_mean": _mean(best_delta),
                "elite_cost_abs_delta_mean": _mean(elite_delta),
                "clean_var_mean": _mean([float(v) for v in c["var_mean"][:count]]),
                "noisy_var_mean": _mean([float(v) for v in n["var_mean"][:count]]),
            }
        )
    return {
        "step_rows": step_rows,
        "step0": step_rows[0] if step_rows else {},
        "final_step": step_rows[-1] if step_rows else {},
    }


def _plan_shift(clean_actions: torch.Tensor, noisy_actions: torch.Tensor) -> dict[str, float]:
    clean = clean_actions.detach().float().cpu()
    noisy = noisy_actions.detach().float().cpu()
    diff = noisy - clean
    plan_l2 = torch.linalg.vector_norm(diff.flatten(1), dim=1).tolist()
    first_l2 = torch.linalg.vector_norm(diff[:, 0].flatten(1), dim=1).tolist()
    plan_dim = max(1, int(diff[0].numel())) if diff.numel() else 1
    first_dim = max(1, int(diff[:, 0].numel() / max(1, diff.size(0)))) if diff.numel() else 1
    return {
        "final_plan_l2_mean": _mean(plan_l2),
        "final_plan_l2_q50": _quantile(plan_l2, 0.5),
        "final_plan_l2_q90": _quantile(plan_l2, 0.9),
        "final_plan_l2_per_dim_mean": _mean([v / math.sqrt(plan_dim) for v in plan_l2]),
        "first_action_l2_mean": _mean(first_l2),
        "first_action_l2_q50": _quantile(first_l2, 0.5),
        "first_action_l2_per_dim_mean": _mean([v / math.sqrt(first_dim) for v in first_l2]),
    }


def _run_row(task: str, seed: int, std_key: str, entry: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_file, tried = phase0.resolve_model_file(str(entry.get("path", "")), str(entry.get("subdir", "")), [])
    base = {
        "training_seed": int(seed),
        "task": task,
        "std_key": std_key,
        "subdir": entry.get("subdir"),
        "run_path": entry.get("path"),
        "model_file": str(model_file) if model_file else None,
        "clean_success": _success(entry, "clean"),
        "pixels_std0.08_success": _success(entry, "pixels_std0.08"),
        "corruption_drop": _success(entry, "clean") - _success(entry, "pixels_std0.08"),
    }
    if model_file is None:
        return {**base, "status": "skipped_missing_model", "model_search_dirs": tried}
    try:
        phase0._ensure_runtime_deps()
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            model = phase0.load_model(str(model_file), device).eval()
            history_size = phase0.infer_history_size(model)
            args.history_size_for_plan = int(history_size)
            future_steps = max(int(args.future_steps), int(args.plan_horizon) + 1, int(args.rollout_horizon) + 1)
            run_path = Path(str(entry.get("path", ""))).expanduser()
            if run_path.parent.name == "ckpt":
                os.environ["STABLEWM_HOME"] = str(run_path.parent.parent)
            batch = phase0.load_dataset_samples(
                dataset_name=phase0.TASK_DATASETS[task],
                state_key=None,
                n_sequences=int(args.n_sequences),
                history_size=int(history_size),
                future_steps=future_steps,
                frameskip=int(args.frameskip),
                img_size=int(args.img_size),
                seed=int(args.state_seed),
                device=device,
            )
            noisy_batch = phase0.make_paired_noisy_batch(
                batch,
                history_size=int(history_size),
                noise_std=float(args.noise_std),
                seed=int(args.state_seed) + 1009,
                corruption_type=str(args.corruption_type),
                corrupt_goal=False,
            )
            raw_action_dim = int(batch["action"].shape[-1])
            clean_solver = _make_solver(
                model,
                args=args,
                n_envs=int(args.n_sequences),
                raw_action_dim=raw_action_dim,
                device=device,
            )
            noisy_solver = _make_solver(
                model,
                args=args,
                n_envs=int(args.n_sequences),
                raw_action_dim=raw_action_dim,
                device=device,
            )
            clean_out = clean_solver(_solver_info(batch, int(history_size)))
            noisy_out = noisy_solver(_solver_info(noisy_batch, int(history_size)))
            clean_steps = _flatten_trace(clean_out["callbacks"]["CEMTraceRecorder"], int(args.n_steps))
            noisy_steps = _flatten_trace(noisy_out["callbacks"]["CEMTraceRecorder"], int(args.n_steps))
            compare = _trace_compare(clean_steps, noisy_steps, int(args.n_steps))
            plan = _plan_shift(clean_out["actions"], noisy_out["actions"])
        return {
            **base,
            "status": "ok",
            "history_size": int(history_size),
            "n_sequences": int(args.n_sequences),
            "noise_std": float(args.noise_std),
            "corrupt_goal": False,
            "plan_horizon": int(args.plan_horizon),
            "action_block": int(args.action_block),
            "raw_action_dim": raw_action_dim,
            "base_action_dim": raw_action_dim // int(args.action_block),
            "cem_num_samples": int(args.num_samples),
            "cem_n_steps": int(args.n_steps),
            "cem_topk": int(args.topk),
            "cem_seed": int(args.cem_seed),
            **plan,
            "step0_seeded_top1_flip_rate": compare["step0"].get("seeded_top1_flip_rate", float("nan")),
            "step0_seeded_topk_jaccard_mean": compare["step0"].get("seeded_topk_jaccard_mean", float("nan")),
            "step0_best_cost_abs_delta_mean": compare["step0"].get("best_cost_abs_delta_mean", float("nan")),
            "final_seeded_top1_flip_rate": compare["final_step"].get("seeded_top1_flip_rate", float("nan")),
            "final_seeded_topk_jaccard_mean": compare["final_step"].get("seeded_topk_jaccard_mean", float("nan")),
            "final_best_cost_abs_delta_mean": compare["final_step"].get("best_cost_abs_delta_mean", float("nan")),
            "trace_steps": compare["step_rows"],
        }
    except Exception as exc:  # noqa: BLE001 - row-level artifact should record failures.
        return {**base, "status": "error", "error": repr(exc)}


def _summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task in TASKS:
        for std_key in STD_KEYS:
            cells = [r for r in rows if r.get("task") == task and r.get("std_key") == std_key and r.get("status") == "ok"]
            if not cells:
                continue
            summary = {
                "task": task,
                "std_key": std_key,
                "training_seeds": sorted(int(r["training_seed"]) for r in cells),
                "n_training_seeds": len(cells),
            }
            for key in (
                "final_plan_l2_per_dim_mean",
                "first_action_l2_per_dim_mean",
                "step0_seeded_top1_flip_rate",
                "final_seeded_top1_flip_rate",
                "step0_seeded_topk_jaccard_mean",
                "final_seeded_topk_jaccard_mean",
                "step0_best_cost_abs_delta_mean",
                "final_best_cost_abs_delta_mean",
            ):
                vals = [float(r[key]) for r in cells]
                summary[f"{key}_mean"] = _mean(vals)
                summary[f"{key}_pstdev"] = _pstdev(vals)
            out.append(summary)
    return out


def _write_markdown(payload: Mapping[str, Any], out: Path) -> None:
    md = out.with_suffix(".md")
    lines = [
        "# Paper 1 CEM Trace Audit",
        "",
        "Offline clean/noisy CEMSolver trace on fixed LeWM checkpoints. This is a reduced-budget planner-side diagnostic, not a closed-loop evaluation.",
        "",
        "| Task | std | final plan L2/dim | first action L2/dim | step0 top1 flip | final top1 flip | final topk Jaccard | final best-cost delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary_rows", []):
        lines.append(
            "| {task} | {std_key} | {plan:.3f} +/- {plan_s:.3f} | {first:.3f} +/- {first_s:.3f} | {s0:.2f} +/- {s0s:.2f} | {sf:.2f} +/- {sfs:.2f} | {tj:.2f} +/- {tjs:.2f} | {bd:.2f} +/- {bds:.2f} |".format(
                task=row["task"],
                std_key=row["std_key"],
                plan=row["final_plan_l2_per_dim_mean_mean"],
                plan_s=row["final_plan_l2_per_dim_mean_pstdev"],
                first=row["first_action_l2_per_dim_mean_mean"],
                first_s=row["first_action_l2_per_dim_mean_pstdev"],
                s0=row["step0_seeded_top1_flip_rate_mean"],
                s0s=row["step0_seeded_top1_flip_rate_pstdev"],
                sf=row["final_seeded_top1_flip_rate_mean"],
                sfs=row["final_seeded_top1_flip_rate_pstdev"],
                tj=row["final_seeded_topk_jaccard_mean_mean"],
                tjs=row["final_seeded_topk_jaccard_mean_pstdev"],
                bd=row["final_best_cost_abs_delta_mean_mean"],
                bds=row["final_best_cost_abs_delta_mean_pstdev"],
            )
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-sequences", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--state-seed", type=int, default=9101)
    parser.add_argument("--cem-seed", type=int, default=1234)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        manifest = _load_manifest(args.manifest_dir, seed)
        for task in args.tasks:
            for std_key in args.std_keys:
                rows.append(_run_row(task, seed, std_key, manifest[task][std_key], args))
                row = rows[-1]
                print(f"[{row.get('status')}] seed={seed} task={task} std={std_key}")
                if args.limit is not None and len(rows) >= args.limit:
                    break
            if args.limit is not None and len(rows) >= args.limit:
                break
        if args.limit is not None and len(rows) >= args.limit:
            break

    payload = {
        "metadata": {
            "schema_version": "paper1-cem-trace-audit-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "No-retraining offline clean/noisy CEMSolver trace over fixed LeWM checkpoints; reduced CEM budget for planner-side diagnostic only.",
            "training_seeds": list(args.seeds),
            "tasks": list(args.tasks),
            "std_keys": list(args.std_keys),
            "n_sequences": int(args.n_sequences),
            "plan_horizon": int(args.plan_horizon),
            "action_block": int(args.action_block),
            "cem_num_samples": int(args.num_samples),
            "cem_n_steps": int(args.n_steps),
            "cem_topk": int(args.topk),
            "cem_seed": int(args.cem_seed),
            "noise_std": float(args.noise_std),
            "corrupt_goal": False,
            "interpretation": "Compares actual CEM update traces under clean/noisy observations with the same solver seed; it is not an adaptive closed-loop guarantee or full-budget evaluation.",
        },
        "rows": rows,
        "summary_rows": _summary(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, args.out)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"wrote {args.out}")
    print("status counts:", counts)


if __name__ == "__main__":
    main()
