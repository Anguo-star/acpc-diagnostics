#!/usr/bin/env python3
"""ACPC-linked adaptive-CEM stability audit with common randomness."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from stable_worldmodel.solver.callbacks.common import Callback

from tools import paper1_phase0_acpc as phase0
from tools.paper1_cem_trace_audit import _make_solver, _solver_info
from tools.paper1_acpc_planner_stability_audit import (
    PROBE_IDENTITIES,
    ROOT,
    TASK_DATASETS,
    _clone_tensor_mapping,
    _git_commit,
    _jsonable,
    _resolve_checkpoint,
    _sha256,
    _shared_pool_costs,
    _validate_severities,
    load_trajectory_blocks,
)
from tools.paper1_operational_protocol import (
    namespace_arguments,
    validate_frozen_execution,
)


SCHEMA_VERSION = "paper1-acpc-adaptive-cem-audit-1.0"
FROZEN_ARGUMENT_NAMES = (
    "method",
    "task",
    "training_seed",
    "checkpoint_role",
    "anonymous_checkpoint_id",
    "dataset_name",
    "probe_family",
    "severities",
    "draws",
    "n_blocks",
    "future_steps",
    "frameskip",
    "img_size",
    "candidate_count",
    "topk",
    "batch_size",
    "n_steps",
    "plan_horizon",
    "action_block",
    "var_scale",
    "trajectory_seed",
    "probe_seed",
    "cem_seed",
    "first_action_tolerance",
    "alignment_atol",
    "alignment_rtol",
)


def normalized_action_rms(nominal: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    if nominal.shape != probe.shape or nominal.ndim < 2:
        raise ValueError("nominal/probe actions must have the same batched shape")
    return (probe.float() - nominal.float()).flatten(1).square().mean(dim=1).sqrt()


def topk_jaccard(nominal: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    if nominal.shape != probe.shape or nominal.ndim != 2:
        raise ValueError("top-k index tensors must share shape (B,k)")
    scores = []
    for left, right in zip(nominal, probe, strict=True):
        left_set = set(int(value) for value in left.tolist())
        right_set = set(int(value) for value in right.tolist())
        union = left_set | right_set
        scores.append(len(left_set & right_set) / len(union) if union else 1.0)
    return torch.tensor(scores, dtype=torch.float32)


class CompactCEMTrace(Callback):
    """Retain planner-relevant summaries without storing candidate tensors."""

    name = "CompactCEMTrace"

    def __init__(self) -> None:
        super().__init__(reduction="none")

    def compute(self, **state: Any) -> dict[str, torch.Tensor]:
        costs = state["costs"].detach().float().cpu()
        mean = state["mean"].detach().float().cpu()
        var = state["var"].detach().float().cpu()
        return {
            "costs": costs,
            "best_index": torch.argmin(costs, dim=1),
            "topk_indices": state["topk_inds"].detach().cpu(),
            "best_cost": costs.min(dim=1).values,
            "mean": mean,
            "std": var.clamp_min(0.0),
        }


def _flatten_trace(
    history: Sequence[Sequence[Mapping[str, torch.Tensor]]],
    *,
    n_steps: int,
) -> list[dict[str, torch.Tensor]]:
    steps: list[dict[str, list[torch.Tensor]]] = [
        {
            "costs": [],
            "best_index": [],
            "topk_indices": [],
            "best_cost": [],
            "mean": [],
            "std": [],
        }
        for _ in range(int(n_steps))
    ]
    for batch_records in history:
        if len(batch_records) != int(n_steps):
            raise RuntimeError("CEM callback did not retain every optimization step")
        for step, record in enumerate(batch_records):
            for key in steps[step]:
                steps[step][key].append(record[key])
    return [
        {key: torch.cat(values, dim=0) for key, values in record.items()}
        for record in steps
    ]


def compare_adaptive_outputs(
    nominal_actions: torch.Tensor,
    probe_actions: torch.Tensor,
    nominal_trace: Sequence[Mapping[str, torch.Tensor]],
    probe_trace: Sequence[Mapping[str, torch.Tensor]],
    *,
    first_action_tolerance: float,
    alignment_atol: float = 1e-7,
    alignment_rtol: float = 1e-6,
) -> dict[str, Any]:
    if len(nominal_trace) != len(probe_trace) or not nominal_trace:
        raise ValueError("nominal/probe traces must have the same nonzero step count")
    first_rms = normalized_action_rms(
        nominal_actions[:, 0],
        probe_actions[:, 0],
    )
    plan_rms = normalized_action_rms(nominal_actions, probe_actions)
    step_rows = []
    proposal_aligned = torch.ones(
        nominal_actions.size(0), dtype=torch.bool
    )
    for step, (nominal, probe) in enumerate(
        zip(nominal_trace, probe_trace, strict=True)
    ):
        nominal_costs = nominal["costs"].float()
        probe_costs = probe["costs"].float()
        if nominal_costs.shape != probe_costs.shape or nominal_costs.ndim != 2:
            raise ValueError("paired CEM costs must share shape (B,K)")
        if nominal_costs.size(1) <= nominal["topk_indices"].size(1):
            raise ValueError("CEM topk must be strictly smaller than candidate count")
        absolute_cost_drift = (probe_costs - nominal_costs).abs()
        max_cost_drift = absolute_cost_drift.max(dim=1).values
        ordered_nominal = torch.sort(
            nominal_costs, dim=1, stable=True
        ).values
        topk_count = int(nominal["topk_indices"].size(1))
        elite_boundary_margin = (
            ordered_nominal[:, topk_count]
            - ordered_nominal[:, topk_count - 1]
        )
        elite_jaccard = topk_jaccard(
            nominal["topk_indices"], probe["topk_indices"]
        )
        elite_membership_same = elite_jaccard == 1.0
        elite_order_same = (
            nominal["topk_indices"] == probe["topk_indices"]
        ).all(dim=1)
        common_drift_certificate = proposal_aligned & (
            elite_boundary_margin > 2.0 * max_cost_drift
        )
        if bool(common_drift_certificate.logical_and(
            ~elite_membership_same
        ).any()):
            raise RuntimeError(
                "aligned-pool elite certificate passed with changed membership"
            )
        mean_close = torch.isclose(
            nominal["mean"],
            probe["mean"],
            atol=float(alignment_atol),
            rtol=float(alignment_rtol),
        ).flatten(1).all(dim=1)
        std_close = torch.isclose(
            nominal["std"],
            probe["std"],
            atol=float(alignment_atol),
            rtol=float(alignment_rtol),
        ).flatten(1).all(dim=1)
        distribution_update_aligned = (
            proposal_aligned & mean_close & std_close
        )
        step_rows.append(
            {
                "step": step,
                "proposal_aligned_by_induction": proposal_aligned.clone(),
                "top1_agreement": nominal["best_index"] == probe["best_index"],
                "elite_jaccard": elite_jaccard,
                "elite_membership_same": elite_membership_same,
                "elite_order_same": elite_order_same,
                "nominal_elite_boundary_margin": elite_boundary_margin,
                "max_aligned_cost_drift": max_cost_drift,
                "aligned_pool_elite_certificate": common_drift_certificate,
                "distribution_update_aligned": distribution_update_aligned,
                "mean_rms_drift": normalized_action_rms(
                    nominal["mean"], probe["mean"]
                ),
                "std_rms_drift": normalized_action_rms(
                    nominal["std"], probe["std"]
                ),
                "best_cost_absolute_drift": (
                    probe["best_cost"] - nominal["best_cost"]
                ).abs(),
            }
        )
        proposal_aligned = distribution_update_aligned
    return {
        "first_action_rms": first_rms,
        "full_plan_rms": plan_rms,
        "first_action_stable": first_rms <= float(first_action_tolerance),
        "all_steps_distribution_aligned": proposal_aligned,
        "step_rows": step_rows,
    }


def _set_stablewm_home(checkpoint: Path) -> None:
    for parent in checkpoint.parents:
        if parent.name == "ckpt":
            os.environ.setdefault("STABLEWM_HOME", str(parent.parent))
            return


def _solve(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    args: argparse.Namespace,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    callback = CompactCEMTrace()
    solver_args = argparse.Namespace(
        action_block=int(args.action_block),
        batch_size=int(args.batch_size),
        num_samples=int(args.candidate_count),
        var_scale=float(args.var_scale),
        n_steps=int(args.n_steps),
        topk=int(args.topk),
        cem_seed=int(seed),
        plan_horizon=int(args.plan_horizon),
        history_size_for_plan=int(history_size),
    )
    solver = _make_solver(
        model,
        args=solver_args,
        n_envs=int(batch["pixels"].shape[0]),
        raw_action_dim=int(batch["action"].shape[-1]),
        device=device,
    )
    solver.callbacks = [callback]
    initial_action = torch.zeros(
        int(batch["pixels"].shape[0]),
        int(args.plan_horizon),
        int(batch["action"].shape[-1]),
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    outputs = solver(
        _solver_info(batch, int(history_size)),
        init_action=initial_action,
    )
    trace = _flatten_trace(
        outputs["callbacks"][callback.output_key],
        n_steps=int(args.n_steps),
    )
    return outputs["actions"].detach().float().cpu(), trace


@torch.inference_mode()
def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    phase0._ensure_runtime_deps()
    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    severities = _validate_severities(args.probe_family, args.severities)
    if int(args.n_steps) < 1:
        raise ValueError("n_steps must be positive")
    if not 0.0 <= float(args.first_action_tolerance):
        raise ValueError("first_action_tolerance must be nonnegative")
    if float(args.alignment_atol) < 0.0 or float(args.alignment_rtol) < 0.0:
        raise ValueError("alignment tolerances must be nonnegative")
    checkpoint = _resolve_checkpoint(
        args.checkpoint,
        [Path(root).expanduser() for root in args.model_root],
    )
    _set_stablewm_home(checkpoint)
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    model = phase0.load_model(str(checkpoint), device).eval()
    history_size = int(phase0.infer_history_size(model))
    batch, blocks = load_trajectory_blocks(
        dataset_name=args.dataset_name or TASK_DATASETS[args.task],
        n_blocks=int(args.n_blocks),
        history_size=history_size,
        future_steps=int(args.future_steps),
        frameskip=int(args.frameskip),
        img_size=int(args.img_size),
        seed=int(args.trajectory_seed),
        device=device,
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    nominal_by_draw: dict[int, tuple[torch.Tensor, list[dict[str, torch.Tensor]]]] = {}
    for draw_index in range(int(args.draws)):
        cem_seed = int(args.cem_seed) + 7919 * draw_index
        nominal_by_draw[draw_index] = _solve(
            model,
            batch,
            history_size=history_size,
            args=args,
            seed=cem_seed,
            device=device,
        )
        for severity_index, severity in enumerate(severities):
            probe_seed = int(args.probe_seed) + 1009 * severity_index + 7919 * draw_index
            try:
                probe_batch = phase0.make_paired_noisy_batch(
                    _clone_tensor_mapping(batch),
                    history_size=history_size,
                    noise_std=float(severity),
                    seed=probe_seed,
                    corruption_type=args.probe_family,
                    corrupt_goal=False,
                )
                probe_actions, probe_trace = _solve(
                    model,
                    probe_batch,
                    history_size=history_size,
                    args=args,
                    seed=cem_seed,
                    device=device,
                )
                nominal_actions, nominal_trace = nominal_by_draw[draw_index]
                comparison = compare_adaptive_outputs(
                    nominal_actions,
                    probe_actions,
                    nominal_trace,
                    probe_trace,
                    first_action_tolerance=float(args.first_action_tolerance),
                    alignment_atol=float(args.alignment_atol),
                    alignment_rtol=float(args.alignment_rtol),
                )
                nominal_plan_cost = _shared_pool_costs(
                    model,
                    batch,
                    nominal_actions.unsqueeze(1),
                    history_size=history_size,
                    device=device,
                ).squeeze(1)
                probe_plan_cost_on_nominal = _shared_pool_costs(
                    model,
                    batch,
                    probe_actions.unsqueeze(1),
                    history_size=history_size,
                    device=device,
                ).squeeze(1)
                decision_regret = (
                    probe_plan_cost_on_nominal - nominal_plan_cost
                )
                for index, block in enumerate(blocks):
                    per_step = []
                    for step in comparison["step_rows"]:
                        per_step.append(
                            {
                                "step": int(step["step"]),
                                "proposal_aligned_by_induction": bool(
                                    step["proposal_aligned_by_induction"][index]
                                ),
                                "top1_agreement": bool(step["top1_agreement"][index]),
                                "elite_jaccard": float(step["elite_jaccard"][index]),
                                "elite_membership_same": bool(
                                    step["elite_membership_same"][index]
                                ),
                                "elite_order_same": bool(
                                    step["elite_order_same"][index]
                                ),
                                "nominal_elite_boundary_margin": float(
                                    step["nominal_elite_boundary_margin"][index]
                                ),
                                "max_aligned_cost_drift": float(
                                    step["max_aligned_cost_drift"][index]
                                ),
                                "aligned_pool_elite_certificate": bool(
                                    step["aligned_pool_elite_certificate"][index]
                                ),
                                "distribution_update_aligned": bool(
                                    step["distribution_update_aligned"][index]
                                ),
                                "mean_rms_drift": float(step["mean_rms_drift"][index]),
                                "std_rms_drift": float(step["std_rms_drift"][index]),
                                "best_cost_absolute_drift": float(
                                    step["best_cost_absolute_drift"][index]
                                ),
                            }
                        )
                    rows.append(
                        {
                            "method": args.method,
                            "task": args.task,
                            "training_seed": int(args.training_seed),
                            "checkpoint_role": args.checkpoint_role,
                            "anonymous_checkpoint_id": args.anonymous_checkpoint_id,
                            "checkpoint_sha256": _sha256(checkpoint),
                            "trajectory_block_index": block.block_index,
                            "trajectory_block_id": block.block_id,
                            "episode_id": block.episode_id,
                            "start_step": block.start_step,
                            "probe_family": args.probe_family,
                            "severity_index": severity_index,
                            "severity": float(severity),
                            "draw_index": draw_index,
                            "probe_seed": probe_seed,
                            "cem_seed": cem_seed,
                            "candidate_count": int(args.candidate_count),
                            "n_steps": int(args.n_steps),
                            "first_action_metric": "RMS in dataset-normalized action coordinates",
                            "first_action_tolerance": float(
                                args.first_action_tolerance
                            ),
                            "first_action_rms": float(
                                comparison["first_action_rms"][index]
                            ),
                            "full_plan_rms": float(comparison["full_plan_rms"][index]),
                            "nominal_plan_clean_cost": float(
                                nominal_plan_cost[index]
                            ),
                            "probe_plan_clean_cost": float(
                                probe_plan_cost_on_nominal[index]
                            ),
                            "clean_decision_regret": float(
                                decision_regret[index]
                            ),
                            "positive_clean_decision_regret": float(
                                decision_regret[index].clamp_min(0.0)
                            ),
                            "first_action_stable": bool(
                                comparison["first_action_stable"][index]
                            ),
                            "step0_top1_agreement": per_step[0]["top1_agreement"],
                            "final_elite_jaccard": per_step[-1]["elite_jaccard"],
                            "all_steps_distribution_aligned": bool(
                                comparison["all_steps_distribution_aligned"][index]
                            ),
                            "per_step": per_step,
                            "common_random_numbers": True,
                            "branches_adapt_independently": True,
                            "behavior_blind": True,
                            "status": "ok",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - retain attempted cells.
                errors.append(
                    {
                        "severity_index": severity_index,
                        "severity": float(severity),
                        "draw_index": draw_index,
                        "error": repr(exc),
                    }
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated(torch.device(device)))
    else:
        peak_memory = 0
    expected_rows = len(blocks) * len(severities) * int(args.draws)
    total_solves = int(args.draws) * (1 + len(severities))
    candidate_calls = (
        len(blocks)
        * int(args.candidate_count)
        * int(args.n_steps)
        * total_solves
    )
    script_path = Path(__file__).resolve()
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "protocol_path": str(args.protocol),
            "protocol_sha256": _sha256(Path(args.protocol)),
            "execution_addendum_path": str(args.execution_addendum),
            "execution_addendum_sha256": _sha256(Path(args.execution_addendum)),
            "status": (
                "complete" if not errors and len(rows) == expected_rows else "partial"
            ),
            "expected_rows": expected_rows,
            "actual_rows": len(rows),
            "error_count": len(errors),
            "decision_target": "adaptive_cem_final_first_action",
            "first_action_metric": "RMS in dataset-normalized action coordinates",
            "first_action_tolerance": float(args.first_action_tolerance),
            "common_random_numbers": True,
            "common_zero_initial_proposal": True,
            "branches_adapt_independently": True,
            "alignment_atol": float(args.alignment_atol),
            "alignment_rtol": float(args.alignment_rtol),
            "method": args.method,
            "task": args.task,
            "training_seed": int(args.training_seed),
            "checkpoint_role": args.checkpoint_role,
            "anonymous_checkpoint_id": args.anonymous_checkpoint_id,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "n_trajectory_blocks": len(blocks),
            "history_size": history_size,
            "probe_family": args.probe_family,
            "severities": severities,
            "draws": int(args.draws),
            "candidate_count": int(args.candidate_count),
            "topk": int(args.topk),
            "n_steps": int(args.n_steps),
            "plan_horizon": int(args.plan_horizon),
            "candidate_forward_calls": candidate_calls,
            "wall_time_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_memory,
        },
        "errors": errors,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["LeWM", "PLDM"], required=True)
    parser.add_argument("--task", choices=sorted(TASK_DATASETS), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint-role", required=True)
    parser.add_argument("--anonymous-checkpoint-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--probe-family", choices=sorted(PROBE_IDENTITIES), required=True)
    parser.add_argument("--severities", type=float, nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--n-blocks", type=int, default=16)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--trajectory-seed", type=int, default=9101)
    parser.add_argument("--probe-seed", type=int, default=20260712)
    parser.add_argument("--cem-seed", type=int, default=1234)
    parser.add_argument("--first-action-tolerance", type=float, default=0.10)
    parser.add_argument("--alignment-atol", type=float, default=1e-7)
    parser.add_argument("--alignment-rtol", type=float, default=1e-6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = _resolve_checkpoint(
        args.checkpoint,
        [Path(root).expanduser() for root in args.model_root],
    )
    _set_stablewm_home(checkpoint)
    args.checkpoint = str(checkpoint)
    validate_frozen_execution(
        protocol_path=args.protocol,
        addendum_path=args.execution_addendum,
        runner_path=Path(__file__),
        shard_id=args.shard_id,
        checkpoint_path=checkpoint,
        output_path=args.out,
        arguments=namespace_arguments(args, FROZEN_ARGUMENT_NAMES),
    )
    payload = run_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if payload["metadata"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
