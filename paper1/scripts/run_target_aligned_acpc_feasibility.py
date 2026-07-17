#!/usr/bin/env python3
"""Run the outcome-blind target-aligned ACPC correctness Gate.

The Gate is intentionally small: four independent task episodes, one
eval-matched current-only CEM step-zero pool, identity parity, dataset action
normalization parity, and deterministic simulator replay. It does not produce
a paper-level ACPC result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import paper1_phase0_acpc as phase0
from tools.paper1_operational_decision_audit import load_trajectory_blocks
from tools.paper1_target_aligned_acpc import (
    SCHEMA_VERSION,
    candidate_response_metrics,
    capture_eval_step_zero_pool,
    fit_action_coordinate_stats,
    inverse_eval_action_coordinates,
    planner_pool_costs,
    planner_pool_predictions,
    planner_temporal_semantics,
    replay_action_pool,
    unpack_candidate_action_blocks,
)
from utils import resolve_h5_dataset_path


RESULT_SCHEMA = "paper1-target-aligned-acpc-feasibility-0.2"


def _canonical_task(task: str | None, dataset_name: str) -> str:
    value = str(task or dataset_name).lower().replace("-", "").replace("_", "")
    if value in {"tworoom", "2room"}:
        return "TwoRoom"
    if "pusht" in value:
        return "PushT"
    if "reacher" in value:
        return "Reacher"
    if "cube" in value:
        return "Cube"
    raise ValueError(f"unsupported target-aligned replay task: {task or dataset_name}")


def _task_state_key(task: str) -> str:
    return {
        "TwoRoom": "proprio",
        "PushT": "state",
        "Reacher": "qpos+qvel",
        "Cube": "qpos+qvel",
    }[task]


def _eval_matched_goal_index(
    *,
    current_index: int,
    plan_horizon: int,
    action_block: int,
    frameskip: int,
    num_steps: int,
) -> int:
    low_level_offset = int(plan_horizon) * int(action_block)
    if low_level_offset % int(frameskip):
        raise ValueError(
            "eval goal offset must align with the loaded observation stride"
        )
    goal_index = int(current_index) + low_level_offset // int(frameskip)
    if not int(current_index) < goal_index < int(num_steps):
        raise ValueError(
            "loaded clip does not contain the eval-matched planner goal"
        )
    return goal_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _set_stablewm_home(checkpoint: Path) -> None:
    for parent in checkpoint.parents:
        if parent.name == "ckpt":
            os.environ.setdefault("STABLEWM_HOME", str(parent.parent))
            return


def _raw_replay_inputs(
    h5_path: Path,
    blocks: list[Any],
    *,
    current_index: int,
    goal_index: int,
    num_steps: int,
    frameskip: int,
    plan_horizon: int,
    action_block: int,
    state_key: str | None = None,
    task: str | None = None,
) -> dict[str, np.ndarray]:
    current_states: list[np.ndarray] = []
    goal_states: list[np.ndarray] = []
    logged_actions: list[np.ndarray] = []
    expected_block_states: list[np.ndarray] = []
    expected_block_pixels: list[np.ndarray] = []
    episode_initial_states: list[np.ndarray] = []
    prefix_actions: list[np.ndarray] = []
    absolute_current_rows: list[int] = []
    absolute_goal_rows: list[int] = []
    with h5py.File(h5_path, "r", swmr=True) as stream:
        resolved_task = (
            _canonical_task(task, "")
            if task is not None
            else ("PushT" if state_key == "state" else "TwoRoom")
        )
        resolved_state_key = _task_state_key(resolved_task)
        state_gate_width: int | None = None

        def read_state(rows: int | np.ndarray) -> np.ndarray:
            if resolved_task in {"TwoRoom", "PushT"}:
                key = "state" if resolved_task == "PushT" else "proprio"
                if key not in stream:
                    raise KeyError(
                        f"dataset does not contain replay state {key!r}"
                    )
                return stream[key][rows].copy()
            required = ("qpos", "qvel")
            if any(key not in stream for key in required):
                raise KeyError(
                    f"{resolved_task} replay requires dataset fields {required}"
                )
            return np.concatenate(
                [stream["qpos"][rows], stream["qvel"][rows]],
                axis=-1,
            )

        def read_goal(row: int) -> np.ndarray:
            if resolved_task in {"TwoRoom", "PushT"}:
                key = "state" if resolved_task == "PushT" else "proprio"
                return stream[key][row].copy()
            if resolved_task == "Reacher":
                return stream["qpos"][row].copy()
            required = (
                "privileged_block_0_pos",
                "privileged_block_0_quat",
            )
            if any(key not in stream for key in required):
                raise KeyError(
                    f"Cube replay requires dataset fields {required}"
                )
            return np.concatenate(
                [
                    stream["privileged_block_0_pos"][row],
                    stream["privileged_block_0_quat"][row],
                ],
                axis=-1,
            )

        offsets = stream["ep_offset"][:]
        all_actions = stream["action"][:]
        action_stats = fit_action_coordinate_stats(all_actions)
        for block in blocks:
            offset = int(offsets[int(block.episode_id)])
            current_row = (
                offset + int(block.start_step) + int(current_index) * int(frameskip)
            )
            resolved_goal_index = (
                int(goal_index) if int(goal_index) >= 0 else int(num_steps) + int(goal_index)
            )
            goal_row = (
                offset
                + int(block.start_step)
                + resolved_goal_index * int(frameskip)
            )
            low_level_steps = int(plan_horizon) * int(action_block)
            action_stop = current_row + low_level_steps
            expected_rows = current_row + np.arange(
                1,
                int(plan_horizon) + 1,
                dtype=np.int64,
            ) * int(action_block)
            current_states.append(read_state(current_row))
            goal_states.append(read_goal(goal_row))
            logged_actions.append(stream["action"][current_row:action_stop].copy())
            expected_block_states.append(read_state(expected_rows))
            expected_block_pixels.append(stream["pixels"][expected_rows].copy())
            episode_initial_states.append(read_state(offset))
            prefix_actions.append(stream["action"][offset:current_row].copy())
            absolute_current_rows.append(current_row)
            absolute_goal_rows.append(goal_row)
        if resolved_task == "Cube":
            # Cube contact impulses can differ in qvel under a different
            # MuJoCo warm-start while qpos and every rendered target remain
            # aligned. T2 is defined on RGB futures, so qpos is the
            # goal-relevant privileged configuration Gate; full qpos/qvel
            # error remains reported below.
            state_gate_width = int(stream["qpos"].shape[-1])
        else:
            state_gate_width = int(current_states[0].shape[-1])
    return {
        "current_states": np.stack(current_states),
        "goal_states": np.stack(goal_states),
        "logged_actions": np.stack(logged_actions),
        "expected_block_states": np.stack(expected_block_states),
        "expected_block_pixels": np.stack(expected_block_pixels),
        "episode_initial_states": np.stack(episode_initial_states),
        "prefix_actions": prefix_actions,
        "absolute_current_rows": np.asarray(absolute_current_rows),
        "absolute_goal_rows": np.asarray(absolute_goal_rows),
        "action_stats": action_stats,
        "state_key": resolved_state_key,
        "state_gate_width": state_gate_width,
        "task": resolved_task,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    phase0._ensure_runtime_deps()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _set_stablewm_home(checkpoint)
    task = _canonical_task(args.task, args.dataset_name)
    state_key = _task_state_key(task)
    h5_path = (
        Path(args.dataset_path).expanduser().resolve()
        if args.dataset_path
        else resolve_h5_dataset_path(args.dataset_name).resolve()
    )
    device = str(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = phase0.load_model(str(checkpoint), device).eval()
    model.requires_grad_(False)
    history_size = int(phase0.infer_history_size(model))
    num_steps = history_size + int(args.future_steps)
    current_index = history_size - 1
    goal_index = _eval_matched_goal_index(
        current_index=current_index,
        plan_horizon=int(args.plan_horizon),
        action_block=int(args.action_block),
        frameskip=int(args.frameskip),
        num_steps=num_steps,
    )
    batch, blocks = load_trajectory_blocks(
        dataset_name=args.dataset_name,
        n_blocks=int(args.n_blocks),
        history_size=history_size,
        future_steps=int(args.future_steps),
        frameskip=int(args.frameskip),
        img_size=int(args.img_size),
        seed=int(args.trajectory_seed),
        device=device,
    )
    candidates, captured_costs = capture_eval_step_zero_pool(
        model,
        batch,
        current_index=current_index,
        goal_index=goal_index,
        action_block=int(args.action_block),
        plan_horizon=int(args.plan_horizon),
        candidate_count=int(args.candidate_count),
        topk=int(args.topk),
        batch_size=int(args.batch_size),
        cem_seed=int(args.cem_seed),
        device=device,
    )
    recomputed_costs = planner_pool_costs(
        model,
        batch,
        candidates,
        current_index=current_index,
        goal_index=goal_index,
        device=device,
        batch_size=int(args.batch_size),
    )
    identity_batch = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    identity_costs = planner_pool_costs(
        model,
        identity_batch,
        candidates.clone(),
        current_index=current_index,
        goal_index=goal_index,
        device=device,
        batch_size=int(args.batch_size),
    )
    nominal_predictions = planner_pool_predictions(
        model,
        batch,
        candidates,
        current_index=current_index,
        device=device,
        batch_size=int(args.batch_size),
    )
    identity_predictions = planner_pool_predictions(
        model,
        identity_batch,
        candidates.clone(),
        current_index=current_index,
        device=device,
        batch_size=int(args.batch_size),
    )
    identity_encoder_response = torch.zeros(
        int(args.n_blocks),
        device=nominal_predictions.device,
    )
    identity_responses = candidate_response_metrics(
        nominal_predictions,
        identity_predictions,
        encoder_response=identity_encoder_response,
    )

    raw = _raw_replay_inputs(
        h5_path,
        blocks,
        current_index=current_index,
        goal_index=goal_index,
        num_steps=num_steps,
        frameskip=int(args.frameskip),
        plan_horizon=int(args.plan_horizon),
        action_block=int(args.action_block),
        state_key=state_key,
        task=task,
    )
    logged_pool = raw["logged_actions"][:, None]
    prefix_kwargs = (
        {
            "prefix_initial_states": raw["episode_initial_states"],
            "prefix_actions": raw["prefix_actions"],
        }
        if task == "PushT"
        else {}
    )
    logged_replay = replay_action_pool(
        task,
        raw["current_states"],
        raw["goal_states"],
        logged_pool,
        action_block=int(args.action_block),
        reset_seed=int(args.replay_seed),
        return_pixels=True,
        **prefix_kwargs,
    )
    logged_replay_error = np.linalg.norm(
        logged_replay["block_states"][:, 0] - raw["expected_block_states"],
        axis=-1,
    )
    state_gate_width = int(raw["state_gate_width"])
    logged_replay_gate_error = np.linalg.norm(
        (
            logged_replay["block_states"][:, 0, :, :state_gate_width]
            - raw["expected_block_states"][:, :, :state_gate_width]
        ),
        axis=-1,
    )
    logged_render_difference = np.abs(
        logged_replay["block_pixels"][:, 0].astype(np.int16)
        - raw["expected_block_pixels"].astype(np.int16)
    )
    logged_render_mae = float(logged_render_difference.mean())
    logged_render_rmse = float(
        np.sqrt(np.mean(logged_render_difference.astype(np.float64) ** 2))
    )
    logged_render_psnr = (
        float("inf")
        if logged_render_rmse == 0.0
        else float(20.0 * np.log10(255.0 / logged_render_rmse))
    )

    unpacked = unpack_candidate_action_blocks(
        candidates,
        action_block=int(args.action_block),
        base_action_dim=int(raw["action_stats"].mean.shape[-1]),
    )
    candidate_raw_actions = inverse_eval_action_coordinates(
        unpacked,
        raw["action_stats"],
    )
    candidate_raw_actions_np = candidate_raw_actions.detach().cpu().numpy()
    candidate_replay = replay_action_pool(
        task,
        raw["current_states"],
        raw["goal_states"],
        candidate_raw_actions_np,
        action_block=int(args.action_block),
        reset_seed=int(args.replay_seed),
        **prefix_kwargs,
    )

    captured_recompute_error = float(
        (captured_costs - recomputed_costs).abs().max().item()
    )
    identity_cost_error = float(
        (recomputed_costs - identity_costs).abs().max().item()
    )
    identity_prediction_error = float(
        (nominal_predictions - identity_predictions).abs().max().item()
    )
    identity_response_max = max(
        float(value.abs().max().item())
        for value in identity_responses.values()
    )
    candidate_endpoint_std = candidate_replay["endpoints"].std(axis=1)
    candidate_endpoint_spread = np.linalg.norm(
        candidate_endpoint_std,
        axis=-1,
    )
    candidate_goal_distance_std = candidate_replay["goal_distances"].std(axis=1)
    candidate_outside_env_bounds = np.mean(
        np.abs(candidate_raw_actions_np) > 1.0
    )
    winner_changes = int(
        torch.count_nonzero(
            torch.argmin(recomputed_costs, dim=1)
            != torch.argmin(identity_costs, dim=1)
        ).item()
    )
    tolerance = float(args.invariant_atol)
    replay_tolerance = (
        float(args.replay_atol)
        if args.replay_atol is not None
        else (
            1e-2
            if task == "PushT"
            else (5e-3 if task == "Cube" else 1e-5)
        )
    )
    render_parity = (
        logged_render_mae <= float(args.render_mae_atol)
        and logged_render_psnr >= float(args.render_psnr_min)
    )
    checks = {
        "single_current_frame": True,
        "five_candidate_steps_produce_five_predictions": (
            nominal_predictions.size(2) == int(args.plan_horizon)
        ),
        "captured_vs_recomputed_cost_parity": (
            captured_recompute_error <= tolerance
        ),
        "identity_cost_parity": identity_cost_error <= tolerance,
        "identity_prediction_parity": identity_prediction_error <= tolerance,
        "identity_response_zero": identity_response_max <= tolerance,
        "identity_winner_parity": winner_changes == 0,
        "logged_dataset_replay_parity": (
            float(logged_replay_gate_error.max()) <= replay_tolerance
        ),
        "logged_dataset_render_parity": bool(render_parity),
        "candidate_endpoint_variation": bool(
            np.all(candidate_endpoint_spread > float(args.minimum_endpoint_spread))
        ),
        "candidate_goal_cost_variation": bool(
            np.all(
                candidate_goal_distance_std
                > float(args.minimum_goal_distance_std)
            )
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    semantics = planner_temporal_semantics(
        observation_steps=1,
        candidate_steps=int(args.plan_horizon),
        action_block=int(args.action_block),
    )
    dataset_stat = h5_path.stat()
    return {
        "metadata": {
            "schema_version": RESULT_SCHEMA,
            "primitive_schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "purpose": "DEV correctness/feasibility only; not paper evidence",
            "behavior_labels_read": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "dataset": str(h5_path),
            "dataset_size_bytes": int(dataset_stat.st_size),
            "dataset_mtime_ns": int(dataset_stat.st_mtime_ns),
            "dataset_full_hash_skipped": True,
            "task": task,
            "device": device,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "semantics": {
            **semantics.to_dict(),
            "model_history_size_for_logged_track": history_size,
            "planner_current_index_in_loaded_clip": current_index,
            "planner_goal_index_in_loaded_clip": goal_index,
            "candidate_contains_logged_history_prefix": False,
            "logged_h8_and_candidate_h5_are_same_object": False,
            "privileged_replay_state_key": state_key,
        },
        "blocks": [
            {
                "block_index": int(block.block_index),
                "episode_id": int(block.episode_id),
                "start_step": int(block.start_step),
                "absolute_current_row": int(raw["absolute_current_rows"][index]),
                "absolute_goal_row": int(raw["absolute_goal_rows"][index]),
            }
            for index, block in enumerate(blocks)
        ],
        "checks": checks,
        "measurements": {
            "captured_recomputed_cost_max_abs": captured_recompute_error,
            "identity_cost_max_abs": identity_cost_error,
            "identity_prediction_max_abs": identity_prediction_error,
            "identity_response_max_abs": identity_response_max,
            "identity_winner_changes": winner_changes,
            "logged_replay_error_per_block_step": logged_replay_error,
            "logged_replay_error_max": float(logged_replay_error.max()),
            "logged_replay_gate_error_per_block_step": (
                logged_replay_gate_error
            ),
            "logged_replay_gate_error_max": float(
                logged_replay_gate_error.max()
            ),
            "state_gate_width": state_gate_width,
            "logged_render_max_abs": int(logged_render_difference.max()),
            "logged_render_mae": logged_render_mae,
            "logged_render_rmse": logged_render_rmse,
            "logged_render_psnr_db": logged_render_psnr,
            "logged_render_nonzero_fraction": float(
                np.mean(logged_render_difference != 0)
            ),
            "candidate_endpoint_spread_per_block": candidate_endpoint_spread,
            "candidate_goal_distance_std_per_block": candidate_goal_distance_std,
            "candidate_raw_action_outside_env_bounds_fraction": float(
                candidate_outside_env_bounds
            ),
            "action_coordinates": raw["action_stats"].to_dict(),
            "candidate_shape": list(candidates.shape),
            "prediction_shape": list(nominal_predictions.shape),
            "replay_tolerance": replay_tolerance,
            "render_mae_tolerance": float(args.render_mae_atol),
            "render_psnr_min_db": float(args.render_psnr_min),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", default="tworoom")
    parser.add_argument("--task", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trajectory-seed", type=int, default=9101)
    parser.add_argument("--cem-seed", type=int, default=1234)
    parser.add_argument("--replay-seed", type=int, default=0)
    parser.add_argument("--invariant-atol", type=float, default=1e-5)
    parser.add_argument("--replay-atol", type=float, default=None)
    parser.add_argument("--render-atol", type=int, default=0)
    parser.add_argument("--render-mae-atol", type=float, default=3.0)
    parser.add_argument("--render-psnr-min", type=float, default=35.0)
    parser.add_argument("--minimum-endpoint-spread", type=float, default=1e-3)
    parser.add_argument("--minimum-goal-distance-std", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["metadata"]["status"],
                "out": str(args.out),
                "checks": payload["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if payload["metadata"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
