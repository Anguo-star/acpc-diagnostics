#!/usr/bin/env python3
"""Run support-matched DEV adjudication for the Paper1 ACPC mechanism.

This is an outcome-blind MVE, not a submission result. Logged H1/H8 signals
are evaluated only against logged held-out futures. Planner H1/H5 signals are
evaluated only against current-only CEM candidates and simulator futures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import stable_pretraining as spt
import torch
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper1.scripts.run_target_aligned_acpc_feasibility import (  # noqa: E402
    _canonical_task,
    _eval_matched_goal_index,
    _jsonable,
    _raw_replay_inputs,
    _set_stablewm_home,
    _task_state_key,
)
from tools import paper1_phase0_acpc as phase0  # noqa: E402
from tools.paper1_acpc_metrics import (  # noqa: E402
    horizon_weighted_stacked_l2,
)
from tools.paper1_trajectory_blocks import (  # noqa: E402
    load_trajectory_blocks,
)
from tools.paper1_target_aligned_acpc import (  # noqa: E402
    candidate_response_metrics,
    capture_eval_step_zero_pool,
    inverse_eval_action_coordinates,
    permute_candidate_actions,
    permute_candidate_time,
    planner_pool_costs,
    planner_pool_predictions,
    prediction_error_drift_certificate,
    replay_action_pool,
    unpack_candidate_action_blocks,
)
from utils import get_img_preprocessor, resolve_h5_dataset_path  # noqa: E402


RESULT_SCHEMA = "paper1-target-aligned-acpc-mve-0.3"
REPLAY_CACHE_SCHEMA = "paper1-target-aligned-replay-cache-0.1"
CONTROL_NAMES = ("correct", "action_zero", "candidate_shuffle", "time_shuffle")
CANDIDATE_BASE_FEATURES = (
    "severity",
    "encoder_response",
    "candidate_action_rms",
    "nominal_h1_displacement",
    "nominal_h5_displacement",
    "nominal_cost_margin",
    "encoder_x_action_rms",
    "encoder_x_nominal_h5_displacement",
)
CANDIDATE_FEATURE_SETS = {
    "strong_simple": CANDIDATE_BASE_FEATURES,
    "plus_correct_h1": CANDIDATE_BASE_FEATURES
    + ("correct_h1_response",),
    "plus_correct_h5": CANDIDATE_BASE_FEATURES
    + ("correct_h1_response", "correct_h5_response"),
    "plus_action_zero_h5_control": CANDIDATE_BASE_FEATURES
    + ("correct_h1_response", "action_zero_h5_response"),
    "plus_candidate_shuffle_h5_control": CANDIDATE_BASE_FEATURES
    + ("correct_h1_response", "candidate_shuffle_h5_response"),
    "plus_time_shuffle_h5_control": CANDIDATE_BASE_FEATURES
    + ("correct_h1_response", "time_shuffle_h5_response"),
}
LOGGED_BASE_FEATURES = ("severity", "encoder_response")
LOGGED_FEATURE_SETS = {
    "encoder": LOGGED_BASE_FEATURES,
    "plus_correct_h1": LOGGED_BASE_FEATURES
    + ("correct_h1_response",),
    "plus_correct_h8": LOGGED_BASE_FEATURES
    + ("correct_h1_response", "correct_h8_response"),
    "plus_action_zero_h8_control": LOGGED_BASE_FEATURES
    + ("correct_h1_response", "action_zero_h8_response"),
    "plus_candidate_shuffle_h8_control": LOGGED_BASE_FEATURES
    + ("correct_h1_response", "candidate_shuffle_h8_response"),
    "plus_time_shuffle_h8_control": LOGGED_BASE_FEATURES
    + ("correct_h1_response", "time_shuffle_h8_response"),
}


def _replay_cache_key(
    *,
    task: str,
    blocks: list[Any],
    current_states: np.ndarray,
    goal_states: np.ndarray,
    candidate_raw_actions: np.ndarray,
    action_block: int,
    replay_seed: int,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "schema": REPLAY_CACHE_SCHEMA,
        "task": str(task),
        "blocks": [
            {
                "episode_id": int(block.episode_id),
                "start_step": int(block.start_step),
            }
            for block in blocks
        ],
        "action_block": int(action_block),
        "replay_seed": int(replay_seed),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for array in (current_states, goal_states, candidate_raw_actions):
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_replay_cache(
    path: Path,
    *,
    expected_key: str,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        schema = str(cache["schema"].item())
        key = str(cache["cache_key"].item())
        if schema != REPLAY_CACHE_SCHEMA:
            raise RuntimeError(
                f"replay cache schema {schema!r} != {REPLAY_CACHE_SCHEMA!r}"
            )
        if key != expected_key:
            raise RuntimeError(
                "replay cache key mismatch; candidate/state support changed"
            )
        return {
            name: cache[name].copy()
            for name in (
                "block_states",
                "endpoints",
                "goal_distances",
                "block_pixels",
            )
        }


def _write_replay_cache(
    path: Path,
    *,
    cache_key: str,
    replay: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            schema=np.asarray(REPLAY_CACHE_SCHEMA),
            cache_key=np.asarray(cache_key),
            block_states=np.asarray(replay["block_states"]),
            endpoints=np.asarray(replay["endpoints"]),
            goal_distances=np.asarray(replay["goal_distances"]),
            block_pixels=np.asarray(replay["block_pixels"]),
        )
    temporary.replace(path)


def _clone_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().contiguous().clone()
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _rank_rows(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("rank input must have shape (B,K)")
    order = torch.argsort(values, dim=1, stable=True)
    ranks = torch.empty_like(order)
    source = torch.arange(values.size(1), device=values.device)[None].expand_as(order)
    ranks.scatter_(1, order, source)
    return ranks


def _safe_spearman(x: list[float], y: list[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(xa) & np.isfinite(ya)
    if valid.sum() < 3:
        return None
    xa = xa[valid]
    ya = ya[valid]
    if np.ptp(xa) == 0.0 or np.ptp(ya) == 0.0:
        return None
    value = float(spearmanr(xa, ya).statistic)
    return value if math.isfinite(value) else None


def _per_group_spearman(
    rows: list[dict[str, Any]],
    *,
    signal: str,
    target: str,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["severity"]) <= 0.0:
            continue
        key = (
            row["trajectory_block_index"],
            row["severity"],
            row["draw_index"],
        )
        groups[key].append(row)
    values: list[float] = []
    for group in groups.values():
        correlation = _safe_spearman(
            [float(row[signal]) for row in group],
            [float(row[target]) for row in group],
        )
        if correlation is not None:
            values.append(correlation)
    return {
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "finite_group_count": len(values),
        "total_group_count": len(groups),
        "values": values,
    }


def _grouped_ridge_predictions(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    target_name: str,
) -> dict[str, Any]:
    filtered = [
        row for row in rows if float(row["severity"]) > 0.0
    ]
    groups = np.asarray(
        [int(row["trajectory_block_index"]) for row in filtered],
        dtype=np.int64,
    )
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return {
            "status": "insufficient_groups",
            "group_count": int(len(unique_groups)),
            "feature_names": list(feature_names),
        }
    x = np.asarray(
        [
            [float(row[name]) for name in feature_names]
            for row in filtered
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row[target_name]) for row in filtered],
        dtype=np.float64,
    )
    predictions = np.full_like(y, np.nan)
    for held_out in unique_groups:
        test = groups == held_out
        train = ~test
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    if not np.isfinite(predictions).all():
        raise RuntimeError("grouped Ridge left non-finite predictions")
    per_group_spearman: list[float] = []
    per_group_mae: list[float] = []
    per_group: list[dict[str, Any]] = []
    for group in unique_groups:
        select = groups == group
        correlation = _safe_spearman(
            predictions[select].tolist(),
            y[select].tolist(),
        )
        if correlation is not None:
            per_group_spearman.append(correlation)
        group_mae = float(
            mean_absolute_error(y[select], predictions[select])
        )
        per_group_mae.append(group_mae)
        per_group.append(
            {
                "trajectory_block_index": int(group),
                "row_count": int(select.sum()),
                "mae": group_mae,
                "spearman": correlation,
            }
        )
    return {
        "status": "complete",
        "group_count": int(len(unique_groups)),
        "row_count": int(len(filtered)),
        "feature_names": list(feature_names),
        "target_name": target_name,
        "r2": float(r2_score(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "spearman": _safe_spearman(
            predictions.tolist(),
            y.tolist(),
        ),
        "mean_within_group_spearman": (
            float(np.mean(per_group_spearman))
            if per_group_spearman
            else None
        ),
        "finite_within_group_count": len(per_group_spearman),
        "mean_group_mae": float(np.mean(per_group_mae)),
        "per_group": per_group,
    }


@torch.inference_mode()
def _encode_current(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    current_index: int,
    embedding_space: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    pixels = batch["pixels"][
        :, current_index : current_index + 1
    ].contiguous()
    for start in range(0, pixels.size(0), int(batch_size)):
        stop = min(start + int(batch_size), pixels.size(0))
        output = model.encode({"pixels": pixels[start:stop].to(device)})
        chunks.append(
            phase0.get_embedding_space(output, embedding_space)[:, 0].detach()
        )
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def _encode_simulator_futures(
    model: Any,
    raw_pixels: np.ndarray,
    *,
    embedding_space: str,
    img_size: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    if raw_pixels.ndim != 6 or raw_pixels.shape[-1] != 3:
        raise ValueError("simulator pixels must have shape (B,K,H,W,C,3)")
    leading = raw_pixels.shape[:3]
    flat = torch.from_numpy(raw_pixels).permute(0, 1, 2, 5, 3, 4)
    flat = flat.reshape(-1, 3, raw_pixels.shape[3], raw_pixels.shape[4])
    transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", int(img_size))
    )
    processed = transform({"pixels": flat})["pixels"]
    encoded: list[torch.Tensor] = []
    for start in range(0, processed.size(0), int(batch_size)):
        stop = min(start + int(batch_size), processed.size(0))
        output = model.encode(
            {"pixels": processed[start:stop].unsqueeze(1).to(device)}
        )
        encoded.append(
            phase0.get_embedding_space(output, embedding_space)[:, 0].detach()
        )
    return torch.cat(encoded, dim=0).reshape(*leading, -1)


def _prediction_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError("prediction target pairs must share shape (B,K,H,D)")
    return {
        "h1": torch.linalg.vector_norm(
            predictions[:, :, 0] - targets[:, :, 0],
            dim=-1,
        ),
        "horizon": horizon_weighted_stacked_l2(predictions, targets),
        "endpoint": torch.linalg.vector_norm(
            predictions[:, :, -1] - targets[:, :, -1],
            dim=-1,
        ),
    }


def _candidate_control_pools(
    candidates: torch.Tensor,
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    shuffled, candidate_permutation = permute_candidate_actions(
        candidates,
        seed=int(seed),
    )
    time_shuffled, time_permutation = permute_candidate_time(
        candidates,
        seed=int(seed) + 17,
    )
    return (
        {
            "correct": candidates,
            "action_zero": torch.zeros_like(candidates),
            "candidate_shuffle": shuffled,
            "time_shuffle": time_shuffled,
        },
        {
            "candidate_permutation": candidate_permutation.cpu(),
            "time_permutation": time_permutation.cpu(),
        },
    )


@torch.inference_mode()
def _candidate_predictions_by_control(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    pools: Mapping[str, torch.Tensor],
    *,
    current_index: int,
    device: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    return {
        name: planner_pool_predictions(
            model,
            batch,
            pool,
            current_index=current_index,
            device=device,
            batch_size=batch_size,
        )
        for name, pool in pools.items()
    }


@torch.inference_mode()
def _logged_track(
    model: Any,
    clean_batch: Mapping[str, torch.Tensor],
    probe_batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    horizon: int,
    embedding_space: str,
    control_seed: int,
) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    clean_output = model.encode(_clone_batch(clean_batch))
    probe_output = model.encode(_clone_batch(probe_batch))
    clean_emb = phase0.get_embedding_space(clean_output, embedding_space).detach()
    probe_emb = phase0.get_embedding_space(probe_output, embedding_space).detach()
    raw_action = clean_batch["action"]
    generator = torch.Generator(device="cpu").manual_seed(int(control_seed))
    batch_permutation = torch.randperm(
        raw_action.size(0),
        generator=generator,
    ).to(raw_action.device)
    time_permutation = torch.stack(
        [
            torch.randperm(raw_action.size(1), generator=generator)
            for _ in range(raw_action.size(0))
        ]
    ).to(raw_action.device)
    batch_index = torch.arange(raw_action.size(0), device=raw_action.device)[:, None]
    control_actions = {
        "correct": raw_action,
        "action_zero": torch.zeros_like(raw_action),
        "candidate_shuffle": raw_action[batch_permutation],
        "time_shuffle": raw_action[batch_index, time_permutation],
    }
    target = clean_emb[
        :, history_size : history_size + int(horizon)
    ]
    encoder_response = horizon_weighted_stacked_l2(
        clean_emb[:, :history_size],
        probe_emb[:, :history_size],
    )
    controls: dict[str, dict[str, torch.Tensor]] = {}
    for name, action in control_actions.items():
        act_emb = model.action_encoder(action)
        clean_chain = phase0._autoregressive_rollout(
            model,
            clean_emb[:, :history_size],
            act_emb,
            history_size,
            int(horizon),
        )
        probe_chain = phase0._autoregressive_rollout(
            model,
            probe_emb[:, :history_size],
            act_emb,
            history_size,
            int(horizon),
        )
        clean_prediction = clean_chain[
            :, history_size : history_size + int(horizon)
        ]
        probe_prediction = probe_chain[
            :, history_size : history_size + int(horizon)
        ]
        controls[name] = {
            "h1_response": torch.linalg.vector_norm(
                probe_prediction[:, 0] - clean_prediction[:, 0],
                dim=-1,
            ),
            "horizon_response": horizon_weighted_stacked_l2(
                clean_prediction,
                probe_prediction,
            ),
            "clean_h1_error": torch.linalg.vector_norm(
                clean_prediction[:, 0] - target[:, 0],
                dim=-1,
            ),
            "probe_h1_error": torch.linalg.vector_norm(
                probe_prediction[:, 0] - target[:, 0],
                dim=-1,
            ),
            "clean_horizon_error": horizon_weighted_stacked_l2(
                clean_prediction,
                target,
            ),
            "probe_horizon_error": horizon_weighted_stacked_l2(
                probe_prediction,
                target,
            ),
        }
    return {
        "encoder_response": encoder_response,
        "controls": controls,
        "batch_permutation": batch_permutation,
        "time_permutation": time_permutation,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if float(args.minimum_true_goal_cost_std) < 0.0:
        raise ValueError("minimum_true_goal_cost_std must be non-negative")
    if not 0.0 <= float(args.minimum_informative_block_fraction) <= 1.0:
        raise ValueError(
            "minimum_informative_block_fraction must lie in [0,1]"
        )

    def progress(stage: str) -> None:
        print(
            f"[target-aligned] stage={stage} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )

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
    spaces = phase0.get_model_spaces(model)
    embedding_space = str(spaces["inference_cost_space"])
    history_size = int(phase0.infer_history_size(model))
    logged_horizon = min(int(args.logged_horizon), int(args.future_steps) - 1)
    if logged_horizon < 1:
        raise ValueError("logged horizon must be positive")
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
    candidates, _captured_costs = capture_eval_step_zero_pool(
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
    nominal_costs = planner_pool_costs(
        model,
        batch,
        candidates,
        current_index=current_index,
        goal_index=goal_index,
        device=device,
        batch_size=int(args.batch_size),
    )
    pools, permutations = _candidate_control_pools(
        candidates,
        seed=int(args.control_seed),
    )
    nominal_predictions = _candidate_predictions_by_control(
        model,
        batch,
        pools,
        current_index=current_index,
        device=device,
        batch_size=int(args.batch_size),
    )
    nominal_current = _encode_current(
        model,
        batch,
        current_index=current_index,
        embedding_space=embedding_space,
        device=device,
        batch_size=int(args.batch_size),
    )
    candidate_action_rms = candidates.float().flatten(2).square().mean(
        dim=2
    ).sqrt()
    current_reference = nominal_current[:, None, None, :].expand_as(
        nominal_predictions["correct"]
    )
    nominal_h1_displacement = torch.linalg.vector_norm(
        nominal_predictions["correct"][:, :, 0] - nominal_current[:, None],
        dim=-1,
    )
    nominal_h5_displacement = horizon_weighted_stacked_l2(
        current_reference,
        nominal_predictions["correct"],
    )
    nominal_cost_margin = nominal_costs - nominal_costs.min(
        dim=1,
        keepdim=True,
    ).values
    progress("planner_pool_and_nominal_predictions_ready")

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
    unpacked = unpack_candidate_action_blocks(
        candidates,
        action_block=int(args.action_block),
        base_action_dim=int(raw["action_stats"].mean.shape[-1]),
    )
    candidate_raw_actions = inverse_eval_action_coordinates(
        unpacked,
        raw["action_stats"],
    ).detach().cpu().numpy()
    prefix_kwargs = (
        {
            "prefix_initial_states": raw["episode_initial_states"],
            "prefix_actions": raw["prefix_actions"],
        }
        if task == "PushT"
        else {}
    )
    replay_cache_key = _replay_cache_key(
        task=task,
        blocks=blocks,
        current_states=raw["current_states"],
        goal_states=raw["goal_states"],
        candidate_raw_actions=candidate_raw_actions,
        action_block=int(args.action_block),
        replay_seed=int(args.replay_seed),
    )
    replay_cache_path = (
        Path(args.replay_cache).expanduser().resolve()
        if args.replay_cache is not None
        else None
    )
    if replay_cache_path is not None and replay_cache_path.exists():
        true_replay = _load_replay_cache(
            replay_cache_path,
            expected_key=replay_cache_key,
        )
        replay_cache_status = "loaded"
    else:
        true_replay = replay_action_pool(
            task,
            raw["current_states"],
            raw["goal_states"],
            candidate_raw_actions,
            action_block=int(args.action_block),
            reset_seed=int(args.replay_seed),
            return_pixels=True,
            **prefix_kwargs,
        )
        if replay_cache_path is not None:
            _write_replay_cache(
                replay_cache_path,
                cache_key=replay_cache_key,
                replay=true_replay,
            )
            replay_cache_status = "written"
        else:
            replay_cache_status = "disabled"
    progress("simulator_candidate_futures_ready")
    true_future_emb = _encode_simulator_futures(
        model,
        true_replay["block_pixels"],
        embedding_space=embedding_space,
        img_size=int(args.img_size),
        device=device,
        batch_size=int(args.target_encode_batch_size),
    )
    progress("simulator_future_targets_encoded")
    nominal_external_errors = _prediction_errors(
        nominal_predictions["correct"],
        true_future_emb,
    )
    true_costs = torch.from_numpy(true_replay["goal_distances"]).float()
    true_ranks = _rank_rows(true_costs)
    nominal_ranks = _rank_rows(nominal_costs)
    nominal_winner = torch.argmin(nominal_costs, dim=1)
    true_best = torch.argmin(true_costs, dim=1)
    true_best_cost = true_costs.gather(1, true_best[:, None]).squeeze(1)
    nominal_selected_true_cost = true_costs.gather(
        1, nominal_winner[:, None]
    ).squeeze(1)

    candidate_rows: list[dict[str, Any]] = []
    logged_rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    severities = [float(value) for value in args.severities]
    for severity_index, severity in enumerate(severities):
        progress(f"probe_severity_{severity:g}_start")
        for draw_index in range(int(args.draws)):
            probe_seed = (
                int(args.probe_seed)
                + 1009 * severity_index
                + 7919 * draw_index
            )
            probe_batch = phase0.make_paired_noisy_batch(
                _clone_batch(batch),
                history_size=history_size,
                noise_std=severity,
                seed=probe_seed,
                corruption_type=str(args.probe_family),
                corrupt_goal=False,
            )
            probe_current = _encode_current(
                model,
                probe_batch,
                current_index=current_index,
                embedding_space=embedding_space,
                device=device,
                batch_size=int(args.batch_size),
            )
            encoder_response = torch.linalg.vector_norm(
                probe_current - nominal_current,
                dim=-1,
            )
            probe_predictions = _candidate_predictions_by_control(
                model,
                probe_batch,
                pools,
                current_index=current_index,
                device=device,
                batch_size=int(args.batch_size),
            )
            response_by_control = {
                name: candidate_response_metrics(
                    nominal_predictions[name],
                    probe_predictions[name],
                    encoder_response=encoder_response,
                )
                for name in CONTROL_NAMES
            }
            probe_external_errors = _prediction_errors(
                probe_predictions["correct"],
                true_future_emb,
            )
            external_error_drift = prediction_error_drift_certificate(
                nominal_predictions["correct"],
                probe_predictions["correct"],
                true_future_emb,
            )
            probe_costs = planner_pool_costs(
                model,
                probe_batch,
                candidates,
                current_index=current_index,
                goal_index=goal_index,
                device=device,
                batch_size=int(args.batch_size),
            )
            probe_ranks = _rank_rows(probe_costs)
            probe_winner = torch.argmin(probe_costs, dim=1)
            exact_stable = nominal_winner == probe_winner
            probe_selected_true_cost = true_costs.gather(
                1, probe_winner[:, None]
            ).squeeze(1)
            for block_index, block in enumerate(blocks):
                cell_summaries.append(
                    {
                        "trajectory_block_index": block_index,
                        "episode_id": int(block.episode_id),
                        "severity": severity,
                        "draw_index": draw_index,
                        "exact_stable": bool(exact_stable[block_index]),
                        "nominal_winner": int(nominal_winner[block_index]),
                        "probe_winner": int(probe_winner[block_index]),
                        "true_best": int(true_best[block_index]),
                        "nominal_true_regret": float(
                            nominal_selected_true_cost[block_index]
                            - true_best_cost[block_index]
                        ),
                        "probe_true_regret": float(
                            probe_selected_true_cost[block_index]
                            - true_best_cost[block_index]
                        ),
                        "probe_induced_true_regret": float(
                            probe_selected_true_cost[block_index]
                            - nominal_selected_true_cost[block_index]
                        ),
                        "encoder_response": float(
                            encoder_response[block_index]
                        ),
                        "candidate_h5_q90": float(
                            torch.quantile(
                                response_by_control["correct"][
                                    "candidate_horizon_response"
                                ][block_index],
                                0.90,
                            )
                        ),
                    }
                )
                for candidate_index in range(candidates.size(1)):
                    row = {
                        "trajectory_block_index": block_index,
                        "episode_id": int(block.episode_id),
                        "severity": severity,
                        "severity_index": severity_index,
                        "draw_index": draw_index,
                        "probe_seed": probe_seed,
                        "candidate_index": candidate_index,
                        "encoder_response": float(
                            encoder_response[block_index]
                        ),
                        "candidate_action_rms": float(
                            candidate_action_rms[
                                block_index, candidate_index
                            ]
                        ),
                        "nominal_h1_displacement": float(
                            nominal_h1_displacement[
                                block_index, candidate_index
                            ]
                        ),
                        "nominal_h5_displacement": float(
                            nominal_h5_displacement[
                                block_index, candidate_index
                            ]
                        ),
                        "nominal_cost_margin": float(
                            nominal_cost_margin[
                                block_index, candidate_index
                            ]
                        ),
                        "encoder_x_action_rms": float(
                            encoder_response[block_index]
                            * candidate_action_rms[
                                block_index, candidate_index
                            ]
                        ),
                        "encoder_x_nominal_h5_displacement": float(
                            encoder_response[block_index]
                            * nominal_h5_displacement[
                                block_index, candidate_index
                            ]
                        ),
                        "nominal_model_cost": float(
                            nominal_costs[block_index, candidate_index]
                        ),
                        "probe_model_cost": float(
                            probe_costs[block_index, candidate_index]
                        ),
                        "absolute_model_cost_drift": float(
                            (
                                probe_costs[block_index, candidate_index]
                                - nominal_costs[block_index, candidate_index]
                            ).abs()
                        ),
                        "true_goal_cost": float(
                            true_costs[block_index, candidate_index]
                        ),
                        "true_rank": int(
                            true_ranks[block_index, candidate_index]
                        ),
                        "nominal_model_rank": int(
                            nominal_ranks[block_index, candidate_index]
                        ),
                        "probe_model_rank": int(
                            probe_ranks[block_index, candidate_index]
                        ),
                        "probe_rank_error": float(
                            (
                                probe_ranks[block_index, candidate_index]
                                - true_ranks[block_index, candidate_index]
                            ).abs()
                        ),
                        "nominal_h1_prediction_error": float(
                            nominal_external_errors["h1"][
                                block_index, candidate_index
                            ]
                        ),
                        "probe_h1_prediction_error": float(
                            probe_external_errors["h1"][
                                block_index, candidate_index
                            ]
                        ),
                        "excess_h1_prediction_error": float(
                            probe_external_errors["h1"][
                                block_index, candidate_index
                            ]
                            - nominal_external_errors["h1"][
                                block_index, candidate_index
                            ]
                        ),
                        "adverse_h1_prediction_degradation": float(
                            external_error_drift[
                                "adverse_h1_error_change"
                            ][block_index, candidate_index]
                        ),
                        "absolute_h1_prediction_error_drift": float(
                            external_error_drift[
                                "absolute_h1_error_drift"
                            ][block_index, candidate_index]
                        ),
                        "correct_h1_error_drift_certificate_slack": float(
                            external_error_drift[
                                "h1_certificate_slack"
                            ][block_index, candidate_index]
                        ),
                        "nominal_h5_prediction_error": float(
                            nominal_external_errors["horizon"][
                                block_index, candidate_index
                            ]
                        ),
                        "probe_h5_prediction_error": float(
                            probe_external_errors["horizon"][
                                block_index, candidate_index
                            ]
                        ),
                        "excess_h5_prediction_error": float(
                            probe_external_errors["horizon"][
                                block_index, candidate_index
                            ]
                            - nominal_external_errors["horizon"][
                                block_index, candidate_index
                            ]
                        ),
                        "adverse_h5_prediction_degradation": float(
                            external_error_drift[
                                "adverse_horizon_error_change"
                            ][block_index, candidate_index]
                        ),
                        "absolute_h5_prediction_error_drift": float(
                            external_error_drift[
                                "absolute_horizon_error_drift"
                            ][block_index, candidate_index]
                        ),
                        "correct_h5_error_drift_certificate_slack": float(
                            external_error_drift[
                                "horizon_certificate_slack"
                            ][block_index, candidate_index]
                        ),
                    }
                    for name in CONTROL_NAMES:
                        metrics = response_by_control[name]
                        row[f"{name}_h1_response"] = float(
                            metrics["candidate_h1_response"][
                                block_index, candidate_index
                            ]
                        )
                        row[f"{name}_h5_response"] = float(
                            metrics["candidate_horizon_response"][
                                block_index, candidate_index
                            ]
                        )
                        row[f"{name}_h5_amplification"] = float(
                            metrics["candidate_horizon_amplification"][
                                block_index, candidate_index
                            ]
                        )
                    candidate_rows.append(row)

            logged = _logged_track(
                model,
                batch,
                probe_batch,
                history_size=history_size,
                horizon=logged_horizon,
                embedding_space=embedding_space,
                control_seed=int(args.control_seed) + severity_index,
            )
            for block_index, block in enumerate(blocks):
                row = {
                    "trajectory_block_index": block_index,
                    "episode_id": int(block.episode_id),
                    "severity": severity,
                    "severity_index": severity_index,
                    "draw_index": draw_index,
                    "probe_seed": probe_seed,
                    "encoder_response": float(
                        logged["encoder_response"][block_index]
                    ),
                }
                for name in CONTROL_NAMES:
                    metrics = logged["controls"][name]
                    h1_change = (
                        metrics["probe_h1_error"][block_index]
                        - metrics["clean_h1_error"][block_index]
                    )
                    h8_change = (
                        metrics["probe_horizon_error"][block_index]
                        - metrics["clean_horizon_error"][block_index]
                    )
                    row[f"{name}_h1_response"] = float(
                        metrics["h1_response"][block_index]
                    )
                    row[f"{name}_h8_response"] = float(
                        metrics["horizon_response"][block_index]
                    )
                    row[f"{name}_excess_h1_error"] = float(h1_change)
                    row[f"{name}_adverse_h1_error"] = float(
                        h1_change.clamp_min(0.0)
                    )
                    row[f"{name}_absolute_h1_error_drift"] = float(
                        h1_change.abs()
                    )
                    row[f"{name}_excess_h8_error"] = float(h8_change)
                    row[f"{name}_adverse_h8_error"] = float(
                        h8_change.clamp_min(0.0)
                    )
                    row[f"{name}_absolute_h8_error_drift"] = float(
                        h8_change.abs()
                    )
                    row[f"{name}_h1_error_drift_certificate_slack"] = float(
                        metrics["h1_response"][block_index]
                        - h1_change.abs()
                    )
                    row[f"{name}_h8_error_drift_certificate_slack"] = float(
                        metrics["horizon_response"][block_index]
                        - h8_change.abs()
                )
                logged_rows.append(row)
        progress(f"probe_severity_{severity:g}_complete")

    candidate_correlations = {}
    for signal in (
        "encoder_response",
        "correct_h1_response",
        "correct_h5_response",
        "action_zero_h5_response",
        "candidate_shuffle_h5_response",
        "time_shuffle_h5_response",
    ):
        candidate_correlations[signal] = {
            "external_excess_h5_error": _per_group_spearman(
                candidate_rows,
                signal=signal,
                target="excess_h5_prediction_error",
            ),
            "external_adverse_h5_degradation": _per_group_spearman(
                candidate_rows,
                signal=signal,
                target="adverse_h5_prediction_degradation",
            ),
            "external_absolute_h5_error_drift": _per_group_spearman(
                candidate_rows,
                signal=signal,
                target="absolute_h5_prediction_error_drift",
            ),
            "internal_absolute_cost_drift": _per_group_spearman(
                candidate_rows,
                signal=signal,
                target="absolute_model_cost_drift",
            ),
            "external_probe_rank_error": _per_group_spearman(
                candidate_rows,
                signal=signal,
                target="probe_rank_error",
            ),
        }

    candidate_grouped_cv = {
        target: {
            name: _grouped_ridge_predictions(
                candidate_rows,
                feature_names=features,
                target_name=target,
            )
            for name, features in CANDIDATE_FEATURE_SETS.items()
        }
        for target in (
            "excess_h5_prediction_error",
            "adverse_h5_prediction_degradation",
            "absolute_h5_prediction_error_drift",
            "absolute_model_cost_drift",
            "probe_rank_error",
        )
    }
    logged_grouped_cv = {
        target: {
            name: _grouped_ridge_predictions(
                logged_rows,
                feature_names=features,
                target_name=target,
            )
            for name, features in LOGGED_FEATURE_SETS.items()
        }
        for target in (
            "correct_excess_h1_error",
            "correct_excess_h8_error",
            "correct_adverse_h1_error",
            "correct_adverse_h8_error",
            "correct_absolute_h1_error_drift",
            "correct_absolute_h8_error_drift",
        )
    }

    identity_candidate_rows = [
        row for row in candidate_rows if float(row["severity"]) == 0.0
    ]
    identity_logged_rows = [
        row for row in logged_rows if float(row["severity"]) == 0.0
    ]
    identity_max = max(
        [
            abs(float(row[key]))
            for row in identity_candidate_rows
            for key in (
                "encoder_response",
                "correct_h1_response",
                "correct_h5_response",
                "absolute_model_cost_drift",
                "excess_h1_prediction_error",
                "excess_h5_prediction_error",
            )
        ]
        + [
            abs(float(row[key]))
            for row in identity_logged_rows
            for key in (
                "encoder_response",
                "correct_h1_response",
                "correct_h8_response",
                "correct_excess_h1_error",
                "correct_excess_h8_error",
            )
        ]
    )
    true_goal_cost_std = np.std(true_replay["goal_distances"], axis=1)
    informative_true_goal_blocks = (
        true_goal_cost_std > float(args.minimum_true_goal_cost_std)
    )
    informative_true_goal_fraction = float(
        np.mean(informative_true_goal_blocks)
    )
    checks = {
        "identity_zero": identity_max <= float(args.invariant_atol),
        "candidate_correct_h1_error_drift_bound": min(
            float(row["correct_h1_error_drift_certificate_slack"])
            for row in candidate_rows
        )
        >= -float(args.invariant_atol),
        "candidate_correct_h5_error_drift_bound": min(
            float(row["correct_h5_error_drift_certificate_slack"])
            for row in candidate_rows
        )
        >= -float(args.invariant_atol),
        "logged_correct_h1_error_drift_bound": min(
            float(row["correct_h1_error_drift_certificate_slack"])
            for row in logged_rows
        )
        >= -float(args.invariant_atol),
        "logged_correct_h8_error_drift_bound": min(
            float(row["correct_h8_error_drift_certificate_slack"])
            for row in logged_rows
        )
        >= -float(args.invariant_atol),
        "candidate_rows_complete": len(candidate_rows)
        == (
            int(args.n_blocks)
            * int(args.candidate_count)
            * len(severities)
            * int(args.draws)
        ),
        "logged_rows_complete": len(logged_rows)
        == int(args.n_blocks) * len(severities) * int(args.draws),
        "true_candidate_cost_informative_block_coverage": bool(
            informative_true_goal_fraction
            >= float(args.minimum_informative_block_fraction)
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    progress(f"complete_{status.lower()}")
    return {
        "metadata": {
            "schema_version": RESULT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "purpose": "DEV branch adjudication only; no paper claim",
            "behavior_labels_read": False,
            "task": task,
            "checkpoint": str(checkpoint),
            "training_seed": args.training_seed,
            "checkpoint_role": args.checkpoint_role,
            "dataset": str(h5_path),
            "device": device,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "design": {
            "n_blocks": int(args.n_blocks),
            "candidate_count": int(args.candidate_count),
            "plan_horizon": int(args.plan_horizon),
            "logged_horizon": logged_horizon,
            "history_size": history_size,
            "current_index": current_index,
            "goal_index": goal_index,
            "goal_offset_low_level_steps": (
                int(args.plan_horizon) * int(args.action_block)
            ),
            "probe_family": args.probe_family,
            "severities": severities,
            "draws": int(args.draws),
            "embedding_space": embedding_space,
            "batch_size_matches_eval": int(args.batch_size) == 1,
            "logged_and_candidate_tracks_pooled": False,
            "privileged_replay_state_key": state_key,
            "decision_event": "direct lexicographic argmin equality",
            "signed_gap_certificate_role": "separate exact audit only",
            "predictive_error_drift_certificate": (
                "reverse-triangle bound under the same correct action "
                "sequence and canonical weighted stacked L2"
            ),
            "target_semantics": {
                "signed_degradation": "probe error minus nominal error",
                "adverse_degradation": "positive part of signed degradation",
                "absolute_error_drift": (
                    "absolute nominal-to-probe prediction-error change"
                ),
            },
            "candidate_controls": list(CONTROL_NAMES),
            "external_candidate_target": (
                "same-state simulator-rendered future encoded as held-out "
                "JEPA target plus privileged true goal distance"
            ),
            "normalization": raw["action_stats"].to_dict(),
            "replay_cache": {
                "status": replay_cache_status,
                "path": (
                    str(replay_cache_path)
                    if replay_cache_path is not None
                    else None
                ),
                "key": replay_cache_key,
                "strict_key_validation": True,
            },
            "true_goal_cost_support": {
                "std_per_block": true_goal_cost_std,
                "informative_block_mask": informative_true_goal_blocks,
                "informative_block_fraction": informative_true_goal_fraction,
                "minimum_std": float(args.minimum_true_goal_cost_std),
                "minimum_fraction": float(
                    args.minimum_informative_block_fraction
                ),
                "neutral_blocks_retained": True,
            },
        },
        "checks": checks,
        "identity_max_abs": identity_max,
        "permutations": permutations,
        "candidate_correlations": candidate_correlations,
        "candidate_grouped_cv": candidate_grouped_cv,
        "logged_grouped_cv": logged_grouped_cv,
        "candidate_rows": candidate_rows,
        "logged_rows": logged_rows,
        "cell_summaries": cell_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument(
        "--checkpoint-role",
        choices=("base", "onset", "endpoint", "other"),
        default=None,
        help="Explicit provenance for legacy checkpoint paths without seed/role tokens.",
    )
    parser.add_argument("--dataset-name", default="tworoom")
    parser.add_argument("--task", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--logged-horizon", type=int, default=8)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-encode-batch-size", type=int, default=16)
    parser.add_argument("--trajectory-seed", type=int, default=9101)
    parser.add_argument("--cem-seed", type=int, default=1234)
    parser.add_argument("--control-seed", type=int, default=4701)
    parser.add_argument("--replay-seed", type=int, default=0)
    parser.add_argument("--replay-cache", type=Path, default=None)
    parser.add_argument("--probe-family", default="gaussian_noise")
    parser.add_argument("--severities", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--probe-seed", type=int, default=20260712)
    parser.add_argument("--invariant-atol", type=float, default=1e-5)
    parser.add_argument(
        "--minimum-true-goal-cost-std",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--minimum-informative-block-fraction",
        type=float,
        default=0.75,
    )
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
    compact = {
        "status": payload["metadata"]["status"],
        "checks": payload["checks"],
        "identity_max_abs": payload["identity_max_abs"],
        "out": str(args.out),
    }
    print(json.dumps(_jsonable(compact), sort_keys=True), flush=True)
    return 0 if payload["metadata"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
