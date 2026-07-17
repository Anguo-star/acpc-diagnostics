#!/usr/bin/env python3
"""Target-aligned ACPC primitives for Paper 1.

This module deliberately keeps two evaluation objects separate:

* logged-action H1/H8 predictions start from the model's recorded history;
* planner-candidate predictions reproduce eval.py: one current observation
  followed by plan_horizon future action blocks.

The distinction is a correctness constraint, not a presentation choice. A
five-token CEM candidate evaluated with three observed frames produces only
three predicted states and is not the planner event used by eval.py.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_worldmodel.solver.callbacks.common import Callback

from tools.paper1_acpc_metrics import horizon_weighted_stacked_l2


SCHEMA_VERSION = "paper1-target-aligned-acpc-0.1"


@dataclass(frozen=True)
class PlannerTemporalSemantics:
    observation_steps: int
    candidate_steps: int
    predicted_steps: int
    action_block: int
    low_level_action_steps: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ActionCoordinateStats:
    """Repository training and eval action-coordinate statistics."""

    count: int
    mean: np.ndarray
    training_sample_std: np.ndarray
    eval_population_std: np.ndarray

    @property
    def max_relative_scale_difference(self) -> float:
        denominator = np.maximum(np.abs(self.eval_population_std), 1e-12)
        return float(
            np.max(
                np.abs(self.training_sample_std - self.eval_population_std)
                / denominator
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "training_sample_std": self.training_sample_std.tolist(),
            "eval_population_std": self.eval_population_std.tolist(),
            "max_relative_scale_difference": self.max_relative_scale_difference,
            "planner_inverse_transform": "eval_population_std",
        }


def planner_temporal_semantics(
    *,
    observation_steps: int,
    candidate_steps: int,
    action_block: int,
) -> PlannerTemporalSemantics:
    """Return the exact transition count implemented by JEPA rollout."""

    observed = int(observation_steps)
    candidates = int(candidate_steps)
    block = int(action_block)
    if observed < 1:
        raise ValueError("observation_steps must be positive")
    if candidates < observed:
        raise ValueError(
            "candidate_steps must be at least observation_steps for LeWM rollout"
        )
    if block < 1:
        raise ValueError("action_block must be positive")
    predicted = candidates - observed + 1
    return PlannerTemporalSemantics(
        observation_steps=observed,
        candidate_steps=candidates,
        predicted_steps=predicted,
        action_block=block,
        low_level_action_steps=candidates * block,
    )


def _resolve_index(length: int, index: int, *, name: str) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved += int(length)
    if not 0 <= resolved < int(length):
        raise IndexError(f"{name}={index} is outside sequence length {length}")
    return resolved


def planner_query_info(
    batch: Mapping[str, torch.Tensor],
    *,
    current_index: int = 0,
    goal_index: int = -1,
) -> dict[str, torch.Tensor]:
    """Build the single-current-frame info used by the real eval planner."""

    if "pixels" not in batch or "action" not in batch:
        raise KeyError("batch must contain pixels and action")
    pixels = batch["pixels"]
    action = batch["action"]
    if pixels.ndim < 3 or action.ndim < 3:
        raise ValueError("pixels/action must have batch and time axes")
    if pixels.shape[:2] != action.shape[:2]:
        raise ValueError("pixels/action batch and time axes must agree")
    current = _resolve_index(pixels.size(1), current_index, name="current_index")
    goal = _resolve_index(pixels.size(1), goal_index, name="goal_index")
    return {
        "pixels": pixels[:, current : current + 1].contiguous(),
        # get_cost expects this key. JEPA rollout replaces it with candidate
        # actions before prediction, matching WorldModelPolicy/CEM behavior.
        "action": action[:, current : current + 1].contiguous(),
        "goal": pixels[:, goal : goal + 1].contiguous(),
    }


def expand_planner_info(
    info: Mapping[str, torch.Tensor],
    *,
    candidate_count: int,
) -> dict[str, torch.Tensor]:
    count = int(candidate_count)
    if count < 1:
        raise ValueError("candidate_count must be positive")
    expanded: dict[str, torch.Tensor] = {}
    for key, value in info.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            raise ValueError(f"planner info {key!r} must be a batched tensor")
        canonical = value.contiguous()
        expanded[key] = canonical.unsqueeze(1).expand(
            canonical.size(0),
            count,
            *canonical.shape[1:],
        )
    return expanded


@torch.inference_mode()
def planner_pool_costs(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    candidates: torch.Tensor,
    *,
    current_index: int = 0,
    goal_index: int = -1,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Evaluate a shared pool through the eval-matched current-only path."""

    if candidates.ndim != 4:
        raise ValueError("candidates must have shape (B,K,H,A)")
    if candidates.size(0) != batch["pixels"].size(0):
        raise ValueError("candidate and trajectory batch sizes must agree")
    if device is not None:
        target_device = torch.device(device)
    elif hasattr(model, "parameters"):
        target_device = next(model.parameters()).device
    else:
        target_device = candidates.device
    base_query = planner_query_info(
        batch,
        current_index=current_index,
        goal_index=goal_index,
    )
    candidate_device = candidates.to(target_device)
    chunk = int(batch_size or candidates.size(0))
    if chunk < 1:
        raise ValueError("batch_size must be positive")
    cost_chunks: list[torch.Tensor] = []
    for start in range(0, candidates.size(0), chunk):
        stop = min(start + chunk, candidates.size(0))
        query = {
            key: value.to(target_device)
            for key, value in expand_planner_info(
                {
                    key: value[start:stop]
                    for key, value in base_query.items()
                },
                candidate_count=candidates.size(1),
            ).items()
        }
        semantics = planner_temporal_semantics(
            observation_steps=query["pixels"].size(2),
            candidate_steps=candidate_device.size(2),
            action_block=1,
        )
        if semantics.observation_steps != 1:
            raise RuntimeError(
                "eval-matched planner query must contain one observation"
            )
        cost_chunks.append(
            model.get_cost(query, candidate_device[start:stop]).detach()
        )
    costs = torch.cat(cost_chunks, dim=0)
    if tuple(costs.shape) != tuple(candidate_device.shape[:2]):
        raise RuntimeError(
            f"model.get_cost returned {tuple(costs.shape)} for "
            f"candidate pool {tuple(candidate_device.shape)}"
        )
    return costs.detach().float().cpu()


class EvalStepZeroCandidateCapture(Callback):
    """Capture the ordered proposal and costs from CEM iteration zero."""

    name = "EvalStepZeroCandidateCapture"

    def __init__(self) -> None:
        super().__init__(reduction="none")

    def compute(self, **state: Any) -> dict[str, torch.Tensor] | None:
        if int(state["step"]) != 0:
            return None
        return {
            "candidates": state["candidates"].detach().cpu(),
            "costs": state["costs"].detach().float().cpu(),
        }


@torch.inference_mode()
def capture_eval_step_zero_pool(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    current_index: int,
    goal_index: int,
    action_block: int,
    plan_horizon: int,
    candidate_count: int,
    topk: int,
    batch_size: int,
    cem_seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the actual eval-style CEM proposal using one current frame."""

    from tools.paper1_cem_trace_audit import _make_solver

    if int(candidate_count) < 2:
        raise ValueError("candidate_count must be at least two")
    if not 1 <= int(topk) <= int(candidate_count):
        raise ValueError("topk must lie in [1,candidate_count]")
    capture = EvalStepZeroCandidateCapture()
    args = argparse.Namespace(
        action_block=int(action_block),
        batch_size=int(batch_size),
        num_samples=int(candidate_count),
        var_scale=1.0,
        n_steps=1,
        topk=int(topk),
        cem_seed=int(cem_seed),
        plan_horizon=int(plan_horizon),
        # PlanConfig.history_len is not consumed by CEM, but one is the only
        # value consistent with the actual query assembled below.
        history_size_for_plan=1,
    )
    raw_action_dim = int(batch["action"].shape[-1])
    solver = _make_solver(
        model,
        args=args,
        n_envs=int(batch["pixels"].shape[0]),
        raw_action_dim=raw_action_dim,
        device=device,
    )
    solver.callbacks = [capture]
    query = planner_query_info(
        batch,
        current_index=current_index,
        goal_index=goal_index,
    )
    output = solver(query)
    history = output["callbacks"][capture.output_key]
    candidate_batches: list[torch.Tensor] = []
    cost_batches: list[torch.Tensor] = []
    for records in history:
        if len(records) != 1:
            raise RuntimeError(
                "step-zero capture must contain one record per solver batch"
            )
        candidate_batches.append(records[0]["candidates"])
        cost_batches.append(records[0]["costs"])
    candidates = torch.cat(candidate_batches, dim=0)
    costs = torch.cat(cost_batches, dim=0)
    expected = (int(batch["pixels"].shape[0]), int(candidate_count))
    if tuple(candidates.shape[:2]) != expected or tuple(costs.shape) != expected:
        raise RuntimeError(
            "captured pool shape mismatch: "
            f"candidates={tuple(candidates.shape)}, costs={tuple(costs.shape)}, "
            f"expected_prefix={expected}"
        )
    semantics = planner_temporal_semantics(
        observation_steps=query["pixels"].size(1),
        candidate_steps=candidates.size(2),
        action_block=action_block,
    )
    if semantics.predicted_steps != int(plan_horizon):
        raise RuntimeError(
            f"eval query predicts {semantics.predicted_steps} steps, "
            f"expected plan_horizon={plan_horizon}"
        )
    return candidates, costs


@torch.inference_mode()
def planner_pool_predictions(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    candidates: torch.Tensor,
    *,
    current_index: int = 0,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Return predicted states only, shaped (B,K,H_plan,D)."""

    if candidates.ndim != 4:
        raise ValueError("candidates must have shape (B,K,H,A)")
    if device is not None:
        target_device = torch.device(device)
    else:
        target_device = next(model.parameters()).device
    base_query = planner_query_info(
        batch,
        current_index=current_index,
        goal_index=current_index,
    )
    base_query.pop("goal")
    candidate_device = candidates.to(target_device)
    chunk = int(batch_size or candidates.size(0))
    if chunk < 1:
        raise ValueError("batch_size must be positive")
    prediction_chunks: list[torch.Tensor] = []
    for start in range(0, candidates.size(0), chunk):
        stop = min(start + chunk, candidates.size(0))
        info = {
            key: value.to(target_device)
            for key, value in expand_planner_info(
                {
                    key: value[start:stop]
                    for key, value in base_query.items()
                },
                candidate_count=candidates.size(1),
            ).items()
        }
        output = model.rollout(info, candidate_device[start:stop])
        prediction_chunks.append(output["predicted_emb"].detach())
    predictions = torch.cat(prediction_chunks, dim=0)
    semantics = planner_temporal_semantics(
        observation_steps=1,
        candidate_steps=candidates.size(2),
        action_block=1,
    )
    expected_steps = 1 + semantics.predicted_steps
    if predictions.ndim != 4 or predictions.size(2) != expected_steps:
        raise RuntimeError(
            f"rollout returned {tuple(predictions.shape)}; expected "
            f"(B,K,{expected_steps},D)"
        )
    return predictions[:, :, 1:].detach()


def unpack_candidate_action_blocks(
    candidates: torch.Tensor | np.ndarray,
    *,
    action_block: int,
    base_action_dim: int | None = None,
) -> torch.Tensor | np.ndarray:
    """Reproduce WorldModelPolicy's horizon/action-block reshape."""

    block = int(action_block)
    if block < 1:
        raise ValueError("action_block must be positive")
    last_dim = int(candidates.shape[-1])
    if last_dim % block:
        raise ValueError(
            f"candidate last dim {last_dim} is not divisible by action_block={block}"
        )
    inferred_dim = last_dim // block
    if base_action_dim is not None and int(base_action_dim) != inferred_dim:
        raise ValueError(
            f"base_action_dim={base_action_dim} disagrees with inferred {inferred_dim}"
        )
    shape = (*candidates.shape[:-2], candidates.shape[-2] * block, inferred_dim)
    return candidates.reshape(shape)


def fit_action_coordinate_stats(
    raw_actions: np.ndarray | torch.Tensor,
) -> ActionCoordinateStats:
    array = (
        raw_actions.detach().cpu().numpy()
        if torch.is_tensor(raw_actions)
        else np.asarray(raw_actions)
    )
    if array.ndim < 2:
        raise ValueError("raw_actions must have a sample axis and action axis")
    array = array.reshape(-1, array.shape[-1]).astype(np.float64, copy=False)
    array = array[~np.isnan(array).any(axis=1)]
    if len(array) < 2:
        raise ValueError("at least two finite action rows are required")
    mean = array.mean(axis=0, keepdims=True)
    eval_std = array.std(axis=0, ddof=0, keepdims=True)
    training_std = array.std(axis=0, ddof=1, keepdims=True)
    # sklearn StandardScaler uses scale=1 for constant features.
    eval_std = np.where(eval_std == 0.0, 1.0, eval_std)
    return ActionCoordinateStats(
        count=int(len(array)),
        mean=mean,
        training_sample_std=training_std,
        eval_population_std=eval_std,
    )


def inverse_eval_action_coordinates(
    normalized_actions: torch.Tensor | np.ndarray,
    stats: ActionCoordinateStats,
) -> torch.Tensor | np.ndarray:
    """Match the sklearn StandardScaler inverse used by eval.py."""

    if normalized_actions.shape[-1] != stats.mean.shape[-1]:
        raise ValueError("normalized action dimension disagrees with statistics")
    if torch.is_tensor(normalized_actions):
        mean = torch.as_tensor(
            stats.mean,
            dtype=normalized_actions.dtype,
            device=normalized_actions.device,
        )
        scale = torch.as_tensor(
            stats.eval_population_std,
            dtype=normalized_actions.dtype,
            device=normalized_actions.device,
        )
        return normalized_actions * scale + mean
    array = np.asarray(normalized_actions)
    return array * stats.eval_population_std + stats.mean


def replay_tworoom_action_pool(
    initial_states: torch.Tensor | np.ndarray,
    goal_states: torch.Tensor | np.ndarray,
    low_level_actions: torch.Tensor | np.ndarray,
    *,
    action_block: int,
    reset_seed: int = 0,
    return_pixels: bool = False,
) -> dict[str, np.ndarray]:
    """Replay a same-state action pool in deterministic TwoRoom dynamics.

    low_level_actions has shape (B,K,L,2) and is already in raw env
    coordinates. States are recorded after every action block.
    """

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    initial = (
        initial_states.detach().cpu().numpy()
        if torch.is_tensor(initial_states)
        else np.asarray(initial_states)
    )
    goals = (
        goal_states.detach().cpu().numpy()
        if torch.is_tensor(goal_states)
        else np.asarray(goal_states)
    )
    actions = (
        low_level_actions.detach().cpu().numpy()
        if torch.is_tensor(low_level_actions)
        else np.asarray(low_level_actions)
    )
    if initial.ndim != 2 or initial.shape[-1] != 2:
        raise ValueError("TwoRoom initial_states must have shape (B,2)")
    if goals.shape != initial.shape:
        raise ValueError("TwoRoom goal_states must match initial_states")
    if actions.ndim != 4 or actions.shape[0] != initial.shape[0]:
        raise ValueError("actions must have shape (B,K,L,2)")
    if actions.shape[-1] != 2:
        raise ValueError("TwoRoom low-level action dimension must be two")
    block = int(action_block)
    if block < 1 or actions.shape[2] % block:
        raise ValueError("low-level action length must be divisible by action_block")
    horizon = actions.shape[2] // block
    endpoints = np.empty((actions.shape[0], actions.shape[1], 2), dtype=np.float32)
    block_states = np.empty(
        (actions.shape[0], actions.shape[1], horizon, 2),
        dtype=np.float32,
    )
    goal_distances = np.empty(actions.shape[:2], dtype=np.float32)
    block_pixels = (
        np.empty(
            (
                actions.shape[0],
                actions.shape[1],
                horizon,
                TwoRoomEnv.IMG_SIZE,
                TwoRoomEnv.IMG_SIZE,
                3,
            ),
            dtype=np.uint8,
        )
        if return_pixels
        else None
    )
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        for batch_index in range(actions.shape[0]):
            for candidate_index in range(actions.shape[1]):
                env.reset(
                    seed=int(reset_seed),
                    options={
                        "state": initial[batch_index],
                        "target_state": goals[batch_index],
                    },
                )
                for step_index, action in enumerate(
                    actions[batch_index, candidate_index]
                ):
                    env.step(action)
                    if (step_index + 1) % block == 0:
                        model_step = (step_index + 1) // block - 1
                        block_states[
                            batch_index, candidate_index, model_step
                        ] = env.agent_position.detach().cpu().numpy()
                        if block_pixels is not None:
                            block_pixels[
                                batch_index, candidate_index, model_step
                            ] = env.render()
                endpoint = env.agent_position.detach().cpu().numpy()
                endpoints[batch_index, candidate_index] = endpoint
                goal_distances[batch_index, candidate_index] = np.linalg.norm(
                    endpoint - goals[batch_index]
                )
    finally:
        env.close()
    result = {
        "block_states": block_states,
        "endpoints": endpoints,
        "goal_distances": goal_distances,
    }
    if block_pixels is not None:
        result["block_pixels"] = block_pixels
    return result


def replay_pusht_action_pool(
    initial_states: torch.Tensor | np.ndarray,
    goal_states: torch.Tensor | np.ndarray,
    low_level_actions: torch.Tensor | np.ndarray,
    *,
    action_block: int,
    reset_seed: int = 0,
    return_pixels: bool = False,
    prefix_initial_states: torch.Tensor | np.ndarray | None = None,
    prefix_actions: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Replay a same-state action pool in deterministic PushT dynamics.

    PushT's stored seven-dimensional state omits the block's linear and
    angular velocity. When episode-prefix data are supplied, replay begins at
    the episode start to recover those hidden velocities. The visible state
    is then snapped back to the exact recorded branch point without stepping
    physics, while the recovered block velocities are preserved.
    """

    from stable_worldmodel.envs.pusht.env import PushT

    initial = (
        initial_states.detach().cpu().numpy()
        if torch.is_tensor(initial_states)
        else np.asarray(initial_states)
    )
    goals = (
        goal_states.detach().cpu().numpy()
        if torch.is_tensor(goal_states)
        else np.asarray(goal_states)
    )
    actions = (
        low_level_actions.detach().cpu().numpy()
        if torch.is_tensor(low_level_actions)
        else np.asarray(low_level_actions)
    )
    prefix_initial = None
    if prefix_initial_states is not None:
        prefix_initial = (
            prefix_initial_states.detach().cpu().numpy()
            if torch.is_tensor(prefix_initial_states)
            else np.asarray(prefix_initial_states)
        )
    if initial.ndim != 2 or initial.shape[-1] != 7:
        raise ValueError("PushT initial_states must have shape (B,7)")
    if goals.shape != initial.shape:
        raise ValueError("PushT goal_states must match initial_states")
    if actions.ndim != 4 or actions.shape[0] != initial.shape[0]:
        raise ValueError("actions must have shape (B,K,L,2)")
    if actions.shape[-1] != 2:
        raise ValueError("PushT low-level action dimension must be two")
    if (prefix_initial is None) != (prefix_actions is None):
        raise ValueError(
            "prefix_initial_states and prefix_actions must be provided together"
        )
    if prefix_initial is not None:
        if prefix_initial.shape != initial.shape:
            raise ValueError("PushT prefix initial states must have shape (B,7)")
        if len(prefix_actions) != initial.shape[0]:
            raise ValueError("PushT prefix action list must have length B")
        for prefix in prefix_actions:
            array = np.asarray(prefix)
            if array.ndim != 2 or array.shape[-1] != 2:
                raise ValueError("each PushT prefix must have shape (L,2)")
    block = int(action_block)
    if block < 1 or actions.shape[2] % block:
        raise ValueError(
            "low-level action length must be divisible by action_block"
        )
    horizon = actions.shape[2] // block
    endpoints = np.empty(
        (actions.shape[0], actions.shape[1], 7), dtype=np.float64
    )
    block_states = np.empty(
        (actions.shape[0], actions.shape[1], horizon, 7),
        dtype=np.float64,
    )
    goal_distances = np.empty(actions.shape[:2], dtype=np.float64)
    block_pixels = (
        np.empty(
            (
                actions.shape[0],
                actions.shape[1],
                horizon,
                224,
                224,
                3,
            ),
            dtype=np.uint8,
        )
        if return_pixels
        else None
    )
    env = PushT(render_mode="rgb_array", resolution=224)

    def restore_visible_state(state: np.ndarray) -> None:
        env.agent.position = state[:2].tolist()
        env.block.position = state[2:4].tolist()
        env.block.angle = float(state[4])
        env.agent.velocity = state[-2:].tolist()

    try:
        for batch_index in range(actions.shape[0]):
            for candidate_index in range(actions.shape[1]):
                reset_state = (
                    prefix_initial[batch_index]
                    if prefix_initial is not None
                    else initial[batch_index]
                )
                env.reset(
                    seed=int(reset_seed),
                    options={
                        "state": reset_state,
                        "goal_state": goals[batch_index],
                    },
                )
                if prefix_initial is not None:
                    restore_visible_state(reset_state)
                    env.block.velocity = (0.0, 0.0)
                    env.block.angular_velocity = 0.0
                    for prefix_action in prefix_actions[batch_index]:
                        env.step(prefix_action)
                    restore_visible_state(initial[batch_index])
                latest_state = initial[batch_index]
                for step_index, action in enumerate(
                    actions[batch_index, candidate_index]
                ):
                    observation, _, _, _, _ = env.step(action)
                    latest_state = observation["state"]
                    if (step_index + 1) % block == 0:
                        model_step = (step_index + 1) // block - 1
                        block_states[
                            batch_index, candidate_index, model_step
                        ] = latest_state
                        if block_pixels is not None:
                            block_pixels[
                                batch_index, candidate_index, model_step
                            ] = env.render()
                endpoints[batch_index, candidate_index] = latest_state
                _, distance = env.eval_state(
                    goals[batch_index], latest_state
                )
                goal_distances[batch_index, candidate_index] = distance
    finally:
        env.close()
    result = {
        "block_states": block_states,
        "endpoints": endpoints,
        "goal_distances": goal_distances,
    }
    if block_pixels is not None:
        result["block_pixels"] = block_pixels
    return result


def replay_reacher_action_pool(
    initial_states: torch.Tensor | np.ndarray,
    goal_states: torch.Tensor | np.ndarray,
    low_level_actions: torch.Tensor | np.ndarray,
    *,
    action_block: int,
    reset_seed: int = 0,
    return_pixels: bool = False,
) -> dict[str, np.ndarray]:
    """Replay a same-state Reacher action pool from full qpos/qvel.

    initial_states concatenates qpos and qvel, while goal_states is the
    future qpos used by the real qpos-match evaluation task. A terminal state
    is held fixed for the remainder of the open-loop horizon so every
    candidate still has exactly one target per model step.
    """

    from stable_worldmodel.envs.dmcontrol.reacher import (
        ReacherDMControlWrapper,
    )

    initial = (
        initial_states.detach().cpu().numpy()
        if torch.is_tensor(initial_states)
        else np.asarray(initial_states)
    )
    goals = (
        goal_states.detach().cpu().numpy()
        if torch.is_tensor(goal_states)
        else np.asarray(goal_states)
    )
    actions = (
        low_level_actions.detach().cpu().numpy()
        if torch.is_tensor(low_level_actions)
        else np.asarray(low_level_actions)
    )
    if initial.ndim != 2:
        raise ValueError("Reacher initial_states must have shape (B,nq+nv)")
    if goals.ndim != 2 or goals.shape[0] != initial.shape[0]:
        raise ValueError("Reacher goal_states must have shape (B,nq)")
    if actions.ndim != 4 or actions.shape[0] != initial.shape[0]:
        raise ValueError("actions must have shape (B,K,L,A)")
    block = int(action_block)
    if block < 1 or actions.shape[2] % block:
        raise ValueError(
            "low-level action length must be divisible by action_block"
        )
    horizon = actions.shape[2] // block
    env = ReacherDMControlWrapper(
        task="qpos_match",
        seed=int(reset_seed),
        render_mode="rgb_array",
    )
    env.reset(seed=int(reset_seed))
    nq = int(env.env.physics.model.nq)
    nv = int(env.env.physics.model.nv)
    action_dim = int(np.prod(env.action_space.shape))
    if initial.shape[-1] != nq + nv:
        raise ValueError(
            f"Reacher initial state dimension must be nq+nv={nq + nv}"
        )
    if goals.shape[-1] != nq:
        raise ValueError(f"Reacher goal state dimension must be nq={nq}")
    if actions.shape[-1] != action_dim:
        raise ValueError(
            f"Reacher low-level action dimension must be {action_dim}"
        )

    state_dim = nq + nv
    endpoints = np.empty(
        (actions.shape[0], actions.shape[1], state_dim),
        dtype=np.float64,
    )
    block_states = np.empty(
        (actions.shape[0], actions.shape[1], horizon, state_dim),
        dtype=np.float64,
    )
    goal_distances = np.empty(actions.shape[:2], dtype=np.float64)
    block_pixels = (
        np.empty(
            (
                actions.shape[0],
                actions.shape[1],
                horizon,
                224,
                224,
                3,
            ),
            dtype=np.uint8,
        )
        if return_pixels
        else None
    )
    try:
        for batch_index in range(actions.shape[0]):
            for candidate_index in range(actions.shape[1]):
                env.reset(seed=int(reset_seed))
                # The predictive target is a fixed 25-step open-loop future.
                # qpos_match only changes early termination, not dynamics, so
                # leave it unset while replaying and score the held-out qpos
                # explicitly below.
                env.env.task.target_qpos = None
                env.set_state(
                    initial[batch_index, :nq],
                    initial[batch_index, nq:],
                )
                terminated = False
                latest_state = initial[batch_index].astype(
                    np.float64, copy=True
                )
                latest_pixels: np.ndarray | None = None
                for step_index, action in enumerate(
                    actions[batch_index, candidate_index]
                ):
                    if not terminated:
                        _, _, terminated, _, info = env.step(action)
                        latest_state = np.concatenate(
                            [info["qpos"], info["qvel"]]
                        )
                        if return_pixels:
                            latest_pixels = env.render()
                    if (step_index + 1) % block == 0:
                        model_step = (step_index + 1) // block - 1
                        block_states[
                            batch_index, candidate_index, model_step
                        ] = latest_state
                        if block_pixels is not None:
                            if latest_pixels is None:
                                latest_pixels = env.render()
                            block_pixels[
                                batch_index, candidate_index, model_step
                            ] = latest_pixels
                endpoints[batch_index, candidate_index] = latest_state
                goal_distances[batch_index, candidate_index] = np.linalg.norm(
                    latest_state[:nq] - goals[batch_index]
                )
    finally:
        env.close()
    result = {
        "block_states": block_states,
        "endpoints": endpoints,
        "goal_distances": goal_distances,
    }
    if block_pixels is not None:
        result["block_pixels"] = block_pixels
    return result


def replay_cube_action_pool(
    initial_states: torch.Tensor | np.ndarray,
    goal_states: torch.Tensor | np.ndarray,
    low_level_actions: torch.Tensor | np.ndarray,
    *,
    action_block: int,
    reset_seed: int = 0,
    return_pixels: bool = False,
) -> dict[str, np.ndarray]:
    """Replay a same-state single-cube pool from full MuJoCo qpos/qvel.

    goal_states concatenates the held-out block position and quaternion.
    Environment construction mirrors config/eval/cube.yaml.
    """

    import mujoco

    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    initial = (
        initial_states.detach().cpu().numpy()
        if torch.is_tensor(initial_states)
        else np.asarray(initial_states)
    )
    goals = (
        goal_states.detach().cpu().numpy()
        if torch.is_tensor(goal_states)
        else np.asarray(goal_states)
    )
    actions = (
        low_level_actions.detach().cpu().numpy()
        if torch.is_tensor(low_level_actions)
        else np.asarray(low_level_actions)
    )
    if initial.ndim != 2:
        raise ValueError("Cube initial_states must have shape (B,nq+nv)")
    if goals.ndim != 2 or goals.shape != (initial.shape[0], 7):
        raise ValueError(
            "Cube goal_states must concatenate position/quaternion as (B,7)"
        )
    if actions.ndim != 4 or actions.shape[0] != initial.shape[0]:
        raise ValueError("actions must have shape (B,K,L,A)")
    block = int(action_block)
    if block < 1 or actions.shape[2] % block:
        raise ValueError(
            "low-level action length must be divisible by action_block"
        )
    horizon = actions.shape[2] // block
    env = CubeEnv(
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        # The model predicts a fixed action horizon. Keep replay open-loop and
        # score distance to the held-out target explicitly after all steps.
        terminate_at_goal=False,
    )
    env.reset(seed=int(reset_seed))
    nq = int(env.model.nq)
    nv = int(env.model.nv)
    action_dim = int(np.prod(env.action_space.shape))
    if initial.shape[-1] != nq + nv:
        raise ValueError(
            f"Cube initial state dimension must be nq+nv={nq + nv}"
        )
    if actions.shape[-1] != action_dim:
        raise ValueError(
            f"Cube low-level action dimension must be {action_dim}"
        )

    state_dim = nq + nv
    endpoints = np.empty(
        (actions.shape[0], actions.shape[1], state_dim),
        dtype=np.float64,
    )
    block_states = np.empty(
        (actions.shape[0], actions.shape[1], horizon, state_dim),
        dtype=np.float64,
    )
    goal_distances = np.empty(actions.shape[:2], dtype=np.float64)
    block_pixels = (
        np.empty(
            (
                actions.shape[0],
                actions.shape[1],
                horizon,
                224,
                224,
                3,
            ),
            dtype=np.uint8,
        )
        if return_pixels
        else None
    )
    try:
        for batch_index in range(actions.shape[0]):
            for candidate_index in range(actions.shape[1]):
                # Rebuilding/modifying MJCF for every branch is unnecessary
                # and dominates Cube runtime. Reset only MjData: the compiled
                # model and visual variations are shared, while qpos/qvel,
                # ctrl, time, warm-start, mocap and success state are restored
                # independently before every candidate.
                mujoco.mj_resetData(env.model, env._data)
                env._reset_next_step = False
                env._success = False
                env.set_target_pos(
                    0,
                    goals[batch_index, :3],
                    goals[batch_index, 3:],
                )
                env.set_state(
                    initial[batch_index, :nq],
                    initial[batch_index, nq:],
                )
                terminated = False
                latest_state = initial[batch_index].astype(
                    np.float64, copy=True
                )
                latest_pixels: np.ndarray | None = None
                for step_index, action in enumerate(
                    actions[batch_index, candidate_index]
                ):
                    if not terminated:
                        _, _, terminated, _, _ = env.step(action)
                        latest_state = np.concatenate(
                            [env._data.qpos.copy(), env._data.qvel.copy()]
                        )
                        if return_pixels:
                            latest_pixels = env.render()
                    if (step_index + 1) % block == 0:
                        model_step = (step_index + 1) // block - 1
                        block_states[
                            batch_index, candidate_index, model_step
                        ] = latest_state
                        if block_pixels is not None:
                            if latest_pixels is None:
                                latest_pixels = env.render()
                            block_pixels[
                                batch_index, candidate_index, model_step
                            ] = latest_pixels
                endpoints[batch_index, candidate_index] = latest_state
                block_position = env._data.joint(
                    "object_joint_0"
                ).qpos[:3].copy()
                goal_distances[batch_index, candidate_index] = np.linalg.norm(
                    block_position - goals[batch_index, :3]
                )
    finally:
        env.close()
    result = {
        "block_states": block_states,
        "endpoints": endpoints,
        "goal_distances": goal_distances,
    }
    if block_pixels is not None:
        result["block_pixels"] = block_pixels
    return result


def replay_action_pool(
    task: str,
    initial_states: torch.Tensor | np.ndarray,
    goal_states: torch.Tensor | np.ndarray,
    low_level_actions: torch.Tensor | np.ndarray,
    *,
    action_block: int,
    reset_seed: int = 0,
    return_pixels: bool = False,
    prefix_initial_states: torch.Tensor | np.ndarray | None = None,
    prefix_actions: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Dispatch a target-aligned replay without changing task semantics."""

    normalized = str(task).lower().replace("-", "").replace("_", "")
    kwargs = {
        "action_block": int(action_block),
        "reset_seed": int(reset_seed),
        "return_pixels": bool(return_pixels),
    }
    if normalized in {"tworoom", "2room"}:
        if prefix_initial_states is not None or prefix_actions is not None:
            raise ValueError("episode-prefix replay is only defined for PushT")
        return replay_tworoom_action_pool(
            initial_states,
            goal_states,
            low_level_actions,
            **kwargs,
        )
    if normalized == "pusht":
        return replay_pusht_action_pool(
            initial_states,
            goal_states,
            low_level_actions,
            prefix_initial_states=prefix_initial_states,
            prefix_actions=prefix_actions,
            **kwargs,
        )
    if normalized in {"reacher", "reacherdmcontrol"}:
        if prefix_initial_states is not None or prefix_actions is not None:
            raise ValueError("Reacher replay does not use episode-prefix state")
        return replay_reacher_action_pool(
            initial_states,
            goal_states,
            low_level_actions,
            **kwargs,
        )
    if normalized in {"cube", "ogbcube"}:
        if prefix_initial_states is not None or prefix_actions is not None:
            raise ValueError("Cube replay does not use episode-prefix state")
        return replay_cube_action_pool(
            initial_states,
            goal_states,
            low_level_actions,
            **kwargs,
        )
    raise ValueError(f"unsupported target-aligned replay task: {task!r}")


def candidate_response_metrics(
    nominal_predictions: torch.Tensor,
    probe_predictions: torch.Tensor,
    *,
    encoder_response: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return per-history, per-candidate H1 and full-horizon responses."""

    nominal = nominal_predictions.detach().float()
    probe = probe_predictions.detach().float()
    if nominal.ndim != 4 or nominal.shape != probe.shape:
        raise ValueError("prediction pairs must share shape (B,K,H,D)")
    if nominal.size(2) < 1:
        raise ValueError("prediction horizon must be positive")
    delta = probe - nominal
    h1 = torch.linalg.vector_norm(delta[:, :, 0], dim=-1)
    full = horizon_weighted_stacked_l2(nominal, probe)
    endpoint = torch.linalg.vector_norm(delta[:, :, -1], dim=-1)
    result = {
        "candidate_h1_response": h1,
        "candidate_horizon_response": full,
        "candidate_endpoint_response": endpoint,
    }
    if encoder_response is not None:
        encoder = encoder_response.detach().float()
        if encoder.ndim != 1 or encoder.size(0) != nominal.size(0):
            raise ValueError("encoder_response must have shape (B,)")
        denominator = encoder[:, None].clamp_min(float(eps))
        result["candidate_h1_amplification"] = h1 / denominator
        result["candidate_horizon_amplification"] = full / denominator
    return result


def prediction_error_drift_certificate(
    nominal_predictions: torch.Tensor,
    probe_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Audit the exact reverse-triangle ACPC prediction-error bound.

    All three tensors must describe the same histories, candidates, horizon,
    embedding space, and (for the two prediction branches) action sequence.
    For the canonical weighted stacked L2 norm, reverse triangle inequality
    gives, sample by sample,

        |d(probe, target) - d(nominal, target)|
            <= d(probe, nominal).

    The right-hand side is available without observing ``targets`` at audit
    time. The target is used here only to validate tightness on held-out or
    simulator-replayed futures. A response computed with zeroed, shuffled,
    or otherwise mismatched actions is not a certificate for the correct-
    action prediction pair.
    """

    nominal = nominal_predictions.detach().float()
    probe = probe_predictions.detach().float()
    target = targets.detach().float()
    if nominal.ndim != 4 or nominal.shape != probe.shape:
        raise ValueError("prediction pairs must share shape (B,K,H,D)")
    if target.shape != nominal.shape:
        raise ValueError("targets must match prediction shape (B,K,H,D)")

    nominal_h1_error = torch.linalg.vector_norm(
        nominal[:, :, 0] - target[:, :, 0], dim=-1
    )
    probe_h1_error = torch.linalg.vector_norm(
        probe[:, :, 0] - target[:, :, 0], dim=-1
    )
    h1_response = torch.linalg.vector_norm(
        probe[:, :, 0] - nominal[:, :, 0], dim=-1
    )
    nominal_horizon_error = horizon_weighted_stacked_l2(nominal, target)
    probe_horizon_error = horizon_weighted_stacked_l2(probe, target)
    horizon_response = horizon_weighted_stacked_l2(nominal, probe)

    signed_h1_change = probe_h1_error - nominal_h1_error
    signed_horizon_change = probe_horizon_error - nominal_horizon_error
    absolute_h1_drift = signed_h1_change.abs()
    absolute_horizon_drift = signed_horizon_change.abs()
    return {
        "nominal_h1_error": nominal_h1_error,
        "probe_h1_error": probe_h1_error,
        "signed_h1_error_change": signed_h1_change,
        "adverse_h1_error_change": signed_h1_change.clamp_min(0.0),
        "absolute_h1_error_drift": absolute_h1_drift,
        "h1_response": h1_response,
        "h1_certificate_slack": h1_response - absolute_h1_drift,
        "nominal_horizon_error": nominal_horizon_error,
        "probe_horizon_error": probe_horizon_error,
        "signed_horizon_error_change": signed_horizon_change,
        "adverse_horizon_error_change": signed_horizon_change.clamp_min(0.0),
        "absolute_horizon_error_drift": absolute_horizon_drift,
        "horizon_response": horizon_response,
        "horizon_certificate_slack": horizon_response
        - absolute_horizon_drift,
    }


def permute_candidate_actions(
    candidates: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Destroy within-history action/target alignment while preserving support."""

    if candidates.ndim != 4 or candidates.size(1) < 2:
        raise ValueError("candidate permutation requires shape (B,K,H,A), K >= 2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutations = torch.stack(
        [
            torch.randperm(candidates.size(1), generator=generator)
            for _ in range(candidates.size(0))
        ]
    ).to(candidates.device)
    batch_index = torch.arange(candidates.size(0), device=candidates.device)[:, None]
    return candidates[batch_index, permutations], permutations


def permute_candidate_time(
    candidates: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Destroy temporal order per candidate while preserving action marginals."""

    if candidates.ndim != 4 or candidates.size(2) < 2:
        raise ValueError("time permutation requires shape (B,K,H,A), H >= 2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutations = torch.stack(
        [
            torch.randperm(candidates.size(2), generator=generator)
            for _ in range(candidates.size(0) * candidates.size(1))
        ]
    ).reshape(candidates.size(0), candidates.size(1), candidates.size(2))
    permutations = permutations.to(candidates.device)
    batch_index = torch.arange(candidates.size(0), device=candidates.device)[:, None, None]
    candidate_index = torch.arange(candidates.size(1), device=candidates.device)[None, :, None]
    return candidates[batch_index, candidate_index, permutations], permutations


__all__ = [
    "ActionCoordinateStats",
    "EvalStepZeroCandidateCapture",
    "PlannerTemporalSemantics",
    "SCHEMA_VERSION",
    "candidate_response_metrics",
    "capture_eval_step_zero_pool",
    "expand_planner_info",
    "fit_action_coordinate_stats",
    "inverse_eval_action_coordinates",
    "permute_candidate_actions",
    "permute_candidate_time",
    "planner_pool_costs",
    "planner_pool_predictions",
    "planner_query_info",
    "planner_temporal_semantics",
    "replay_tworoom_action_pool",
    "unpack_candidate_action_blocks",
]
