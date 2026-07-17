#!/usr/bin/env python3
"""ACPC-to-planner fixed-pool stability audit for Paper 1.

This runner is deliberately narrower than a robustness evaluator.  It samples
independent trajectory blocks from the current nominal dataset, captures the
actual ordered step-0 CEM proposal, and evaluates the same candidate pool on
nominal and additionally probed histories.  The primary event is an exact,
tie-aware winner comparison.  No behavior labels are read.

The module also exposes pure tensor helpers used by unit tests and by the
later risk-controlled cascade.  Formal result generation is allowed only when
the frozen operational protocol and an execution addendum are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.solver.callbacks.common import Callback

from tools import paper1_phase0_acpc as phase0
from tools.paper1_cem_trace_audit import _make_solver, _solver_info
from tools.paper1_operational_protocol import (
    namespace_arguments,
    validate_frozen_execution,
)
from utils import get_column_normalizer, get_img_preprocessor, resolve_h5_dataset_path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "paper1-acpc-planner-stability-audit-1.0"
TASK_DATASETS = dict(phase0.TASK_DATASETS)
PROBE_IDENTITIES = {
    "gaussian_noise": 0.0,
    "gaussian_blur": 1.0,
    "resize": 1.0,
}
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
    "pool_index",
    "n_blocks",
    "future_steps",
    "frameskip",
    "img_size",
    "candidate_count",
    "topk",
    "batch_size",
    "plan_horizon",
    "action_block",
    "trajectory_seed",
    "probe_seed",
    "cem_seed",
    "invariant_atol",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _clone_tensor_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in values.items()
    }


def _top2_margin(costs: torch.Tensor) -> torch.Tensor:
    if costs.ndim != 2 or costs.size(1) < 2:
        raise ValueError("costs must have shape (B,K) with K >= 2")
    sorted_costs = torch.sort(costs, dim=1, stable=True).values
    return sorted_costs[:, 1] - sorted_costs[:, 0]


def tie_aware_decision_metrics(
    nominal_costs: torch.Tensor,
    probe_costs: torch.Tensor,
    *,
    invariant_atol: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return exact lexicographic winner and signed-gap metrics.

    ``torch.argmin`` selects the first minimum and therefore implements the
    protocol's ordered/lexicographic tie rule.  Positive strict signed gap is
    equivalent to the nominal winner remaining the *unique* probe winner.  It
    is intentionally not treated as equivalent when a zero-gap tie is broken
    in favor of the nominal winner.
    """

    nominal = nominal_costs.detach().float()
    probe = probe_costs.detach().float()
    if nominal.ndim != 2 or probe.shape != nominal.shape or nominal.size(1) < 2:
        raise ValueError("nominal/probe costs must share shape (B,K), K >= 2")
    if not bool(torch.isfinite(nominal).all()) or not bool(torch.isfinite(probe).all()):
        raise ValueError("candidate costs must be finite")

    nominal_winner = torch.argmin(nominal, dim=1)
    probe_winner = torch.argmin(probe, dim=1)
    exact_stable = nominal_winner == probe_winner
    winner_index = nominal_winner.unsqueeze(1)

    nominal_winner_cost = nominal.gather(1, winner_index)
    probe_winner_reference_cost = probe.gather(1, winner_index)
    nominal_gap = nominal - nominal_winner_cost
    signed_drift = probe - nominal
    winner_signed_drift = signed_drift.gather(1, winner_index)
    signed_probe_gap = nominal_gap + signed_drift - winner_signed_drift
    direct_probe_gap = probe - probe_winner_reference_cost
    if not torch.allclose(
        signed_probe_gap,
        direct_probe_gap,
        rtol=0.0,
        atol=float(invariant_atol),
    ):
        raise RuntimeError("signed perturbed-gap identity mismatch")

    competitor_gap = signed_probe_gap.clone()
    competitor_gap.scatter_(1, winner_index, float("inf"))
    min_competitor_gap = competitor_gap.min(dim=1).values
    strict_unique_same_winner = min_competitor_gap > 0.0
    if bool(strict_unique_same_winner.logical_and(~exact_stable).any()):
        raise RuntimeError("positive strict gap with changed tie-aware winner")

    probe_min = probe.min(dim=1, keepdim=True).values
    probe_tie_count = torch.isclose(
        probe,
        probe_min,
        rtol=0.0,
        atol=float(invariant_atol),
    ).sum(dim=1)
    nominal_min = nominal.min(dim=1, keepdim=True).values
    nominal_tie_count = torch.isclose(
        nominal,
        nominal_min,
        rtol=0.0,
        atol=float(invariant_atol),
    ).sum(dim=1)

    absolute_drift = signed_drift.abs()
    return {
        "nominal_winner": nominal_winner,
        "probe_winner": probe_winner,
        "exact_stable": exact_stable,
        "strict_unique_same_winner": strict_unique_same_winner,
        "min_signed_probe_gap": min_competitor_gap,
        "nominal_margin": _top2_margin(nominal),
        "probe_margin": _top2_margin(probe),
        "nominal_tie_count": nominal_tie_count,
        "probe_tie_count": probe_tie_count,
        "signed_drift": signed_drift,
        "max_absolute_drift": absolute_drift.max(dim=1).values,
        "mean_absolute_drift": absolute_drift.mean(dim=1),
        "nominal_cost_std": nominal.std(dim=1, unbiased=False),
        "probe_cost_std": probe.std(dim=1, unbiased=False),
    }


def mse_cost_acpc_metrics(
    nominal_costs: torch.Tensor,
    probe_costs: torch.Tensor,
    nominal_final: torch.Tensor,
    probe_final: torch.Tensor,
    goal_final: torch.Tensor,
    *,
    topk: int,
    invariant_atol: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Evaluate the exact MSE-cost bound and its top-1/elite certificates.

    LeWM uses C(x,g)=||x-g||_2^2 at the final predicted latent.  Therefore

        |C(x,g)-C(y,g)|
        <= ||x-y||_2 (||x-g||_2 + ||y-g||_2).

    The first factor is the final-step cost-space ACPC response.  Candidate-wise
    upper bounds can be compared with clean top-1 and elite-boundary margins
    without estimating an unknown global Lipschitz constant.
    """

    nominal = nominal_costs.detach().float()
    probe = probe_costs.detach().float()
    nominal_z = nominal_final.detach().float()
    probe_z = probe_final.detach().float()
    goal = goal_final.detach().float()
    if nominal.ndim != 2 or probe.shape != nominal.shape:
        raise ValueError("nominal/probe costs must share shape (B,K)")
    if nominal_z.shape != probe_z.shape or nominal_z.ndim != 3:
        raise ValueError("nominal/probe final latents must share shape (B,K,D)")
    if nominal_z.shape[:2] != nominal.shape:
        raise ValueError("latent and cost candidate axes must agree")
    if goal.ndim == 2:
        goal = goal.unsqueeze(1).expand_as(nominal_z)
    if goal.shape != nominal_z.shape:
        raise ValueError("goal_final must have shape (B,D) or (B,K,D)")
    candidate_count = int(nominal.size(1))
    if not 1 <= int(topk) < candidate_count:
        raise ValueError("topk must lie in [1,K) for an elite-boundary certificate")

    acpc = torch.linalg.vector_norm(probe_z - nominal_z, dim=-1)
    nominal_goal = torch.linalg.vector_norm(nominal_z - goal, dim=-1)
    probe_goal = torch.linalg.vector_norm(probe_z - goal, dim=-1)
    bound = acpc * (nominal_goal + probe_goal)
    actual_drift = (probe - nominal).abs()
    bound_slack = bound - actual_drift
    bound_holds = bound_slack >= -float(invariant_atol)

    order = torch.argsort(nominal, dim=1, stable=True)
    probe_order = torch.argsort(probe, dim=1, stable=True)
    winner = order[:, :1]
    winner_bound = bound.gather(1, winner)
    clean_gap = nominal - nominal.gather(1, winner)
    top1_slack = clean_gap - bound - winner_bound
    top1_slack.scatter_(1, winner, float("inf"))
    min_top1_slack = top1_slack.min(dim=1).values
    top1_certificate = min_top1_slack > 0.0

    elite_indices = order[:, : int(topk)]
    nonelite_indices = order[:, int(topk) :]
    elite_upper = (
        nominal.gather(1, elite_indices) + bound.gather(1, elite_indices)
    ).max(dim=1).values
    nonelite_lower = (
        nominal.gather(1, nonelite_indices) - bound.gather(1, nonelite_indices)
    ).min(dim=1).values
    elite_certificate_slack = nonelite_lower - elite_upper
    elite_certificate = elite_certificate_slack > 0.0
    nominal_elite = elite_indices
    probe_elite = probe_order[:, : int(topk)]
    elite_jaccard = []
    elite_set_stable = []
    for left, right in zip(nominal_elite, probe_elite, strict=True):
        left_set = set(int(value) for value in left.tolist())
        right_set = set(int(value) for value in right.tolist())
        elite_set_stable.append(left_set == right_set)
        elite_jaccard.append(
            len(left_set & right_set) / len(left_set | right_set)
        )
    elite_set_stable_t = torch.tensor(elite_set_stable, dtype=torch.bool)
    elite_jaccard_t = torch.tensor(elite_jaccard, dtype=torch.float32)

    if bool(top1_certificate.logical_and(
        torch.argmin(nominal, dim=1) != torch.argmin(probe, dim=1)
    ).any()):
        raise RuntimeError("ACPC top-1 certificate passed with a changed winner")
    if bool(elite_certificate.logical_and(~elite_set_stable_t).any()):
        raise RuntimeError("ACPC elite certificate passed with a changed elite set")

    return {
        "cost_space_acpc_final_l2": acpc,
        "mse_cost_drift_upper_bound": bound,
        "mse_bound_slack": bound_slack,
        "mse_bound_holds": bound_holds,
        "max_cost_space_acpc_final_l2": acpc.max(dim=1).values,
        "q90_cost_space_acpc_final_l2": torch.quantile(acpc, 0.90, dim=1),
        "mean_cost_space_acpc_final_l2": acpc.mean(dim=1),
        "max_mse_cost_drift_upper_bound": bound.max(dim=1).values,
        "minimum_mse_bound_slack": bound_slack.min(dim=1).values,
        "all_mse_bounds_hold": bound_holds.all(dim=1),
        "acpc_top1_certificate": top1_certificate,
        "acpc_top1_certificate_slack": min_top1_slack,
        "acpc_elite_certificate": elite_certificate,
        "acpc_elite_certificate_slack": elite_certificate_slack,
        "nominal_elite_boundary_margin": (
            nominal.gather(1, nonelite_indices[:, :1]).squeeze(1)
            - nominal.gather(1, elite_indices[:, -1:]).squeeze(1)
        ),
        "exact_elite_set_stable": elite_set_stable_t,
        "elite_jaccard": elite_jaccard_t,
    }


def candidate_latent_acpc_metrics(
    nominal: torch.Tensor,
    probe: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize candidate-wise latent drift for a fixed rollout step."""

    nominal_z = nominal.detach().float()
    probe_z = probe.detach().float()
    if nominal_z.shape != probe_z.shape or nominal_z.ndim != 3:
        raise ValueError("candidate latents must share shape (B,K,D)")
    acpc = torch.linalg.vector_norm(probe_z - nominal_z, dim=-1)
    return {
        "candidate_acpc_l2": acpc,
        "max_candidate_acpc_l2": acpc.max(dim=1).values,
        "q90_candidate_acpc_l2": torch.quantile(acpc, 0.90, dim=1),
        "mean_candidate_acpc_l2": acpc.mean(dim=1),
    }


def choose_unique_episode_indices(
    clip_indices: Sequence[Sequence[int]],
    *,
    n_blocks: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Choose one dataset clip per distinct episode.

    Returns ``(dataset_index, episode_id, start_step)`` in a seeded random
    episode order.  Starts are sampled independently within selected episodes.
    This helper is pure and unit-testable.
    """

    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    episodes = sorted({int(pair[0]) for pair in clip_indices})
    if len(episodes) < n_blocks:
        raise ValueError(
            f"requested {n_blocks} trajectory blocks but only {len(episodes)} episodes are available"
        )
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(episodes), generator=generator)[:n_blocks]
    selected = [episodes[int(index)] for index in permutation]
    selected_set = set(selected)
    candidates: dict[int, list[tuple[int, int]]] = {episode: [] for episode in selected}
    for dataset_index, pair in enumerate(clip_indices):
        episode, start = int(pair[0]), int(pair[1])
        if episode in selected_set:
            candidates[episode].append((dataset_index, start))
    result: list[tuple[int, int, int]] = []
    for episode in selected:
        choices = candidates[episode]
        if not choices:
            raise RuntimeError(f"selected episode {episode} has no valid clip")
        position = int(torch.randint(len(choices), (1,), generator=generator).item())
        dataset_index, start = choices[position]
        result.append((int(dataset_index), int(episode), int(start)))
    return result


@dataclass(frozen=True)
class TrajectoryBlock:
    block_index: int
    dataset_index: int
    episode_id: int
    start_step: int

    @property
    def block_id(self) -> str:
        return f"episode-{self.episode_id}"


def load_trajectory_blocks(
    *,
    dataset_name: str,
    n_blocks: int,
    history_size: int,
    future_steps: int,
    frameskip: int,
    img_size: int,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[TrajectoryBlock]]:
    """Load one transformed clip from each independently selected episode."""

    num_steps = int(history_size) + int(future_steps)
    h5_path = resolve_h5_dataset_path(dataset_name)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path),
        num_steps=num_steps,
        frameskip=int(frameskip),
        keys_to_load=["pixels", "action"],
        transform=None,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", int(img_size)),
        get_column_normalizer(dataset, "action", "action"),
    )
    chosen = choose_unique_episode_indices(
        dataset.clip_indices,
        n_blocks=int(n_blocks),
        seed=int(seed),
    )
    samples = [dataset[dataset_index] for dataset_index, _episode, _start in chosen]
    batch = {
        "pixels": torch.stack([sample["pixels"] for sample in samples]).to(device),
        "action": torch.nan_to_num(
            torch.stack([sample["action"] for sample in samples]),
            0.0,
        ).to(device),
    }
    blocks = [
        TrajectoryBlock(
            block_index=index,
            dataset_index=dataset_index,
            episode_id=episode,
            start_step=start,
        )
        for index, (dataset_index, episode, start) in enumerate(chosen)
    ]
    if len({block.episode_id for block in blocks}) != len(blocks):
        raise RuntimeError("trajectory block sampler returned duplicate episodes")
    return batch, blocks


class StepZeroCandidateCapture(Callback):
    """Capture actual ordered CEM step-0 candidates and nominal costs."""

    name = "StepZeroCandidateCapture"

    def __init__(self) -> None:
        super().__init__(reduction="none")

    def compute(self, **state: Any) -> dict[str, torch.Tensor] | None:
        if int(state["step"]) != 0:
            return None
        return {
            "candidates": state["candidates"].detach().cpu(),
            "costs": state["costs"].detach().float().cpu(),
        }


def _capture_step_zero_pool(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    action_block: int,
    plan_horizon: int,
    candidate_count: int,
    topk: int,
    batch_size: int,
    cem_seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    capture = StepZeroCandidateCapture()
    args = argparse.Namespace(
        action_block=int(action_block),
        batch_size=int(batch_size),
        num_samples=int(candidate_count),
        var_scale=1.0,
        n_steps=1,
        topk=int(topk),
        cem_seed=int(cem_seed),
        plan_horizon=int(plan_horizon),
        history_size_for_plan=int(history_size),
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
    output = solver(_solver_info(batch, int(history_size)))
    history = output["callbacks"][capture.output_key]
    candidate_batches: list[torch.Tensor] = []
    cost_batches: list[torch.Tensor] = []
    for records in history:
        if len(records) != 1:
            raise RuntimeError("step-0 capture must contain exactly one record per solver batch")
        candidate_batches.append(records[0]["candidates"])
        cost_batches.append(records[0]["costs"])
    candidates = torch.cat(candidate_batches, dim=0)
    costs = torch.cat(cost_batches, dim=0)
    expected = (int(batch["pixels"].shape[0]), int(candidate_count))
    if tuple(candidates.shape[:2]) != expected or tuple(costs.shape) != expected:
        raise RuntimeError(
            f"captured step-0 pool shape mismatch: candidates={tuple(candidates.shape)}, costs={tuple(costs.shape)}, expected prefix={expected}"
        )
    return candidates, costs


def _expanded_solver_info(
    batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    candidate_count: int,
) -> dict[str, torch.Tensor]:
    info = _solver_info(batch, int(history_size))
    expanded: dict[str, torch.Tensor] = {}
    for key, value in info.items():
        # Dataset transforms can return channels-last-strided tensors while a
        # no-op probe assembled with ``torch.cat`` is contiguous NCHW.  Equal
        # pixel values must not take different model kernels merely because of
        # storage layout, so canonicalise before adding the candidate axis.
        canonical = value.contiguous()
        expanded[key] = canonical.unsqueeze(1).expand(
            int(value.shape[0]),
            int(candidate_count),
            *value.shape[1:],
        )
    return expanded


@torch.inference_mode()
def _shared_pool_cost_details(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    candidates_cpu: torch.Tensor,
    *,
    history_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    candidates = candidates_cpu.to(device)
    info = _expanded_solver_info(
        batch,
        history_size=int(history_size),
        candidate_count=int(candidates.shape[1]),
    )
    costs = model.get_cost(info, candidates)
    if costs.ndim != 2 or tuple(costs.shape) != tuple(candidates.shape[:2]):
        raise RuntimeError(
            f"model.get_cost returned {tuple(costs.shape)} for candidate pool {tuple(candidates.shape)}"
        )
    predicted = info.get("predicted_emb")
    goal = info.get("goal_emb")
    if not torch.is_tensor(predicted) or predicted.ndim != 4:
        raise RuntimeError("model.get_cost did not expose predicted_emb with shape (B,K,T,D)")
    if not torch.is_tensor(goal) or goal.ndim != 3:
        raise RuntimeError("model.get_cost did not expose goal_emb with shape (B,T,D)")
    final = predicted[..., -1, :]
    goal_final = goal[..., -1, :]
    if tuple(final.shape[:2]) != tuple(costs.shape):
        raise RuntimeError("predicted final latent and candidate-cost axes differ")
    return {
        "costs": costs.detach().float().cpu(),
        "first": predicted[..., 0, :].detach().float().cpu(),
        "final": final.detach().float().cpu(),
        "goal_final": goal_final.detach().float().cpu(),
    }


@torch.inference_mode()
def _shared_pool_costs(
    model: Any,
    batch: Mapping[str, torch.Tensor],
    candidates_cpu: torch.Tensor,
    *,
    history_size: int,
    device: str,
) -> torch.Tensor:
    """Compatibility path for models that expose only get_cost."""

    candidates = candidates_cpu.to(device)
    info = _expanded_solver_info(
        batch,
        history_size=int(history_size),
        candidate_count=int(candidates.shape[1]),
    )
    costs = model.get_cost(info, candidates)
    if costs.ndim != 2 or tuple(costs.shape) != tuple(candidates.shape[:2]):
        raise RuntimeError(
            f"model.get_cost returned {tuple(costs.shape)} for candidate pool "
            f"{tuple(candidates.shape)}"
        )
    return costs.detach().float().cpu()


def _resolve_checkpoint(path: str, model_roots: Sequence[Path]) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file():
        return checkpoint.resolve()
    model_file, tried = phase0.resolve_model_file(path, checkpoint.name, model_roots)
    if model_file is None:
        raise FileNotFoundError(f"could not resolve checkpoint {path!r}; tried {tried}")
    return model_file.resolve()


def _validate_severities(family: str, severities: Sequence[float]) -> list[float]:
    if family not in PROBE_IDENTITIES:
        raise ValueError(f"unsupported probe family: {family}")
    values = [float(value) for value in severities]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("severities must be a non-empty finite sequence")
    identity = PROBE_IDENTITIES[family]
    if not math.isclose(values[0], identity, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"first severity must be identity {identity} for {family}")
    if len(set(values)) != len(values):
        raise ValueError("severities must be unique")
    if family in {"gaussian_noise", "gaussian_blur"} and values != sorted(values):
        raise ValueError(f"{family} severities must be ordered weak-to-strong")
    if family == "resize" and values != sorted(values, reverse=True):
        raise ValueError("resize severities must be ordered weak-to-strong")
    return values


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    phase0._ensure_runtime_deps()
    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    severities = _validate_severities(args.probe_family, args.severities)
    if int(args.candidate_count) < 2:
        raise ValueError("candidate_count must be at least 2")
    if int(args.topk) < 1 or int(args.topk) > int(args.candidate_count):
        raise ValueError("topk must lie in [1,candidate_count]")
    if int(args.draws) < 1:
        raise ValueError("draws must be positive")

    checkpoint = _resolve_checkpoint(
        args.checkpoint,
        [Path(root).expanduser() for root in args.model_root],
    )
    for parent in checkpoint.parents:
        if parent.name == "ckpt":
            os.environ.setdefault("STABLEWM_HOME", str(parent.parent))
            break
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    model_started = time.perf_counter()
    model = phase0.load_model(str(checkpoint), device).eval()
    model_load_time = time.perf_counter() - model_started
    history_size = int(phase0.infer_history_size(model))

    data_started = time.perf_counter()
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
    data_time = time.perf_counter() - data_started

    pool_started = time.perf_counter()
    candidates, captured_nominal_costs = _capture_step_zero_pool(
        model,
        batch,
        history_size=history_size,
        action_block=int(args.action_block),
        plan_horizon=int(args.plan_horizon),
        candidate_count=int(args.candidate_count),
        topk=int(args.topk),
        batch_size=int(args.batch_size),
        cem_seed=int(args.cem_seed),
        device=device,
    )
    # The solver capture is used only to obtain the actual ordered proposal.
    # Recompute nominal costs through the exact same public path used for every
    # probe branch.  Mixing the solver's internal batching path with
    # ``_shared_pool_costs`` created a non-zero identity drift even though the
    # ordered winner was unchanged.
    nominal_details = _shared_pool_cost_details(
        model,
        batch,
        candidates,
        history_size=history_size,
        device=device,
    )
    nominal_costs = nominal_details["costs"]
    if captured_nominal_costs.shape != nominal_costs.shape:
        raise RuntimeError(
            "captured and recomputed nominal cost shapes differ: "
            f"{tuple(captured_nominal_costs.shape)} vs {tuple(nominal_costs.shape)}"
        )
    captured_nominal_max_abs_difference = float(
        (captured_nominal_costs - nominal_costs).abs().max().item()
    )
    pool_time = time.perf_counter() - pool_started

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    candidate_hashes = [_tensor_sha256(candidates[index]) for index in range(len(blocks))]
    for severity_index, severity in enumerate(severities):
        for draw_index in range(int(args.draws)):
            draw_seed = (
                int(args.probe_seed)
                + 1009 * severity_index
                + 7919 * draw_index
            )
            try:
                probe_batch = phase0.make_paired_noisy_batch(
                    _clone_tensor_mapping(batch),
                    history_size=history_size,
                    noise_std=float(severity),
                    seed=int(draw_seed),
                    corruption_type=str(args.probe_family),
                    corrupt_goal=False,
                )
                probe_details = _shared_pool_cost_details(
                    model,
                    probe_batch,
                    candidates,
                    history_size=history_size,
                    device=device,
                )
                probe_costs = probe_details["costs"]
                if not torch.allclose(
                    nominal_details["goal_final"],
                    probe_details["goal_final"],
                    rtol=0.0,
                    atol=float(args.invariant_atol),
                ):
                    raise RuntimeError("history-only probe changed the goal embedding")
                metrics = tie_aware_decision_metrics(
                    nominal_costs,
                    probe_costs,
                    invariant_atol=float(args.invariant_atol),
                )
                acpc_metrics = mse_cost_acpc_metrics(
                    nominal_costs,
                    probe_costs,
                    nominal_details["final"],
                    probe_details["final"],
                    nominal_details["goal_final"],
                    topk=int(args.topk),
                    invariant_atol=float(args.invariant_atol),
                )
                h1_acpc_metrics = candidate_latent_acpc_metrics(
                    nominal_details["first"],
                    probe_details["first"],
                )
                for block_index, block in enumerate(blocks):
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
                            "dataset_index": block.dataset_index,
                            "episode_id": block.episode_id,
                            "start_step": block.start_step,
                            "probe_family": args.probe_family,
                            "severity_index": int(severity_index),
                            "severity": float(severity),
                            "draw_index": int(draw_index),
                            "draw_seed": int(draw_seed),
                            "pool_index": int(args.pool_index),
                            "candidate_count": int(candidates.shape[1]),
                            "candidate_pool_sha256": candidate_hashes[block_index],
                            "candidate_source": "actual_cem_step0_ordered_proposal",
                            "cem_seed": int(args.cem_seed),
                            "nominal_winner": int(metrics["nominal_winner"][block_index]),
                            "probe_winner": int(metrics["probe_winner"][block_index]),
                            "exact_stable": bool(metrics["exact_stable"][block_index]),
                            "strict_unique_same_winner": bool(
                                metrics["strict_unique_same_winner"][block_index]
                            ),
                            "min_signed_probe_gap": float(
                                metrics["min_signed_probe_gap"][block_index]
                            ),
                            "nominal_margin": float(metrics["nominal_margin"][block_index]),
                            "probe_margin": float(metrics["probe_margin"][block_index]),
                            "nominal_tie_count": int(metrics["nominal_tie_count"][block_index]),
                            "probe_tie_count": int(metrics["probe_tie_count"][block_index]),
                            "max_absolute_cost_drift": float(
                                metrics["max_absolute_drift"][block_index]
                            ),
                            "mean_absolute_cost_drift": float(
                                metrics["mean_absolute_drift"][block_index]
                            ),
                            "nominal_cost_std": float(metrics["nominal_cost_std"][block_index]),
                            "probe_cost_std": float(metrics["probe_cost_std"][block_index]),
                            "max_cost_space_acpc_final_l2": float(
                                acpc_metrics["max_cost_space_acpc_final_l2"][block_index]
                            ),
                            "q90_cost_space_acpc_final_l2": float(
                                acpc_metrics["q90_cost_space_acpc_final_l2"][block_index]
                            ),
                            "mean_cost_space_acpc_final_l2": float(
                                acpc_metrics["mean_cost_space_acpc_final_l2"][block_index]
                            ),
                            "max_candidate_h1_acpc_l2": float(
                                h1_acpc_metrics["max_candidate_acpc_l2"][block_index]
                            ),
                            "q90_candidate_h1_acpc_l2": float(
                                h1_acpc_metrics["q90_candidate_acpc_l2"][block_index]
                            ),
                            "mean_candidate_h1_acpc_l2": float(
                                h1_acpc_metrics["mean_candidate_acpc_l2"][block_index]
                            ),
                            "max_mse_cost_drift_upper_bound": float(
                                acpc_metrics["max_mse_cost_drift_upper_bound"][block_index]
                            ),
                            "minimum_mse_bound_slack": float(
                                acpc_metrics["minimum_mse_bound_slack"][block_index]
                            ),
                            "all_mse_bounds_hold": bool(
                                acpc_metrics["all_mse_bounds_hold"][block_index]
                            ),
                            "acpc_top1_certificate": bool(
                                acpc_metrics["acpc_top1_certificate"][block_index]
                            ),
                            "acpc_top1_certificate_slack": float(
                                acpc_metrics["acpc_top1_certificate_slack"][block_index]
                            ),
                            "acpc_elite_certificate": bool(
                                acpc_metrics["acpc_elite_certificate"][block_index]
                            ),
                            "acpc_elite_certificate_slack": float(
                                acpc_metrics["acpc_elite_certificate_slack"][block_index]
                            ),
                            "nominal_elite_boundary_margin": float(
                                acpc_metrics["nominal_elite_boundary_margin"][block_index]
                            ),
                            "exact_elite_set_stable": bool(
                                acpc_metrics["exact_elite_set_stable"][block_index]
                            ),
                            "elite_jaccard": float(
                                acpc_metrics["elite_jaccard"][block_index]
                            ),
                            "candidate_action_std": float(
                                candidates[block_index].float().std(unbiased=False)
                            ),
                            "goal_corrupted": False,
                            "behavior_blind": True,
                            "status": "ok",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - every attempted cell is retained.
                errors.append(
                    {
                        "probe_family": args.probe_family,
                        "severity_index": int(severity_index),
                        "severity": float(severity),
                        "draw_index": int(draw_index),
                        "error": repr(exc),
                    }
                )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated(torch.device(device)))
    else:
        peak_memory = 0
    wall_time = time.perf_counter() - started
    expected_rows = len(blocks) * len(severities) * int(args.draws)
    status = "complete" if not errors and len(rows) == expected_rows else "partial"
    script_path = Path(__file__).resolve()
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "protocol_path": str(args.protocol),
            "protocol_sha256": _sha256(Path(args.protocol)) if args.protocol else None,
            "execution_addendum_path": str(args.execution_addendum) if args.execution_addendum else None,
            "execution_addendum_sha256": (
                _sha256(Path(args.execution_addendum)) if args.execution_addendum else None
            ),
            "status": status,
            "expected_rows": expected_rows,
            "actual_rows": len(rows),
            "error_count": len(errors),
            "behavior_blind": True,
            "decision_target": "fixed_pool_tie_aware_top1",
            "cost_readout": "final_latent_squared_l2_to_fixed_goal",
            "acpc_cost_bound": (
                "|delta cost| <= ||z_probe-z_nominal||_2 * "
                "(||z_nominal-goal||_2 + ||z_probe-goal||_2)"
            ),
            "short_horizon_comparator": "same_pool_candidate_conditioned_H1_ACPC",
            "candidate_source": "actual_cem_step0_ordered_proposal",
            "nominal_cost_reference": "shared_pool_recompute",
            "captured_nominal_costs_used_for_decision": False,
            "captured_nominal_max_absolute_difference": (
                captured_nominal_max_abs_difference
            ),
            "independent_unit": "trajectory episode block",
            "nested_units": ["probe draw", "candidate pool", "candidate"],
            "method": args.method,
            "task": args.task,
            "training_seed": int(args.training_seed),
            "checkpoint_role": args.checkpoint_role,
            "anonymous_checkpoint_id": args.anonymous_checkpoint_id,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "dataset_name": args.dataset_name or TASK_DATASETS[args.task],
            "n_trajectory_blocks": len(blocks),
            "history_size": history_size,
            "probe_family": args.probe_family,
            "severities": severities,
            "draws": int(args.draws),
            "pool_index": int(args.pool_index),
            "trajectory_seed": int(args.trajectory_seed),
            "probe_seed": int(args.probe_seed),
            "cem_seed": int(args.cem_seed),
            "candidate_count": int(args.candidate_count),
            "topk": int(args.topk),
            "plan_horizon": int(args.plan_horizon),
            "action_block": int(args.action_block),
            "goal_corrupted": False,
            "model_load_time_seconds": model_load_time,
            "data_time_seconds": data_time,
            "step0_pool_time_seconds": pool_time,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_bytes": peak_memory,
            "candidate_forward_calls_pool_capture": (
                len(blocks) * int(args.candidate_count)
            ),
            "candidate_forward_calls_nominal_recompute": (
                len(blocks) * int(args.candidate_count)
            ),
            "candidate_forward_calls_nominal": (
                2 * len(blocks) * int(args.candidate_count)
            ),
            "candidate_forward_calls_probe": (
                len(blocks)
                * int(args.candidate_count)
                * len(severities)
                * int(args.draws)
            ),
            "candidate_forward_calls_total": (
                len(blocks)
                * int(args.candidate_count)
                * (2 + len(severities) * int(args.draws))
            ),
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
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--pool-index", type=int, default=0)
    parser.add_argument("--n-blocks", type=int, default=16)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--trajectory-seed", type=int, default=9101)
    parser.add_argument("--probe-seed", type=int, default=20260712)
    parser.add_argument("--cem-seed", type=int, default=1234)
    parser.add_argument("--invariant-atol", type=float, default=1e-6)
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
    print("status:", payload["metadata"]["status"])
    return 0 if payload["metadata"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
