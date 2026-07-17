#!/usr/bin/env python3
"""Build the V1 SMPR sensitivity, control, and two-task oracle MVE artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from paper1.scripts import smpr_sensitivity as canonical
from tools import paper1_phase0_acpc as phase0
from tools import paper1_semantic_margin as semantic
from tools.paper1_acpc_metrics import per_anchor_clean_transition_scale
from utils import resolve_h5_dataset_path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT")
STD_KEYS = ("0.0", "0.08")
MODEL_ROOTS = {
    "TwoRoom": "/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms",
    "PushT": "/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht",
}
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
NOISE_DRAW_LEVELS = (1, 5)
RADIUS_QUANTILES = (0.80, 0.90, 0.95)
LOCAL_QUANTILES = (0.10, 0.25, 0.35, 0.50)
MARGIN_DELTAS = (0.00, 0.05, 0.10, 0.25)
LABEL_BINNINGS = ("median", "quartile", "fixed_physical")
COLLAPSE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    _require(isinstance(payload, dict), f"{path}: JSON root must be an object")
    return payload


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{name}: value is not finite")
    return result


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    return canonical._distribution(values)


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "same_state_tube_radius": _finite(
            metrics["same_state_tube_radius"], name="tube radius"
        ),
        "smpr": _finite(metrics["smpr"], name="SMPR"),
        "same_state_radius_distribution": _distribution(
            metrics["same_state_radius_per_anchor"]
        ),
        "different_state_distance_distribution": _distribution(
            metrics["different_state_distance_per_pair"]
        ),
        "raw_margin_distribution": _distribution(
            metrics["raw_margin_to_tube_per_pair"]
        ),
        "pair_count": int(metrics["different_state_distance_per_pair"].numel()),
    }


@lru_cache(maxsize=None)
def _column_stats(task_home: str, dataset_name: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    old_home = os.environ.get("STABLEWM_HOME")
    os.environ["STABLEWM_HOME"] = task_home
    try:
        path = resolve_h5_dataset_path(dataset_name)
    finally:
        if old_home is None:
            os.environ.pop("STABLEWM_HOME", None)
        else:
            os.environ["STABLEWM_HOME"] = old_home
    with h5py.File(path, "r") as stream:
        values = np.asarray(stream[key])
    values = values[~np.isnan(values).any(axis=1)]
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def _denormalize(
    values: torch.Tensor,
    *,
    task_home: str,
    dataset_name: str,
    key: str,
) -> torch.Tensor:
    mean, std = _column_stats(task_home, dataset_name, key)
    mean_t = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
    std_t = torch.as_tensor(std, dtype=values.dtype, device=values.device)
    return values * std_t + mean_t


def _quartile_labels(feature: torch.Tensor) -> torch.Tensor:
    _require(feature.ndim == 2 and feature.size(0) >= 2, "quartile feature shape")
    labels = torch.zeros(feature.size(0), dtype=torch.long, device=feature.device)
    multiplier = 1
    for dim in range(feature.size(1)):
        q = torch.quantile(
            feature[:, dim].float(),
            torch.tensor([0.25, 0.50, 0.75], device=feature.device),
        )
        bucket = (
            (feature[:, dim] > q[0]).long()
            + (feature[:, dim] > q[1]).long()
            + (feature[:, dim] > q[2]).long()
        )
        labels += multiplier * bucket
        multiplier *= 4
    return labels


def _fixed_physical_labels(
    *,
    task: str,
    state_raw: torch.Tensor,
    target_raw: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    if task == "TwoRoom":
        agent_side = (state_raw[:, 0] >= 112.0).long()
        target_side = (target_raw[:, 0] >= 112.0).long()
        return (
            agent_side + 2 * target_side,
            "fixed wall-side cells at environment WALL_CENTER=112 for agent and target",
        )
    block = state_raw[:, 2:4]
    pusher = state_raw[:, :2]
    goal_block = target_raw[:, 2:4]
    relation = block - goal_block
    angle_delta = torch.atan2(
        torch.sin(state_raw[:, 4] - target_raw[:, 4]),
        torch.cos(state_raw[:, 4] - target_raw[:, 4]),
    )
    contact = (torch.linalg.vector_norm(pusher - block, dim=1) <= 70.0).long()
    labels = (
        (relation[:, 0] >= 0).long()
        + 2 * (relation[:, 1] >= 0).long()
        + 4 * (angle_delta >= 0).long()
        + 8 * contact
    )
    return (
        labels,
        "sequence-goal block x/y/angle side plus pusher-block contact proxy (70 px)",
    )


def _labels(
    *,
    task: str,
    state_norm: torch.Tensor,
    state_raw: torch.Tensor,
    target_raw: torch.Tensor,
    binning: str,
) -> tuple[torch.Tensor, str]:
    if binning == "median":
        labels, rule, _ = semantic._task_grounded_labels(task, state_norm)
        return labels, rule
    if binning == "quartile":
        feature = semantic._task_feature(task, state_norm).float()
        if task == "TwoRoom":
            feature = feature[:, :1]
        else:
            feature = feature[:, : min(3, feature.size(1))]
        return _quartile_labels(feature), "task feature marginal quartile cells"
    if binning == "fixed_physical":
        return _fixed_physical_labels(
            task=task,
            state_raw=state_raw,
            target_raw=target_raw,
        )
    raise ValueError(f"unknown label binning: {binning}")


def _select_pairs(
    *,
    state: torch.Tensor,
    labels: torch.Tensor | None,
    local_quantile: float,
    mode: str,
    oracle_value: torch.Tensor | None = None,
    oracle_delta: float = 0.10,
) -> dict[str, Any]:
    _require(state.ndim == 2 and state.size(0) >= 2, "state pair matrix shape")
    n = state.size(0)
    dist = torch.cdist(state.float(), state.float(), p=2)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=state.device)
    threshold = torch.quantile(dist[offdiag], float(local_quantile))
    anchors: list[int] = []
    neighbors: list[int] = []
    for anchor in range(n):
        if mode == "far":
            candidates = torch.nonzero(offdiag[anchor], as_tuple=False).flatten()
            neighbor = candidates[torch.argmax(dist[anchor, candidates])]
        else:
            valid = offdiag[anchor] & (dist[anchor] <= threshold)
            if mode == "cross_label":
                _require(labels is not None, "cross-label selection needs labels")
                valid &= labels != labels[anchor]
            elif mode == "same_label":
                _require(labels is not None, "same-label selection needs labels")
                valid &= labels == labels[anchor]
            elif mode == "oracle":
                _require(oracle_value is not None, "oracle selection needs values")
                valid &= torch.abs(oracle_value - oracle_value[anchor]) > oracle_delta
            else:
                raise ValueError(f"unknown pair mode: {mode}")
            if not bool(valid.any()):
                continue
            candidates = torch.nonzero(valid, as_tuple=False).flatten()
            neighbor = candidates[torch.argmin(dist[anchor, candidates])]
        anchors.append(anchor)
        neighbors.append(int(neighbor))
    return {
        "pair_anchor_indices": torch.tensor(
            anchors, dtype=torch.long, device=state.device
        ),
        "pair_neighbor_indices": torch.tensor(
            neighbors, dtype=torch.long, device=state.device
        ),
        "pair_count": len(anchors),
        "skipped_anchor_count": n - len(anchors),
        "state_distance_threshold": float(threshold.detach().cpu()),
        "selected_state_distance": (
            dist[
                torch.tensor(anchors, dtype=torch.long, device=state.device),
                torch.tensor(neighbors, dtype=torch.long, device=state.device),
            ]
            if anchors
            else torch.empty(0, device=state.device)
        ),
    }


def _oracle_value(
    *,
    task: str,
    state_raw: torch.Tensor,
    target_raw: torch.Tensor,
) -> tuple[torch.Tensor, str, float]:
    if task == "TwoRoom":
        door = torch.tensor([112.0, 49.0], device=state_raw.device)
        same_room = (state_raw[:, 0] < 112.0) == (target_raw[:, 0] < 112.0)
        direct = torch.linalg.vector_norm(state_raw - target_raw, dim=1)
        through_door = (
            torch.linalg.vector_norm(state_raw - door, dim=1)
            + torch.linalg.vector_norm(target_raw - door, dim=1)
        )
        return (
            torch.where(same_room, direct, through_door) / 224.0,
            "environment-geometry shortest-path proxy via fixed wall center 112 and door center (112,49)",
            0.10,
        )
    pusher = state_raw[:, :2]
    block = state_raw[:, 2:4]
    goal_block = target_raw[:, 2:4]
    pos_cost = torch.linalg.vector_norm(block - goal_block, dim=1) / 512.0
    angle = torch.abs(
        torch.atan2(
            torch.sin(state_raw[:, 4] - target_raw[:, 4]),
            torch.cos(state_raw[:, 4] - target_raw[:, 4]),
        )
    ) / math.pi
    pusher_block = torch.linalg.vector_norm(pusher - block, dim=1) / 512.0
    return (
        pos_cost + angle + pusher_block,
        "sequence-goal object-pose cost plus pusher-object distance; HDF5 lacks simulator goal_state",
        0.10,
    )


def _rollout_set(
    *,
    model: Any,
    clean_init: torch.Tensor,
    noisy_init: Sequence[torch.Tensor],
    actions: torch.Tensor,
    history_size: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = phase0._autoregressive_rollout(
        model, clean_init, actions, history_size, horizon
    )[:, history_size : history_size + horizon]
    noisy = torch.stack(
        [
            phase0._autoregressive_rollout(
                model, value, actions, history_size, horizon
            )[:, history_size : history_size + horizon]
            for value in noisy_init
        ],
        dim=1,
    )
    return clean, noisy


def _different_rollout(
    *,
    model: Any,
    clean_emb: torch.Tensor,
    actions: torch.Tensor,
    pair: Mapping[str, Any],
    history_size: int,
    horizon: int,
) -> torch.Tensor:
    anchors = pair["pair_anchor_indices"]
    neighbors = pair["pair_neighbor_indices"]
    _require(int(anchors.numel()) > 0, "pair selection produced zero pairs")
    chain = phase0._autoregressive_rollout(
        model,
        clean_emb.index_select(0, neighbors)[:, :history_size],
        actions.index_select(0, anchors),
        history_size,
        horizon,
    )
    return chain[:, history_size : history_size + horizon]


def _compute(
    prepared: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    clean_rollout: torch.Tensor | None = None,
    noisy_rollout: torch.Tensor | None = None,
    different_rollout: torch.Tensor | None = None,
    transition_scale: torch.Tensor | None = None,
    noise_draws: int = 5,
    radius_q: float = 0.90,
    margin_delta: float = 0.10,
) -> dict[str, Any]:
    clean = prepared["clean_rollout"] if clean_rollout is None else clean_rollout
    noisy = prepared["noisy_rollout"] if noisy_rollout is None else noisy_rollout
    different = (
        _different_rollout(
            model=prepared["model"],
            clean_emb=prepared["clean_emb"],
            actions=prepared["act_emb"],
            pair=pair,
            history_size=prepared["history_size"],
            horizon=prepared["horizon"],
        )
        if different_rollout is None
        else different_rollout
    )
    scale = prepared["transition_scale"] if transition_scale is None else transition_scale
    return semantic.compute_smpr_v2_from_rollouts(
        clean_rollout=clean,
        noisy_rollout=noisy[:, :noise_draws],
        different_state_rollout=different,
        pair_anchor_indices=pair["pair_anchor_indices"],
        clean_transition_scale=scale,
        radius_quantile=radius_q,
        margin_delta_norm=margin_delta,
        eps=1e-8,
    )


def _prepare(
    *,
    task: str,
    std_key: str,
    entry: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase0._ensure_runtime_deps()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_file, tried = phase0.resolve_model_file(
        str(entry.get("path", "")),
        str(entry.get("subdir", "")),
        [Path(MODEL_ROOTS[task])],
    )
    _require(model_file is not None, f"{task}/{std_key}: checkpoint missing; {tried}")
    model_file = model_file.resolve()
    canonical._set_task_home(entry)
    task_home = os.environ["STABLEWM_HOME"]
    model = phase0.load_model(str(model_file), device)
    history_size = phase0.infer_history_size(model)
    future_steps = max(args.future_steps, args.rollout_horizon + 1)
    batch = phase0.load_dataset_samples(
        dataset_name=phase0.TASK_DATASETS[task],
        state_key=semantic.SEMANTIC_STATE_KEYS[task],
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.anchor_seed,
        device=device,
    )
    target_batch = None
    if task == "TwoRoom":
        target_batch = phase0.load_dataset_samples(
            dataset_name=phase0.TASK_DATASETS[task],
            state_key="pos_target",
            n_sequences=args.n_sequences,
            history_size=history_size,
            future_steps=future_steps,
            frameskip=args.frameskip,
            img_size=args.img_size,
            seed=args.anchor_seed,
            device=device,
        )
    draw_seeds = [args.anchor_seed + 1009 + 7919 * draw for draw in range(5)]
    noisy_batches = [
        phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=seed,
            corruption_type="gaussian_noise",
            corrupt_goal=False,
        )
        for seed in draw_seeds
    ]
    with torch.no_grad():
        clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
        noisy_outputs = [
            phase0.encode_sequences(model, phase0._clone_batch(value))
            for value in noisy_batches
        ]
        embedding_space = args.embedding_space or phase0.get_model_spaces(model)[
            "inference_cost_space"
        ]
        clean_emb = phase0.get_embedding_space(clean_outputs, embedding_space).detach()
        noisy_emb = [
            phase0.get_embedding_space(value, embedding_space).detach()
            for value in noisy_outputs
        ]
        act_emb = clean_outputs["act_emb"].detach()
        horizon = min(
            args.rollout_horizon,
            act_emb.size(1) - history_size + 1,
            clean_emb.size(1) - history_size,
        )
        clean_rollout, noisy_rollout = _rollout_set(
            model=model,
            clean_init=clean_emb[:, :history_size],
            noisy_init=[value[:, :history_size] for value in noisy_emb],
            actions=act_emb,
            history_size=history_size,
            horizon=horizon,
        )
        observed = clean_emb[:, history_size : history_size + horizon]
        scale = per_anchor_clean_transition_scale(
            observed,
            initial_clean_state=clean_emb[:, history_size - 1],
            transition_quantile=0.50,
        )
    state_index = min(batch["state"].size(1) - 1, history_size + horizon - 1)
    state_norm = batch["state"][:, state_index].reshape(args.n_sequences, -1)
    state_raw_all = _denormalize(
        batch["state"],
        task_home=task_home,
        dataset_name=phase0.TASK_DATASETS[task],
        key=semantic.SEMANTIC_STATE_KEYS[task],
    )
    state_raw = state_raw_all[:, state_index].reshape(args.n_sequences, -1)
    if task == "TwoRoom":
        assert target_batch is not None
        target_raw_all = _denormalize(
            target_batch["state"],
            task_home=task_home,
            dataset_name=phase0.TASK_DATASETS[task],
            key="pos_target",
        )
        target_raw = target_raw_all[:, state_index].reshape(args.n_sequences, -1)
    else:
        target_raw = state_raw_all[:, -1].reshape(args.n_sequences, -1)
    return {
        "task": task,
        "std_key": std_key,
        "model": model,
        "model_file": model_file,
        "checkpoint_sha256": _sha256(model_file),
        "history_size": history_size,
        "horizon": horizon,
        "clean_emb": clean_emb,
        "noisy_emb": noisy_emb,
        "act_emb": act_emb,
        "clean_rollout": clean_rollout,
        "noisy_rollout": noisy_rollout,
        "transition_scale": scale,
        "state_norm": state_norm,
        "state_raw": state_raw,
        "target_raw": target_raw,
        "embedding_space": embedding_space,
        "wall_time": time.perf_counter() - started,
    }


def _static_sensitivity(canonical_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in canonical_payload.get("rows", []):
        _require(source.get("status") == "ok", "canonical SMPR row not ok")
        same = torch.tensor(source["same_state_radius_per_noise_draw"], dtype=torch.float64)
        different = torch.tensor(source["different_state_distance_per_pair"], dtype=torch.float64)
        for draws in NOISE_DRAW_LEVELS:
            same_anchor = same[:, :draws].mean(dim=1)
            for radius_q in RADIUS_QUANTILES:
                tube = torch.quantile(same_anchor, radius_q)
                for delta in MARGIN_DELTAS:
                    margins = different - tube
                    rows.append(
                        {
                            "status": "ok",
                            "sensitivity_type": "full_calibration_distribution_grid",
                            "task": source["task"],
                            "std_key": source["std_key"],
                            "training_seed": 3072,
                            "noise_draws": draws,
                            "radius_quantile": radius_q,
                            "local_state_quantile": source["local_state_quantile"],
                            "margin_delta_norm": delta,
                            "label_binning": "median",
                            "pair_count": int(different.numel()),
                            "skipped_anchor_count": int(source["semantic_skipped_anchor_count"]),
                            "label_count": int(source["semantic_label_count"]),
                            "same_state_tube_radius": float(tube),
                            "smpr": float((margins > delta).float().mean()),
                            "same_state_radius_distribution": _distribution(same_anchor),
                            "different_state_distance_distribution": _distribution(different),
                            "raw_margin_distribution": _distribution(margins),
                        }
                    )
    return rows


def _pairing_sensitivity(prepared: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for binning in LABEL_BINNINGS:
        labels, rule = _labels(
            task=prepared["task"],
            state_norm=prepared["state_norm"],
            state_raw=prepared["state_raw"],
            target_raw=prepared["target_raw"],
            binning=binning,
        )
        for local_q in LOCAL_QUANTILES:
            pair = _select_pairs(
                state=prepared["state_norm"],
                labels=labels,
                local_quantile=local_q,
                mode="cross_label",
            )
            base = {
                "sensitivity_type": "pairing_mve",
                "task": prepared["task"],
                "std_key": prepared["std_key"],
                "training_seed": 3072,
                "noise_draws": 5,
                "radius_quantile": 0.90,
                "local_state_quantile": local_q,
                "margin_delta_norm": 0.10,
                "label_binning": binning,
                "label_rule": rule,
                "label_count": int(torch.unique(labels).numel()),
                "pair_count": pair["pair_count"],
                "skipped_anchor_count": pair["skipped_anchor_count"],
            }
            if pair["pair_count"] == 0:
                rows.append({**base, "status": "explicit_zero_pairs"})
                continue
            metrics = _compute(prepared, pair)
            rows.append({**base, "status": "ok", **_metric_summary(metrics)})
    return rows


def _joint(metrics: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    return (
        float(metrics["same_state_tube_radius"]) <= float(protocol["tau_atr"])
        and float(metrics["smpr"]) >= float(protocol["tau_smpr"])
    )


def _control_rows(
    prepared: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    task = prepared["task"]
    labels, label_rule = _labels(
        task=task,
        state_norm=prepared["state_norm"],
        state_raw=prepared["state_raw"],
        target_raw=prepared["target_raw"],
        binning="median",
    )
    true_pair = _select_pairs(
        state=prepared["state_norm"],
        labels=labels,
        local_quantile=0.35,
        mode="cross_label",
    )
    _require(true_pair["pair_count"] > 0, "default control pair count is zero")
    common = {
        "task": task,
        "std_key": prepared["std_key"],
        "training_seed": 3072,
        "checkpoint_sha256": prepared["checkpoint_sha256"],
        "label_rule": label_rule,
    }
    rows = []

    def add(control: str, pair: Mapping[str, Any], metrics: Mapping[str, Any], **extra: Any) -> None:
        summary = _metric_summary(metrics)
        rows.append(
            {
                **common,
                "status": "ok",
                "control": control,
                "pair_count": pair["pair_count"],
                "skipped_anchor_count": pair["skipped_anchor_count"],
                **summary,
                "joint_gate_pass": _joint(summary, protocol),
                **extra,
            }
        )

    true_different = _different_rollout(
        model=prepared["model"],
        clean_emb=prepared["clean_emb"],
        actions=prepared["act_emb"],
        pair=true_pair,
        history_size=prepared["history_size"],
        horizon=prepared["horizon"],
    )
    default_metrics = _compute(
        prepared, true_pair, different_rollout=true_different
    )
    add("task_grounded_default", true_pair, default_metrics)

    generator = torch.Generator().manual_seed(9101 + int(float(prepared["std_key"]) * 1000))
    permutation = torch.randperm(labels.numel(), generator=generator).to(labels.device)
    random_pair = _select_pairs(
        state=prepared["state_norm"],
        labels=labels.index_select(0, permutation),
        local_quantile=0.35,
        mode="cross_label",
    )
    if random_pair["pair_count"]:
        add("state_label_permutation", random_pair, _compute(prepared, random_pair))

    same_pair = _select_pairs(
        state=prepared["state_norm"],
        labels=labels,
        local_quantile=0.35,
        mode="same_label",
    )
    if same_pair["pair_count"]:
        add("same_label_nearest_neighbor", same_pair, _compute(prepared, same_pair))

    far_pair = _select_pairs(
        state=prepared["state_norm"],
        labels=None,
        local_quantile=0.35,
        mode="far",
    )
    add("global_far_neighbor", far_pair, _compute(prepared, far_pair))

    identical_noisy = prepared["clean_rollout"].unsqueeze(1).expand_as(
        prepared["noisy_rollout"]
    )
    add(
        "identical_clean_noisy_positive",
        true_pair,
        _compute(
            prepared,
            true_pair,
            noisy_rollout=identical_noisy,
            different_rollout=true_different,
        ),
    )

    perm_actions = torch.roll(prepared["act_emb"], shifts=1, dims=0)
    perm_clean, perm_noisy = _rollout_set(
        model=prepared["model"],
        clean_init=prepared["clean_emb"][:, : prepared["history_size"]],
        noisy_init=[
            value[:, : prepared["history_size"]] for value in prepared["noisy_emb"]
        ],
        actions=perm_actions,
        history_size=prepared["history_size"],
        horizon=prepared["horizon"],
    )
    perm_different = _different_rollout(
        model=prepared["model"],
        clean_emb=prepared["clean_emb"],
        actions=perm_actions,
        pair=true_pair,
        history_size=prepared["history_size"],
        horizon=prepared["horizon"],
    )
    add(
        "action_permutation",
        true_pair,
        _compute(
            prepared,
            true_pair,
            clean_rollout=perm_clean,
            noisy_rollout=perm_noisy,
            different_rollout=perm_different,
        ),
    )

    center = prepared["clean_rollout"].mean(dim=(0, 1), keepdim=True)
    for collapse in COLLAPSE_LEVELS:
        clean = (1.0 - collapse) * prepared["clean_rollout"] + collapse * center
        noisy = (
            (1.0 - collapse) * prepared["noisy_rollout"]
            + collapse * center.unsqueeze(1)
        )
        different = (1.0 - collapse) * true_different + collapse * center
        fixed = _compute(
            prepared,
            true_pair,
            clean_rollout=clean,
            noisy_rollout=noisy,
            different_rollout=different,
            transition_scale=prepared["transition_scale"],
        )
        dynamic = _compute(
            prepared,
            true_pair,
            clean_rollout=clean,
            noisy_rollout=noisy,
            different_rollout=different,
            transition_scale=(1.0 - collapse) * prepared["transition_scale"],
        )
        fixed_summary = _metric_summary(fixed)
        dynamic_summary = _metric_summary(dynamic)
        rows.append(
            {
                **common,
                "status": "ok",
                "control": (
                    "constant_collapsed_rollout"
                    if collapse == 1.0
                    else "progressive_collapse"
                ),
                "collapse_lambda": collapse,
                "pair_count": true_pair["pair_count"],
                "skipped_anchor_count": true_pair["skipped_anchor_count"],
                "fixed_original_scale": fixed_summary,
                "recomputed_collapsed_scale": dynamic_summary,
                "same_state_tube_radius": dynamic_summary["same_state_tube_radius"],
                "smpr": dynamic_summary["smpr"],
                "joint_gate_pass": _joint(dynamic_summary, protocol),
            }
        )
    return rows


def _oracle_row(
    prepared: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    values, rule, delta = _oracle_value(
        task=prepared["task"],
        state_raw=prepared["state_raw"],
        target_raw=prepared["target_raw"],
    )
    pair = _select_pairs(
        state=prepared["state_norm"],
        labels=None,
        local_quantile=0.35,
        mode="oracle",
        oracle_value=values,
        oracle_delta=delta,
    )
    base = {
        "task": prepared["task"],
        "std_key": prepared["std_key"],
        "training_seed": 3072,
        "checkpoint_sha256": prepared["checkpoint_sha256"],
        "oracle_rule": rule,
        "oracle_value_delta_threshold": delta,
        "oracle_value_distribution": _distribution(values),
        "pair_count": pair["pair_count"],
        "skipped_anchor_count": pair["skipped_anchor_count"],
        "selected_state_distance_distribution": _distribution(
            pair["selected_state_distance"]
        ),
        "oracle_scope": (
            "environment_geometry_oracle_proxy"
            if prepared["task"] == "TwoRoom"
            else "sequence_goal_state_derived_oracle_proxy"
        ),
    }
    if not pair["pair_count"]:
        return {**base, "status": "explicit_zero_pairs"}
    summary = _metric_summary(_compute(prepared, pair))
    return {
        **base,
        "status": "ok",
        **summary,
        "joint_gate_pass": _joint(summary, protocol),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_")


def _write_tables(
    *,
    sensitivity_path: Path,
    controls_path: Path,
    sensitivity_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> None:
    sensitivity_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{SMPR V1 sensitivity summaries. Distribution parameters use all 36 CAL checkpoints; pair-selection parameters use the two-task four-checkpoint MVE.}",
        r"\label{tab:smpr-sensitivity}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Factor & Level & Mean SMPR & Mean pair count \\",
        r"\midrule",
    ]
    slices = []
    full = [row for row in sensitivity_rows if row["sensitivity_type"] == "full_calibration_distribution_grid"]
    for field, levels, defaults in (
        ("noise_draws", NOISE_DRAW_LEVELS, {"radius_quantile": 0.90, "margin_delta_norm": 0.10}),
        ("radius_quantile", RADIUS_QUANTILES, {"noise_draws": 5, "margin_delta_norm": 0.10}),
        ("margin_delta_norm", MARGIN_DELTAS, {"noise_draws": 5, "radius_quantile": 0.90}),
    ):
        for level in levels:
            rows = [
                row for row in full
                if row[field] == level
                and all(row[key] == value for key, value in defaults.items())
            ]
            slices.append((field, level, rows))
    pairing = [row for row in sensitivity_rows if row["sensitivity_type"] == "pairing_mve" and row["status"] == "ok"]
    for level in LOCAL_QUANTILES:
        slices.append(("local_state_quantile", level, [row for row in pairing if row["label_binning"] == "median" and row["local_state_quantile"] == level]))
    for level in LABEL_BINNINGS:
        slices.append(("label_binning", level, [row for row in pairing if row["local_state_quantile"] == 0.35 and row["label_binning"] == level]))
    for field, level, rows in slices:
        if not rows:
            continue
        sensitivity_lines.append(
            f"{_latex_escape(field)} & {_latex_escape(level)} & "
            f"{_mean([float(row['smpr']) for row in rows]):.3f} & "
            f"{_mean([float(row['pair_count']) for row in rows]):.1f} \\\\"
        )
    sensitivity_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_path.write_text("\n".join(sensitivity_lines), encoding="utf-8")

    control_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{SMPR controls on the TwoRoom+PushT base/endpoint MVE.}",
        r"\label{tab:smpr-controls}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Control & Rows & Mean SMPR & Joint passes \\",
        r"\midrule",
    ]
    for control in sorted({str(row["control"]) for row in control_rows}):
        rows = [row for row in control_rows if row["control"] == control]
        control_lines.append(
            f"{_latex_escape(control)} & {len(rows)} & "
            f"{_mean([float(row['smpr']) for row in rows]):.3f} & "
            f"{sum(bool(row['joint_gate_pass']) for row in rows)} \\\\"
        )
    control_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    controls_path.write_text("\n".join(control_lines), encoding="utf-8")


def _plot(path: Path, control_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    collapsed = [
        row for row in control_rows
        if row["control"] in {"progressive_collapse", "constant_collapsed_rollout"}
    ]
    levels = list(COLLAPSE_LEVELS)
    tube = []
    different = []
    margin = []
    smpr_dynamic = []
    smpr_fixed = []
    for level in levels:
        rows = [row for row in collapsed if row["collapse_lambda"] == level]
        dynamic = [row["recomputed_collapsed_scale"] for row in rows]
        fixed = [row["fixed_original_scale"] for row in rows]
        tube.append(_mean([float(row["same_state_tube_radius"]) for row in dynamic]))
        different.append(_mean([float(row["different_state_distance_distribution"]["q50"]) for row in dynamic]))
        margin.append(_mean([float(row["raw_margin_distribution"]["q50"]) for row in dynamic]))
        smpr_dynamic.append(_mean([float(row["smpr"]) for row in dynamic]))
        smpr_fixed.append(_mean([float(row["smpr"]) for row in fixed]))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    axes[0].plot(levels, tube, marker="o", label="same-state tube")
    axes[0].plot(levels, different, marker="o", label="different-state distance")
    axes[0].plot(levels, margin, marker="o", label="raw margin")
    axes[0].set_title("Radius--margin decomposition")
    axes[0].set_ylabel("Normalized value (MVE mean)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(levels, smpr_dynamic, marker="o", label="recomputed scale")
    axes[1].plot(levels, smpr_fixed, marker="o", label="fixed original scale")
    axes[1].set_title("Progressive collapse control")
    axes[1].set_ylabel("SMPR")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel(r"Collapse $\lambda$")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evals",
        type=Path,
        default=ROOT / "assets/paper1_data/training_seed_eval_manifests/lewm_seed3072_evals.json",
    )
    parser.add_argument(
        "--canonical-smpr",
        type=Path,
        default=ROOT / "assets/paper1_data/smpr_calibration_lewm_seed3072_v2.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    parser.add_argument("--n-sequences", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--anchor-seed", type=int, default=9101)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-sensitivity",
        type=Path,
        default=ROOT / "assets/paper1_data/smpr_sensitivity_v2.json",
    )
    parser.add_argument(
        "--out-controls",
        type=Path,
        default=ROOT / "assets/paper1_data/smpr_controls_v2.json",
    )
    parser.add_argument(
        "--out-oracle",
        type=Path,
        default=ROOT / "assets/paper1_data/smpr_oracle_guard_v2.json",
    )
    parser.add_argument(
        "--sensitivity-table",
        type=Path,
        default=ROOT / "paper1/tables/table_smpr_sensitivity.tex",
    )
    parser.add_argument(
        "--controls-table",
        type=Path,
        default=ROOT / "paper1/tables/table_smpr_controls.tex",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "assets/paper1_figs/fig_smpr_radius_margin_decomposition.png",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _require(args.n_sequences >= 2, "n_sequences must be at least two")
    protocol_bytes = args.protocol.read_bytes()
    _require(hashlib.sha256(protocol_bytes).hexdigest() == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    protocol = _load(args.protocol)
    evals = _load(args.evals)
    canonical_payload = _load(args.canonical_smpr)
    _require(
        canonical_payload.get("metadata", {}).get("schema_version")
        == "paper1-smpr-v2-merged-1.0",
        "canonical SMPR schema mismatch",
    )
    sensitivity_rows = _static_sensitivity(canonical_payload)
    control_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    mve_metadata = []
    requested = [(task, std_key) for task in args.tasks for std_key in args.std_keys]
    for index, (task, std_key) in enumerate(requested, start=1):
        print(f"[{index}/{len(requested)}] SMPR controls {task} std{std_key}", flush=True)
        entry = evals.get(task, {}).get(std_key)
        _require(isinstance(entry, Mapping), f"manifest row missing: {task}/{std_key}")
        prepared = _prepare(task=task, std_key=std_key, entry=entry, args=args)
        sensitivity_rows.extend(_pairing_sensitivity(prepared))
        control_rows.extend(_control_rows(prepared, protocol))
        oracle_rows.append(_oracle_row(prepared, protocol))
        mve_metadata.append(
            {
                "task": task,
                "std_key": std_key,
                "checkpoint_sha256": prepared["checkpoint_sha256"],
                "model_file": str(prepared["model_file"]),
                "wall_time": prepared["wall_time"],
                "embedding_space": prepared["embedding_space"],
            }
        )
        del prepared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    constant_rows = [row for row in control_rows if row["control"] == "constant_collapsed_rollout"]
    identical_rows = [row for row in control_rows if row["control"] == "identical_clean_noisy_positive"]
    correctness = {
        "constant_collapse_rows": len(constant_rows),
        "constant_collapse_joint_passes": sum(bool(row["joint_gate_pass"]) for row in constant_rows),
        "constant_collapse_rejected_all_mve_rows": bool(constant_rows) and not any(bool(row["joint_gate_pass"]) for row in constant_rows),
        "identical_positive_rows": len(identical_rows),
        "identical_positive_max_tube_radius": max(float(row["same_state_tube_radius"]) for row in identical_rows),
        "all_noncollapse_control_pair_counts_positive": all(
            int(row["pair_count"]) > 0
            for row in control_rows
            if row["control"] != "constant_collapsed_rollout"
        ),
    }
    control_index = {
        (str(row["task"]), str(row["std_key"]), str(row["control"]), row.get("collapse_lambda")): row
        for row in control_rows
    }
    default_keys = [(task, std_key) for task in args.tasks for std_key in args.std_keys]
    default_smpr = [
        float(control_index[(task, std_key, "task_grounded_default", None)]["smpr"])
        for task, std_key in default_keys
    ]
    random_smpr = [
        float(control_index[(task, std_key, "state_label_permutation", None)]["smpr"])
        for task, std_key in default_keys
    ]
    action_smpr = [
        float(control_index[(task, std_key, "action_permutation", None)]["smpr"])
        for task, std_key in default_keys
    ]
    far_smpr = [
        float(control_index[(task, std_key, "global_far_neighbor", None)]["smpr"])
        for task, std_key in default_keys
    ]
    preconstant_collapse_changes = []
    for task, std_key in default_keys:
        reference = float(
            control_index[(task, std_key, "progressive_collapse", 0.0)]["smpr"]
        )
        for level in (0.25, 0.5, 0.75):
            value = float(
                control_index[(task, std_key, "progressive_collapse", level)]["smpr"]
            )
            preconstant_collapse_changes.append(abs(value - reference))
    claim_decisions = {
        "task_grounded_minus_random_smpr_mean": _mean(
            [left - right for left, right in zip(default_smpr, random_smpr)]
        ),
        "task_grounded_strictly_better_than_random_all_mve_rows": all(
            left > right for left, right in zip(default_smpr, random_smpr)
        ),
        "task_grounded_increment_established": False,
        "action_permutation_max_abs_smpr_change": max(
            abs(left - right) for left, right in zip(default_smpr, action_smpr)
        ),
        "action_relevance_increment_established": False,
        "far_neighbor_minus_near_smpr_mean": _mean(
            [left - right for left, right in zip(far_smpr, default_smpr)]
        ),
        "preconstant_progressive_collapse_max_abs_smpr_change": max(
            preconstant_collapse_changes
        ),
        "progressive_collapse_sensitivity_established": False,
        "full_constant_collapse_correctly_rejected": correctness[
            "constant_collapse_rejected_all_mve_rows"
        ],
        "recommended_claim_scope": (
            "proxy-level guard correctness only; task-label, action-relevance, and "
            "progressive anti-collapse increments are not established on the MVE"
        ),
    }
    _require(correctness["constant_collapse_rejected_all_mve_rows"], "constant collapse passed the joint gate")
    _require(correctness["identical_positive_max_tube_radius"] <= 1e-8, "identical clean/noisy control has nonzero radius")
    source_paths = {
        "protocol": args.protocol,
        "eval_manifest": args.evals,
        "canonical_smpr": args.canonical_smpr,
        "runner": Path(__file__).resolve(),
        "semantic_margin": ROOT / "tools/paper1_semantic_margin.py",
        "canonical_smpr_runner": ROOT / "paper1/scripts/smpr_sensitivity.py",
    }
    common_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "status": "complete",
        "protocol_hash": FROZEN_PROTOCOL_SHA256,
        "threshold_search_allowed": False,
        "behavior_labels_used": False,
        "tasks": list(args.tasks),
        "std_keys": list(args.std_keys),
        "training_seed": 3072,
        "training_seed_semantics": "one independently trained LeWM checkpoint family",
        "mve_scope": "TwoRoom+PushT base+endpoint, 32 anchors, five fixed noise draws",
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "source_hashes": {key: _sha256(path) for key, path in source_paths.items()},
        "mve_checkpoints": mve_metadata,
        "missing_rows": [],
        "errors": [],
    }
    sensitivity = {
        "metadata": {
            **common_metadata,
            "schema_version": "paper1-smpr-sensitivity-v2-1.0",
            "factorial_scope": (
                "noise_draws x radius_q x margin_delta on all 36 CAL rows; "
                "local_quantile x label_binning on the four-checkpoint MVE"
            ),
            "levels": {
                "noise_draws": NOISE_DRAW_LEVELS,
                "radius_quantile": RADIUS_QUANTILES,
                "local_state_quantile": LOCAL_QUANTILES,
                "margin_delta_norm": MARGIN_DELTAS,
                "label_binning": LABEL_BINNINGS,
            },
        },
        "count_contract": {
            "full_calibration_grid_rows": sum(row["sensitivity_type"] == "full_calibration_distribution_grid" for row in sensitivity_rows),
            "pairing_mve_rows": sum(row["sensitivity_type"] == "pairing_mve" for row in sensitivity_rows),
            "total_rows": len(sensitivity_rows),
            "explicit_zero_pair_rows": sum(
                row.get("status") == "explicit_zero_pairs"
                for row in sensitivity_rows
            ),
            "pairing_mve_min_pair_count": min(
                int(row["pair_count"])
                for row in sensitivity_rows
                if row["sensitivity_type"] == "pairing_mve"
            ),
            "pairing_mve_max_pair_count": max(
                int(row["pair_count"])
                for row in sensitivity_rows
                if row["sensitivity_type"] == "pairing_mve"
            ),
        },
        "rows": sensitivity_rows,
    }
    controls = {
        "metadata": {
            **common_metadata,
            "schema_version": "paper1-smpr-controls-v2-1.0",
            "constant_collapse_expected_to_fail": True,
            "scale_counterfactuals": ["fixed_original_scale", "recomputed_collapsed_scale"],
        },
        "correctness_gates": correctness,
        "claim_decisions": claim_decisions,
        "count_contract": {"rows": len(control_rows)},
        "rows": control_rows,
    }
    oracle = {
        "metadata": {
            **common_metadata,
            "schema_version": "paper1-smpr-oracle-guard-v2-1.0",
            "claim_scope": (
                "TwoRoom environment-geometry oracle proxy and PushT sequence-goal "
                "state-derived oracle proxy; not a four-task simulator-oracle result"
            ),
            "pusht_simulator_goal_state_available_in_hdf5": False,
        },
        "count_contract": {"expected_rows": 4, "observed_rows": len(oracle_rows)},
        "rows": oracle_rows,
    }
    _write_json(args.out_sensitivity, sensitivity)
    _write_json(args.out_controls, controls)
    _write_json(args.out_oracle, oracle)
    _write_tables(
        sensitivity_path=args.sensitivity_table,
        controls_path=args.controls_table,
        sensitivity_rows=sensitivity_rows,
        control_rows=control_rows,
    )
    _plot(args.figure, control_rows)
    _require(args.protocol.read_bytes() == protocol_bytes, "SMPR consumer modified frozen protocol")
    print(
        f"wrote {len(sensitivity_rows)} sensitivity rows, "
        f"{len(control_rows)} control rows, {len(oracle_rows)} oracle rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
