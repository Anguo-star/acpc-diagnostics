"""Run Paper 1 task-semantic margin pass-rate probes.

This is a lightweight selective-ACPC guard: for each checkpoint, compare the
same-state clean/noisy rollout radius against hard semantic-different clean
rollout distances on a task-specific state proxy. It is intended to complement
ACPC/PCC/CRA/MAF with one compact semantic discriminability readout per task.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_acpc_metrics import (
    horizon_weighted_stacked_l2,
    uniform_horizon_weights,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_MANIFEST_DIR = DATA_DIR / "training_seed_eval_manifests"
DEFAULT_OUT = DATA_DIR / "semantic_margin_passrate_lewm_three_seed.json"
SEEDS = (3072, 3073, 3074)
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEMANTIC_STATE_KEYS = {
    "TwoRoom": "pos_agent",
    "PushT": "state",
    "Reacher": "observation",
    "Cube": "observation",
}
SEMANTIC_FACTORS = {
    "TwoRoom": "agent room/doorway/topology proxy from pos_agent",
    "PushT": "T-block and pusher pose proxy from state",
    "Reacher": "joint/target geometry proxy from observation",
    "Cube": "cube pose and gripper-object proxy from observation",
}
LOCAL_FEATURE_FACTORS = {
    "TwoRoom": "local agent x-coordinate room/doorway-side contrast within nearby positions",
    "PushT": "local T-block pose contrast within nearby full states",
    "Reacher": "local target/end-effector relation proxy contrast within nearby observations",
    "Cube": "local cube pose/goal-relation proxy contrast within nearby observations",
}
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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().float().cpu(), q))


def compute_smpr_v2_from_rollouts(
    *,
    clean_rollout: torch.Tensor,
    noisy_rollout: torch.Tensor,
    different_state_rollout: torch.Tensor,
    pair_anchor_indices: torch.Tensor,
    clean_transition_scale: torch.Tensor,
    radius_quantile: float = 0.90,
    margin_delta_norm: float = 0.10,
    weights: torch.Tensor | Sequence[float] | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute task-grounded SMPR v2 from aligned canonical rollouts.

    clean_rollout has shape (B,H,D). noisy_rollout has shape (B,M,H,D)
    and is aggregated by a within-anchor mean, matching ATR. Each row in
    different_state_rollout uses the action sequence of the corresponding
    anchor named by pair_anchor_indices. Different-state distances and the
    positive margin use the same per-anchor clean-transition units.
    """

    if clean_rollout.ndim != 3:
        raise ValueError("clean_rollout must have shape (B,H,D)")
    if noisy_rollout.ndim != 4:
        raise ValueError("noisy_rollout must have shape (B,M,H,D)")
    if different_state_rollout.ndim != 3:
        raise ValueError("different_state_rollout must have shape (P,H,D)")
    if pair_anchor_indices.ndim != 1:
        raise ValueError("pair_anchor_indices must have shape (P,)")
    if clean_transition_scale.ndim != 1:
        raise ValueError("clean_transition_scale must have shape (B,)")
    batch, horizon, feature_dim = clean_rollout.shape
    if noisy_rollout.shape[0] != batch or noisy_rollout.shape[2:] != (
        horizon,
        feature_dim,
    ):
        raise ValueError("noisy_rollout must match clean rollout B/H/D")
    if noisy_rollout.shape[1] < 1:
        raise ValueError("noisy_rollout must contain at least one draw")
    if different_state_rollout.shape[1:] != (horizon, feature_dim):
        raise ValueError("different_state_rollout must match clean rollout H/D")
    if different_state_rollout.shape[0] != pair_anchor_indices.numel():
        raise ValueError("one pair anchor index is required per different rollout")
    if clean_transition_scale.shape != (batch,):
        raise ValueError("clean_transition_scale must contain one value per anchor")
    if pair_anchor_indices.dtype == torch.bool or pair_anchor_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("pair_anchor_indices must have an integer dtype")
    if pair_anchor_indices.numel() and (
        int(pair_anchor_indices.min()) < 0
        or int(pair_anchor_indices.max()) >= batch
    ):
        raise ValueError("pair_anchor_indices contains an out-of-range anchor")
    for name, value in (
        ("radius_quantile", radius_quantile),
        ("margin_delta_norm", margin_delta_norm),
        ("eps", eps),
    ):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= float(radius_quantile) <= 1.0:
        raise ValueError("radius_quantile must be in [0,1]")
    if float(margin_delta_norm) < 0.0:
        raise ValueError("margin_delta_norm must be non-negative")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    for name, value in (
        ("clean_rollout", clean_rollout),
        ("noisy_rollout", noisy_rollout),
        ("different_state_rollout", different_state_rollout),
        ("clean_transition_scale", clean_transition_scale),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or infinity")
    if bool((clean_transition_scale < 0).any()):
        raise ValueError("clean_transition_scale must be non-negative")
    if (
        noisy_rollout.device != clean_rollout.device
        or different_state_rollout.device != clean_rollout.device
        or pair_anchor_indices.device != clean_rollout.device
        or clean_transition_scale.device != clean_rollout.device
    ):
        raise ValueError("all SMPR tensors must share one device")

    alpha = (
        uniform_horizon_weights(
            horizon,
            dtype=clean_rollout.dtype,
            device=clean_rollout.device,
        )
        if weights is None
        else torch.as_tensor(
            weights,
            dtype=clean_rollout.dtype,
            device=clean_rollout.device,
        )
    )
    repeated_clean = clean_rollout.unsqueeze(1).expand_as(noisy_rollout)
    same_raw_per_draw = horizon_weighted_stacked_l2(
        repeated_clean,
        noisy_rollout,
        weights=alpha,
    )
    same_norm_per_draw = same_raw_per_draw / (
        clean_transition_scale.unsqueeze(1) + float(eps)
    )
    same_norm_per_anchor = same_norm_per_draw.mean(dim=1)
    quantile_work = (
        same_norm_per_anchor.float()
        if same_norm_per_anchor.dtype in (torch.float16, torch.bfloat16)
        else same_norm_per_anchor
    )
    tube_radius = torch.quantile(quantile_work, float(radius_quantile))

    anchor_indices = pair_anchor_indices.to(dtype=torch.long)
    pair_clean = clean_rollout.index_select(0, anchor_indices)
    different_raw = horizon_weighted_stacked_l2(
        pair_clean,
        different_state_rollout,
        weights=alpha,
    )
    pair_scale = clean_transition_scale.index_select(0, anchor_indices)
    different_norm = different_raw / (pair_scale + float(eps))
    margins_to_tube = different_norm - tube_radius
    passes = margins_to_tube > float(margin_delta_norm)
    aligned_same_radius = same_norm_per_anchor.index_select(0, anchor_indices)
    margins_to_anchor_radius = different_norm - aligned_same_radius
    smpr = (
        passes.float().mean()
        if passes.numel()
        else torch.full(
            (),
            float("nan"),
            dtype=clean_rollout.dtype,
            device=clean_rollout.device,
        )
    )
    return {
        "radius_metric": "horizon_weighted_stacked_l2_v2",
        "rollout_horizon": horizon,
        "horizon_weights": alpha,
        "noise_draw_aggregation": "per_anchor_mean_then_checkpoint_radius_quantile",
        "radius_quantile": float(radius_quantile),
        "margin_delta_norm": float(margin_delta_norm),
        "same_state_radius_per_noise_draw": same_norm_per_draw,
        "same_state_radius_per_anchor": same_norm_per_anchor,
        "same_state_tube_radius": tube_radius,
        "different_state_distance_per_pair": different_norm,
        "raw_margin_to_tube_per_pair": margins_to_tube,
        "raw_margin_to_anchor_radius_per_pair": margins_to_anchor_radius,
        "pair_pass": passes,
        "smpr": smpr,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def _pstdev(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def _success(entry: Mapping[str, Any], metric: str) -> float:
    return float(entry.get("metrics", {}).get(metric, {}).get("mean", float("nan")))


def _task_feature(task: str, state_final: torch.Tensor) -> torch.Tensor:
    dim = state_final.size(1)
    if task == "TwoRoom" and dim >= 1:
        return state_final[:, :1]
    if task == "PushT" and dim >= 5:
        return state_final[:, 2:5]
    if task == "Reacher" and dim >= 4:
        return state_final[:, -2:]
    if task == "Cube" and dim >= 6:
        return torch.cat([state_final[:, :3], state_final[:, -3:]], dim=1)
    return state_final


def _local_feature_metrics(
    *,
    task: str,
    state_final: torch.Tensor,
    latent_dist: torch.Tensor,
    same_radius: torch.Tensor,
    local_quantile: float,
    margin_delta: float,
) -> dict[str, Any]:
    n = state_final.size(0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=state_final.device)
    state_dist = torch.cdist(state_final, state_final, p=2)
    feature = _task_feature(task, state_final)
    feature_dist = torch.cdist(feature, feature, p=2)
    candidate_state = state_dist[offdiag]
    if candidate_state.numel() == 0:
        return {
            "semantic_state_key": SEMANTIC_STATE_KEYS[task],
            "semantic_factor": LOCAL_FEATURE_FACTORS[task],
            "semantic_pair_rule": "local task-feature contrast within state-distance quantile",
            "semantic_pair_count": 0,
            "semantic_distance_threshold": float("nan"),
            "same_state_noisy_radius_median": _safe_quantile(same_radius, 0.5),
            "semantic_diff_l2_median": float("nan"),
            "semantic_margin_median": float("nan"),
            "semantic_margin_pass_rate": float("nan"),
            "semantic_discriminability_ratio": float("nan"),
            "local_state_distance_median": float("nan"),
            "local_task_feature_delta_median": float("nan"),
        }
    threshold = torch.quantile(candidate_state, float(local_quantile))
    hard_diffs = []
    aligned_same = []
    chosen_state = []
    chosen_feature = []
    for i in range(n):
        row = offdiag[i] & (state_dist[i] <= threshold)
        if bool(row.any()):
            candidates = torch.nonzero(row, as_tuple=False).flatten()
            j = candidates[torch.argmax(feature_dist[i, candidates])]
            hard_diffs.append(latent_dist[i, j])
            aligned_same.append(same_radius[i])
            chosen_state.append(state_dist[i, j])
            chosen_feature.append(feature_dist[i, j])
    if not hard_diffs:
        hard = torch.empty(0, device=state_final.device)
        same = torch.empty(0, device=state_final.device)
        state_delta = torch.empty(0, device=state_final.device)
        feature_delta = torch.empty(0, device=state_final.device)
    else:
        hard = torch.stack(hard_diffs)
        same = torch.stack(aligned_same)
        state_delta = torch.stack(chosen_state)
        feature_delta = torch.stack(chosen_feature)
    margins = hard - same
    passes = margins > float(margin_delta)
    same_med = _safe_quantile(same, 0.5)
    diff_med = _safe_quantile(hard, 0.5)
    return {
        "semantic_state_key": SEMANTIC_STATE_KEYS[task],
        "semantic_factor": LOCAL_FEATURE_FACTORS[task],
        "semantic_pair_rule": "local task-feature contrast within state-distance quantile",
        "semantic_pair_count": int(hard.numel()),
        "semantic_distance_threshold": float(threshold.detach().cpu()),
        "same_state_noisy_radius_median": same_med,
        "semantic_diff_l2_median": diff_med,
        "semantic_margin_median": _safe_quantile(margins, 0.5),
        "semantic_margin_pass_rate": float(passes.float().mean().detach().cpu()) if passes.numel() else float("nan"),
        "semantic_discriminability_ratio": diff_med / same_med if same_med > 0 else float("nan"),
        "local_state_distance_median": _safe_quantile(state_delta, 0.5),
        "local_task_feature_delta_median": _safe_quantile(feature_delta, 0.5),
    }



def _task_grounded_labels(task: str, state_final: torch.Tensor) -> tuple[torch.Tensor, str, torch.Tensor]:
    """Programmatic task-label proxy for near-boundary pair selection.

    The labels are deliberately simple and derived from logged task state. They
    are not human/oracle semantic annotations; they only make the pair selection
    more task-grounded than a global distance quantile.
    """
    feature = _task_feature(task, state_final).float()
    if feature.ndim != 2 or feature.size(0) == 0:
        labels = torch.zeros(state_final.size(0), dtype=torch.long, device=state_final.device)
        return labels, "degenerate single label", feature
    med = torch.median(feature, dim=0).values
    if task == "TwoRoom":
        x = feature[:, 0]
        labels = (x > torch.median(x)).long()
        return labels, "agent doorway/room-side split from local x-coordinate", feature[:, :1]
    if task == "PushT":
        x = feature[:, 0]
        y = feature[:, 1] if feature.size(1) > 1 else feature[:, 0]
        theta = feature[:, 2] if feature.size(1) > 2 else feature[:, -1]
        labels = ((x > med[0]).long() + 2 * (y > med[1]).long() + 4 * (theta > med[min(2, feature.size(1) - 1)]).long())
        return labels, "T-block pose cell from x/y/theta median splits", feature[:, : min(3, feature.size(1))]
    if task == "Reacher":
        x = feature[:, 0]
        y = feature[:, 1] if feature.size(1) > 1 else feature[:, 0]
        radius = torch.linalg.vector_norm(feature[:, : min(2, feature.size(1))], dim=1)
        labels = ((x > med[0]).long() + 2 * (y > med[min(1, feature.size(1) - 1)]).long() + 4 * (radius > torch.median(radius)).long())
        return labels, "target/end-effector relation quadrant plus distance bin", feature[:, : min(2, feature.size(1))]
    if task == "Cube":
        split = max(1, feature.size(1) // 2)
        left = feature[:, :split]
        right = feature[:, -split:]
        relation = left - right
        dist = torch.linalg.vector_norm(relation, dim=1)
        x = relation[:, 0]
        y = relation[:, 1] if relation.size(1) > 1 else relation[:, 0]
        labels = ((x > torch.median(x)).long() + 2 * (y > torch.median(y)).long() + 4 * (dist > torch.median(dist)).long())
        return labels, "cube-pose/goal-relation cell from relation direction and distance", relation
    labels = torch.zeros(state_final.size(0), dtype=torch.long, device=state_final.device)
    return labels, "fallback single label", feature


def _task_grounded_near_boundary_metrics(
    *,
    task: str,
    state_final: torch.Tensor,
    latent_dist: torch.Tensor,
    same_radius: torch.Tensor,
    local_quantile: float,
    margin_delta: float,
) -> dict[str, Any]:
    n = state_final.size(0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=state_final.device)
    state_dist = torch.cdist(state_final, state_final, p=2)
    labels, label_rule, feature = _task_grounded_labels(task, state_final)
    feature_dist = torch.cdist(feature.float(), feature.float(), p=2)
    candidate_state = state_dist[offdiag]
    if candidate_state.numel() == 0:
        threshold = torch.tensor(float("nan"), device=state_final.device)
    else:
        threshold = torch.quantile(candidate_state, float(local_quantile))
    hard_diffs = []
    aligned_same = []
    chosen_state = []
    chosen_feature = []
    skipped = 0
    for i in range(n):
        label_diff = labels != labels[i]
        row = offdiag[i] & label_diff & (state_dist[i] <= threshold)
        if not bool(row.any()):
            skipped += 1
            continue
        candidates = torch.nonzero(row, as_tuple=False).flatten()
        # Near-boundary proxy: choose the closest state that crosses the task label.
        j = candidates[torch.argmin(state_dist[i, candidates])]
        hard_diffs.append(latent_dist[i, j])
        aligned_same.append(same_radius[i])
        chosen_state.append(state_dist[i, j])
        chosen_feature.append(feature_dist[i, j])
    if hard_diffs:
        hard = torch.stack(hard_diffs)
        same = torch.stack(aligned_same)
        state_delta = torch.stack(chosen_state)
        feature_delta = torch.stack(chosen_feature)
    else:
        hard = torch.empty(0, device=state_final.device)
        same = torch.empty(0, device=state_final.device)
        state_delta = torch.empty(0, device=state_final.device)
        feature_delta = torch.empty(0, device=state_final.device)
    margins = hard - same
    passes = margins > float(margin_delta)
    same_med = _safe_quantile(same, 0.5)
    diff_med = _safe_quantile(hard, 0.5)
    unique_labels = torch.unique(labels.detach()).numel()
    return {
        "semantic_state_key": SEMANTIC_STATE_KEYS[task],
        "semantic_factor": f"task-grounded near-boundary proxy: {label_rule}",
        "semantic_pair_rule": "task-grounded label contrast within closest state-distance neighborhood",
        "semantic_pair_count": int(hard.numel()),
        "semantic_skipped_anchor_count": int(skipped),
        "semantic_label_count": int(unique_labels),
        "semantic_label_rule": label_rule,
        "semantic_distance_threshold": float(threshold.detach().cpu()) if torch.isfinite(threshold) else float("nan"),
        "same_state_noisy_radius_median": same_med,
        "semantic_diff_l2_median": diff_med,
        "semantic_margin_median": _safe_quantile(margins, 0.5),
        "semantic_margin_pass_rate": float(passes.float().mean().detach().cpu()) if passes.numel() else float("nan"),
        "semantic_discriminability_ratio": diff_med / same_med if same_med > 0 else float("nan"),
        "local_state_distance_median": _safe_quantile(state_delta, 0.5),
        "local_task_feature_delta_median": _safe_quantile(feature_delta, 0.5),
    }


def _semantic_metrics(
    *,
    model,
    task: str,
    batch: Mapping[str, torch.Tensor],
    noisy_batch: Mapping[str, torch.Tensor],
    history_size: int,
    rollout_horizon: int,
    embedding_space: str,
    semantic_quantile: float,
    local_quantile: float,
    margin_delta: float,
    pair_rule: str,
) -> dict[str, Any]:
    clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
    noisy_outputs = phase0.encode_sequences(model, phase0._clone_batch(noisy_batch))
    clean_emb = phase0.get_embedding_space(clean_outputs, embedding_space).detach()
    noisy_emb = phase0.get_embedding_space(noisy_outputs, embedding_space).detach()
    act_emb = clean_outputs["act_emb"].detach()
    max_steps = min(rollout_horizon, max(0, act_emb.size(1) - history_size + 1))
    clean_chain = phase0._autoregressive_rollout(model, clean_emb[:, :history_size], act_emb, history_size, max_steps)
    noisy_chain = phase0._autoregressive_rollout(model, noisy_emb[:, :history_size], act_emb, history_size, max_steps)
    final_idx = history_size + max_steps - 1 if max_steps else history_size - 1
    clean_final = clean_chain[:, final_idx].float()
    noisy_final = noisy_chain[:, final_idx].float()
    same_radius = torch.linalg.vector_norm(clean_final - noisy_final, dim=-1)

    state = batch["state"].float()
    state_idx = min(state.size(1) - 1, final_idx)
    state_final = state[:, state_idx].reshape(state.size(0), -1)
    state_dist = torch.cdist(state_final, state_final, p=2)
    latent_dist = torch.cdist(clean_final, clean_final, p=2)
    n = state_dist.size(0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=state_dist.device)
    if pair_rule == "local_task_feature_contrast":
        return _local_feature_metrics(
            task=task,
            state_final=state_final,
            latent_dist=latent_dist,
            same_radius=same_radius,
            local_quantile=local_quantile,
            margin_delta=margin_delta,
        )
    if pair_rule == "task_grounded_near_boundary":
        return _task_grounded_near_boundary_metrics(
            task=task,
            state_final=state_final,
            latent_dist=latent_dist,
            same_radius=same_radius,
            local_quantile=local_quantile,
            margin_delta=margin_delta,
        )
    candidate_state = state_dist[offdiag]
    if candidate_state.numel() == 0:
        return {
            "semantic_state_key": SEMANTIC_STATE_KEYS[task],
            "semantic_factor": SEMANTIC_FACTORS[task],
            "semantic_pair_rule": "hard semantic-different pair above state-distance quantile",
            "semantic_pair_count": 0,
            "semantic_distance_threshold": float("nan"),
            "same_state_noisy_radius_median": _safe_quantile(same_radius, 0.5),
            "semantic_diff_l2_median": float("nan"),
            "semantic_margin_median": float("nan"),
            "semantic_margin_pass_rate": float("nan"),
            "semantic_discriminability_ratio": float("nan"),
        }
    threshold = torch.quantile(candidate_state, float(semantic_quantile))
    valid = offdiag & (state_dist >= threshold)
    hard_diffs = []
    aligned_same = []
    for i in range(n):
        row = valid[i]
        if bool(row.any()):
            hard_diffs.append(torch.min(latent_dist[i][row]))
            aligned_same.append(same_radius[i])
    if not hard_diffs:
        hard = torch.empty(0, device=clean_final.device)
        same = torch.empty(0, device=clean_final.device)
    else:
        hard = torch.stack(hard_diffs)
        same = torch.stack(aligned_same)
    margins = hard - same
    passes = margins > float(margin_delta)
    same_med = _safe_quantile(same, 0.5)
    diff_med = _safe_quantile(hard, 0.5)
    return {
        "semantic_state_key": SEMANTIC_STATE_KEYS[task],
        "semantic_factor": SEMANTIC_FACTORS[task],
        "semantic_pair_rule": "hard semantic-different pair above state-distance quantile",
        "semantic_pair_count": int(hard.numel()),
        "semantic_distance_threshold": float(threshold.detach().cpu()),
        "same_state_noisy_radius_median": same_med,
        "semantic_diff_l2_median": diff_med,
        "semantic_margin_median": _safe_quantile(margins, 0.5),
        "semantic_margin_pass_rate": float(passes.float().mean().detach().cpu()) if passes.numel() else float("nan"),
        "semantic_discriminability_ratio": diff_med / same_med if same_med > 0 else float("nan"),
    }


def _run_row(task: str, std_key: str, entry: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_file, tried = phase0.resolve_model_file(str(entry.get("path", "")), str(entry.get("subdir", "")), [])
    base = {
        "training_seed": int(args.training_seed),
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
            model = phase0.load_model(str(model_file), device)
            history_size = phase0.infer_history_size(model)
            future_steps = max(args.future_steps, args.rollout_horizon + 1)
            state_key = SEMANTIC_STATE_KEYS[task]
            run_path = Path(str(entry.get("path", ""))).expanduser()
            if run_path.parent.name == "ckpt":
                os.environ["STABLEWM_HOME"] = str(run_path.parent.parent)
            batch = phase0.load_dataset_samples(
                dataset_name=phase0.TASK_DATASETS[task],
                state_key=state_key,
                n_sequences=args.n_sequences,
                history_size=history_size,
                future_steps=future_steps,
                frameskip=args.frameskip,
                img_size=args.img_size,
                seed=args.seed,
                device=device,
            )
            noisy_batch = phase0.make_paired_noisy_batch(
                batch,
                history_size=history_size,
                noise_std=args.noise_std,
                seed=args.seed + 1009,
                corruption_type=args.corruption_type,
                corrupt_goal=False,
            )
            spaces = phase0.get_model_spaces(model)
            embedding_space = args.embedding_space or spaces["inference_cost_space"]
            metrics = _semantic_metrics(
                model=model,
                task=task,
                batch=batch,
                noisy_batch=noisy_batch,
                history_size=history_size,
                rollout_horizon=args.rollout_horizon,
                embedding_space=embedding_space,
                semantic_quantile=args.semantic_quantile,
                local_quantile=args.local_quantile,
                margin_delta=args.margin_delta,
                pair_rule=args.pair_rule,
            )
            return {
                **base,
                "status": "ok",
                "history_size": int(history_size),
                "n_sequences": int(args.n_sequences),
                "rollout_horizon": int(args.rollout_horizon),
                "embedding_space": embedding_space,
                "noise_std": float(args.noise_std),
                **metrics,
            }
    except Exception as exc:  # noqa: BLE001 - row-level artifact should record failures.
        return {**base, "status": "error", "error": repr(exc)}


def _summaries(rows: Sequence[dict[str, Any]], std_keys: Sequence[str] | None = None) -> list[dict[str, Any]]:
    out = []
    keys = list(std_keys) if std_keys is not None else sorted({str(r["std_key"]) for r in rows})
    for task in TASKS:
        for std_key in keys:
            task_rows = [r for r in rows if r["task"] == task and r["std_key"] == std_key and r["status"] == "ok"]
            if not task_rows:
                continue
            summary = {"task": task, "std_key": std_key, "training_seeds": sorted(int(r["training_seed"]) for r in task_rows), "n_training_seeds": len(task_rows)}
            for key in (
                "semantic_margin_pass_rate",
                "semantic_discriminability_ratio",
                "same_state_noisy_radius_median",
                "semantic_diff_l2_median",
                "semantic_margin_median",
                "local_state_distance_median",
                "local_task_feature_delta_median",
            ):
                if key not in task_rows[0]:
                    continue
                vals = [float(r[key]) for r in task_rows]
                summary[f"{key}_mean"] = _mean(vals)
                summary[f"{key}_pstdev"] = _pstdev(vals)
            out.append(summary)
    return out


def _iter_rows(manifest: Mapping[str, Any], tasks: Sequence[str], std_keys: Sequence[str]) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    for task in tasks:
        for std_key in std_keys:
            yield task, std_key, manifest[task][std_key]


def run_for_seed(seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    args.training_seed = seed
    manifest = _load(args.manifest_dir / f"lewm_seed{seed}_evals.json")
    rows = []
    for task, std_key, entry in _iter_rows(manifest, args.tasks, args.std_keys):
        rows.append(_run_row(task, std_key, entry, args))
        if args.limit is not None and len(rows) >= args.limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--semantic-quantile", type=float, default=0.5)
    parser.add_argument("--local-quantile", type=float, default=0.35)
    parser.add_argument("--pair-rule", choices=["global_state_quantile", "local_task_feature_contrast", "task_grounded_near_boundary"], default="global_state_quantile")
    parser.add_argument("--margin-delta", type=float, default=0.0)
    parser.add_argument("--state-key-seed", dest="seed", type=int, default=9101)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", choices=["raw", "normalized"], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows.extend(run_for_seed(seed, args))
    coverage = {
        f"{task}:{std_key}": sorted(int(r["training_seed"]) for r in rows if r["task"] == task and r["std_key"] == std_key and r["status"] == "ok")
        for task in args.tasks
        for std_key in args.std_keys
    }
    payload = {
        "metadata": {
            "schema_version": "paper1-semantic-margin-passrate-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "fixed semantic guard over requested LeWM training seeds and training-noise std keys; no retraining",
            "semantic_state_keys": SEMANTIC_STATE_KEYS,
            "semantic_factors": SEMANTIC_FACTORS,
            "std_keys": list(args.std_keys),
            "training_seeds": list(args.seeds),
            "semantic_quantile": float(args.semantic_quantile),
            "local_quantile": float(args.local_quantile),
            "pair_rule": args.pair_rule,
            "margin_delta": float(args.margin_delta),
        },
        "rows": rows,
        "summary_rows": _summaries(rows, args.std_keys),
        "coverage": coverage,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    try:
        display_out = args.out.relative_to(ROOT)
    except ValueError:
        display_out = args.out
    print(f"wrote {display_out}")
    print("status counts:", counts)


if __name__ == "__main__":
    main()
