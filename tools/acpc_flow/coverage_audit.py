from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from acpc_flow import sample_latent_noise
from tools.repr_analysis.analyze_repr import (
    infer_history_size,
    load_dataset_samples,
    load_model,
    sample_random_future_actions,
    spearman_corr,
    to_serializable,
)
from tools.repr_analysis.noise_sensitivity import _add_eval_corruption


DATASET_ROOTS = {
    "tworoom": "lewm-tworooms",
    "reacher": "lewm-reacher",
    "pusht": "lewm-pusht",
    "cube": "lewm-cube",
}
DATASET_NAMES = {
    "tworoom": "tworoom",
    "reacher": "reacher",
    "pusht": "pusht_expert_train",
    "cube": "ogbench/cube_single_expert",
}
STATE_KEYS = {
    "tworoom": "proprio",
    "reacher": "observation",
    "pusht": "state",
    "cube": "observation",
}
FEATURE_LEVELS = ("encoder_feat", "emb", "predictor_hidden", "pred_emb")
EPS = 1e-8


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.size(-1))


def _q(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().float().cpu(), q))


def _q_finite(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    x = x.detach().float()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.cpu(), q))


def _quantiles(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        f"{prefix}_mean": float(x.mean().cpu()),
        f"{prefix}_median": _q(x, 0.50),
        f"{prefix}_q75": _q(x, 0.75),
        f"{prefix}_q90": _q(x, 0.90),
        f"{prefix}_q95": _q(x, 0.95),
        f"{prefix}_q99": _q(x, 0.99),
    }


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.detach().float().mean().cpu())


def _project_features(model, feat_seq: torch.Tensor) -> torch.Tensor:
    b, t = feat_seq.shape[:2]
    flat = feat_seq.reshape(b * t, feat_seq.size(-1))
    emb = model.projector(flat).reshape(b, t, -1)
    normalize = getattr(model, "normalize_embeddings", None)
    if callable(normalize):
        emb = normalize(emb)
    return emb


def _predictor_levels(model, emb_seq: torch.Tensor, act_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.predictor(emb_seq, act_emb)
    b, t = hidden.shape[:2]
    pred_flat = model.pred_proj(hidden.reshape(b * t, hidden.size(-1)))
    pred_emb = pred_flat.reshape(b, t, -1)
    return hidden, pred_emb


def _add_predictor_levels(model, info: dict[str, torch.Tensor], history_size: int) -> dict[str, torch.Tensor]:
    if "emb" not in info or "act_emb" not in info:
        return info
    emb = info["emb"][:, :history_size]
    act_emb = info["act_emb"][:, :history_size]
    hidden, pred_emb = _predictor_levels(model, emb, act_emb)
    info["predictor_hidden"] = hidden.detach()
    info["pred_emb"] = pred_emb.detach()
    return info


def _context_view(x: torch.Tensor, history_size: int) -> torch.Tensor:
    return x[:, :history_size]


def _noise_like(clean: torch.Tensor, std: float, mode: str) -> torch.Tensor:
    if std <= 0:
        return torch.zeros_like(clean)
    return sample_latent_noise(
        clean,
        std_min=std,
        std_max=std,
        mode=mode,
        relative=True,
        sample_per_token=True,
    )


def _scheduled_std(t: float, sigma_max: float, schedule: str) -> float:
    t = max(0.0, min(1.0, float(t)))
    if schedule == "variance_preserving":
        return float(sigma_max) * math.sqrt(t)
    if schedule == "linear":
        return float(sigma_max) * t
    raise ValueError(f"Unsupported noise schedule: {schedule}")


def _first_covering_t(values: torch.Tensor, thresholds_by_t: list[tuple[float, float]]) -> torch.Tensor:
    out = torch.full_like(values.float(), float("nan"))
    unresolved = torch.ones_like(values, dtype=torch.bool)
    for t, threshold in sorted(thresholds_by_t, key=lambda item: item[0]):
        covered = unresolved & (values <= float(threshold))
        out[covered] = float(t)
        unresolved = unresolved & ~covered
    return out


def _build_action_candidates(
    action: torch.Tensor,
    *,
    history_size: int,
    future_steps: int,
    random_action_trials: int,
    seed: int,
) -> torch.Tensor:
    if future_steps < 2:
        raise ValueError("future_steps must be >= 2 for candidate-rank metrics")
    if random_action_trials < 1:
        raise ValueError("random_action_trials must be >= 1 for candidate-rank metrics")
    future_action_steps = future_steps - 1
    if action.size(1) < history_size + future_action_steps:
        raise ValueError(
            "Loaded action sequence is too short for candidate-rank metrics: "
            f"got {action.size(1)}, need {history_size + future_action_steps}"
        )
    b = action.size(0)
    expert_future = action[:, history_size : history_size + future_action_steps]
    expert_candidate = action[:, : history_size + future_action_steps].unsqueeze(1)
    random_future = sample_random_future_actions(
        expert_future, n_trials=random_action_trials, seed=seed
    )
    history = action[:, :history_size].unsqueeze(1).expand(
        b, random_action_trials, history_size, -1
    )
    random_candidates = torch.cat([history, random_future], dim=2)
    return torch.cat([expert_candidate, random_candidates], dim=1)


def _rollout_from_context(
    model,
    context_emb: torch.Tensor,
    action_candidates: torch.Tensor,
    *,
    history_size: int,
    future_steps: int,
) -> torch.Tensor:
    b, n_candidates, action_steps = action_candidates.shape[:3]
    init = context_emb.unsqueeze(1).expand(
        b, n_candidates, history_size, context_emb.size(-1)
    )
    chain = init.reshape(b * n_candidates, history_size, context_emb.size(-1)).clone()
    act_flat = action_candidates.reshape(b * n_candidates, action_steps, action_candidates.size(-1))
    act_emb = model.action_encoder(act_flat)
    for step in range(future_steps):
        action_window = act_emb[:, step : step + history_size]
        if action_window.size(1) < history_size:
            break
        pred = model.predict(chain[:, -history_size:], action_window)[:, -1:]
        chain = torch.cat([chain, pred], dim=1)
    return chain


def _candidate_costs_from_context(
    model,
    context_emb: torch.Tensor,
    goal_emb: torch.Tensor,
    action_candidates: torch.Tensor,
    *,
    history_size: int,
    future_steps: int,
) -> torch.Tensor:
    b, n_candidates = action_candidates.shape[:2]
    chain = _rollout_from_context(
        model,
        context_emb,
        action_candidates,
        history_size=history_size,
        future_steps=future_steps,
    )
    pred_final = chain[:, -1].reshape(b, n_candidates, -1)
    goal_final = goal_emb[:, -1].unsqueeze(1).expand_as(pred_final)
    cost_type = str(getattr(model, "inference_cost_type", "mse")).lower()
    if cost_type == "cosine":
        return 1.0 - F.cosine_similarity(pred_final, goal_final, dim=-1)
    return F.mse_loss(pred_final, goal_final, reduction="none").sum(dim=-1)


def _candidate_rank_metrics(clean_costs: torch.Tensor, other_costs: torch.Tensor, topk: int) -> dict[str, float]:
    clean_cpu = clean_costs.detach().float().cpu()
    other_cpu = other_costs.detach().float().cpu()
    spearmans = [spearman_corr(c, o) for c, o in zip(clean_cpu, other_cpu) if c.numel() > 1]
    spearman_t = torch.tensor(spearmans, dtype=torch.float32)
    k = max(1, min(int(topk), clean_cpu.size(1)))
    clean_top = torch.topk(clean_cpu, k=k, largest=False).indices
    other_top = torch.topk(other_cpu, k=k, largest=False).indices
    overlaps = []
    for clean_row, other_row in zip(clean_top, other_top):
        overlaps.append(len(set(clean_row.tolist()) & set(other_row.tolist())) / float(k))
    overlap_t = torch.tensor(overlaps, dtype=torch.float32)
    clean_sorted = torch.sort(clean_cpu, dim=1).values
    if clean_cpu.size(1) > 1:
        margins = clean_sorted[:, 1] - clean_sorted[:, 0]
    else:
        margins = torch.empty(0, dtype=torch.float32)
    return {
        "candidate_count": float(clean_cpu.size(1)),
        "candidate_rank_spearman": _safe_mean(spearman_t),
        "candidate_rank_spearman_median": _q(spearman_t, 0.50),
        "candidate_top1_flip_rate": float((clean_cpu.argmin(dim=1) != other_cpu.argmin(dim=1)).float().mean()),
        "candidate_topk": float(k),
        "candidate_topk_overlap_rate": _safe_mean(overlap_t),
        "candidate_margin_clean_q10": _q(margins, 0.10),
        "candidate_margin_clean_q50": _q(margins, 0.50),
    }


def _clean_knn_distance(clean: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(clean.float(), clean.float())
    eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    dist = dist.masked_fill(eye, float("inf"))
    k_eff = min(k, max(1, dist.size(0) - 1))
    return dist.topk(k_eff, largest=False).values.mean(dim=1).clamp_min(1e-6)


def _paired_rank_and_crossing(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    state: torch.Tensor | None,
    *,
    k: int,
) -> dict[str, float]:
    dist = torch.cdist(corrupt.float(), clean.float())
    paired = dist.diag()
    paired_rank = (dist < paired[:, None]).sum(dim=1).float() + 1.0
    result = {
        "paired_clean_rank_median": _q(paired_rank, 0.50),
        "paired_clean_rank_q90": _q(paired_rank, 0.90),
    }
    if state is None:
        result.update(
            {
                "wrong_label_nn_rate": float("nan"),
                "closer_to_wrong_than_pair_rate": float("nan"),
                "same_label_topk_rate": float("nan"),
                "state_proxy_missing": 1.0,
            }
        )
        return result

    state = state.float()
    state_dist = torch.cdist(state, state)
    state_knn = _clean_knn_distance(state, k=k)
    wrong = state_dist > state_knn[:, None]
    nearest = dist.argmin(dim=1)
    wrong_nn = wrong[torch.arange(dist.size(0), device=dist.device), nearest]
    closer_wrong = (dist.masked_fill(~wrong, float("inf")) < paired[:, None]).any(dim=1)
    topk = dist.topk(min(k, dist.size(1)), largest=False).indices
    same_topk = (~wrong.gather(1, topk)).float().mean(dim=1)
    result.update(
        {
            "wrong_label_nn_rate": float(wrong_nn.float().mean().cpu()),
            "closer_to_wrong_than_pair_rate": float(closer_wrong.float().mean().cpu()),
            "same_label_topk_rate": float(same_topk.mean().cpu()),
            "state_proxy_missing": 0.0,
        }
    )
    return result


def _anisotropy(delta: torch.Tensor) -> dict[str, float]:
    x = delta.float() - delta.float().mean(dim=0, keepdim=True)
    if x.size(0) < 2:
        return {
            "effective_rank": float("nan"),
            "lambda_max_over_trace": float("nan"),
            "top1_eigen_ratio": float("nan"),
            "top5_eigen_ratio": float("nan"),
        }
    cov = x.T @ x / float(x.size(0) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0).sort(descending=True).values
    trace = eig.sum().clamp_min(1e-12)
    p = eig / trace
    entropy_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return {
        "effective_rank": float(entropy_rank.cpu()),
        "lambda_max_over_trace": float((eig[0] / trace).cpu()),
        "top1_eigen_ratio": float((eig[0] / trace).cpu()),
        "top5_eigen_ratio": float((eig[:5].sum() / trace).cpu()),
    }


def _task_alignment(delta: torch.Tensor, clean: torch.Tensor, state: torch.Tensor | None, k: int) -> dict[str, float]:
    if state is None:
        return {"task_alignment_mean": float("nan"), "task_alignment_q90": float("nan"), "task_alignment_q95": float("nan")}
    state_dist = torch.cdist(state.float(), state.float())
    state_knn = _clean_knn_distance(state.float(), k=k)
    wrong = state_dist > state_knn[:, None]
    clean_diff = clean[None, :, :] - clean[:, None, :]
    delta_n = F.normalize(delta.float(), dim=-1, eps=1e-8)
    diff_n = F.normalize(clean_diff.float(), dim=-1, eps=1e-8)
    cos = (delta_n[:, None, :] * diff_n).sum(dim=-1).masked_fill(~wrong, -1.0)
    max_align = cos.max(dim=1).values
    return {
        "task_alignment_mean": float(max_align.mean().cpu()),
        "task_alignment_q90": _q(max_align, 0.90),
        "task_alignment_q95": _q(max_align, 0.95),
    }


def _synthetic_coverage(
    clean: torch.Tensor,
    delta_norm: torch.Tensor,
    std_grid: list[float],
    mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    coverage: dict[str, dict[str, float]] = {}
    radii: dict[str, dict[str, float]] = {}
    clean_shape = clean.unsqueeze(1)
    for std in std_grid:
        noise = sample_latent_noise(
            clean_shape,
            std_min=std,
            std_max=std,
            mode=mode,
            relative=True,
            sample_per_token=True,
        ).squeeze(1)
        radius = torch.linalg.vector_norm(noise, dim=-1)
        key = f"{std:.4g}"
        radii[key] = {"q90": _q(radius, 0.90), "q95": _q(radius, 0.95), "q99": _q(radius, 0.99)}
        coverage[key] = {
            "q90": float((delta_norm <= radii[key]["q90"]).float().mean().cpu()),
            "q95": float((delta_norm <= radii[key]["q95"]).float().mean().cpu()),
            "q99": float((delta_norm <= radii[key]["q99"]).float().mean().cpu()),
        }
    return coverage, radii


def _predict_gap(model, clean_ctx: torch.Tensor, corrupt_ctx: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    clean_pred = model.predict(clean_ctx, act)
    corrupt_pred = model.predict(corrupt_ctx, act)
    return (clean_pred - corrupt_pred).pow(2).mean(dim=-1).reshape(-1)


def _level_gap(
    model,
    level: str,
    clean: torch.Tensor,
    other: torch.Tensor,
    act_emb: torch.Tensor,
) -> torch.Tensor:
    if level == "encoder_feat":
        clean_emb = _project_features(model, clean)
        other_emb = _project_features(model, other)
        return _predict_gap(model, clean_emb, other_emb, act_emb)
    if level == "emb":
        return _predict_gap(model, clean, other, act_emb)
    if level == "predictor_hidden":
        b, t = clean.shape[:2]
        clean_pred = model.pred_proj(clean.reshape(b * t, clean.size(-1))).reshape(b, t, -1)
        other_pred = model.pred_proj(other.reshape(b * t, other.size(-1))).reshape(b, t, -1)
        return (clean_pred - other_pred).pow(2).mean(dim=-1).reshape(-1)
    if level == "pred_emb":
        return (clean - other).pow(2).mean(dim=-1).reshape(-1)
    raise ValueError(f"Unsupported feature level: {level}")


def _synthetic_acpc_gaps(
    model,
    level: str,
    clean_seq: torch.Tensor,
    act: torch.Tensor,
    std_grid: list[float],
    mode: str,
) -> dict[str, dict[str, float]]:
    gaps: dict[str, dict[str, float]] = {}
    for std in std_grid:
        noise = _noise_like(clean_seq, std, mode)
        gap = _level_gap(model, level, clean_seq, clean_seq + noise, act)
        gaps[f"{std:.4g}"] = {
            "mean": float(gap.mean().cpu()),
            "q90": _q(gap, 0.90),
            "q95": _q(gap, 0.95),
        }
    return gaps


def _coverage_strength(metrics: Mapping[str, Any]) -> str:
    ratio_q90 = metrics.get("ratio_to_knn_q90", float("inf"))
    wrong_rate = metrics.get("wrong_label_nn_rate", float("nan"))
    closer_wrong = metrics.get("closer_to_wrong_than_pair_rate", float("nan"))
    cov_004 = metrics.get("coverage_q95_by_alpha", {}).get("0.04", 0.0)
    cov_008 = metrics.get("coverage_q95_by_alpha", {}).get("0.08", 0.0)
    pixel_gap = metrics.get("acpc_gap_q90", float("nan"))
    synth = metrics.get("synthetic_acpc_gap_q90_by_alpha", {})
    max_synth_gap = max([v for v in synth.values() if not math.isnan(v)], default=float("nan"))
    if (not math.isnan(wrong_rate) and wrong_rate >= 0.30) or (
        not math.isnan(closer_wrong) and closer_wrong >= 0.40
    ):
        return "no_go"
    if (
        not math.isnan(pixel_gap)
        and not math.isnan(max_synth_gap)
        and max_synth_gap > 0
        and pixel_gap > 2.0 * max_synth_gap
    ):
        return "no_go"
    if ratio_q90 >= 0.8 or (not math.isnan(wrong_rate) and wrong_rate >= 0.15) or cov_008 < 0.50:
        return "low"
    if ratio_q90 < 0.3 and (math.isnan(wrong_rate) or wrong_rate < 0.05) and cov_004 > 0.80:
        return "high"
    return "medium"


def _encode(model, pixels: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
    info = model.encode({"pixels": pixels, "action": action})
    return {k: v.detach() for k, v in info.items() if torch.is_tensor(v)}


def _audit_space(
    *,
    model,
    level: str,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    act_emb: torch.Tensor,
    state: torch.Tensor | None,
    std_grid: list[float],
    noise_mode: str,
    knn_k: int,
    candidate_rank: Mapping[str, float] | None,
    t_calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    clean_flat = _flatten(clean)
    corrupt_flat = _flatten(corrupt)
    delta = corrupt_flat - clean_flat
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    knn = _clean_knn_distance(clean_flat, k=knn_k)
    ratio = delta_norm / knn
    state_flat = _flatten(state) if state is not None else None
    coverage, radii = _synthetic_coverage(clean_flat, delta_norm, std_grid, noise_mode)
    gap = _level_gap(model, level, clean, corrupt, act_emb)
    synth_gaps = _synthetic_acpc_gaps(model, level, clean, act_emb, std_grid, noise_mode)

    metrics: dict[str, Any] = {
        **_quantiles("delta_norm", delta_norm),
        **_quantiles("ratio_to_knn", ratio),
        **_paired_rank_and_crossing(clean_flat, corrupt_flat, state_flat, k=knn_k),
        **_anisotropy(delta),
        **_task_alignment(delta, clean_flat, state_flat, k=knn_k),
        "coverage_q95_by_alpha": {k: v["q95"] for k, v in coverage.items()},
        "coverage_by_alpha": coverage,
        "synthetic_radius_by_alpha": radii,
        "acpc_gap_mean": float(gap.mean().cpu()),
        "acpc_gap_q90": _q(gap, 0.90),
        "acpc_gap_q95": _q(gap, 0.95),
        "synthetic_acpc_gap_q90_by_alpha": {k: v["q90"] for k, v in synth_gaps.items()},
        "synthetic_acpc_gap_by_alpha": synth_gaps,
    }
    if candidate_rank is None:
        metrics["candidate_rank_metrics_computed"] = False
    else:
        metrics["candidate_rank_metrics_computed"] = True
        metrics.update(candidate_rank)
    if t_calibration is not None:
        metrics["t_calibration"] = dict(t_calibration)
        for key in (
            "t_star_radius_median",
            "t_star_radius_q90",
            "t_star_radius_q95",
            "t_star_acpc_median",
            "t_star_acpc_q90",
            "t_star_acpc_q95",
            "wrong_label_rate_at_t_star",
            "clean_false_positive_rate",
            "is_t_start_separable_from_clean",
        ):
            metrics[key] = t_calibration.get(key, float("nan"))
    metrics["coverage_decision"] = _coverage_strength(metrics)
    return metrics


def _amplification_metrics(
    clean_levels: Mapping[str, torch.Tensor],
    corrupt_levels: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    norms: dict[str, torch.Tensor] = {}
    for level in FEATURE_LEVELS:
        if level in clean_levels and level in corrupt_levels:
            delta = _flatten(corrupt_levels[level]) - _flatten(clean_levels[level])
            norms[level] = torch.linalg.vector_norm(delta, dim=-1)
    metrics: dict[str, float] = {}
    pairs = {
        "amp_P": ("encoder_feat", "emb"),
        "amp_B": ("emb", "predictor_hidden"),
        "amp_R": ("predictor_hidden", "pred_emb"),
        "amp_total": ("encoder_feat", "pred_emb"),
    }
    for name, (src, dst) in pairs.items():
        if src not in norms or dst not in norms:
            continue
        amp = norms[dst] / (norms[src] + EPS)
        metrics.update(_quantiles(name, amp))
    return metrics


def _candidate_rank_bundle(
    model,
    batch: Mapping[str, torch.Tensor],
    clean_info: Mapping[str, torch.Tensor],
    corrupt_info: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    future_steps: int,
    random_action_trials: int,
    topk: int,
    std_grid: list[float],
    noise_mode: str,
    seed: int,
) -> dict[str, Any]:
    try:
        candidates = _build_action_candidates(
            batch["action"],
            history_size=history_size,
            future_steps=future_steps,
            random_action_trials=random_action_trials,
            seed=seed,
        )
        clean_context = _context_view(clean_info["emb"], history_size)
        corrupt_context = _context_view(corrupt_info["emb"], history_size)
        clean_goal = clean_info["emb"][:, -1:]
        clean_costs = _candidate_costs_from_context(
            model,
            clean_context,
            clean_goal,
            candidates,
            history_size=history_size,
            future_steps=future_steps,
        )
        corrupt_costs = _candidate_costs_from_context(
            model,
            corrupt_context,
            clean_goal,
            candidates,
            history_size=history_size,
            future_steps=future_steps,
        )
        synth_encoder: dict[str, dict[str, float]] = {}
        synth_emb: dict[str, dict[str, float]] = {}
        clean_feat = _context_view(clean_info["encoder_feat"], history_size)
        for std in std_grid:
            key = f"{std:.4g}"
            feat_source = clean_feat + _noise_like(clean_feat, std, noise_mode)
            feat_context = _project_features(model, feat_source)
            feat_costs = _candidate_costs_from_context(
                model,
                feat_context,
                clean_goal,
                candidates,
                history_size=history_size,
                future_steps=future_steps,
            )
            emb_context = clean_context + _noise_like(clean_context, std, noise_mode)
            emb_costs = _candidate_costs_from_context(
                model,
                emb_context,
                clean_goal,
                candidates,
                history_size=history_size,
                future_steps=future_steps,
            )
            synth_encoder[key] = _candidate_rank_metrics(clean_costs, feat_costs, topk)
            synth_emb[key] = _candidate_rank_metrics(clean_costs, emb_costs, topk)
        return {
            "computed": True,
            "pixel": _candidate_rank_metrics(clean_costs, corrupt_costs, topk),
            "synthetic_encoder_by_alpha": synth_encoder,
            "synthetic_emb_by_alpha": synth_emb,
            "synthetic_predictor_hidden": {
                "computed": False,
                "reason": "requires hidden-state injection inside autoregressive rollout",
            },
        }
    except Exception as exc:  # pragma: no cover - surfaced in audit artifact.
        return {"computed": False, "reason": str(exc)}


def _t_calibration_metrics(
    model,
    level: str,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    *,
    act_emb: torch.Tensor,
    state: torch.Tensor | None,
    t_grid: list[float],
    sigma_max: float,
    noise_schedule: str,
    noise_mode: str,
    knn_k: int,
) -> dict[str, Any]:
    clean_flat = _flatten(clean)
    corrupt_flat = _flatten(corrupt)
    delta_norm = torch.linalg.vector_norm(corrupt_flat - clean_flat, dim=-1)
    pixel_gap = _level_gap(model, level, clean, corrupt, act_emb)
    state_flat = _flatten(state) if state is not None else None

    radius_thresholds: list[tuple[float, float]] = []
    gap_thresholds: list[tuple[float, float]] = []
    radius_by_t: dict[str, dict[str, float]] = {}
    gap_by_t: dict[str, dict[str, float]] = {}
    crossing_by_t: dict[str, dict[str, float]] = {}
    for t in t_grid:
        std = _scheduled_std(t, sigma_max, noise_schedule)
        source = clean + _noise_like(clean, std, noise_mode)
        source_flat = _flatten(source)
        radius = torch.linalg.vector_norm(source_flat - clean_flat, dim=-1)
        gap = _level_gap(model, level, clean, source, act_emb)
        key = f"{t:.4g}"
        radius_by_t[key] = {
            "std": std,
            "q90": _q(radius, 0.90),
            "q95": _q(radius, 0.95),
            "q99": _q(radius, 0.99),
        }
        gap_by_t[key] = {
            "std": std,
            "q90": _q(gap, 0.90),
            "q95": _q(gap, 0.95),
            "q99": _q(gap, 0.99),
        }
        radius_thresholds.append((float(t), radius_by_t[key]["q95"]))
        gap_thresholds.append((float(t), gap_by_t[key]["q95"]))
        crossing_by_t[key] = _paired_rank_and_crossing(clean_flat, source_flat, state_flat, k=knn_k)

    t_star_radius = _first_covering_t(delta_norm, radius_thresholds)
    t_star_acpc = _first_covering_t(pixel_gap, gap_thresholds)
    t_ref = _q_finite(t_star_radius, 0.90)
    if math.isnan(t_ref):
        wrong_at_t = float("nan")
    else:
        nearest_t = min(t_grid, key=lambda t: abs(float(t) - t_ref))
        wrong_at_t = crossing_by_t[f"{nearest_t:.4g}"].get("wrong_label_nn_rate", float("nan"))
    radius_uncovered = float(torch.isnan(t_star_radius).float().mean().cpu())
    acpc_uncovered = float(torch.isnan(t_star_acpc).float().mean().cpu())
    separable = (
        not math.isnan(t_ref)
        and t_ref <= 0.4
        and (math.isnan(wrong_at_t) or wrong_at_t < 0.15)
        and radius_uncovered < 0.20
    )
    return {
        "t_grid": [float(t) for t in t_grid],
        "noise_schedule": noise_schedule,
        "t_sigma_max": float(sigma_max),
        "synthetic_radius_by_t": radius_by_t,
        "synthetic_acpc_gap_by_t": gap_by_t,
        "synthetic_crossing_by_t": crossing_by_t,
        "t_star_radius_median": _q_finite(t_star_radius, 0.50),
        "t_star_radius_q90": _q_finite(t_star_radius, 0.90),
        "t_star_radius_q95": _q_finite(t_star_radius, 0.95),
        "t_star_acpc_median": _q_finite(t_star_acpc, 0.50),
        "t_star_acpc_q90": _q_finite(t_star_acpc, 0.90),
        "t_star_acpc_q95": _q_finite(t_star_acpc, 0.95),
        "wrong_label_rate_at_t_star": wrong_at_t,
        "clean_false_positive_rate": 0.0,
        "is_t_start_separable_from_clean": bool(separable),
        "radius_uncovered_rate": radius_uncovered,
        "acpc_uncovered_rate": acpc_uncovered,
        "selection_note": "radius/acpc oracle grid audit; not a learned t_start estimator",
    }


def _v2_decision(
    level: str,
    metrics: Mapping[str, Any],
    amp: Mapping[str, float],
) -> tuple[str, str, str]:
    strength = str(metrics.get("coverage_decision", "no_go"))
    rank_flip = float(metrics.get("candidate_top1_flip_rate", float("nan")))
    rank_unstable = not math.isnan(rank_flip) and rank_flip >= 0.25
    amp_p = float(amp.get("amp_P_q90", float("nan")))
    amp_b = float(amp.get("amp_B_q90", float("nan")))
    amp_r = float(amp.get("amp_R_q90", float("nan")))
    t_sep = bool(metrics.get("is_t_start_separable_from_clean", False))
    t_q90 = float(metrics.get("t_star_radius_q90", float("nan")))

    if strength == "no_go":
        return "no_go", "coverage/crossing gate failed", "do_not_train"
    if level == "emb" and not math.isnan(amp_p) and amp_p >= 1.25 and strength in {"medium", "high"}:
        return "encoder_projector_small_train", "encoder projector P amplifies a covered shift", "stage_b_encoder_projector"
    if level == "pred_emb" and not math.isnan(amp_r) and (amp_r >= 1.25 or rank_unstable):
        return "predictor_projector_small_train", "pred_proj R amplifies residual shift or destabilizes candidate rank", "stage_c_predictor_projector"
    if level == "predictor_hidden" and not math.isnan(amp_b) and amp_b >= 1.25:
        return "pixel_paired_source_candidate", "predictor backbone B amplifies residual shift", "audit_backbone_or_pixel_paired_source"
    if t_sep and (math.isnan(t_q90) or t_q90 <= 0.4):
        return "t_conditioned_fm_candidate", "t grid is locally separable with low crossing", "stage_d_oracle_t_upper_bound"
    if strength == "low":
        return "weak_local_only", "only weak local synthetic coverage", "no_scale_training"
    return "pixel_paired_source_candidate", "coverage is plausible but needs paired-source validation", "paired_source_or_small_offline_ablation"


def _build_decision_table(
    task: str,
    results: dict[str, dict[str, Any]],
    amplification: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, by_corr in results.items():
        for corr, metrics in by_corr.items():
            amp = amplification.get(corr, {})
            for amp_key, amp_value in amp.items():
                metrics[amp_key] = amp_value
            decision, reason, action = _v2_decision(level, metrics, amp)
            metrics["v2_decision"] = decision
            metrics["decision_reason"] = reason
            metrics["recommended_next_action"] = action
            rows.append(
                {
                    "task": task,
                    "stressor": corr,
                    "level": level,
                    "decision": decision,
                    "reason": reason,
                    "recommended_next_action": action,
                }
            )
    return rows


def _recommendation(decision_table: list[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = [str(row["decision"]) for row in decision_table]
    if "encoder_projector_small_train" in decisions:
        action = "stage_b_encoder_projector"
        worth = True
        reason = "v2 audit found a covered projector-amplified shift; run encoder projector-as-transport only."
    elif "predictor_projector_small_train" in decisions:
        action = "stage_c_predictor_projector"
        worth = True
        reason = "v2 audit found predictor projection amplification or candidate-rank instability."
    elif "t_conditioned_fm_candidate" in decisions:
        action = "stage_d_oracle_t_upper_bound"
        worth = False
        reason = "Run only oracle-t upper bound before implementing learned t_start."
    elif any(d != "no_go" for d in decisions):
        action = "paired_source_or_small_offline_ablation"
        worth = False
        reason = "Some levels are not no-go, but evidence is not strong enough for training."
    else:
        action = "do_not_train"
        worth = False
        reason = "All audited paths remain no-go."
    return {
        "worth_doing": worth,
        "recommended_next_action": action,
        "reason": reason,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.stablewm_home:
        os.environ["STABLEWM_HOME"] = args.stablewm_home
    task = args.task
    dataset_name = args.dataset_name or DATASET_NAMES[task]
    state_key = args.state_key if args.state_key != "auto" else STATE_KEYS.get(task)
    std_grid = [float(x) for x in args.synthetic_std_grid.split(",") if x]
    t_grid = [float(x) for x in args.t_grid.split(",") if x]
    candidate_std_grid = [float(x) for x in args.candidate_synthetic_std_grid.split(",") if x]
    t_sigma_max = args.t_sigma_max if args.t_sigma_max is not None else max(std_grid)

    model = load_model(args.checkpoint, args.device)
    model.eval()
    history_size = infer_history_size(model)
    batch = load_dataset_samples(
        dataset_name=dataset_name,
        state_key=state_key,
        n_sequences=args.num_samples,
        history_size=history_size,
        future_steps=max(1, args.future_steps),
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.seed,
        device=args.device,
    )
    with torch.no_grad():
        clean_info = _add_predictor_levels(model, _encode(model, batch["pixels"], batch["action"]), history_size)
        state = batch.get("state")
        state_context = _context_view(state, history_size) if state is not None else None
        act_emb = clean_info["act_emb"][:, :history_size]
        clean_levels = {
            level: _context_view(clean_info[level], history_size)
            for level in FEATURE_LEVELS
            if level in clean_info
        }

        results: dict[str, dict[str, Any]] = {level: {} for level in FEATURE_LEVELS}
        amplification: dict[str, dict[str, float]] = {}
        candidate_rank_metrics: dict[str, Any] = {}
        for c_idx, spec in enumerate(args.corruption):
            ctype, mag_s = spec.split(":", 1)
            mag = float(mag_s)
            corrupt_pixels = _add_eval_corruption(
                batch["pixels"], mag, args.seed + 7919 * (c_idx + 1), corruption_type=ctype
            )
            corrupt_info = _add_predictor_levels(model, _encode(model, corrupt_pixels, batch["action"]), history_size)
            corrupt_levels = {
                level: _context_view(corrupt_info[level], history_size)
                for level in FEATURE_LEVELS
                if level in corrupt_info
            }
            label = _corruption_label(ctype, mag)
            amp = _amplification_metrics(clean_levels, corrupt_levels)
            amplification[label] = amp
            rank_bundle = _candidate_rank_bundle(
                model,
                batch,
                clean_info,
                corrupt_info,
                history_size=history_size,
                future_steps=args.future_steps,
                random_action_trials=args.candidate_random_trials,
                topk=args.candidate_topk,
                std_grid=candidate_std_grid,
                noise_mode=args.synthetic_noise_mode,
                seed=args.seed + 1543 * (c_idx + 1),
            )
            candidate_rank_metrics[label] = rank_bundle
            for level in FEATURE_LEVELS:
                if level not in clean_levels or level not in corrupt_levels:
                    continue
                t_metrics = _t_calibration_metrics(
                    model,
                    level,
                    clean_levels[level],
                    corrupt_levels[level],
                    act_emb=act_emb,
                    state=state_context,
                    t_grid=t_grid,
                    sigma_max=t_sigma_max,
                    noise_schedule=args.noise_schedule,
                    noise_mode=args.synthetic_noise_mode,
                    knn_k=args.knn_k,
                )
                candidate_pixel = rank_bundle.get("pixel") if rank_bundle.get("computed") else None
                metrics = _audit_space(
                    model=model,
                    level=level,
                    clean=clean_levels[level],
                    corrupt=corrupt_levels[level],
                    act_emb=act_emb,
                    state=state_context,
                    std_grid=std_grid,
                    noise_mode=args.synthetic_noise_mode,
                    knn_k=args.knn_k,
                    candidate_rank=candidate_pixel,
                    t_calibration=t_metrics,
                )
                if rank_bundle.get("computed"):
                    if level == "encoder_feat":
                        metrics["candidate_rank_synthetic_encoder_by_alpha"] = rank_bundle[
                            "synthetic_encoder_by_alpha"
                        ]
                    if level == "emb":
                        metrics["candidate_rank_synthetic_emb_by_alpha"] = rank_bundle[
                            "synthetic_emb_by_alpha"
                        ]
                    if level == "predictor_hidden":
                        metrics["candidate_rank_synthetic_predictor_hidden"] = rank_bundle[
                            "synthetic_predictor_hidden"
                        ]
                results[level][label] = metrics

    results = {level: by_corr for level, by_corr in results.items() if by_corr}
    decision_table = _build_decision_table(task, results, amplification)
    recommendation = _recommendation(decision_table)
    return {
        "schema_version": "acpc-flow-coverage-v2",
        "task": task,
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "num_samples": args.num_samples,
        "history_size": history_size,
        "future_steps": args.future_steps,
        "feature_spaces": [level for level in FEATURE_LEVELS if level in results],
        "synthetic_noise_grid": std_grid,
        "candidate_synthetic_noise_grid": candidate_std_grid,
        "t_grid": t_grid,
        "noise_schedule": args.noise_schedule,
        "candidate_rank_metrics_computed": any(
            bool(v.get("computed")) for v in candidate_rank_metrics.values()
        ),
        "candidate_rank_metrics": candidate_rank_metrics,
        "amplification": amplification,
        "results": results,
        "decision_table": decision_table,
        "recommendation": recommendation,
        "overall_decisions": [row["decision"] for row in decision_table],
    }


def _corruption_label(ctype: str, magnitude: float) -> str:
    if ctype == "gaussian_noise":
        return f"gaussian_std{magnitude:g}"
    if ctype == "gaussian_blur":
        return f"blur_ks{int(magnitude)}"
    if ctype == "resize":
        return f"resize_factor{magnitude:g}"
    return f"{ctype}_{magnitude:g}"


def _write_outputs(report: Mapping[str, Any], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    with json_path.open("w") as f:
        json.dump(to_serializable(report), f, indent=2, sort_keys=True)
    rows = []
    for space, by_corr in report["results"].items():
        for corr, metrics in by_corr.items():
            flat = {"feature_space": space, "corruption": corr}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    continue
                flat[k] = v
            rows.append(flat)
    fieldnames = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ACPC-Flow coverage audit v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(DATASET_NAMES), default="tworoom")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--stablewm-home", default=None)
    parser.add_argument("--state-key", default="auto")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--future-steps", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic-std-grid", default="0.01,0.02,0.04,0.08,0.12")
    parser.add_argument("--candidate-synthetic-std-grid", default="0.03,0.05,0.08")
    parser.add_argument("--synthetic-noise-mode", default="token_std")
    parser.add_argument("--t-grid", default="0.0,0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--t-sigma-max", type=float, default=None)
    parser.add_argument("--noise-schedule", choices=("linear", "variance_preserving"), default="linear")
    parser.add_argument("--candidate-random-trials", type=int, default=16)
    parser.add_argument("--candidate-topk", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--corruption",
        action="append",
        default=None,
        help="Corruption spec type:magnitude; repeatable.",
    )
    parser.add_argument("--output-dir", default="assets/paper1_data")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    if args.corruption is None:
        args.corruption = [
            "gaussian_noise:0.03",
            "gaussian_noise:0.05",
            "gaussian_noise:0.08",
            "gaussian_blur:7",
            "resize:0.5",
        ]

    report = run_audit(args)
    date = datetime.utcnow().strftime("%Y%m%d")
    prefix = args.prefix or f"acpc_flow_coverage_v2_{args.task}_{Path(args.checkpoint).parent.name}_{date}"
    json_path, csv_path = _write_outputs(report, Path(args.output_dir), prefix)
    print(f"[coverage_audit] wrote {json_path}")
    print(f"[coverage_audit] wrote {csv_path}")
    print("[decision_table]")
    for row in report["decision_table"]:
        metrics = report["results"][row["level"]][row["stressor"]]
        ratio_q90 = metrics["ratio_to_knn_q90"]
        wrong_nn = metrics["wrong_label_nn_rate"]
        print(
            f"  {row['stressor']} {row['level']}: "
            f"decision={row['decision']} "
            f"action={row['recommended_next_action']} "
            f"ratio_q90={ratio_q90:.3g} wrong_nn={wrong_nn:.3g}"
        )
    print("recommendation: " + report["recommendation"]["reason"])


if __name__ == "__main__":
    main()
