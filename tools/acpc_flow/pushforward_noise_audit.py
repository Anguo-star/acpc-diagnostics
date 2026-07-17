from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from tools.acpc_flow.coverage_audit import (
    DATASET_NAMES,
    FEATURE_LEVELS,
    STATE_KEYS,
    _add_predictor_levels,
    _amplification_metrics,
    _build_action_candidates,
    _candidate_costs_from_context,
    _candidate_rank_metrics,
    _clean_knn_distance,
    _context_view,
    _corruption_label,
    _encode,
    _flatten,
    _level_gap,
    _paired_rank_and_crossing,
    _project_features,
    _q,
    _quantiles,
)
from tools.repr_analysis.analyze_repr import (
    infer_history_size,
    load_dataset_samples,
    load_model,
    to_serializable,
)
from tools.repr_analysis.noise_sensitivity import _add_eval_corruption


EPS = 1e-8
DEFAULT_LOWRANK_RANKS = (1, 2, 4, 8, 16)
DEFAULT_REPLAY_FAMILIES = ("isotropic", "diagonal", "lowrank_r4", "lowrank_r8", "pixel_paired")
Z95 = 1.6448536269514722


def _parse_float_grid(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_int_grid(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _randn(shape: Sequence[int], *, like: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=like.device)
    generator.manual_seed(int(seed))
    return torch.randn(tuple(shape), device=like.device, dtype=like.dtype, generator=generator)


def _covariance_decomposition(delta: torch.Tensor) -> dict[str, torch.Tensor]:
    x = delta.detach().float()
    mean = x.mean(dim=0)
    centered = x - mean
    n = x.size(0)
    if n < 2:
        d = x.size(1)
        cov = torch.zeros(d, d, dtype=x.dtype, device=x.device)
    else:
        cov = centered.T @ centered / float(n - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    eigvecs = eigvecs[:, order]
    return {"mean": mean, "centered": centered, "cov": cov, "eigvals": eigvals, "eigvecs": eigvecs}


def covariance_geometry(delta: torch.Tensor, reference_delta: torch.Tensor | None = None, subspace_rank: int = 5) -> dict[str, float]:
    """Summarise empirical pushforward covariance geometry for flattened deltas."""
    decomp = _covariance_decomposition(delta)
    cov = decomp["cov"]
    eigvals = decomp["eigvals"]
    trace = eigvals.sum().clamp_min(EPS)
    p = eigvals / trace
    entropy_rank = torch.exp(-(p * torch.log(p.clamp_min(EPS))).sum())
    top = lambda k: float((eigvals[: min(k, eigvals.numel())].sum() / trace).cpu())
    fro = cov.pow(2).sum().clamp_min(EPS)
    diag_energy = cov.diag().pow(2).sum() / fro
    out = {
        "mean_shift_norm": float(torch.linalg.vector_norm(decomp["mean"]).cpu()),
        "cov_trace": float(trace.cpu()),
        "cov_effective_rank": float(entropy_rank.cpu()),
        "effective_rank": float(entropy_rank.cpu()),
        "lambda_max_over_trace": float((eigvals[0] / trace).cpu()) if eigvals.numel() else float("nan"),
        "top1_energy": top(1),
        "top5_energy": top(5),
        "top10_energy": top(10),
        "top1_eigen_ratio": top(1),
        "top5_eigen_ratio": top(5),
        "diagonal_energy_ratio": float(diag_energy.cpu()),
        "offdiag_energy_ratio": float((1.0 - diag_energy).cpu()),
        "family_subspace_overlap": float("nan"),
    }
    if reference_delta is not None and reference_delta.numel() > 0:
        ref = _covariance_decomposition(reference_delta)
        k = min(int(subspace_rank), decomp["eigvecs"].size(1), ref["eigvecs"].size(1))
        if k > 0:
            u = decomp["eigvecs"][:, :k]
            v = ref["eigvecs"][:, :k]
            out["family_subspace_overlap"] = float(((u.T @ v).pow(2).sum() / float(k)).cpu())
    return out


def _chi_square_radius_q95(dim: int) -> float:
    dim = max(int(dim), 1)
    # Wilson-Hilferty approximation for chi-square 0.95 without scipy.
    return math.sqrt(dim * (1.0 - 2.0 / (9.0 * dim) + Z95 * math.sqrt(2.0 / (9.0 * dim))) ** 3)


def sample_isotropic_trace(delta: torch.Tensor, *, seed: int) -> torch.Tensor:
    decomp = _covariance_decomposition(delta)
    dim = delta.size(1)
    var = float((decomp["eigvals"].sum() / max(dim, 1)).cpu())
    return _randn(delta.shape, like=delta, seed=seed) * math.sqrt(max(var, 0.0))


def sample_diagonal_empirical(delta: torch.Tensor, *, seed: int, include_mean: bool = True) -> torch.Tensor:
    decomp = _covariance_decomposition(delta)
    var = decomp["centered"].var(dim=0, unbiased=delta.size(0) > 1).clamp_min(0.0)
    sample = _randn(delta.shape, like=delta, seed=seed) * var.sqrt()
    if include_mean:
        sample = sample + decomp["mean"]
    return sample


def sample_lowrank_diag_empirical(delta: torch.Tensor, rank: int, *, seed: int, include_mean: bool = True) -> torch.Tensor:
    decomp = _covariance_decomposition(delta)
    eigvals = decomp["eigvals"]
    eigvecs = decomp["eigvecs"]
    dim = delta.size(1)
    r = min(max(int(rank), 0), dim, eigvals.numel())
    sample = torch.zeros_like(delta)
    if r > 0:
        z = _randn((delta.size(0), r), like=delta, seed=seed)
        sample = sample + (z * eigvals[:r].sqrt()) @ eigvecs[:, :r].T
    if r < dim:
        approx = (eigvecs[:, :r] * eigvals[:r].unsqueeze(0)) @ eigvecs[:, :r].T if r > 0 else torch.zeros_like(decomp["cov"])
        residual_var = (decomp["cov"].diag() - approx.diag()).clamp_min(0.0)
        sample = sample + _randn(delta.shape, like=delta, seed=seed + 17) * residual_var.sqrt()
    if include_mean:
        sample = sample + decomp["mean"]
    return sample


def sample_empirical_mixture(delta_pool: torch.Tensor, n: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=delta_pool.device)
    generator.manual_seed(int(seed))
    idx = torch.randint(delta_pool.size(0), (int(n),), device=delta_pool.device, generator=generator)
    return delta_pool[idx]


def _radius_coverage(delta_norm: torch.Tensor, synthetic_delta: torch.Tensor) -> dict[str, float]:
    radius = torch.linalg.vector_norm(synthetic_delta.float(), dim=-1)
    q90 = _q(radius, 0.90)
    q95 = _q(radius, 0.95)
    q99 = _q(radius, 0.99)
    return {
        "radius_q90": q90,
        "radius_q95": q95,
        "radius_q99": q99,
        "coverage_q90": float((delta_norm <= q90).float().mean().cpu()),
        "coverage_q95": float((delta_norm <= q95).float().mean().cpu()),
        "coverage_q99": float((delta_norm <= q99).float().mean().cpu()),
    }


def _mahalanobis_quantiles(delta: torch.Tensor, shrinkage: float) -> dict[str, float]:
    decomp = _covariance_decomposition(delta)
    centered = decomp["centered"]
    eigvals = decomp["eigvals"]
    eigvecs = decomp["eigvecs"]
    trace = eigvals.sum()
    ridge = float(shrinkage) * float((trace / max(delta.size(1), 1)).cpu())
    denom = eigvals + max(ridge, EPS)
    coords = centered @ eigvecs
    maha = (coords.pow(2) / denom.clamp_min(EPS)).sum(dim=-1).sqrt()
    return {"mahalanobis_q50": _q(maha, 0.50), "mahalanobis_q90": _q(maha, 0.90), "mahalanobis_q95": _q(maha, 0.95)}


def _safe_radius(clean: torch.Tensor, state: torch.Tensor | None, *, k: int) -> dict[str, float]:
    if state is None:
        return {"safe_radius_q95": float("nan"), "safe_radius_q50": float("nan")}
    state_dist = torch.cdist(state.float(), state.float())
    state_knn = _clean_knn_distance(state.float(), k=k)
    wrong = state_dist > state_knn[:, None]
    clean_dist = torch.cdist(clean.float(), clean.float()).masked_fill(~wrong, float("inf"))
    nearest_wrong = clean_dist.min(dim=1).values
    nearest_wrong = nearest_wrong[torch.isfinite(nearest_wrong)]
    if nearest_wrong.numel() == 0:
        return {"safe_radius_q95": float("nan"), "safe_radius_q50": float("nan")}
    # Radius below this value keeps 95% of tokens inside their state-proxy basin.
    return {"safe_radius_q95": _q(nearest_wrong, 0.05), "safe_radius_q50": _q(nearest_wrong, 0.50)}


def _direction_and_norm_match(pixel_delta: torch.Tensor, synthetic_delta: torch.Tensor) -> dict[str, float]:
    cos = F.cosine_similarity(pixel_delta.float(), synthetic_delta.float(), dim=-1, eps=EPS)
    pixel_norm = torch.linalg.vector_norm(pixel_delta.float(), dim=-1).clamp_min(EPS)
    synth_norm = torch.linalg.vector_norm(synthetic_delta.float(), dim=-1)
    ratio = synth_norm / pixel_norm
    return {
        "delta_direction_cosine_vs_pixel": float(cos.mean().cpu()),
        "delta_direction_cosine_q50_vs_pixel": _q(cos, 0.50),
        "norm_ratio_vs_pixel": _q(ratio, 0.50),
        "norm_ratio_mean_vs_pixel": float(ratio.mean().cpu()),
    }


def _rank_flip_bound(clean_costs: torch.Tensor, other_costs: torch.Tensor) -> dict[str, float]:
    clean_sorted = torch.sort(clean_costs.detach().float(), dim=1).values
    if clean_sorted.size(1) < 2:
        return {"rank_flip_bound_mean": float("nan"), "rank_flip_bound_q90": float("nan")}
    margin = (clean_sorted[:, 1] - clean_sorted[:, 0]).clamp_min(EPS)
    drift = (other_costs.detach().float() - clean_costs.detach().float()).abs().max(dim=1).values
    bound = (2.0 * drift / margin).clamp(max=1.0)
    return {"rank_flip_bound_mean": float(bound.mean().cpu()), "rank_flip_bound_q90": _q(bound, 0.90)}


def _candidate_replay_context(
    model,
    batch: Mapping[str, torch.Tensor],
    clean_info: Mapping[str, torch.Tensor],
    corrupt_info: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    future_steps: int,
    random_action_trials: int,
    seed: int,
) -> dict[str, Any]:
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
    pixel_costs = _candidate_costs_from_context(
        model,
        corrupt_context,
        clean_goal,
        candidates,
        history_size=history_size,
        future_steps=future_steps,
    )
    return {
        "computed": True,
        "candidates": candidates,
        "clean_goal": clean_goal,
        "clean_costs": clean_costs,
        "pixel_costs": pixel_costs,
    }


def _candidate_replay_metrics(
    model,
    level: str,
    source_seq: torch.Tensor,
    replay_context: Mapping[str, Any] | None,
    *,
    history_size: int,
    future_steps: int,
    topk: int,
) -> dict[str, float]:
    if replay_context is None or not replay_context.get("computed"):
        return {
            "candidate_replay_computed": 0.0,
            "rank_spearman_match_error": float("nan"),
            "topk_overlap_match_error": float("nan"),
            "rank_flip_bound_mean": float("nan"),
            "rank_flip_bound_q90": float("nan"),
        }
    if level == "encoder_feat":
        context = _project_features(model, source_seq)
    elif level == "emb":
        context = source_seq
    else:
        return {
            "candidate_replay_computed": 0.0,
            "rank_spearman_match_error": float("nan"),
            "topk_overlap_match_error": float("nan"),
            "rank_flip_bound_mean": float("nan"),
            "rank_flip_bound_q90": float("nan"),
        }
    costs = _candidate_costs_from_context(
        model,
        context,
        replay_context["clean_goal"],
        replay_context["candidates"],
        history_size=history_size,
        future_steps=future_steps,
    )
    pixel_rank = _candidate_rank_metrics(replay_context["clean_costs"], replay_context["pixel_costs"], topk)
    synth_rank = _candidate_rank_metrics(replay_context["clean_costs"], costs, topk)
    out = {
        "candidate_replay_computed": 1.0,
        "rank_spearman_match_error": abs(
            synth_rank["candidate_rank_spearman"] - pixel_rank["candidate_rank_spearman"]
        ),
        "topk_overlap_match_error": abs(
            synth_rank["candidate_topk_overlap_rate"] - pixel_rank["candidate_topk_overlap_rate"]
        ),
        "candidate_rank_spearman": synth_rank["candidate_rank_spearman"],
        "candidate_top1_flip_rate": synth_rank["candidate_top1_flip_rate"],
        "candidate_topk_overlap_rate": synth_rank["candidate_topk_overlap_rate"],
    }
    out.update(_rank_flip_bound(replay_context["clean_costs"], costs))
    return out


def _family_replay_metrics(
    *,
    model,
    level: str,
    clean_seq: torch.Tensor,
    pixel_delta: torch.Tensor,
    synthetic_delta: torch.Tensor,
    act_emb: torch.Tensor,
    state_flat: torch.Tensor | None,
    replay_context: Mapping[str, Any] | None,
    pixel_gap_q90: float,
    history_size: int,
    future_steps: int,
    topk: int,
    knn_k: int,
) -> dict[str, Any]:
    clean_flat = _flatten(clean_seq)
    source_flat = clean_flat + synthetic_delta
    source_seq = source_flat.reshape_as(clean_seq)
    synth_gap = _level_gap(model, level, clean_seq, source_seq, act_emb)
    crossing = _paired_rank_and_crossing(clean_flat, source_flat, state_flat, k=knn_k)
    out: dict[str, Any] = {
        **_direction_and_norm_match(pixel_delta, synthetic_delta),
        "ACPC_gap_q90": _q(synth_gap, 0.90),
        "ACPC_gap_q95": _q(synth_gap, 0.95),
        "ACPC_gap_match_error": abs(_q(synth_gap, 0.90) - pixel_gap_q90) / max(abs(pixel_gap_q90), EPS),
        "crossing_rate": crossing.get("closer_to_wrong_than_pair_rate", float("nan")),
        "wrong_nn_rate": crossing.get("wrong_label_nn_rate", float("nan")),
    }
    out.update(_candidate_replay_metrics(
        model,
        level,
        source_seq,
        replay_context,
        history_size=history_size,
        future_steps=future_steps,
        topk=topk,
    ))
    return out


def structured_family_samples(
    delta: torch.Tensor,
    *,
    ranks: Sequence[int],
    mixture_delta: torch.Tensor | None,
    seed: int,
) -> dict[str, torch.Tensor]:
    samples = {
        "isotropic": sample_isotropic_trace(delta, seed=seed + 101),
        "diagonal": sample_diagonal_empirical(delta, seed=seed + 211),
        "pixel_paired": delta.detach().clone(),
    }
    for rank in ranks:
        samples[f"lowrank_r{int(rank)}"] = sample_lowrank_diag_empirical(delta, int(rank), seed=seed + 307 + int(rank))
    if mixture_delta is not None and mixture_delta.numel() > 0:
        samples["mixture"] = sample_empirical_mixture(mixture_delta, delta.size(0), seed=seed + 409)
    return samples


def _corruption_family(label: str) -> str:
    if label.startswith("gaussian_"):
        return "gaussian"
    if label.startswith("blur_"):
        return "blur"
    if label.startswith("resize_"):
        return "resize"
    return label.split("_", 1)[0]


def audit_pushforward_space(
    *,
    model,
    level: str,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    act_emb: torch.Tensor,
    state: torch.Tensor | None,
    replay_context: Mapping[str, Any] | None,
    ranks: Sequence[int],
    replay_families: Sequence[str],
    mixture_delta: torch.Tensor | None,
    reference_delta: torch.Tensor | None,
    knn_k: int,
    seed: int,
    future_steps: int,
    history_size: int,
    topk: int,
    amplification: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    clean_flat = _flatten(clean)
    corrupt_flat = _flatten(corrupt)
    state_flat = _flatten(state) if state is not None else None
    delta = corrupt_flat - clean_flat
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1)
    knn = _clean_knn_distance(clean_flat, k=knn_k)
    ratio = delta_norm / knn
    pixel_gap = _level_gap(model, level, clean, corrupt, act_emb)

    samples = structured_family_samples(delta, ranks=ranks, mixture_delta=mixture_delta, seed=seed)
    family_metrics: dict[str, dict[str, Any]] = {}
    for name, synth_delta in samples.items():
        cov = _radius_coverage(delta_norm, synth_delta)
        family_metrics[name] = {
            **cov,
            **_paired_rank_and_crossing(clean_flat, clean_flat + synth_delta, state_flat, k=knn_k),
        }
        if name in replay_families:
            family_metrics[name].update(_family_replay_metrics(
                model=model,
                level=level,
                clean_seq=clean,
                pixel_delta=delta,
                synthetic_delta=synth_delta,
                act_emb=act_emb,
                state_flat=state_flat,
                replay_context=replay_context,
                pixel_gap_q90=_q(pixel_gap, 0.90),
                history_size=history_size,
                future_steps=future_steps,
                topk=topk,
                knn_k=knn_k,
            ))

    lowrank_coverages = {
        int(rank): family_metrics.get(f"lowrank_r{int(rank)}", {}).get("coverage_q95", float("nan"))
        for rank in ranks
    }
    required_rank = next((rank for rank, cov in lowrank_coverages.items() if not math.isnan(cov) and cov >= 0.95), -1)
    safe = _safe_radius(clean_flat, state_flat, k=knn_k)
    delta_q95 = _q(delta_norm, 0.95)
    required_isotropic_std = delta_q95 / _chi_square_radius_q95(delta.size(1))
    best_family, best_metrics = _best_replay_family(family_metrics)
    decision, action = _pushforward_decision(family_metrics, safe, best_family, best_metrics)

    metrics: dict[str, Any] = {
        **_quantiles("delta_norm", delta_norm),
        **_quantiles("ratio_to_clean_knn", ratio),
        **_quantiles("ratio_to_knn", ratio),
        **_paired_rank_and_crossing(clean_flat, corrupt_flat, state_flat, k=knn_k),
        **covariance_geometry(delta, reference_delta=reference_delta),
        **_mahalanobis_quantiles(delta, shrinkage=0.05),
        **safe,
        "required_isotropic_std": required_isotropic_std,
        "required_lowrank_rank_for_coverage": float(required_rank),
        "coverage_isotropic_q95": family_metrics["isotropic"]["coverage_q95"],
        "coverage_diag_q95": family_metrics["diagonal"]["coverage_q95"],
        "coverage_mixture_q95": family_metrics.get("mixture", {}).get("coverage_q95", float("nan")),
        "crossing_rate_isotropic": family_metrics["isotropic"].get("closer_to_wrong_than_pair_rate", float("nan")),
        "crossing_rate_diag": family_metrics["diagonal"].get("closer_to_wrong_than_pair_rate", float("nan")),
        "crossing_rate_lowrank": _best_lowrank_crossing(family_metrics, ranks),
        "coverage_safe_conflict": bool(
            not math.isnan(safe["safe_radius_q95"]) and delta_q95 > safe["safe_radius_q95"]
        ),
        "ACPC_gap_q90": _q(pixel_gap, 0.90),
        "ACPC_gap_q95": _q(pixel_gap, 0.95),
        "acpc_gap_q90": _q(pixel_gap, 0.90),
        "acpc_gap_q95": _q(pixel_gap, 0.95),
        "best_replay_family": best_family,
        "delta_direction_cosine_vs_pixel": best_metrics.get("delta_direction_cosine_vs_pixel", float("nan")),
        "norm_ratio_vs_pixel": best_metrics.get("norm_ratio_vs_pixel", float("nan")),
        "ACPC_gap_match_error": best_metrics.get("ACPC_gap_match_error", float("nan")),
        "rank_spearman_match_error": best_metrics.get("rank_spearman_match_error", float("nan")),
        "topk_overlap_match_error": best_metrics.get("topk_overlap_match_error", float("nan")),
        "rank_flip_bound_mean": best_metrics.get("rank_flip_bound_mean", float("nan")),
        "rank_flip_bound_q90": best_metrics.get("rank_flip_bound_q90", float("nan")),
        "decision": decision,
        "recommended_next_action": action,
        "synthetic_families": family_metrics,
    }
    for rank in ranks:
        key = f"lowrank_r{int(rank)}"
        metrics[f"coverage_lowrank_r{int(rank)}_q95"] = family_metrics[key]["coverage_q95"]
        metrics[f"crossing_lowrank_r{int(rank)}"] = family_metrics[key].get(
            "closer_to_wrong_than_pair_rate", float("nan")
        )
    if amplification:
        metrics.update(amplification)
    if replay_context is not None and replay_context.get("computed"):
        pixel_rank = _candidate_rank_metrics(replay_context["clean_costs"], replay_context["pixel_costs"], topk)
        metrics.update(pixel_rank)
        metrics.update(_rank_flip_bound(replay_context["clean_costs"], replay_context["pixel_costs"]))
    return metrics


def _best_lowrank_crossing(family_metrics: Mapping[str, Mapping[str, Any]], ranks: Sequence[int]) -> float:
    pairs = []
    for rank in ranks:
        metrics = family_metrics.get(f"lowrank_r{int(rank)}", {})
        cov = metrics.get("coverage_q95", float("nan"))
        crossing = metrics.get("closer_to_wrong_than_pair_rate", float("nan"))
        if not math.isnan(cov) and not math.isnan(crossing):
            pairs.append((cov, -crossing, crossing))
    if not pairs:
        return float("nan")
    return sorted(pairs, reverse=True)[0][2]


def _best_replay_family(family_metrics: Mapping[str, Mapping[str, Any]]) -> tuple[str, Mapping[str, Any]]:
    candidates = []
    for name, metrics in family_metrics.items():
        if name == "pixel_paired":
            continue
        acpc = float(metrics.get("ACPC_gap_match_error", float("nan")))
        rank = float(metrics.get("rank_spearman_match_error", 0.0))
        topk = float(metrics.get("topk_overlap_match_error", 0.0))
        coverage = float(metrics.get("coverage_q95", 0.0))
        crossing = float(metrics.get("closer_to_wrong_than_pair_rate", 0.0))
        if math.isnan(acpc):
            continue
        score = acpc + rank + topk + max(0.0, 0.80 - coverage) + max(0.0, crossing - 0.20)
        candidates.append((score, name, metrics))
    if not candidates:
        return "none", {}
    _, name, metrics = sorted(candidates, key=lambda item: item[0])[0]
    return name, metrics


def _pushforward_decision(
    family_metrics: Mapping[str, Mapping[str, Any]],
    safe: Mapping[str, float],
    best_family: str,
    best_metrics: Mapping[str, Any],
) -> tuple[str, str]:
    iso_cov = float(family_metrics["isotropic"].get("coverage_q95", 0.0))
    diag_cov = float(family_metrics["diagonal"].get("coverage_q95", 0.0))
    diag_cross = float(family_metrics["diagonal"].get("closer_to_wrong_than_pair_rate", float("nan")))
    lowrank_items = [
        (name, metrics)
        for name, metrics in family_metrics.items()
        if name.startswith("lowrank_r")
    ]
    lowrank_best = max(lowrank_items, key=lambda item: float(item[1].get("coverage_q95", 0.0)), default=(None, {}))
    lowrank_cov = float(lowrank_best[1].get("coverage_q95", 0.0)) if lowrank_best[0] else 0.0
    lowrank_cross = float(lowrank_best[1].get("closer_to_wrong_than_pair_rate", float("nan"))) if lowrank_best[0] else float("nan")
    mixture_cov = float(family_metrics.get("mixture", {}).get("coverage_q95", float("nan")))
    pixel_cross = float(family_metrics["pixel_paired"].get("closer_to_wrong_than_pair_rate", float("nan")))

    if not math.isnan(pixel_cross) and pixel_cross >= 0.40:
        return "needs_semantic_guard", "audit_semantic_guard_or_training_time_only"
    if lowrank_cov >= max(0.80, iso_cov + 0.10) and (math.isnan(lowrank_cross) or lowrank_cross < 0.20):
        return "lowrank_diag_candidate", f"offline_replay_then_train_mve_if_{best_family}_holds"
    if diag_cov >= max(0.80, iso_cov + 0.10) and (math.isnan(diag_cross) or diag_cross < 0.20):
        return "diagonal_candidate", "offline_replay_then_train_mve_if_diagonal_holds"
    if not math.isnan(mixture_cov) and mixture_cov >= max(0.90, iso_cov + 0.10):
        return "family_mixture_candidate", "audit_family_conditioning_before_training"
    if family_metrics["pixel_paired"].get("coverage_q95", 0.0) >= 0.95 and max(diag_cov, lowrank_cov) < 0.80:
        return "pixel_paired_upper_bound_only", "do_not_train_global_latent_noise"
    if max(diag_cov, lowrank_cov) <= iso_cov + 0.05:
        return "isotropic_no_go", "revise_structured_family_or_stop"
    if not math.isnan(safe.get("safe_radius_q95", float("nan"))) and safe["safe_radius_q95"] <= 0:
        return "no_go", "state_proxy_safe_radius_invalid"
    return "training_time_only", "offline_replay_required_before_training"


def _build_family_references(delta_bank: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[tuple[str, str], torch.Tensor]:
    refs: dict[tuple[str, str], torch.Tensor] = {}
    for level in FEATURE_LEVELS:
        by_family: dict[str, list[torch.Tensor]] = {}
        for label, by_level in delta_bank.items():
            if level not in by_level:
                continue
            by_family.setdefault(_corruption_family(label), []).append(by_level[level])
        for family, deltas in by_family.items():
            refs[(level, family)] = torch.cat(deltas, dim=0)
    return refs


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.stablewm_home:
        os.environ["STABLEWM_HOME"] = args.stablewm_home
    task = args.task
    dataset_name = args.dataset_name or DATASET_NAMES[task]
    state_key = args.state_key if args.state_key != "auto" else STATE_KEYS.get(task)
    ranks = _parse_int_grid(args.lowrank_ranks)
    replay_families = tuple(x.strip() for x in args.replay_families.split(",") if x.strip())

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

        corrupt_infos: dict[str, dict[str, torch.Tensor]] = {}
        corrupt_levels_bank: dict[str, dict[str, torch.Tensor]] = {}
        delta_bank: dict[str, dict[str, torch.Tensor]] = {}
        amplification: dict[str, dict[str, float]] = {}
        replay_contexts: dict[str, Mapping[str, Any]] = {}
        for c_idx, spec in enumerate(args.corruption):
            ctype, mag_s = spec.split(":", 1)
            mag = float(mag_s)
            label = _corruption_label(ctype, mag)
            corrupt_pixels = _add_eval_corruption(
                batch["pixels"], mag, args.seed + 7919 * (c_idx + 1), corruption_type=ctype
            )
            corrupt_info = _add_predictor_levels(model, _encode(model, corrupt_pixels, batch["action"]), history_size)
            corrupt_infos[label] = corrupt_info
            corrupt_levels = {
                level: _context_view(corrupt_info[level], history_size)
                for level in FEATURE_LEVELS
                if level in corrupt_info
            }
            corrupt_levels_bank[label] = corrupt_levels
            amplification[label] = _amplification_metrics(clean_levels, corrupt_levels)
            delta_bank[label] = {
                level: _flatten(corrupt_levels[level]) - _flatten(clean_levels[level])
                for level in corrupt_levels
                if level in clean_levels
            }
            try:
                replay_contexts[label] = _candidate_replay_context(
                    model,
                    batch,
                    clean_info,
                    corrupt_info,
                    history_size=history_size,
                    future_steps=args.future_steps,
                    random_action_trials=args.candidate_random_trials,
                    seed=args.seed + 1543 * (c_idx + 1),
                )
            except Exception as exc:  # pragma: no cover - saved in artifact.
                replay_contexts[label] = {"computed": False, "reason": str(exc)}

        family_refs = _build_family_references(delta_bank)
        results: dict[str, dict[str, Any]] = {level: {} for level in FEATURE_LEVELS}
        for label, corrupt_levels in corrupt_levels_bank.items():
            family = _corruption_family(label)
            for level in FEATURE_LEVELS:
                if level not in clean_levels or level not in corrupt_levels:
                    continue
                mixture_delta = family_refs.get((level, family))
                reference_delta = family_refs.get((level, family))
                metrics = audit_pushforward_space(
                    model=model,
                    level=level,
                    clean=clean_levels[level],
                    corrupt=corrupt_levels[level],
                    act_emb=act_emb,
                    state=state_context,
                    replay_context=replay_contexts.get(label),
                    ranks=ranks,
                    replay_families=replay_families,
                    mixture_delta=mixture_delta,
                    reference_delta=reference_delta,
                    knn_k=args.knn_k,
                    seed=args.seed + 421 * (len(results[level]) + 1),
                    future_steps=args.future_steps,
                    history_size=history_size,
                    topk=args.candidate_topk,
                    amplification=amplification.get(label),
                )
                results[level][label] = metrics

    results = {level: by_corr for level, by_corr in results.items() if by_corr}
    decision_table = _build_decision_table(task, args.checkpoint, results)
    return {
        "schema_version": "latent-pushforward-audit-v1",
        "task": task,
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "num_samples": args.num_samples,
        "history_size": history_size,
        "future_steps": args.future_steps,
        "feature_spaces": [level for level in FEATURE_LEVELS if level in results],
        "lowrank_ranks": ranks,
        "replay_families": list(replay_families),
        "corruptions": list(args.corruption),
        "definitions": {
            "coverage_*_q95": "Fraction of pixel-induced delta norms below the synthetic family's sampled q95 radius.",
            "safe_radius_q95": "Conservative state-proxy basin radius: q05 nearest wrong-state clean distance.",
            "diagonal_energy_ratio": "Frobenius covariance energy on the diagonal.",
            "family_subspace_overlap": "Top-k PCA subspace overlap with pooled same-corruption-family deltas.",
            "isotropic_family": "Zero-mean Gaussian with empirical trace/dimension variance.",
            "diagonal_lowrank_families": "Empirical mean plus diagonal or low-rank-plus-diagonal covariance.",
            "mixture_family": "Empirical same-corruption-family bootstrap upper-bound sampler.",
        },
        "results": results,
        "decision_table": decision_table,
        "overall_decisions": [row["decision"] for row in decision_table],
    }


def _build_decision_table(task: str, checkpoint: str, results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    table = []
    ckpt_name = Path(checkpoint).parent.name
    for level, by_corr in results.items():
        for stressor, metrics in by_corr.items():
            table.append(
                {
                    "task": task,
                    "checkpoint": ckpt_name,
                    "level": level,
                    "stressor": stressor,
                    "decision": metrics.get("decision"),
                    "recommended_next_action": metrics.get("recommended_next_action"),
                    "coverage_isotropic_q95": metrics.get("coverage_isotropic_q95"),
                    "coverage_diag_q95": metrics.get("coverage_diag_q95"),
                    "coverage_lowrank_r4_q95": metrics.get("coverage_lowrank_r4_q95"),
                    "coverage_lowrank_r8_q95": metrics.get("coverage_lowrank_r8_q95"),
                    "crossing_rate_diag": metrics.get("crossing_rate_diag"),
                    "crossing_rate_lowrank": metrics.get("crossing_rate_lowrank"),
                    "ACPC_gap_match_error": metrics.get("ACPC_gap_match_error"),
                    "rank_spearman_match_error": metrics.get("rank_spearman_match_error"),
                    "best_replay_family": metrics.get("best_replay_family"),
                }
            )
    return table


def _write_outputs(report: Mapping[str, Any], output_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}_summary.md"
    with json_path.open("w") as f:
        json.dump(to_serializable(report), f, indent=2, sort_keys=True)
    rows = []
    for level, by_corr in report["results"].items():
        for corr, metrics in by_corr.items():
            flat = {"feature_space": level, "corruption": corr}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    continue
                flat[key] = value
            rows.append(flat)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(_format_summary_md(report), encoding="utf-8")
    return json_path, csv_path, md_path


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "nan"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _format_summary_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Latent Pushforward Audit Summary",
        "",
        f"- schema: `{report['schema_version']}`",
        f"- task: `{report['task']}`",
        f"- checkpoint: `{Path(report['checkpoint']).parent.name}`",
        f"- num_samples: `{report['num_samples']}`",
        f"- lowrank_ranks: `{','.join(str(r) for r in report['lowrank_ranks'])}`",
        "",
        "| Level | Stressor | Decision | iso cov | diag cov | lr4 cov | lr8 cov | diag cross | lr cross | best replay | ACPC err | rank err |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in report["decision_table"]:
        lines.append(
            "| {level} | {stressor} | {decision} | {iso} | {diag} | {lr4} | {lr8} | {dc} | {lc} | {best} | {ae} | {re} |".format(
                level=row["level"],
                stressor=row["stressor"],
                decision=row["decision"],
                iso=_fmt(row["coverage_isotropic_q95"]),
                diag=_fmt(row["coverage_diag_q95"]),
                lr4=_fmt(row.get("coverage_lowrank_r4_q95")),
                lr8=_fmt(row.get("coverage_lowrank_r8_q95")),
                dc=_fmt(row["crossing_rate_diag"]),
                lc=_fmt(row["crossing_rate_lowrank"]),
                best=row.get("best_replay_family"),
                ae=_fmt(row["ACPC_gap_match_error"]),
                re=_fmt(row["rank_spearman_match_error"]),
            )
        )
    lines.extend(
        [
            "",
            "Reading: coverage is q95 radius coverage of measured pixel deltas; crossing is closer-to-wrong-than-pair under the state proxy.",
            "No training is performed by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pushforward noise geometry and replay audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=sorted(DATASET_NAMES), default="tworoom")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--stablewm-home", default=None)
    parser.add_argument("--state-key", default="auto")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--future-steps", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=3073)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate-random-trials", type=int, default=16)
    parser.add_argument("--candidate-topk", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--lowrank-ranks", default="1,2,4,8,16")
    parser.add_argument("--replay-families", default=",".join(DEFAULT_REPLAY_FAMILIES))
    parser.add_argument(
        "--corruption",
        action="append",
        default=None,
        help="Corruption spec type:magnitude; repeatable.",
    )
    parser.add_argument("--output-dir", default="assets/paper2_data")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    if args.corruption is None:
        args.corruption = [
            "gaussian_noise:0.03",
            "gaussian_noise:0.05",
            "gaussian_noise:0.08",
            "gaussian_blur:15",
            "resize:0.25",
        ]

    report = run_audit(args)
    date = datetime.utcnow().strftime("%Y%m%d")
    prefix = args.prefix or f"latent_pushforward_audit_{args.task}_{Path(args.checkpoint).parent.name}_{date}"
    json_path, csv_path, md_path = _write_outputs(report, Path(args.output_dir), prefix)
    print(f"[pushforward_noise_audit] wrote {json_path}")
    print(f"[pushforward_noise_audit] wrote {csv_path}")
    print(f"[pushforward_noise_audit] wrote {md_path}")
    print("[decision_table]")
    for row in report["decision_table"]:
        print(
            f"  {row['stressor']} {row['level']}: "
            f"decision={row['decision']} action={row['recommended_next_action']} "
            f"iso={_fmt(row['coverage_isotropic_q95'])} diag={_fmt(row['coverage_diag_q95'])} "
            f"lr4={_fmt(row.get('coverage_lowrank_r4_q95'))}"
        )


if __name__ == "__main__":
    main()
