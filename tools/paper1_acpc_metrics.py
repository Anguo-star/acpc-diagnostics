"""Canonical paired-rollout metrics for Paper 1.

The theorem-aligned ATR uses one weighted, stacked horizon L2 radius per
anchor.  The legacy stepwise q90 is retained under the explicit
``stepwise_rollout_q90`` name only; it is not an ATR estimate.

Rollouts passed to this module are assumed to already be in the projection and
embedding space required by the frozen protocol.  Their final two dimensions
must be ``(H, D)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch


DEFAULT_ROLLOUT_HORIZON = 8
DEFAULT_ATR_QUANTILE = 0.90
DEFAULT_TRANSITION_QUANTILE = 0.50
DEFAULT_EPS = 1e-8
CANONICAL_RADIUS_METRIC = "horizon_weighted_stacked_l2_v2"
LEGACY_STEPWISE_FIELD = "stepwise_rollout_q90"
MEAN_AGGREGATION_SEMANTICS = "per_anchor_mean_then_checkpoint_quantile"
MEDIAN_AGGREGATION_SEMANTICS = "per_anchor_median_then_checkpoint_quantile"

NoiseDrawAggregation = str | Callable[[torch.Tensor, int], torch.Tensor]


def _validate_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return result


def _validate_eps(eps: float) -> float:
    if isinstance(eps, bool):
        raise TypeError("eps must be a positive finite real number, not bool")
    try:
        result = float(eps)
    except (TypeError, ValueError) as exc:
        raise TypeError("eps must be a positive finite real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")
    return result


def _validate_rollout(rollout: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(rollout):
        raise TypeError(f"{name} must be a torch.Tensor")
    if rollout.ndim < 2:
        raise ValueError(f"{name} must have shape (..., H, D), got {tuple(rollout.shape)}")
    if rollout.shape[-2] < 1 or rollout.shape[-1] < 1 or rollout.numel() == 0:
        raise ValueError(f"{name} must have non-empty (..., H, D) dimensions")
    if not rollout.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {rollout.dtype}")
    if not bool(torch.isfinite(rollout).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return rollout


def _promote_rollout_pair(
    clean_rollout: torch.Tensor,
    noisy_rollout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = _validate_rollout(clean_rollout, name="clean_rollout")
    noisy = _validate_rollout(noisy_rollout, name="noisy_rollout")
    if clean.device != noisy.device:
        raise ValueError(
            "clean_rollout and noisy_rollout must be on the same device, "
            f"got {clean.device} and {noisy.device}"
        )
    if clean.shape[-2:] != noisy.shape[-2:]:
        raise ValueError(
            "clean_rollout and noisy_rollout must share (H, D), got "
            f"{tuple(clean.shape[-2:])} and {tuple(noisy.shape[-2:])}"
        )
    dtype = torch.promote_types(clean.dtype, noisy.dtype)
    return clean.to(dtype=dtype), noisy.to(dtype=dtype)


def _normalise_dim(dim: int, ndim: int, *, name: str) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(f"{name} must be an integer")
    normalised = dim + ndim if dim < 0 else dim
    if normalised < 0 or normalised >= ndim:
        raise ValueError(f"{name}={dim} is invalid for {ndim} leading dimensions")
    return normalised


def _align_rollout_pair(
    clean_rollout: torch.Tensor,
    noisy_rollout: torch.Tensor,
    *,
    noise_draw_dim: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int | None, tuple[int, ...], torch.Tensor]:
    """Align a clean rollout with optional noisy draws.

    With draws, ``clean_rollout`` may omit the draw dimension or may contain
    exact repeats along it.  Requiring a shared clean rollout prevents a draw
    axis from silently becoming an extra anchor axis.
    """

    clean, noisy = _promote_rollout_pair(clean_rollout, noisy_rollout)
    noisy_prefix = tuple(noisy.shape[:-2])
    if noise_draw_dim is None:
        if clean.shape != noisy.shape:
            raise ValueError(
                "rollout shapes must match when noise_draw_dim is None, got "
                f"{tuple(clean.shape)} and {tuple(noisy.shape)}"
            )
        return clean, noisy, None, noisy_prefix, clean

    draw_dim = _normalise_dim(
        noise_draw_dim, len(noisy_prefix), name="noise_draw_dim"
    )
    anchor_shape = noisy_prefix[:draw_dim] + noisy_prefix[draw_dim + 1 :]
    if clean.shape == noisy.shape:
        clean_base = clean.select(draw_dim, 0)
        repeated = clean_base.unsqueeze(draw_dim).expand_as(clean)
        if not torch.equal(clean, repeated):
            raise ValueError(
                "clean_rollout must be identical across noise draws for each anchor"
            )
        clean_aligned = clean
    elif tuple(clean.shape[:-2]) == anchor_shape:
        clean_base = clean
        clean_aligned = clean.unsqueeze(draw_dim).expand_as(noisy)
    else:
        raise ValueError(
            "with noise_draw_dim, clean_rollout must omit that dimension or contain "
            "exact repeats; got clean/noisy shapes "
            f"{tuple(clean.shape)} and {tuple(noisy.shape)}"
        )
    return clean_aligned, noisy, draw_dim, anchor_shape, clean_base


def uniform_horizon_weights(
    horizon: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the canonical uniform ``alpha_k = 1 / H`` weights."""

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    dtype = dtype or torch.get_default_dtype()
    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point():
        raise TypeError(f"weights require a floating-point dtype, got {dtype}")
    return torch.full((horizon,), 1.0 / horizon, dtype=dtype, device=device)


def _validated_horizon_weights(
    weights: torch.Tensor | Sequence[float] | None,
    *,
    horizon: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if weights is None:
        return uniform_horizon_weights(
            horizon, dtype=reference.dtype, device=reference.device
        )
    try:
        result = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
    except (TypeError, ValueError) as exc:
        raise TypeError("horizon weights must be a one-dimensional numeric sequence") from exc
    if result.shape != (horizon,):
        raise ValueError(
            f"horizon weights must have shape ({horizon},), got {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("horizon weights contain NaN or infinity")
    if bool((result < 0).any()):
        raise ValueError("horizon weights must be non-negative")
    total = result.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("horizon weights must have a positive finite sum")
    tolerance = 1e-4 if reference.dtype in (torch.float16, torch.bfloat16) else 1e-6
    if not math.isclose(float(total), 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(
            f"horizon weights must sum to 1, got {float(total):.12g}"
        )
    return result / total


def weighted_stacked_rollout(
    rollout: torch.Tensor,
    *,
    weights: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Apply ``sqrt(alpha_k)`` to each step and stack to ``(..., H * D)``."""

    rollout = _validate_rollout(rollout, name="rollout")
    horizon = rollout.shape[-2]
    alpha = _validated_horizon_weights(
        weights, horizon=horizon, reference=rollout
    )
    weight_shape = (1,) * (rollout.ndim - 2) + (horizon, 1)
    stacked = (rollout * alpha.sqrt().reshape(weight_shape)).flatten(start_dim=-2)
    if not bool(torch.isfinite(stacked).all()):
        raise ValueError("weighted stacked rollout is non-finite")
    return stacked


def horizon_weighted_stacked_l2(
    clean_rollout: torch.Tensor,
    noisy_rollout: torch.Tensor,
    *,
    weights: torch.Tensor | Sequence[float] | None = None,
) -> torch.Tensor:
    """Return one unnormalised weighted stacked L2 value per leading unit."""

    clean, noisy = _promote_rollout_pair(clean_rollout, noisy_rollout)
    if clean.shape != noisy.shape:
        raise ValueError(
            "horizon_weighted_stacked_l2 requires matching shapes, got "
            f"{tuple(clean.shape)} and {tuple(noisy.shape)}"
        )
    clean_stacked = weighted_stacked_rollout(clean, weights=weights)
    noisy_stacked = weighted_stacked_rollout(noisy, weights=weights)
    radius = torch.linalg.vector_norm(clean_stacked - noisy_stacked, dim=-1)
    if not bool(torch.isfinite(radius).all()):
        raise ValueError("horizon weighted stacked L2 is non-finite")
    return radius


def _quantile(
    values: torch.Tensor,
    q: float,
    *,
    dim: int | None = None,
) -> torch.Tensor:
    work = values
    if work.dtype in (torch.float16, torch.bfloat16):
        work = work.float()
    return torch.quantile(work, q, dim=dim)


def per_anchor_clean_transition_scale(
    clean_rollout: torch.Tensor,
    *,
    initial_clean_state: torch.Tensor | None = None,
    transition_quantile: float = DEFAULT_TRANSITION_QUANTILE,
) -> torch.Tensor:
    """Compute each anchor's q50 clean-transition L2 scale.

    By default only transitions inside the supplied rollout are used, so an
    ``H``-step rollout contributes ``H - 1`` clean transitions.  Passing
    ``initial_clean_state`` explicitly adds the transition into the first
    rollout step.  This choice is never inferred implicitly.
    """

    clean = _validate_rollout(clean_rollout, name="clean_rollout")
    q = _validate_probability(transition_quantile, name="transition_quantile")
    if initial_clean_state is None:
        if clean.shape[-2] < 2:
            raise ValueError(
                "at least two clean rollout steps are required when "
                "initial_clean_state is not supplied"
            )
        transitions = clean[..., 1:, :] - clean[..., :-1, :]
    else:
        if not torch.is_tensor(initial_clean_state):
            raise TypeError("initial_clean_state must be a torch.Tensor")
        expected_shape = clean.shape[:-2] + (clean.shape[-1],)
        if initial_clean_state.shape != expected_shape:
            raise ValueError(
                "initial_clean_state must have shape "
                f"{tuple(expected_shape)}, got {tuple(initial_clean_state.shape)}"
            )
        if initial_clean_state.device != clean.device:
            raise ValueError("initial_clean_state and clean_rollout must share a device")
        if not initial_clean_state.is_floating_point():
            raise TypeError("initial_clean_state must have a floating-point dtype")
        if not bool(torch.isfinite(initial_clean_state).all()):
            raise ValueError("initial_clean_state contains NaN or infinity")
        dtype = torch.promote_types(clean.dtype, initial_clean_state.dtype)
        clean = clean.to(dtype=dtype)
        initial = initial_clean_state.to(dtype=dtype)
        states = torch.cat((initial.unsqueeze(-2), clean), dim=-2)
        transitions = states[..., 1:, :] - states[..., :-1, :]

    transition_l2 = torch.linalg.vector_norm(transitions, dim=-1)
    scale = _quantile(transition_l2, q, dim=-1)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("per-anchor clean transition scale is non-finite")
    return scale


def _coerce_per_anchor_scale(
    scale: Any,
    *,
    anchor_shape: tuple[int, ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    try:
        result = torch.as_tensor(scale, dtype=reference.dtype, device=reference.device)
    except (TypeError, ValueError) as exc:
        raise TypeError("clean_transition_scale must be numeric") from exc
    if tuple(result.shape) != anchor_shape:
        raise ValueError(
            "clean_transition_scale must contain exactly one value per anchor: "
            f"expected shape {anchor_shape}, got {tuple(result.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("clean_transition_scale contains NaN or infinity")
    if bool((result < 0).any()):
        raise ValueError("clean_transition_scale must be non-negative")
    return result


def _aggregation_semantics(aggregation: NoiseDrawAggregation) -> str:
    if aggregation == "mean":
        return MEAN_AGGREGATION_SEMANTICS
    if aggregation == "median":
        return MEDIAN_AGGREGATION_SEMANTICS
    if callable(aggregation):
        name = getattr(aggregation, "__name__", aggregation.__class__.__name__)
        return f"per_anchor_callable_{name}_then_checkpoint_quantile"
    raise ValueError("noise_draw_aggregation must be 'mean', 'median', or a callable")


def aggregate_noise_draw_radii(
    per_draw_radii: torch.Tensor,
    *,
    noise_draw_dim: int,
    aggregation: NoiseDrawAggregation = "mean",
) -> torch.Tensor:
    """Aggregate draws conditionally within each anchor."""

    if not torch.is_tensor(per_draw_radii):
        raise TypeError("per_draw_radii must be a torch.Tensor")
    if not per_draw_radii.is_floating_point():
        raise TypeError("per_draw_radii must have a floating-point dtype")
    if per_draw_radii.numel() == 0 or not bool(torch.isfinite(per_draw_radii).all()):
        raise ValueError("per_draw_radii must be non-empty and finite")
    dim = _normalise_dim(noise_draw_dim, per_draw_radii.ndim, name="noise_draw_dim")
    _aggregation_semantics(aggregation)
    if aggregation == "mean":
        result = per_draw_radii.mean(dim=dim)
    elif aggregation == "median":
        result = _quantile(per_draw_radii, 0.50, dim=dim)
    else:
        result = aggregation(per_draw_radii, dim)
        if not torch.is_tensor(result):
            raise TypeError("noise draw aggregation callable must return a torch.Tensor")
    expected_shape = per_draw_radii.shape[:dim] + per_draw_radii.shape[dim + 1 :]
    if result.shape != expected_shape:
        raise ValueError(
            "noise draw aggregation returned the wrong shape: expected "
            f"{tuple(expected_shape)}, got {tuple(result.shape)}"
        )
    if not result.is_floating_point() or not bool(torch.isfinite(result).all()):
        raise ValueError("noise draw aggregation must return finite floating-point values")
    return result


def checkpoint_radius_quantile(
    per_anchor_radii: torch.Tensor,
    *,
    quantile: float = DEFAULT_ATR_QUANTILE,
) -> torch.Tensor:
    """Take a checkpoint-level quantile over already aggregated anchors."""

    if not torch.is_tensor(per_anchor_radii):
        raise TypeError("per_anchor_radii must be a torch.Tensor")
    if not per_anchor_radii.is_floating_point():
        raise TypeError("per_anchor_radii must have a floating-point dtype")
    if per_anchor_radii.numel() == 0:
        raise ValueError("per_anchor_radii must contain at least one anchor")
    if not bool(torch.isfinite(per_anchor_radii).all()):
        raise ValueError("per_anchor_radii contains NaN or infinity")
    q = _validate_probability(quantile, name="quantile")
    return _quantile(per_anchor_radii.reshape(-1), q)


def stepwise_rollout_q90(
    clean_rollout: torch.Tensor,
    noisy_rollout: torch.Tensor,
    *,
    noise_draw_dim: int | None = None,
) -> torch.Tensor:
    """Return the legacy q90 over flattened stepwise L2 distances.

    This compatibility statistic pools ``(... x H)`` distances.  It is
    intentionally neither per-anchor nor transition-normalised and must not be
    reported as ATR.
    """

    clean, noisy, _, _, _ = _align_rollout_pair(
        clean_rollout, noisy_rollout, noise_draw_dim=noise_draw_dim
    )
    stepwise = torch.linalg.vector_norm(clean - noisy, dim=-1)
    return checkpoint_radius_quantile(stepwise, quantile=0.90)


def compute_acpc_horizon_metrics(
    clean_rollout: torch.Tensor,
    noisy_rollout: torch.Tensor,
    *,
    clean_transition_scale: torch.Tensor | Sequence[float] | float | None = None,
    initial_clean_state: torch.Tensor | None = None,
    weights: torch.Tensor | Sequence[float] | None = None,
    noise_draw_dim: int | None = None,
    noise_draw_aggregation: NoiseDrawAggregation = "mean",
    atr_quantile: float = DEFAULT_ATR_QUANTILE,
    transition_quantile: float = DEFAULT_TRANSITION_QUANTILE,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Compute canonical horizon ATR and the named legacy compatibility field.

    A noisy draw axis must be named with ``noise_draw_dim``.  Its values are
    aggregated within anchor before the checkpoint quantile; they are never
    pooled directly into that quantile.
    """

    eps_value = _validate_eps(eps)
    q = _validate_probability(atr_quantile, name="atr_quantile")
    transition_q = _validate_probability(
        transition_quantile, name="transition_quantile"
    )
    aggregation_semantics = _aggregation_semantics(noise_draw_aggregation)
    clean, noisy, draw_dim, anchor_shape, clean_base = _align_rollout_pair(
        clean_rollout, noisy_rollout, noise_draw_dim=noise_draw_dim
    )
    alpha = _validated_horizon_weights(
        weights, horizon=clean.shape[-2], reference=clean
    )
    unnormalised = horizon_weighted_stacked_l2(clean, noisy, weights=alpha)

    if clean_transition_scale is None:
        scale = per_anchor_clean_transition_scale(
            clean_base,
            initial_clean_state=initial_clean_state,
            transition_quantile=transition_q,
        )
    else:
        if initial_clean_state is not None:
            raise ValueError(
                "initial_clean_state cannot be combined with an explicit "
                "clean_transition_scale"
            )
        scale = _coerce_per_anchor_scale(
            clean_transition_scale, anchor_shape=anchor_shape, reference=clean
        )

    denominator = scale + eps_value
    if draw_dim is not None:
        denominator = denominator.unsqueeze(draw_dim)
    per_draw_radius = unnormalised / denominator
    if not bool(torch.isfinite(per_draw_radius).all()):
        raise ValueError(
            "normalised horizon radius is non-finite; increase eps or use a wider dtype"
        )

    if draw_dim is None:
        per_anchor_radius = per_draw_radius
    else:
        per_anchor_radius = aggregate_noise_draw_radii(
            per_draw_radius,
            noise_draw_dim=draw_dim,
            aggregation=noise_draw_aggregation,
        )
    atr = checkpoint_radius_quantile(per_anchor_radius, quantile=q)
    legacy_q90 = stepwise_rollout_q90(
        clean, noisy, noise_draw_dim=draw_dim
    )

    return {
        "radius_metric": CANONICAL_RADIUS_METRIC,
        "normalization": "per_anchor_clean_transition_l2_q50",
        "initial_clean_state_included": initial_clean_state is not None,
        "horizon": clean.shape[-2],
        "horizon_weights": alpha,
        "atr_quantile": q,
        "noise_draw_aggregation": aggregation_semantics,
        "clean_transition_scale": scale,
        "horizon_radius_per_noise_draw": per_draw_radius,
        "horizon_radius_per_anchor": per_anchor_radius,
        "atr": atr,
        LEGACY_STEPWISE_FIELD: legacy_q90,
        "stepwise_rollout_q90_is_atr": False,
    }


__all__ = [
    "CANONICAL_RADIUS_METRIC",
    "DEFAULT_ATR_QUANTILE",
    "DEFAULT_EPS",
    "DEFAULT_ROLLOUT_HORIZON",
    "DEFAULT_TRANSITION_QUANTILE",
    "LEGACY_STEPWISE_FIELD",
    "MEAN_AGGREGATION_SEMANTICS",
    "MEDIAN_AGGREGATION_SEMANTICS",
    "aggregate_noise_draw_radii",
    "checkpoint_radius_quantile",
    "compute_acpc_horizon_metrics",
    "horizon_weighted_stacked_l2",
    "per_anchor_clean_transition_scale",
    "stepwise_rollout_q90",
    "uniform_horizon_weights",
    "weighted_stacked_rollout",
]
