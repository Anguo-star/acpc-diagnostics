"""Deprecated reader for the released IR/DR-v1 schema.

Historical result files are immutable because their hashes are part of the
experiment record. This module remains only for exact v1 API compatibility;
current analysis and rendering code use :mod:`ir_sr_compat` and the IR/SR-v2
schema.
"""

from __future__ import annotations

from typing import Any


_EXACT_KEY_MAP = {
    "atr": "ir_raw",
    "atr_q80": "ir_raw_q80",
    "atr_q90": "ir_raw_q90",
    "atr_q95": "ir_raw_q95",
    "atr_horizon_v2_q90": "ir_raw_q90",
    "atr_normalized_q90": "ir_relative_q90",
    "atr_normalized_q90_mean": "ir_relative_q90_mean",
    "atr_normalized_q90_pstdev": "ir_relative_q90_pstdev",
    "smpr": "dr",
    "smpr_delta0": "dr_delta0",
    "smpr_delta005": "dr_delta005",
    "smpr_delta010": "dr_delta010",
    "tau_atr": "ir_threshold",
    "tau_smpr": "dr_threshold",
    "base_atr": "base_ir_raw",
    "endpoint_atr": "endpoint_ir_raw",
    "delta_atr": "delta_ir_raw",
    "base_atr_rel": "base_ir_relative",
    "endpoint_atr_rel": "endpoint_ir_relative",
    "base_smpr": "base_dr",
    "endpoint_smpr": "endpoint_dr",
    "delta_smpr": "delta_dr",
    "atr_threshold_margin": "ir_threshold_margin",
    "smpr_threshold_margin": "dr_threshold_margin",
}


def _key_to_ir_dr(key: str) -> str:
    if key in _EXACT_KEY_MAP:
        return _EXACT_KEY_MAP[key]
    parts = key.split("_")
    renamed = [
        "ir" if part == "atr" else "dr" if part == "smpr" else part
        for part in parts
    ]
    return "_".join(renamed)


def to_ir_dr(value: Any) -> Any:
    """Recursively rename frozen diagnostic fields without changing values."""

    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            renamed = _key_to_ir_dr(str(key))
            if renamed in converted:
                raise ValueError(
                    f"legacy IR/DR-v1 field collision while converting {key!r}"
                )
            converted[renamed] = to_ir_dr(item)
        return converted
    if isinstance(value, list):
        return [to_ir_dr(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_ir_dr(item) for item in value)
    return value
