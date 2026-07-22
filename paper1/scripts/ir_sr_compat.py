"""Read frozen diagnostic artifacts into the current IR/SR schema.

Historical ATR/SMPR artifacts and released IR/DR-v1 artifacts are immutable
because their hashes are part of the experiment record.  Current paper code
calls :func:`to_ir_sr` immediately after loading them; downstream analysis and
new derived artifacts use only the current IR/SR names.
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
    "smpr": "sr",
    "smpr_delta0": "sr_delta0",
    "smpr_delta005": "sr_delta005",
    "smpr_delta010": "sr_delta010",
    "dr": "sr",
    "dr_delta0": "sr_delta0",
    "dr_delta005": "sr_delta005",
    "dr_delta010": "sr_delta010",
    "tau_atr": "ir_threshold",
    "tau_smpr": "sr_threshold",
    "tau_dr": "sr_threshold",
    "tau_sr": "sr_threshold",
    "base_atr": "base_ir_raw",
    "endpoint_atr": "endpoint_ir_raw",
    "delta_atr": "delta_ir_raw",
    "base_atr_rel": "base_ir_relative",
    "endpoint_atr_rel": "endpoint_ir_relative",
    "base_smpr": "base_sr",
    "endpoint_smpr": "endpoint_sr",
    "delta_smpr": "delta_sr",
    "base_dr": "base_sr",
    "endpoint_dr": "endpoint_sr",
    "delta_dr": "delta_sr",
    "atr_threshold_margin": "ir_threshold_margin",
    "smpr_threshold_margin": "sr_threshold_margin",
    "dr_threshold_margin": "sr_threshold_margin",
}


def _key_to_ir_sr(key: str) -> str:
    if key in _EXACT_KEY_MAP:
        return _EXACT_KEY_MAP[key]
    parts = key.split("_")
    renamed = [
        "ir"
        if part == "atr"
        else "sr"
        if part in {"smpr", "dr"}
        else part
        for part in parts
    ]
    return "_".join(renamed)


def to_ir_sr(value: Any) -> Any:
    """Recursively canonicalize legacy diagnostic fields without changing values.

    Equal duplicate aliases are accepted so a transition payload may expose a
    legacy and canonical spelling together.  If aliases collapse onto the same
    IR/SR key with different values, conversion fails instead of silently
    choosing one representation.
    """

    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            renamed = _key_to_ir_sr(str(key))
            canonical_item = to_ir_sr(item)
            if renamed in converted:
                if converted[renamed] != canonical_item:
                    raise ValueError(
                        "IR/SR field collision with inconsistent values while "
                        f"converting {key!r}"
                    )
                continue
            converted[renamed] = canonical_item
        return converted
    if isinstance(value, list):
        return [to_ir_sr(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_ir_sr(item) for item in value)
    return value
