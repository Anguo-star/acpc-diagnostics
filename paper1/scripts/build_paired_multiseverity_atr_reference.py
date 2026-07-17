#!/usr/bin/env python3
"""Build a deterministic SMPR reference view from one validated ATR raw shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "paper1/config/paired_multiseverity_protocol_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def build_reference(
    *,
    raw_path: Path,
    protocol_path: Path,
    training_seed: int,
) -> dict[str, Any]:
    raw_path = _rooted(raw_path)
    protocol_path = _rooted(protocol_path)
    protocol_hash_path = protocol_path.with_suffix(".sha256")
    sidecar = protocol_hash_path.read_text(encoding="utf-8").split()
    _require(len(sidecar) == 2, "invalid protocol hash sidecar")
    protocol_sha = _sha256(protocol_path)
    _require(sidecar[0] == protocol_sha, "protocol hash sidecar mismatch")
    protocol = _load(protocol_path)

    base_protocol_path = _rooted(Path(protocol["diagnostic"]["base_protocol_path"]))
    _require(
        _sha256(base_protocol_path) == protocol["diagnostic"]["base_protocol_sha256"],
        "base frozen protocol hash mismatch",
    )
    base_protocol = _load(base_protocol_path)
    raw = _load(raw_path)
    metadata = raw.get("metadata", {})
    rows = raw.get("rows", [])
    _require(
        metadata.get("schema_version") == "paper1-acpc-phase0-0.2",
        "raw ATR schema mismatch",
    )
    _require(metadata.get("status_counts") == {"ok": 2}, "raw ATR rows are incomplete")
    _require(metadata.get("missing_rows") == [], "raw ATR has missing rows")
    _require(metadata.get("errors") == [], "raw ATR has errors")
    _require(isinstance(rows, list) and len(rows) == 2, "expected exactly two ATR rows")
    _require(
        metadata.get("script_sha256") == protocol["source_hashes"]["acpc_runner"],
        "raw ATR runner hash mismatch",
    )
    _require(
        metadata.get("metric_implementation_sha256")
        == protocol["source_hashes"]["canonical_metric"],
        "raw canonical metric hash mismatch",
    )
    manifest_path = str(metadata.get("source_paths", {}).get("LeWM", ""))
    _require(
        manifest_path.endswith(f"lewm_seed{training_seed}_evals.json"),
        "raw ATR training-seed manifest mismatch",
    )
    raw_protocol = metadata.get("protocol", {})
    expected_raw_protocol = {
        "radius_metric": base_protocol["radius_metric"],
        "rollout_horizon": protocol["diagnostic"]["rollout_horizon"],
        "horizon_weights": base_protocol["horizon_weights"],
        "atr_quantile": base_protocol["atr_quantile"],
        "normalization": base_protocol["normalization"],
        "num_noise_draws": protocol["diagnostic"]["noise_draws"],
        "anchor_seed": protocol["diagnostic"]["anchor_seed"],
    }
    for name, expected in expected_raw_protocol.items():
        _require(
            raw_protocol.get(name) == expected,
            f"raw ATR metadata protocol mismatch: {name}",
        )

    tasks = {str(row.get("task")) for row in rows}
    families = {str(row.get("corruption_type")) for row in rows}
    severities = {_finite(row.get("noise_std"), "severity") for row in rows}
    _require(len(tasks) == 1 and len(families) == 1 and len(severities) == 1, "raw ATR shard is mixed")
    task = next(iter(tasks))
    family = next(iter(families))
    severity = next(iter(severities))
    _require(family in protocol["stressors"], "unsupported stressor family")
    _require(
        any(
            math.isclose(severity, float(value), rel_tol=0.0, abs_tol=1e-12)
            for value in protocol["stressors"][family]["prospective_nonidentity"]
        ),
        "severity is outside the frozen prospective grid",
    )
    expected_draws = [
        protocol["diagnostic"]["anchor_seed"] + 1009 + 7919 * index
        for index in range(protocol["diagnostic"]["noise_draws"])
    ]
    by_std = {str(row.get("std_key")): row for row in rows}
    _require(set(by_std) == {"0.0", "0.08"}, "raw ATR checkpoint pair mismatch")

    output_rows: list[dict[str, Any]] = []
    for std_key in ("0.0", "0.08"):
        row = by_std[std_key]
        _require(row.get("status") == "ok", f"{std_key}: ATR row is not ok")
        _require(row.get("method") == "LeWM", f"{std_key}: model family mismatch")
        _require(row.get("n_sequences") == protocol["diagnostic"]["anchor_count"], f"{std_key}: anchor count mismatch")
        _require(row.get("num_noise_draws") == protocol["diagnostic"]["noise_draws"], f"{std_key}: draw count mismatch")
        _require(row.get("noise_draw_seeds") == expected_draws, f"{std_key}: draw seeds mismatch")
        _require(row.get("rollout_horizon_actual") == protocol["diagnostic"]["rollout_horizon"], f"{std_key}: horizon mismatch")
        _require(row.get("corrupt_goal") is False, f"{std_key}: goal corruption mismatch")
        _require(row.get("embedding_space") == protocol["diagnostic"]["embedding_space"], f"{std_key}: embedding space mismatch")
        _require(row.get("radius_metric") == base_protocol["radius_metric"], f"{std_key}: radius metric mismatch")
        horizon = protocol["diagnostic"]["rollout_horizon"]
        _require(row.get("horizon") == horizon, f"{std_key}: stored horizon mismatch")
        weights = row.get("horizon_weights")
        _require(
            isinstance(weights, list)
            and len(weights) == horizon
            and all(
                math.isclose(float(value), 1.0 / horizon, rel_tol=0.0, abs_tol=1e-12)
                for value in weights
            ),
            f"{std_key}: row uniform horizon weights mismatch",
        )
        _require(row.get("normalization") == "per_anchor_clean_transition_l2_q50", f"{std_key}: row normalization mismatch")
        _require(row.get("atr_quantile") == base_protocol["atr_quantile"], f"{std_key}: ATR quantile mismatch")
        _require(row.get("noise_draw_seed_rule") == "seed+1009+7919*draw_index", f"{std_key}: draw seed rule mismatch")
        output_rows.append(
            {
                "status": "ok",
                "model_family": "LeWM",
                "training_seed": training_seed,
                "task": task,
                "std_key": std_key,
                "subdir": row.get("subdir"),
                "model_file": row.get("model_file"),
                "corruption_type": family,
                "stressor_severity": severity,
                "atr_horizon_v2_q90": _finite(
                    row.get("atr_horizon_v2_q90"),
                    f"{std_key}: atr_horizon_v2_q90",
                ),
            }
        )

    return {
        "metadata": {
            "schema_version": "paper1-acpc-horizon-v2-1.0",
            "artifact_role": "paired_multiseverity_atr_reference",
            "status": "complete",
            "protocol_sha256": protocol_sha,
            "base_protocol_sha256": protocol["diagnostic"]["base_protocol_sha256"],
            "source_path": _relative(raw_path),
            "source_sha256": _sha256(raw_path),
            "model_family": "LeWM",
            "training_seed": training_seed,
            "task": task,
            "corruption_type": family,
            "stressor_severity": severity,
            "implementation_hashes": {
                "acpc_runner": protocol["source_hashes"]["acpc_runner"],
                "canonical_metric": protocol["source_hashes"]["canonical_metric"],
            },
        },
        "rows": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_reference(
        raw_path=args.raw,
        protocol_path=args.protocol,
        training_seed=args.training_seed,
    )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    out = _rooted(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite different reference: {out}")
        print(f"unchanged {out}")
        return 0
    out.write_bytes(encoded)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
