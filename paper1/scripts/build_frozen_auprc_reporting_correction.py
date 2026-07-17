#!/usr/bin/env python3
"""Correct tied-score CAL Average Precision without changing protocol v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.frozen_external_validation import _auprc, _joint_values


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_strict(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _legacy_row_order_ap(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    ranked = sorted(
        zip(y_true, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    positives = sum(bool(truth) for truth, _score in ranked)
    values: list[float] = []
    true_positive = 0
    for rank, (truth, _score) in enumerate(ranked, start=1):
        if truth:
            true_positive += 1
            values.append(true_positive / rank)
    return sum(values) / positives


def build_correction(
    *,
    protocol_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    protocol_original = protocol_path.read_bytes()
    protocol_mtime = protocol_path.stat().st_mtime_ns
    protocol = _load_strict(protocol_path)
    audit = _load_strict(audit_path)
    protocol_sha = _sha256(protocol_path)
    audit_sha = _sha256(audit_path)
    _require(protocol.get("status") == "frozen", "protocol v1 is not frozen")
    _require(protocol.get("immutable") is True, "protocol v1 is not immutable")
    _require(protocol.get("calibration_audit_sha256") == audit_sha, "CAL audit hash mismatch")
    rows = audit.get("calibration_rows", [])
    _require(isinstance(rows, list) and len(rows) == 36, "expected 36 CAL rows")
    tau_atr = float(protocol["tau_atr"])
    tau_smpr = float(protocol["tau_smpr"])
    y_true = [bool(row["behavior_label"]) for row in rows]
    scores = [
        _joint_values(
            atr=float(row["atr_horizon_v2_q90"]),
            smpr=float(row["smpr"]),
            tau_atr=tau_atr,
            tau_smpr=tau_smpr,
        )[2]
        for row in rows
    ]
    legacy_value = _legacy_row_order_ap(y_true, scores)
    corrected_value = _auprc(y_true, scores)
    _require(
        abs(legacy_value - float(protocol["calibration_metrics"]["auprc"])) <= 1e-12,
        "protocol CAL AUPRC does not match the identified legacy implementation",
    )
    _require(
        corrected_value == _auprc(list(reversed(y_true)), list(reversed(scores))),
        "corrected Average Precision is not permutation invariant",
    )
    script_path = Path(__file__).resolve()
    scorer_path = ROOT / "paper1" / "scripts" / "frozen_external_validation.py"
    source_paths = {
        "protocol_v1": str(protocol_path),
        "calibration_audit": str(audit_path),
        "tie_aware_scorer": str(scorer_path),
    }
    source_hashes = {name: _sha256(Path(path)) for name, path in source_paths.items()}
    artifact = {
        "metadata": {
            "schema_version": "paper1-frozen-auprc-reporting-correction-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "protocol_v1_sha256": protocol_sha,
            "protocol_v1_status": "immutable_and_unmodified",
            "correction_scope": "reporting_only",
            "thresholds_changed": False,
            "gate_decisions_changed": False,
            "external_rerun_required": "score/apply artifacts only; no model inference rerun",
            "status": "complete",
            "missing_rows": [],
            "errors": [],
        },
        "calibration": {
            "row_count": len(rows),
            "positive_count": sum(y_true),
            "legacy_reported_auprc": legacy_value,
            "legacy_definition": (
                "row-order-dependent ungrouped Average Precision within exact score ties"
            ),
            "corrected_auprc": corrected_value,
            "corrected_definition": (
                "tie-aware stepwise Average Precision; exact joint_score ties "
                "enter each retrieval set together"
            ),
            "absolute_difference": corrected_value - legacy_value,
        },
        "protocol_note": (
            "Protocol v1 thresholds, metric definitions, and all gate decisions remain "
            "authoritative. Only the descriptive CAL AUPRC field is superseded for reporting."
        ),
    }
    _require(set(source_paths) == set(source_hashes), "source provenance keys differ")
    _require(protocol_path.read_bytes() == protocol_original, "correction changed protocol bytes")
    _require(protocol_path.stat().st_mtime_ns == protocol_mtime, "correction changed protocol mtime")
    return artifact


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = build_correction(protocol_path=args.protocol, audit_path=args.audit)
    _write_exclusive(args.out, artifact)
    correction = artifact["calibration"]
    print(
        f"wrote {args.out} (CAL AP {correction['legacy_reported_auprc']:.6f} "
        f"-> {correction['corrected_auprc']:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
