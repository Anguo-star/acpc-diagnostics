#!/usr/bin/env python3
"""Audit training rho as a privileged Gaussian-only metadata baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path}: JSON root must be an object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _confusion(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum((not label) and (not prediction) for label, prediction in zip(labels, predictions))
    fp = sum((not label) and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and (not prediction) for label, prediction in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(labels),
        "positive_n": sum(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        ),
        "false_pass_rate": fp / (fp + tn) if fp + tn else None,
    }


def _select_threshold(calibration_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rhos = sorted({float(row["training_rho"]) for row in calibration_rows})
    candidates = [rhos[0] - 1e-9]
    candidates.extend((left + right) / 2.0 for left, right in zip(rhos, rhos[1:]))
    candidates.append(rhos[-1] + 1e-9)
    labels = [_as_bool(row["behavior_label"]) for row in calibration_rows]
    scored = []
    for threshold in candidates:
        predictions = [float(row["training_rho"]) >= threshold for row in calibration_rows]
        metrics = _confusion(labels, predictions)
        score = metrics["balanced_accuracy"]
        _require(score is not None, "CAL rho audit lacks both behavior classes")
        scored.append(
            (
                score,
                -(metrics["false_pass_rate"] or 0.0),
                threshold,
                metrics,
            )
        )
    best = max(scored)
    return {
        "threshold": best[2],
        "direction": "pass_if_training_rho_ge_threshold",
        "selection_split": "CAL",
        "selection_rule": "maximize balanced accuracy; tie-break lower false-pass rate, then higher rho",
        "calibration_metrics": best[3],
        "candidate_threshold_count": len(candidates),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(
    calibration_path: Path,
    lewm_path: Path,
    pldm_path: Path,
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(_sha256(protocol_path) == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    calibration_payload = _load(calibration_path)
    calibration_rows = calibration_payload.get("calibration_rows", [])
    _require(len(calibration_rows) == 36, "CAL rho rows mismatch")
    selection = _select_threshold(calibration_rows)
    rows = []
    for path, expected_family in ((lewm_path, "LeWM"), (pldm_path, "PLDM")):
        for source in _read_csv(path):
            _require(source["model_family"] == expected_family, f"{path}: family mismatch")
            _require(source["protocol_sha256"] == FROZEN_PROTOCOL_SHA256, f"{path}: protocol mismatch")
            expected_source_split = "TEST" if expected_family == "LeWM" else "E2"
            _require(
                source["split_name"] == expected_source_split,
                f"{path}: split mismatch",
            )
            rho = float(source["training_rho"])
            label = _as_bool(source["behavior_label"])
            rows.append(
                {
                    "model_family": expected_family,
                    "training_family_id": source.get(
                        "training_family_id",
                        f"lewm_seed{source['training_seed']}",
                    ),
                    "training_seed": int(source["training_seed"]),
                    "task": source["task"],
                    "training_rho": rho,
                    "behavior_label": label,
                    "rho_privileged_prediction": rho >= selection["threshold"],
                    "split_name": "E1" if expected_family == "LeWM" else "E2",
                    "protocol_hash": FROZEN_PROTOCOL_SHA256,
                    "baseline_scope": "Gaussian-only privileged training metadata",
                    "external_leaderboard_eligible": False,
                }
            )
    _require(len(rows) == 108, "external rho row count mismatch")
    split_metrics = {}
    for split in ("E1", "E2", "E1+E2"):
        selected = rows if split == "E1+E2" else [row for row in rows if row["split_name"] == split]
        split_metrics[split] = _confusion(
            [_as_bool(row["behavior_label"]) for row in selected],
            [_as_bool(row["rho_privileged_prediction"]) for row in selected],
        )
    summary = {
        "metadata": {
            "schema_version": "paper1-gaussian-rho-confound-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "protocol_hash": FROZEN_PROTOCOL_SHA256,
            "threshold_search_split": "CAL only",
            "external_threshold_search_allowed": False,
            "external_leaderboard_eligible": False,
            "scope": "Gaussian training sweep only; undefined across repair families/stressors",
            "source_paths": {
                "calibration": str(calibration_path),
                "lewm_external": str(lewm_path),
                "pldm_external": str(pldm_path),
                "protocol": str(protocol_path),
            },
            "source_hashes": {
                "calibration": _sha256(calibration_path),
                "lewm_external": _sha256(lewm_path),
                "pldm_external": _sha256(pldm_path),
                "protocol": FROZEN_PROTOCOL_SHA256,
            },
            "missing_rows": [],
            "errors": [],
        },
        "threshold_selection": selection,
        "external_metrics": split_metrics,
        "count_contract": {"expected_rows": 108, "observed_rows": len(rows)},
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "paper1/results/frozen_diagnostic_protocol_calibration.json")
    parser.add_argument("--lewm-external", type=Path, default=ROOT / "paper1/results/frozen_external_validation_rows_v3.csv")
    parser.add_argument("--pldm-external", type=Path, default=ROOT / "paper1/results/external_validation/pldm_frozen_rows_v2.csv")
    parser.add_argument("--protocol", type=Path, default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json")
    parser.add_argument("--out-rows", type=Path, default=ROOT / "paper1/results/diagnostic_baselines/gaussian_rho_confound_rows.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows, summary = build(
        args.calibration,
        args.lewm_external,
        args.pldm_external,
        args.protocol,
    )
    _write_csv(args.out_rows, rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} privileged Gaussian rho rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
