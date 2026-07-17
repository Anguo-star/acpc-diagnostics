#!/usr/bin/env python3
"""Merge behavior-blind baseline shards and calibrate simple thresholds on CAL only."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
BASELINES = (
    "encoder_q90",
    "h1_q90",
    "action_shuffled_h8_q90",
    "action_zeroed_h8_q90",
    "time_shuffled_h8_q90",
    "atr_h8_q90",
)
CALIBRATED_BASELINES = (*BASELINES, "clean_score", "smpr")
BASELINE_DIRECTIONS = {
    **{metric: "pass_if_value_le_threshold" for metric in BASELINES},
    "clean_score": "pass_if_value_ge_threshold",
    "smpr": "pass_if_value_ge_threshold",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    _require(isinstance(payload, dict), f"{path}: top-level JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected finite numeric") from exc
    _require(math.isfinite(result), f"{name}: value is not finite")
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _balanced(labels: Sequence[bool], predictions: Sequence[bool]) -> tuple[float | None, dict[str, int]]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum((not label) and (not prediction) for label, prediction in zip(labels, predictions))
    fp = sum((not label) and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and (not prediction) for label, prediction in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    score = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return score, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    groups: dict[float, list[bool]] = defaultdict(list)
    for label, score in zip(labels, scores):
        groups[score].append(label)
    tp = fp = 0
    recall = area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        tp += sum(group)
        fp += len(group) - sum(group)
        next_recall = tp / positives
        area += (next_recall - recall) * tp / (tp + fp)
        recall = next_recall
    return area


def _select_threshold(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    direction: str = "pass_if_value_le_threshold",
) -> dict[str, Any]:
    _require(
        direction in {"pass_if_value_le_threshold", "pass_if_value_ge_threshold"},
        f"{metric}: unsupported threshold direction",
    )
    values = sorted({_finite(row[metric], name=metric) for row in rows})
    _require(values, f"{metric}: no calibration values")
    scale = max(1.0, max(abs(value) for value in values))
    candidates = [values[0] - scale * 1e-9]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(values, values[1:])
    )
    candidates.append(values[-1] + scale * 1e-9)
    labels = [_as_bool(row["behavior_label"]) for row in rows]
    scored = []
    for threshold in candidates:
        predictions = [
            (
                float(row[metric]) <= threshold
                if direction == "pass_if_value_le_threshold"
                else float(row[metric]) >= threshold
            )
            for row in rows
        ]
        balanced, confusion = _balanced(labels, predictions)
        _require(balanced is not None, f"{metric}: CAL lacks both classes")
        false_pass_rate = (
            confusion["fp"] / (confusion["fp"] + confusion["tn"])
            if confusion["fp"] + confusion["tn"]
            else 0.0
        )
        strict_threshold_tiebreak = (
            -threshold
            if direction == "pass_if_value_le_threshold"
            else threshold
        )
        scored.append(
            (
                balanced,
                -false_pass_rate,
                strict_threshold_tiebreak,
                threshold,
                confusion,
            )
        )
    selected = max(scored)
    return {
        "metric": metric,
        "threshold": selected[3],
        "direction": direction,
        "selection_split": "CAL",
        "selection_rule": (
            "maximize balanced accuracy; tie-break lower false-pass rate, "
            "then lower threshold"
        ),
        "calibration_balanced_accuracy": selected[0],
        "calibration_confusion": selected[4],
        "candidate_threshold_count": len(candidates),
    }


def _calibration_behavior(path: Path) -> dict[tuple[str, int, str, float], dict[str, Any]]:
    payload = _load(path)
    rows = payload.get("calibration_rows", [])
    result = {}
    for row in rows:
        key = ("LeWM", 3072, str(row["task"]), float(row["training_rho"]))
        result[key] = dict(row)
    _require(len(result) == 36, "CAL behavior coverage mismatch")
    return result


def _external_behavior(
    lewm_path: Path,
    pldm_path: Path,
) -> dict[tuple[str, int, str, float], dict[str, Any]]:
    result: dict[tuple[str, int, str, float], dict[str, Any]] = {}
    for path, family in ((lewm_path, "LeWM"), (pldm_path, "PLDM")):
        for row in _read_csv(path):
            _require(row.get("model_family") == family, f"{path}: family mismatch")
            _require(row.get("protocol_sha256") == FROZEN_PROTOCOL_SHA256, f"{path}: protocol mismatch")
            key = (
                family,
                int(row["training_seed"]),
                str(row["task"]),
                float(row["training_rho"]),
            )
            _require(key not in result, f"{path}: duplicate behavior row")
            result[key] = dict(row)
    _require(len(result) == 108, "E1/E2 behavior coverage mismatch")
    return result


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    threshold: float,
    *,
    direction: str = "pass_if_value_le_threshold",
) -> dict[str, Any]:
    labels = [_as_bool(row["behavior_label"]) for row in rows]
    values = [_finite(row[metric], name=metric) for row in rows]
    predictions = [
        value <= threshold
        if direction == "pass_if_value_le_threshold"
        else value >= threshold
        for value in values
    ]
    balanced, confusion = _balanced(labels, predictions)
    tp, tn, fp, fn = (
        confusion["tp"],
        confusion["tn"],
        confusion["fp"],
        confusion["fn"],
    )
    return {
        "metric": metric,
        "n": len(rows),
        "positive_n": sum(labels),
        "threshold": threshold,
        "balanced_accuracy": balanced,
        "auprc": _average_precision(
            labels,
            [
                -value
                if direction == "pass_if_value_le_threshold"
                else value
                for value in values
            ],
        ),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "false_pass_rate": fp / (fp + tn) if fp + tn else None,
        **confusion,
    }


def _fixed_prediction_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    prediction_field: str,
    score_field: str,
    threshold_label: str,
) -> dict[str, Any]:
    labels = [_as_bool(row["behavior_label"]) for row in rows]
    predictions = [_as_bool(row[prediction_field]) for row in rows]
    scores = [_finite(row[score_field], name=score_field) for row in rows]
    balanced, confusion = _balanced(labels, predictions)
    tp, tn, fp, fn = (
        confusion["tp"],
        confusion["tn"],
        confusion["fp"],
        confusion["fn"],
    )
    return {
        "metric": metric,
        "n": len(rows),
        "positive_n": sum(labels),
        "threshold": threshold_label,
        "balanced_accuracy": balanced,
        "auprc": _average_precision(labels, scores),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "false_pass_rate": fp / (fp + tn) if fp + tn else None,
        **confusion,
    }


def build(
    *,
    input_dir: Path,
    protocol_path: Path,
    calibration_path: Path,
    lewm_behavior_path: Path,
    pldm_behavior_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _require(_sha256(protocol_path) == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    paths = sorted(input_dir.glob("baseline_*.json"))
    _require(len(paths) == 56, f"expected 56 baseline shards, got {len(paths)}")
    all_rows: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    source_metadata: dict[str, Any] = {}
    natural_keys: set[tuple[Any, ...]] = set()
    for path in paths:
        payload = _load(path)
        metadata = payload.get("metadata", {})
        _require(metadata.get("schema_version") == "paper1-diagnostic-baseline-raw-1.0", f"{path}: schema mismatch")
        _require(metadata.get("status") == "complete", f"{path}: incomplete")
        _require(metadata.get("behavior_blind") is True, f"{path}: behavior leaked")
        _require(metadata.get("threshold_search_available") is False, f"{path}: threshold search exposed")
        _require(metadata.get("protocol_hash") == FROZEN_PROTOCOL_SHA256, f"{path}: protocol mismatch")
        _require(metadata.get("missing_rows") == [] and metadata.get("errors") == [], f"{path}: missing/error rows")
        source_key = path.stem
        source_paths[source_key] = str(path)
        source_hashes[source_key] = _sha256(path)
        source_metadata[source_key] = metadata
        for row in payload.get("rows", []):
            _require(row.get("status") == "ok", f"{path}: row not ok")
            for metric in BASELINES:
                _finite(row.get(metric), name=f"{path}/{metric}")
            key = (
                row.get("training_family_id"),
                row.get("task"),
                float(row.get("training_rho")),
                row.get("stressor_family"),
                row.get("branch"),
            )
            _require(key not in natural_keys, f"{path}: duplicate baseline row")
            natural_keys.add(key)
            all_rows.append(dict(row))
    expected_by_split = {"CAL": 36, "E1": 72, "E2": 36, "E3-L": 48, "E3-P": 16, "E4": 64}
    observed_by_split = {
        split: sum(row["split_name"] == split for row in all_rows)
        for split in expected_by_split
    }
    _require(observed_by_split == expected_by_split, f"baseline split coverage mismatch: {observed_by_split}")
    _require(len(all_rows) == 272, "baseline total row count mismatch")

    cal_behavior = _calibration_behavior(calibration_path)
    external_behavior = _external_behavior(lewm_behavior_path, pldm_behavior_path)
    cal_rows = []
    for row in all_rows:
        if row["split_name"] != "CAL":
            continue
        key = (
            row["model_family"],
            int(row["training_seed"]),
            row["task"],
            float(row["training_rho"]),
        )
        behavior = cal_behavior[key]
        cal_rows.append(
            {
                **row,
                "behavior_label": behavior["behavior_label"],
                "clean_score": _finite(behavior.get("clean_score"), name="CAL/clean_score"),
                "smpr": _finite(behavior.get("smpr"), name="CAL/smpr"),
            }
        )
    _require(len(cal_rows) == 36, "CAL baseline join mismatch")
    thresholds = {
        metric: _select_threshold(
            cal_rows,
            metric,
            direction=BASELINE_DIRECTIONS[metric],
        )
        for metric in CALIBRATED_BASELINES
    }

    heldout_rows = []
    for row in all_rows:
        if row["split_name"] not in {"E1", "E2"}:
            continue
        key = (
            row["model_family"],
            int(row["training_seed"]),
            row["task"],
            float(row["training_rho"]),
        )
        behavior = external_behavior[key]
        joined = {
            **row,
            "clean_score": behavior["clean_score"],
            "stress_score": behavior["stress_score"],
            "smpr": _finite(behavior.get("smpr"), name="heldout/smpr"),
            "behavior_label": _as_bool(behavior["behavior_label"]),
            "joint_score": behavior["joint_score"],
            "frozen_gate_pass": _as_bool(behavior["frozen_gate_pass"]),
            "protocol_hash": FROZEN_PROTOCOL_SHA256,
        }
        for metric, selection in thresholds.items():
            joined[f"{metric}_pass"] = (
                float(joined[metric]) <= selection["threshold"]
                if selection["direction"] == "pass_if_value_le_threshold"
                else float(joined[metric]) >= selection["threshold"]
            )
        heldout_rows.append(joined)
    _require(len(heldout_rows) == 108, "heldout baseline join mismatch")
    summary_rows = []
    for split in ("E1", "E2", "E1+E2"):
        rows = (
            heldout_rows
            if split == "E1+E2"
            else [row for row in heldout_rows if row["split_name"] == split]
        )
        for metric, selection in thresholds.items():
            summary_rows.append(
                {
                    "split_name": split,
                    **_metric_summary(
                        rows,
                        metric,
                        selection["threshold"],
                        direction=selection["direction"],
                    ),
                }
            )
        summary_rows.append(
            {
                "split_name": split,
                **_fixed_prediction_summary(
                    rows,
                    metric="atr_smpr_joint",
                    prediction_field="frozen_gate_pass",
                    score_field="joint_score",
                    threshold_label="frozen_joint_gate",
                ),
            }
        )
    script_path = Path(__file__).resolve()
    source_paths.update(
        {
            "protocol": str(protocol_path),
            "calibration_behavior": str(calibration_path),
            "lewm_behavior": str(lewm_behavior_path),
            "pldm_behavior": str(pldm_behavior_path),
            "merge_builder": str(script_path),
        }
    )
    source_hashes.update(
        {
            "protocol": FROZEN_PROTOCOL_SHA256,
            "calibration_behavior": _sha256(calibration_path),
            "lewm_behavior": _sha256(lewm_behavior_path),
            "pldm_behavior": _sha256(pldm_behavior_path),
            "merge_builder": _sha256(script_path),
        }
    )
    combined = {
        "metadata": {
            "schema_version": "paper1-diagnostic-baseline-all-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "status": "complete",
            "status_counts": {"ok": len(all_rows)},
            "missing_rows": [],
            "errors": [],
            "behavior_blind_rows": True,
            "threshold_search_split": "CAL only",
            "external_threshold_search_allowed": False,
            "protocol_hash": FROZEN_PROTOCOL_SHA256,
            "count_by_split": observed_by_split,
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "source_metadata": source_metadata,
            "calibrated_thresholds": thresholds,
        },
        "rows": all_rows,
    }
    return combined, heldout_rows, summary_rows


def _write_table(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        row for row in summary_rows if row["split_name"] == "E1+E2"
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{CAL-frozen diagnostic baselines on the combined E1+E2 external rows. Thresholds are never selected on external behavior; clean score is not training-free.}",
        r"\label{tab:diagnostic-baselines}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Diagnostic & Balanced accuracy & AUPRC \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["metric"]).replace("_", r"\_")
        lines.append(
            f"{label} & "
            f"{float(row['balanced_accuracy']):.3f} & "
            f"{float(row['auprc']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figure(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in summary_rows if row["split_name"] == "E1+E2"]
    labels = [str(row["metric"]).replace("_q90", "").replace("_", " ") for row in rows]
    y = list(range(len(rows)))
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), sharey=True)
    for axis, field, title in (
        (axes[0], "balanced_accuracy", "Balanced accuracy"),
        (axes[1], "auprc", "AUPRC"),
    ):
        values = [float(row[field]) for row in rows]
        axis.barh(y, values, color="#4c78a8")
        axis.set_xlim(0.0, 1.0)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        for index, value in enumerate(values):
            axis.text(min(value + 0.015, 0.96), index, f"{value:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    figure.suptitle("CAL-frozen diagnostic comparison on E1+E2")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "paper1/results/remediation_phase3_baseline_sources")
    parser.add_argument("--protocol", type=Path, default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "paper1/results/frozen_diagnostic_protocol_calibration.json")
    parser.add_argument("--lewm-behavior", type=Path, default=ROOT / "paper1/results/frozen_external_validation_rows_v3.csv")
    parser.add_argument("--pldm-behavior", type=Path, default=ROOT / "paper1/results/external_validation/pldm_frozen_rows_v2.csv")
    parser.add_argument("--out-json", type=Path, default=ROOT / "paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json")
    parser.add_argument("--out-rows", type=Path, default=ROOT / "paper1/results/diagnostic_baselines/heldout_baseline_rows.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "paper1/results/diagnostic_baselines/heldout_baseline_summary.csv")
    parser.add_argument("--table", type=Path, default=ROOT / "paper1/tables/table_diagnostic_baselines.tex")
    parser.add_argument("--figure", type=Path, default=ROOT / "assets/paper1_figs/fig_diagnostic_baseline_external.png")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    combined, heldout_rows, summary_rows = build(
        input_dir=args.input_dir,
        protocol_path=args.protocol,
        calibration_path=args.calibration,
        lewm_behavior_path=args.lewm_behavior,
        pldm_behavior_path=args.pldm_behavior,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.out_rows, heldout_rows)
    _write_csv(args.out_summary, summary_rows)
    _write_table(args.table, summary_rows)
    _write_figure(args.figure, summary_rows)
    print(
        f"wrote {len(combined['rows'])} baseline rows and "
        f"{len(heldout_rows)} E1/E2 joins"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
