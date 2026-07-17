#!/usr/bin/env python3
"""Apply the immutable ATR+SMPR gate to the E4 target-view falsification set."""

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
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHO_KEYS = ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
BRANCHES = ("full_sequence", "target_view")
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
BASELINE_FIELDS = (
    "encoder_q90",
    "h1_q90",
    "action_shuffled_h8_q90",
    "action_zeroed_h8_q90",
    "time_shuffled_h8_q90",
)
ROW_FIELDS = (
    "model_family",
    "training_family_id",
    "training_seed",
    "training_seed_semantics",
    "evaluation_seeds",
    "evaluation_seed_semantics",
    "task",
    "training_rho",
    "branch",
    "checkpoint_sha256",
    "clean_score",
    "stress_score",
    "clean_score_by_evaluation_seed",
    "stress_score_by_evaluation_seed",
    "base_clean_score",
    "base_stress_score",
    "best_stress_score",
    "recovery_score_threshold",
    "behavioral_onset",
    "normalized_recovery",
    "clean_constraint_pass",
    "behavior_label",
    "rho_privileged_baseline_pass",
    *BASELINE_FIELDS,
    *(f"{field}_pass" for field in BASELINE_FIELDS),
    "atr_horizon_v2_q90",
    "smpr",
    "joint_score",
    "frozen_gate_pass",
    "false_pass",
    "false_negative",
    "split_name",
    "protocol_hash",
    "diagnostics_sha256",
    "behavior_source_sha256",
)
PAIR_FIELDS = (
    "model_family",
    "training_seed",
    "task",
    "training_rho",
    "full_sequence_checkpoint_sha256",
    "target_view_checkpoint_sha256",
    "full_sequence_stress_score",
    "target_view_stress_score",
    "delta_behavior_full_minus_target",
    "full_sequence_behavior_label",
    "target_view_behavior_label",
    "behavior_collapse",
    "full_sequence_atr",
    "target_view_atr",
    "delta_atr_target_minus_full",
    "full_sequence_smpr",
    "target_view_smpr",
    "delta_smpr_full_minus_target",
    "full_sequence_joint_score",
    "target_view_joint_score",
    "delta_joint_full_minus_target",
    "full_sequence_gate_pass",
    "target_view_gate_pass",
    "collapse_but_target_gate_pass",
    "matched_behavior_preference",
    "matched_joint_preference",
    "matched_ordering_correct",
    "rho_privileged_baseline_tie",
    "split_name",
    "protocol_hash",
)


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


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot compute an empty mean")
    return sum(values) / len(values)


def _score_metric(block: Mapping[str, Any], *, name: str) -> tuple[float, list[float]]:
    values = [_finite(value, name=f"{name}/value") for value in block.get("values", [])]
    _require(len(values) == 3, f"{name}: expected three evaluation values")
    _require(block.get("eval_seeds") == [42, 43, 44], f"{name}: eval seed mismatch")
    mean_value = _finite(block.get("mean"), name=f"{name}/mean")
    _require(math.isclose(mean_value, _mean(values), abs_tol=1e-8), f"{name}: mean mismatch")
    return mean_value, values


def _protocol(path: Path) -> tuple[dict[str, Any], bytes, int]:
    original = path.read_bytes()
    payload = json.loads(original.decode("utf-8"), parse_constant=_reject_constant)
    _require(hashlib.sha256(original).hexdigest() == FROZEN_PROTOCOL_SHA256, "protocol hash mismatch")
    _require(payload.get("status") == "frozen", "protocol is not frozen")
    _require(payload.get("immutable") is True, "protocol is not immutable")
    _require(
        payload.get("external_policy", {}).get("threshold_search_allowed") is False,
        "external threshold search enabled",
    )
    return payload, original, path.stat().st_mtime_ns


def _assert_protocol_unchanged(path: Path, original: bytes, mtime_ns: int) -> None:
    _require(path.read_bytes() == original, "consumer changed protocol bytes")
    _require(path.stat().st_mtime_ns == mtime_ns, "consumer changed protocol mtime")


def _joint(
    *,
    atr: float,
    smpr: float,
    tau_atr: float,
    tau_smpr: float,
) -> tuple[float, bool]:
    atr_margin = (tau_atr - atr) / (abs(tau_atr) + 1e-12)
    smpr_margin = (smpr - tau_smpr) / (abs(tau_smpr) + 1e-12)
    return min(atr_margin, smpr_margin), atr <= tau_atr and smpr >= tau_smpr


def _load_diagnostics(
    *,
    atr_paths: Mapping[str, Path],
    smpr_paths: Mapping[str, Path],
    protocol: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, str], dict[str, str]]:
    tau_atr = _finite(protocol.get("tau_atr"), name="tau_atr")
    tau_smpr = _finite(protocol.get("tau_smpr"), name="tau_smpr")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    expected = {(task, rho) for task in TASKS for rho in RHO_KEYS}
    for branch in BRANCHES:
        atr_path = atr_paths[branch]
        smpr_path = smpr_paths[branch]
        atr_payload = _load(atr_path)
        smpr_payload = _load(smpr_path)
        for payload, role in (
            (atr_payload, "target_view_external_atr"),
            (smpr_payload, "target_view_external_smpr"),
        ):
            metadata = payload.get("metadata", {})
            _require(metadata.get("artifact_role") == role, f"{branch}: wrong diagnostic role")
            _require(metadata.get("status") == "complete", f"{branch}: diagnostic incomplete")
            _require(metadata.get("behavior_blind") is True, f"{branch}: behavior leaked")
            _require(metadata.get("threshold_search_allowed") is False, f"{branch}: threshold search enabled")
            _require(metadata.get("protocol_hash") == FROZEN_PROTOCOL_SHA256, f"{branch}: protocol mismatch")
            _require(metadata.get("branch") == branch, f"{branch}: branch mismatch")
            _require(metadata.get("split_name") == "E4", f"{branch}: split mismatch")
        atr_rows = {
            (str(row.get("task")), str(row.get("std_key"))): row
            for row in atr_payload.get("rows", [])
        }
        smpr_rows = {
            (str(row.get("task")), str(row.get("std_key"))): row
            for row in smpr_payload.get("rows", [])
        }
        _require(set(atr_rows) == expected, f"{branch}: ATR coverage mismatch")
        _require(set(smpr_rows) == expected, f"{branch}: SMPR coverage mismatch")
        atr_hash = _sha256(atr_path)
        smpr_hash = _sha256(smpr_path)
        source_paths[f"{branch}_atr"] = str(atr_path)
        source_paths[f"{branch}_smpr"] = str(smpr_path)
        source_hashes[f"{branch}_atr"] = atr_hash
        source_hashes[f"{branch}_smpr"] = smpr_hash
        for task, rho in expected:
            atr_row = atr_rows[(task, rho)]
            smpr_row = smpr_rows[(task, rho)]
            name = f"E4/{branch}/{task}/{rho}"
            _require(atr_row.get("status") == "ok", f"{name}: ATR not ok")
            _require(smpr_row.get("status") == "ok", f"{name}: SMPR not ok")
            _require(
                atr_row.get("checkpoint_sha256") == smpr_row.get("checkpoint_sha256"),
                f"{name}: checkpoint mismatch",
            )
            atr = _finite(atr_row.get("atr_horizon_v2_q90"), name=f"{name}/ATR")
            reference = _finite(
                smpr_row.get("reference_atr_horizon_v2_q90"),
                name=f"{name}/reference-ATR",
            )
            _require(math.isclose(atr, reference, rel_tol=1e-6, abs_tol=1e-6), f"{name}: ATR reference mismatch")
            smpr = _finite(smpr_row.get("smpr"), name=f"{name}/SMPR")
            joint_score, gate_pass = _joint(
                atr=atr,
                smpr=smpr,
                tau_atr=tau_atr,
                tau_smpr=tau_smpr,
            )
            result[(task, rho, branch)] = {
                "atr": atr,
                "smpr": smpr,
                "joint_score": joint_score,
                "gate_pass": gate_pass,
                "checkpoint_sha256": str(atr_row.get("checkpoint_sha256")),
                "diagnostics_sha256": hashlib.sha256(
                    (atr_hash + smpr_hash + str(atr_row.get("checkpoint_sha256"))).encode("ascii")
                ).hexdigest(),
            }
    _require(len(result) == 64, "E4 diagnostic row count mismatch")
    return result, source_paths, source_hashes


def _calibration_blocks(
    *,
    path: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], str]:
    expected_hash = str(protocol.get("calibration_audit_sha256"))
    actual_hash = _sha256(path)
    _require(actual_hash == expected_hash, "calibration audit hash mismatch")
    payload = _load(path)
    _require(
        payload.get("metadata", {}).get("schema_version")
        == "paper1-frozen-diagnostic-protocol-calibration-1.0",
        "calibration audit schema mismatch",
    )
    onset = {
        str(row.get("task")): _finite(row.get("behavioral_onset"), name="behavioral onset")
        for row in payload.get("calibration_blocks", [])
    }
    _require(set(onset) == set(TASKS), "calibration onset coverage mismatch")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("calibration_rows", []):
        grouped[str(row.get("task"))].append(row)
    result: dict[str, dict[str, float]] = {}
    for task in TASKS:
        _require(len(grouped[task]) == 9, f"{task}: calibration row count mismatch")
        values = {}
        for field in (
            "base_clean_score",
            "base_stress_score",
            "best_stress_score",
            "recovery_score_threshold",
        ):
            unique = {
                _finite(row.get(field), name=f"{task}/{field}")
                for row in grouped[task]
            }
            _require(len(unique) == 1, f"{task}: calibration {field} changed within block")
            values[field] = next(iter(unique))
        values["behavioral_onset"] = onset[task]
        result[task] = values
    return result, actual_hash


def _manifest_rows(
    path: Path,
) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]], Mapping[str, Any]]:
    payload = _load(path)
    metadata = payload.get("_metadata", {})
    _require(
        metadata.get("schema_version")
        == "paper1-target-view-diagnostic-manifest-1.0",
        "target-view manifest schema mismatch",
    )
    _require(metadata.get("status") == "ok", "target-view manifest incomplete")
    _require(metadata.get("actual_rows") == 64, "target-view manifest count mismatch")
    rows = {
        (str(row.get("task")), str(row.get("std_key")), str(row.get("branch"))): row
        for row in payload.get("rows", [])
    }
    expected = {
        (task, rho, branch)
        for task in TASKS
        for rho in RHO_KEYS
        for branch in BRANCHES
    }
    _require(set(rows) == expected, "target-view behavior coverage mismatch")
    return rows, metadata


def _baseline_rows(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str], Mapping[str, Any]],
    dict[str, float],
    Mapping[str, Any],
]:
    payload = _load(path)
    metadata = payload.get("metadata", {})
    _require(
        metadata.get("schema_version") == "paper1-diagnostic-baseline-all-1.0",
        "diagnostic baseline schema mismatch",
    )
    _require(metadata.get("status") == "complete", "diagnostic baseline incomplete")
    _require(metadata.get("errors") == [], "diagnostic baseline errors present")
    _require(metadata.get("missing_rows") == [], "diagnostic baseline rows missing")
    _require(metadata.get("behavior_blind_rows") is True, "diagnostic baseline rows are not behavior blind")
    _require(
        metadata.get("external_threshold_search_allowed") is False,
        "diagnostic baseline external threshold search enabled",
    )
    _require(metadata.get("threshold_search_split") == "CAL only", "diagnostic baseline threshold split mismatch")
    _require(metadata.get("protocol_hash") == FROZEN_PROTOCOL_SHA256, "diagnostic baseline protocol mismatch")
    thresholds = {}
    selections = metadata.get("calibrated_thresholds", {})
    for field in BASELINE_FIELDS:
        selection = selections.get(field, {})
        _require(selection.get("selection_split") == "CAL", f"{field}: baseline threshold not CAL-frozen")
        _require(
            selection.get("direction") == "pass_if_value_le_threshold",
            f"{field}: baseline threshold direction mismatch",
        )
        thresholds[field] = _finite(selection.get("threshold"), name=f"{field}/threshold")
    rows = {}
    for row in payload.get("rows", []):
        if row.get("split_name") != "E4":
            continue
        _require(row.get("status") == "ok", "E4 baseline row not ok")
        _require(row.get("model_family") == "LeWM", "E4 baseline model mismatch")
        _require(int(row.get("training_seed")) == 3072, "E4 baseline training seed mismatch")
        _require(row.get("reference_atr_match") is True, "E4 baseline/reference ATR mismatch")
        key = (
            str(row.get("task")),
            f"{float(row.get('std_key')):.2f}",
            str(row.get("branch")),
        )
        _require(key not in rows, "duplicate E4 baseline row")
        for field in BASELINE_FIELDS:
            _finite(row.get(field), name=f"E4/{key}/{field}")
        rows[key] = row
    expected = {
        (task, rho, branch)
        for task in TASKS
        for rho in RHO_KEYS
        for branch in BRANCHES
    }
    _require(set(rows) == expected, "E4 baseline coverage mismatch")
    return rows, thresholds, metadata


def _build_rows(
    *,
    diagnostics: Mapping[tuple[str, str, str], Mapping[str, Any]],
    baseline_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    baseline_thresholds: Mapping[str, float],
    behavior_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    calibration: Mapping[str, Mapping[str, float]],
    clean_tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        block = calibration[task]
        denominator = max(block["best_stress_score"] - block["base_stress_score"], 1e-12)
        for rho in RHO_KEYS:
            for branch in BRANCHES:
                name = f"E4/{branch}/{task}/{rho}"
                behavior = behavior_rows[(task, rho, branch)]
                diag = diagnostics[(task, rho, branch)]
                baseline = baseline_rows[(task, rho, branch)]
                checkpoint = behavior.get("checkpoint", {})
                _require(checkpoint.get("sha256") == diag["checkpoint_sha256"], f"{name}: behavior/diagnostic checkpoint mismatch")
                _require(
                    baseline.get("checkpoint_sha256") == diag["checkpoint_sha256"],
                    f"{name}: baseline/diagnostic checkpoint mismatch",
                )
                clean, clean_values = _score_metric(
                    behavior.get("metrics", {}).get("clean", {}),
                    name=f"{name}/clean",
                )
                stress, stress_values = _score_metric(
                    behavior.get("metrics", {}).get("pixels_std0.08", {}),
                    name=f"{name}/stress",
                )
                clean_pass = clean >= block["base_clean_score"] - clean_tolerance
                label = stress >= block["recovery_score_threshold"] and clean_pass
                rho_baseline = float(rho) >= block["behavioral_onset"]
                gate = bool(diag["gate_pass"])
                baseline_values = {}
                for field in BASELINE_FIELDS:
                    value = _finite(baseline.get(field), name=f"{name}/{field}")
                    baseline_values[field] = value
                    baseline_values[f"{field}_pass"] = value <= baseline_thresholds[field]
                rows.append(
                    {
                        "model_family": "LeWM",
                        "training_family_id": f"lewm_seed3072_{branch}",
                        "training_seed": 3072,
                        "training_seed_semantics": "one existing LeWM training run per task/std/branch",
                        "evaluation_seeds": "42;43;44",
                        "evaluation_seed_semantics": "repeated closed-loop evaluation of a fixed checkpoint",
                        "task": task,
                        "training_rho": float(rho),
                        "branch": branch,
                        "checkpoint_sha256": diag["checkpoint_sha256"],
                        "clean_score": clean,
                        "stress_score": stress,
                        "clean_score_by_evaluation_seed": ";".join(str(value) for value in clean_values),
                        "stress_score_by_evaluation_seed": ";".join(str(value) for value in stress_values),
                        **block,
                        "normalized_recovery": (stress - block["base_stress_score"]) / denominator,
                        "clean_constraint_pass": clean_pass,
                        "behavior_label": label,
                        "rho_privileged_baseline_pass": rho_baseline,
                        **baseline_values,
                        "atr_horizon_v2_q90": diag["atr"],
                        "smpr": diag["smpr"],
                        "joint_score": diag["joint_score"],
                        "frozen_gate_pass": gate,
                        "false_pass": gate and not label,
                        "false_negative": label and not gate,
                        "split_name": "E4",
                        "protocol_hash": FROZEN_PROTOCOL_SHA256,
                        "diagnostics_sha256": diag["diagnostics_sha256"],
                        "behavior_source_sha256": hashlib.sha256(
                            "".join(
                                source["sha256"]
                                for metric in behavior.get("metrics", {}).values()
                                for source in metric.get("source_files", [])
                            ).encode("ascii")
                        ).hexdigest(),
                    }
                )
    _require(len(rows) == 64, "E4 scored row count mismatch")
    return rows


def _preference(left: float, right: float, *, left_name: str, right_name: str) -> str:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return left_name if left > right else right_name


def _build_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row["task"]), f"{float(row['training_rho']):.2f}", str(row["branch"])): row
        for row in rows
    }
    result = []
    for task in TASKS:
        for rho in RHO_KEYS:
            full = index[(task, rho, "full_sequence")]
            target = index[(task, rho, "target_view")]
            behavior_preference = _preference(
                float(full["stress_score"]),
                float(target["stress_score"]),
                left_name="full_sequence",
                right_name="target_view",
            )
            joint_preference = _preference(
                float(full["joint_score"]),
                float(target["joint_score"]),
                left_name="full_sequence",
                right_name="target_view",
            )
            collapse = bool(full["behavior_label"]) and not bool(target["behavior_label"])
            result.append(
                {
                    "model_family": "LeWM",
                    "training_seed": 3072,
                    "task": task,
                    "training_rho": float(rho),
                    "full_sequence_checkpoint_sha256": full["checkpoint_sha256"],
                    "target_view_checkpoint_sha256": target["checkpoint_sha256"],
                    "full_sequence_stress_score": full["stress_score"],
                    "target_view_stress_score": target["stress_score"],
                    "delta_behavior_full_minus_target": float(full["stress_score"]) - float(target["stress_score"]),
                    "full_sequence_behavior_label": full["behavior_label"],
                    "target_view_behavior_label": target["behavior_label"],
                    "behavior_collapse": collapse,
                    "full_sequence_atr": full["atr_horizon_v2_q90"],
                    "target_view_atr": target["atr_horizon_v2_q90"],
                    "delta_atr_target_minus_full": float(target["atr_horizon_v2_q90"]) - float(full["atr_horizon_v2_q90"]),
                    "full_sequence_smpr": full["smpr"],
                    "target_view_smpr": target["smpr"],
                    "delta_smpr_full_minus_target": float(full["smpr"]) - float(target["smpr"]),
                    "full_sequence_joint_score": full["joint_score"],
                    "target_view_joint_score": target["joint_score"],
                    "delta_joint_full_minus_target": float(full["joint_score"]) - float(target["joint_score"]),
                    "full_sequence_gate_pass": full["frozen_gate_pass"],
                    "target_view_gate_pass": target["frozen_gate_pass"],
                    "collapse_but_target_gate_pass": collapse and bool(target["frozen_gate_pass"]),
                    "matched_behavior_preference": behavior_preference,
                    "matched_joint_preference": joint_preference,
                    "matched_ordering_correct": (
                        ""
                        if behavior_preference == "tie"
                        else behavior_preference == joint_preference
                    ),
                    "rho_privileged_baseline_tie": (
                        full["rho_privileged_baseline_pass"]
                        == target["rho_privileged_baseline_pass"]
                    ),
                    "split_name": "E4",
                    "protocol_hash": FROZEN_PROTOCOL_SHA256,
                }
            )
    _require(len(result) == 32, "E4 matched pair count mismatch")
    return result


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
        values = groups[score]
        tp += sum(values)
        fp += len(values) - sum(values)
        next_recall = tp / positives
        area += (next_recall - recall) * tp / (tp + fp)
        recall = next_recall
    return area


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str,
    score_key: str | None = None,
    higher_score_is_positive: bool = True,
) -> dict[str, Any]:
    labels = [bool(row["behavior_label"]) for row in rows]
    predictions = [bool(row[prediction_key]) for row in rows]
    if score_key is None:
        score_key = "joint_score" if prediction_key == "frozen_gate_pass" else "training_rho"
    score_sign = 1.0 if higher_score_is_positive else -1.0
    scores = [score_sign * float(row[score_key]) for row in rows]
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum((not label) and (not prediction) for label, prediction in zip(labels, predictions))
    fp = sum((not label) and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and (not prediction) for label, prediction in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(rows),
        "positive_n": sum(labels),
        "predicted_positive_n": sum(predictions),
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
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "auprc": _average_precision(labels, scores),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _plot(path: Path, pairs: Sequence[Mapping[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for task in TASKS:
        task_rows = [row for row in pairs if row["task"] == task]
        x = [float(row["training_rho"]) for row in task_rows]
        axes[0].plot(x, [float(row["delta_behavior_full_minus_target"]) for row in task_rows], marker="o", label=task)
        axes[1].plot(x, [float(row["delta_joint_full_minus_target"]) for row in task_rows], marker="o", label=task)
    axes[0].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_title("Behavior: full minus target-view")
    axes[1].set_title("Frozen joint score: full minus target-view")
    for axis in axes:
        axis.set_xlabel("Training rho")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Difference")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json")
    parser.add_argument("--calibration-audit", type=Path, default=ROOT / "paper1/results/frozen_diagnostic_protocol_calibration.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets/paper1_data/target_view_diagnostic_manifest_v1.json")
    parser.add_argument("--full-atr", type=Path, default=ROOT / "paper1/results/remediation_phase2_external_sources/target_view/full_sequence/acpc_horizon_v2_checkpoint_bound.json")
    parser.add_argument("--full-smpr", type=Path, default=ROOT / "paper1/results/remediation_phase2_external_sources/target_view/full_sequence/smpr_v2_checkpoint_bound.json")
    parser.add_argument("--target-atr", type=Path, default=ROOT / "paper1/results/remediation_phase2_external_sources/target_view/target_view/acpc_horizon_v2_checkpoint_bound.json")
    parser.add_argument("--target-smpr", type=Path, default=ROOT / "paper1/results/remediation_phase2_external_sources/target_view/target_view/smpr_v2_checkpoint_bound.json")
    parser.add_argument("--baseline-diagnostics", type=Path, default=ROOT / "paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json")
    parser.add_argument("--out-rows", type=Path, default=ROOT / "paper1/results/external_validation/target_view_frozen_rows.csv")
    parser.add_argument("--out-pairs", type=Path, default=ROOT / "paper1/results/external_validation/target_view_matched_pairs.csv")
    parser.add_argument("--out-summary", type=Path, default=ROOT / "paper1/results/external_validation/target_view_frozen_summary.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "assets/paper1_figs/fig_target_view_frozen_gate.png")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protocol, protocol_bytes, protocol_mtime = _protocol(args.protocol)
    # Diagnostics are loaded and scored before any E4 behavior source.
    diagnostics, diagnostic_paths, diagnostic_hashes = _load_diagnostics(
        atr_paths={"full_sequence": args.full_atr, "target_view": args.target_atr},
        smpr_paths={"full_sequence": args.full_smpr, "target_view": args.target_smpr},
        protocol=protocol,
    )
    baseline_rows, baseline_thresholds, baseline_metadata = _baseline_rows(
        args.baseline_diagnostics
    )
    calibration, calibration_hash = _calibration_blocks(
        path=args.calibration_audit,
        protocol=protocol,
    )
    behavior_rows, manifest_metadata = _manifest_rows(args.manifest)
    rows = _build_rows(
        diagnostics=diagnostics,
        baseline_rows=baseline_rows,
        baseline_thresholds=baseline_thresholds,
        behavior_rows=behavior_rows,
        calibration=calibration,
        clean_tolerance=_finite(
            protocol.get("gaussian_behavior_label", {}).get("clean_tolerance_pp"),
            name="clean tolerance",
        ),
    )
    pairs = _build_pairs(rows)
    branch_metrics = {
        branch: {
            "frozen_gate": _metrics(
                [row for row in rows if row["branch"] == branch],
                prediction_key="frozen_gate_pass",
            ),
            "rho_privileged_baseline": _metrics(
                [row for row in rows if row["branch"] == branch],
                prediction_key="rho_privileged_baseline_pass",
            ),
            "simple_diagnostic_baselines": {
                field: _metrics(
                    [row for row in rows if row["branch"] == branch],
                    prediction_key=f"{field}_pass",
                    score_key=field,
                    higher_score_is_positive=False,
                )
                for field in BASELINE_FIELDS
            },
        }
        for branch in BRANCHES
    }
    comparable = [row for row in pairs if row["matched_ordering_correct"] != ""]
    collapse_rows = [row for row in pairs if row["behavior_collapse"]]
    collapse_counterexamples = [
        row for row in pairs if row["collapse_but_target_gate_pass"]
    ]
    false_negatives = [row for row in rows if row["false_negative"]]
    false_passes = [row for row in rows if row["false_pass"]]
    source_paths = {
        "protocol": str(args.protocol),
        "calibration_audit": str(args.calibration_audit),
        "target_view_manifest": str(args.manifest),
        "baseline_diagnostics": str(args.baseline_diagnostics),
        **diagnostic_paths,
    }
    source_hashes = {
        "protocol": FROZEN_PROTOCOL_SHA256,
        "calibration_audit": calibration_hash,
        "target_view_manifest": _sha256(args.manifest),
        "baseline_diagnostics": _sha256(args.baseline_diagnostics),
        **diagnostic_hashes,
    }
    script_path = Path(__file__).resolve()
    source_paths["validator"] = str(script_path.relative_to(ROOT))
    source_hashes["validator"] = _sha256(script_path)
    summary = {
        "metadata": {
            "schema_version": "paper1-target-view-frozen-gate-validation-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "status": "complete",
            "split_name": "E4",
            "protocol_hash": FROZEN_PROTOCOL_SHA256,
            "threshold_search_allowed": False,
            "behavior_loaded_after_frozen_diagnostic_scoring": True,
            "target_view_interpretation": "failed mechanism falsification, not a successful repair",
            "manifest_legacy_conflict_count": manifest_metadata.get("legacy_conflict_count"),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "baseline_diagnostics_status": baseline_metadata.get("status"),
            "baseline_threshold_selection_split": baseline_metadata.get("threshold_search_split"),
            "baseline_thresholds": baseline_thresholds,
            "missing_baseline_fields": [],
            "errors": [],
            "missing_rows": [],
        },
        "count_contract": {
            "expected_rows": 64,
            "observed_rows": len(rows),
            "expected_pairs": 32,
            "observed_pairs": len(pairs),
            "expected_baseline_rows": 64,
            "observed_baseline_rows": len(baseline_rows),
        },
        "branch_metrics": branch_metrics,
        "overall_frozen_gate_metrics": _metrics(rows, prediction_key="frozen_gate_pass"),
        "overall_rho_privileged_baseline_metrics": _metrics(rows, prediction_key="rho_privileged_baseline_pass"),
        "overall_simple_diagnostic_baseline_metrics": {
            field: _metrics(
                rows,
                prediction_key=f"{field}_pass",
                score_key=field,
                higher_score_is_positive=False,
            )
            for field in BASELINE_FIELDS
        },
        "matched_pair_ordering": {
            "comparable_pairs": len(comparable),
            "correct_pairs": sum(bool(row["matched_ordering_correct"]) for row in comparable),
            "accuracy": (
                sum(bool(row["matched_ordering_correct"]) for row in comparable) / len(comparable)
                if comparable
                else None
            ),
            "rho_privileged_baseline_ties": sum(bool(row["rho_privileged_baseline_tie"]) for row in pairs),
            "rho_privileged_baseline_pair_discrimination": 0.0,
        },
        "behavior_collapse": {
            "count": len(collapse_rows),
            "target_gate_pass_counterexample_count": len(collapse_counterexamples),
            "target_gate_pass_counterexamples": collapse_counterexamples,
        },
        "false_pass_count": len(false_passes),
        "false_pass_rows": false_passes,
        "false_negative_count": len(false_negatives),
        "false_negative_rows": false_negatives,
    }
    _write_csv(args.out_rows, rows, ROW_FIELDS, force=args.force)
    _write_csv(args.out_pairs, pairs, PAIR_FIELDS, force=args.force)
    _write_json(args.out_summary, summary, force=args.force)
    _plot(args.figure, pairs, force=args.force)
    _assert_protocol_unchanged(args.protocol, protocol_bytes, protocol_mtime)
    print(f"wrote {len(rows)} E4 rows and {len(pairs)} matched pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
