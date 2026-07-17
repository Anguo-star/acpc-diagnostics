#!/usr/bin/env python3
"""Summarize target-aligned ACPC DEV adjudication without selecting a story.

The input artifacts retain raw rows, so this script can apply the corrected
target semantics to both schema 0.1 and 0.2 runs.  Signed degradation,
adverse degradation, absolute prediction-error drift, and exact certificate
validity are deliberately reported as separate objects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper1.scripts.run_target_aligned_acpc_mve import (
    CANDIDATE_FEATURE_SETS,
    LOGGED_FEATURE_SETS,
    _grouped_ridge_predictions,
)


SCHEMA = "paper1-target-aligned-acpc-adjudication-0.1"
CANDIDATE_TARGETS = (
    "excess_h5_prediction_error",
    "adverse_h5_prediction_degradation",
    "absolute_h5_prediction_error_drift",
)
LOGGED_TARGETS = (
    "correct_excess_h8_error",
    "correct_adverse_h8_error",
    "correct_absolute_h8_error_drift",
)


def _finite(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    return result[np.isfinite(result)]


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    finite = _finite(values)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "q00_q25_q50_q75_q95_q100": np.quantile(
            finite, [0.0, 0.25, 0.50, 0.75, 0.95, 1.0]
        ).tolist(),
    }


def _enrich_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        h1_change = float(row["excess_h1_prediction_error"])
        h5_change = float(row["excess_h5_prediction_error"])
        row.setdefault("adverse_h1_prediction_degradation", max(h1_change, 0.0))
        row.setdefault("absolute_h1_prediction_error_drift", abs(h1_change))
        row.setdefault(
            "correct_h1_error_drift_certificate_slack",
            float(row["correct_h1_response"]) - abs(h1_change),
        )
        row.setdefault("adverse_h5_prediction_degradation", max(h5_change, 0.0))
        row.setdefault("absolute_h5_prediction_error_drift", abs(h5_change))
        row.setdefault(
            "correct_h5_error_drift_certificate_slack",
            float(row["correct_h5_response"]) - abs(h5_change),
        )
        result.append(row)
    return result


def _enrich_logged_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for horizon in ("h1", "h8"):
            change = float(row[f"correct_excess_{horizon}_error"])
            row.setdefault(
                f"correct_adverse_{horizon}_error", max(change, 0.0)
            )
            row.setdefault(
                f"correct_absolute_{horizon}_error_drift", abs(change)
            )
            row.setdefault(
                f"correct_{horizon}_error_drift_certificate_slack",
                float(row[f"correct_{horizon}_response"]) - abs(change),
            )
        result.append(row)
    return result


def _grouped_models(
    rows: list[dict[str, Any]],
    *,
    feature_sets: Mapping[str, tuple[str, ...]],
    targets: Iterable[str],
) -> dict[str, dict[str, Any]]:
    return {
        target: {
            name: _grouped_ridge_predictions(
                rows,
                feature_names=features,
                target_name=target,
            )
            for name, features in feature_sets.items()
        }
        for target in targets
    }


def _relative_mae_reduction(correct: float, comparator: float) -> float:
    if not math.isfinite(correct) or not math.isfinite(comparator):
        return float("nan")
    if comparator <= 0.0:
        return 0.0 if correct == comparator else float("-inf")
    return (comparator - correct) / comparator


def _material_gate(
    models: Mapping[str, Mapping[str, Any]],
    *,
    correct_name: str,
    shallow_name: str,
    destroyed_names: tuple[str, ...],
    mae_fraction: float = 0.05,
    rho_increment: float = 0.05,
) -> dict[str, Any]:
    correct = models[correct_name]
    shallow = models[shallow_name]
    best_destroyed_name = min(
        destroyed_names, key=lambda name: float(models[name]["mae"])
    )
    best_destroyed = models[best_destroyed_name]
    reductions = {
        "versus_shallow": _relative_mae_reduction(
            float(correct["mae"]), float(shallow["mae"])
        ),
        "versus_best_destroyed": _relative_mae_reduction(
            float(correct["mae"]), float(best_destroyed["mae"])
        ),
    }
    rho_gains = {
        "versus_shallow": float(correct["mean_within_group_spearman"])
        - float(shallow["mean_within_group_spearman"]),
        "versus_best_destroyed": float(
            correct["mean_within_group_spearman"]
        )
        - float(best_destroyed["mean_within_group_spearman"]),
    }
    shallow_by_group = {
        int(item["trajectory_block_index"]): item
        for item in shallow["per_group"]
    }
    destroyed_by_group = {
        int(item["trajectory_block_index"]): item
        for item in best_destroyed["per_group"]
    }
    paired_groups = []
    for item in correct["per_group"]:
        group = int(item["trajectory_block_index"])
        if group not in shallow_by_group or group not in destroyed_by_group:
            continue
        paired_groups.append(
            {
                "trajectory_block_index": group,
                "correct_better_than_shallow": float(item["mae"])
                < float(shallow_by_group[group]["mae"]),
                "correct_better_than_best_destroyed": float(item["mae"])
                < float(destroyed_by_group[group]["mae"]),
            }
        )
    both_wins = sum(
        item["correct_better_than_shallow"]
        and item["correct_better_than_best_destroyed"]
        for item in paired_groups
    )
    mae_pass = all(value >= mae_fraction for value in reductions.values())
    rho_pass = all(value >= rho_increment for value in rho_gains.values())
    majority_pass = bool(paired_groups) and both_wins > len(paired_groups) / 2
    return {
        "pass": bool(mae_pass and (rho_pass or majority_pass)),
        "correct_name": correct_name,
        "shallow_name": shallow_name,
        "best_destroyed_name": best_destroyed_name,
        "mae": {
            "correct": float(correct["mae"]),
            "shallow": float(shallow["mae"]),
            "best_destroyed": float(best_destroyed["mae"]),
        },
        "relative_mae_reduction": reductions,
        "within_group_rho_increment": rho_gains,
        "block_direction": {
            "both_win_count": int(both_wins),
            "paired_group_count": len(paired_groups),
            "majority_pass": majority_pass,
            "counterexample_blocks": [
                item["trajectory_block_index"]
                for item in paired_groups
                if not (
                    item["correct_better_than_shallow"]
                    and item["correct_better_than_best_destroyed"]
                )
            ],
        },
        "thresholds": {
            "minimum_relative_mae_reduction": mae_fraction,
            "minimum_rho_increment_or_block_majority": rho_increment,
        },
    }


def _pseudo_bound_audit(
    rows: list[dict[str, Any]],
    *,
    target_name: str,
    signals: Iterable[str],
    atol: float = 1e-5,
) -> dict[str, Any]:
    selected = [row for row in rows if float(row["severity"]) > 0.0]
    target = np.asarray([float(row[target_name]) for row in selected])
    result = {}
    for signal in signals:
        bound = np.asarray([float(row[signal]) for row in selected])
        result[signal] = {
            "coverage": float(np.mean(target <= bound + atol)),
            "violation_rate": float(np.mean(target > bound + atol)),
            "maximum_violation": float(np.max(target - bound)),
            "mean_bound": float(np.mean(bound)),
            "mean_target": float(np.mean(target)),
            "mean_slack": float(np.mean(bound - target)),
        }
    return result


def summarize_artifact(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    candidates = _enrich_candidate_rows(raw["candidate_rows"])
    logged = _enrich_logged_rows(raw["logged_rows"])
    candidate_models = _grouped_models(
        candidates,
        feature_sets=CANDIDATE_FEATURE_SETS,
        targets=CANDIDATE_TARGETS,
    )
    logged_models = _grouped_models(
        logged,
        feature_sets=LOGGED_FEATURE_SETS,
        targets=LOGGED_TARGETS,
    )
    nonidentity_candidates = [
        row for row in candidates if float(row["severity"]) > 0.0
    ]
    response = np.asarray(
        [float(row["correct_h5_response"]) for row in nonidentity_candidates]
    )
    drift = np.asarray(
        [
            float(row["absolute_h5_prediction_error_drift"])
            for row in nonidentity_candidates
        ]
    )
    utilization = drift[response > 1e-12] / response[response > 1e-12]
    return {
        "source": str(path),
        "source_schema": raw["metadata"].get("schema_version"),
        "source_status": raw["metadata"].get("status"),
        "task": raw["metadata"].get("task"),
        "checkpoint": raw["metadata"].get("checkpoint"),
        "training_seed": raw["metadata"].get("training_seed"),
        "checkpoint_role": raw["metadata"].get("checkpoint_role"),
        "design": raw.get("design", {}),
        "target_distributions": {
            target: _distribution(
                float(row[target])
                for row in nonidentity_candidates
            )
            for target in CANDIDATE_TARGETS
        },
        "response_distributions": {
            "correct_h5_response": _distribution(response),
            "correct_h5_bound_utilization": _distribution(utilization),
        },
        "candidate_grouped_cv": candidate_models,
        "logged_grouped_cv": logged_models,
        "candidate_gates": {
            "signed_degradation_original_dev_gate": _material_gate(
                candidate_models["excess_h5_prediction_error"],
                correct_name="plus_correct_h5",
                shallow_name="plus_correct_h1",
                destroyed_names=(
                    "plus_action_zero_h5_control",
                    "plus_candidate_shuffle_h5_control",
                    "plus_time_shuffle_h5_control",
                ),
            ),
            "adverse_degradation_semantic_reanalysis": _material_gate(
                candidate_models["adverse_h5_prediction_degradation"],
                correct_name="plus_correct_h5",
                shallow_name="plus_correct_h1",
                destroyed_names=(
                    "plus_action_zero_h5_control",
                    "plus_candidate_shuffle_h5_control",
                    "plus_time_shuffle_h5_control",
                ),
            ),
            "absolute_error_drift_ranking_secondary": _material_gate(
                candidate_models["absolute_h5_prediction_error_drift"],
                correct_name="plus_correct_h5",
                shallow_name="plus_correct_h1",
                destroyed_names=(
                    "plus_action_zero_h5_control",
                    "plus_candidate_shuffle_h5_control",
                    "plus_time_shuffle_h5_control",
                ),
            ),
        },
        "logged_gates": {
            target: _material_gate(
                logged_models[target],
                correct_name="plus_correct_h8",
                shallow_name="plus_correct_h1",
                destroyed_names=(
                    "plus_action_zero_h8_control",
                    "plus_candidate_shuffle_h8_control",
                    "plus_time_shuffle_h8_control",
                ),
            )
            for target in LOGGED_TARGETS
        },
        "candidate_correct_action_certificate": {
            "minimum_h5_slack": float(
                min(
                    row["correct_h5_error_drift_certificate_slack"]
                    for row in candidates
                )
            ),
            "role": "mathematical invariant, not independent evidence",
        },
        "candidate_pseudo_bound_audit": _pseudo_bound_audit(
            candidates,
            target_name="absolute_h5_prediction_error_drift",
            signals=(
                "correct_h5_response",
                "correct_h1_response",
                "encoder_response",
                "action_zero_h5_response",
                "candidate_shuffle_h5_response",
                "time_shuffle_h5_response",
            ),
        ),
        "logged_correct_action_certificate": {
            "minimum_h8_slack": float(
                min(
                    row["correct_h8_error_drift_certificate_slack"]
                    for row in logged
                )
            ),
            "role": "mathematical invariant, not independent evidence",
        },
        "provenance_warning": (
            "adverse/absolute semantic reanalysis was added after the "
            "TwoRoom endpoint was inspected; it is DEV evidence and must be "
            "frozen before PushT or other held-out tasks"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "metadata": {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "DEV target-aligned branch adjudication only",
            "story_selected": False,
        },
        "artifacts": [summarize_artifact(path) for path in args.inputs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        item["checkpoint"]: {
            "signed_gate": item["candidate_gates"][
                "signed_degradation_original_dev_gate"
            ]["pass"],
            "adverse_gate": item["candidate_gates"][
                "adverse_degradation_semantic_reanalysis"
            ]["pass"],
            "absolute_drift_gate": item["candidate_gates"][
                "absolute_error_drift_ranking_secondary"
            ]["pass"],
            "minimum_correct_h5_slack": item[
                "candidate_correct_action_certificate"
            ]["minimum_h5_slack"],
        }
        for item in result["artifacts"]
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
