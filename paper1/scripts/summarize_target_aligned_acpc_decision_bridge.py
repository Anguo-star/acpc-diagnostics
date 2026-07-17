#!/usr/bin/env python3
"""Exploratory fixed-pool decision bridge for target-aligned ACPC MVE rows.

This audit uses the decision and privileged replay fields already committed by
the MVE. It does not change the predictive-error targets or the frozen PushT
gate. The ordered-pool flip is internal (T1); selected-candidate true regret is
a bounded, one-plan simulator endpoint (T3), not closed-loop behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper1.scripts.run_target_aligned_acpc_mve import (  # noqa: E402
    _grouped_ridge_predictions,
)


SCHEMA = "paper1-target-aligned-acpc-decision-bridge-0.2"
BASE_FEATURES = (
    "severity",
    "encoder_response",
    "nominal_top1_top2_margin",
    "nominal_model_cost_std",
    "candidate_action_rms_mean",
    "candidate_action_rms_std",
    "nominal_h1_displacement_mean",
    "nominal_h5_displacement_mean",
    "nominal_h5_displacement_std",
)
FEATURE_SETS = {
    "strong_simple": BASE_FEATURES,
    "plus_correct_h1": BASE_FEATURES
    + (
        "correct_h1_winner",
        "correct_h1_runnerup",
        "correct_h1_pool_max",
        "correct_h1_gap_risk",
    ),
    "plus_correct_h5": BASE_FEATURES
    + (
        "correct_h1_winner",
        "correct_h1_runnerup",
        "correct_h1_pool_max",
        "correct_h1_gap_risk",
        "correct_h5_winner",
        "correct_h5_runnerup",
        "correct_h5_pool_max",
        "correct_h5_gap_risk",
    ),
    "plus_action_zero_h5_control": BASE_FEATURES
    + (
        "correct_h1_winner",
        "correct_h1_runnerup",
        "correct_h1_pool_max",
        "correct_h1_gap_risk",
        "action_zero_h5_winner",
        "action_zero_h5_runnerup",
        "action_zero_h5_pool_max",
        "action_zero_h5_gap_risk",
    ),
    "plus_candidate_shuffle_h5_control": BASE_FEATURES
    + (
        "correct_h1_winner",
        "correct_h1_runnerup",
        "correct_h1_pool_max",
        "correct_h1_gap_risk",
        "candidate_shuffle_h5_winner",
        "candidate_shuffle_h5_runnerup",
        "candidate_shuffle_h5_pool_max",
        "candidate_shuffle_h5_gap_risk",
    ),
    "plus_time_shuffle_h5_control": BASE_FEATURES
    + (
        "correct_h1_winner",
        "correct_h1_runnerup",
        "correct_h1_pool_max",
        "correct_h1_gap_risk",
        "time_shuffle_h5_winner",
        "time_shuffle_h5_runnerup",
        "time_shuffle_h5_pool_max",
        "time_shuffle_h5_gap_risk",
    ),
}


def _safe_metric(metric, y: np.ndarray, p: np.ndarray) -> float | None:
    try:
        value = float(metric(y, p))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _grouped_logistic_predictions(
    rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    target_name: str,
) -> dict[str, Any]:
    filtered = [row for row in rows if float(row["severity"]) > 0.0]
    groups = np.asarray(
        [int(row["trajectory_block_index"]) for row in filtered],
        dtype=np.int64,
    )
    unique_groups = np.unique(groups)
    x = np.asarray(
        [[float(row[name]) for name in feature_names] for row in filtered],
        dtype=np.float64,
    )
    y = np.asarray([int(bool(row[target_name])) for row in filtered])
    probabilities = np.full(y.shape, np.nan, dtype=np.float64)
    for held_out in unique_groups:
        test = groups == held_out
        train = ~test
        if np.unique(y[train]).size < 2:
            probabilities[test] = float(y[train].mean())
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        )
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("grouped logistic left non-finite probabilities")
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    per_group_log_loss = []
    for group in unique_groups:
        select = groups == group
        per_group_log_loss.append(
            float(log_loss(y[select], clipped[select], labels=[0, 1]))
        )
    return {
        "feature_names": list(feature_names),
        "target_name": target_name,
        "row_count": int(len(y)),
        "group_count": int(len(unique_groups)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
        "auprc": _safe_metric(average_precision_score, y, probabilities),
        "auroc": _safe_metric(roc_auc_score, y, probabilities),
        "mean_group_log_loss": float(np.mean(per_group_log_loss)),
    }


def _pool_signal_features(
    rows: Iterable[Mapping[str, Any]],
    *,
    signal: str,
    prefix: str,
) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["nominal_model_rank"]))
    values = np.asarray([float(row[signal]) for row in ordered])
    gaps = np.asarray([float(row["nominal_cost_margin"]) for row in ordered])
    if len(values) < 2 or gaps[0] != 0.0:
        raise RuntimeError("ordered pool must contain winner and runner-up")
    gap_risk = np.max((values[0] + values[1:]) / (gaps[1:] + 1e-6))
    return {
        f"{prefix}_winner": float(values[0]),
        f"{prefix}_runnerup": float(values[1]),
        f"{prefix}_pool_max": float(values.max()),
        f"{prefix}_gap_risk": float(gap_risk),
    }


def build_cell_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_groups: dict[tuple[int, float, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in payload["candidate_rows"]:
        candidate_groups[
            (
                int(row["trajectory_block_index"]),
                float(row["severity"]),
                int(row["draw_index"]),
            )
        ].append(row)
    result = []
    for source in payload["cell_summaries"]:
        key = (
            int(source["trajectory_block_index"]),
            float(source["severity"]),
            int(source["draw_index"]),
        )
        candidates = candidate_groups[key]
        if not candidates:
            raise RuntimeError(f"missing candidate rows for cell {key}")
        nominal_margins = np.sort(
            [float(row["nominal_cost_margin"]) for row in candidates]
        )
        induced = float(source["probe_induced_true_regret"])
        row = {
            **dict(source),
            "flip": not bool(source["exact_stable"]),
            "adverse_probe_induced_true_regret": max(induced, 0.0),
            "absolute_probe_induced_true_regret": abs(induced),
            "nominal_top1_top2_margin": float(nominal_margins[1]),
            "nominal_model_cost_std": float(
                np.std([row["nominal_model_cost"] for row in candidates])
            ),
            "candidate_action_rms_mean": float(
                np.mean([row["candidate_action_rms"] for row in candidates])
            ),
            "candidate_action_rms_std": float(
                np.std([row["candidate_action_rms"] for row in candidates])
            ),
            "nominal_h1_displacement_mean": float(
                np.mean(
                    [row["nominal_h1_displacement"] for row in candidates]
                )
            ),
            "nominal_h5_displacement_mean": float(
                np.mean(
                    [row["nominal_h5_displacement"] for row in candidates]
                )
            ),
            "nominal_h5_displacement_std": float(
                np.std(
                    [row["nominal_h5_displacement"] for row in candidates]
                )
            ),
            **_pool_signal_features(
                candidates,
                signal="correct_h1_response",
                prefix="correct_h1",
            ),
            **_pool_signal_features(
                candidates,
                signal="correct_h5_response",
                prefix="correct_h5",
            ),
            **_pool_signal_features(
                candidates,
                signal="action_zero_h5_response",
                prefix="action_zero_h5",
            ),
            **_pool_signal_features(
                candidates,
                signal="candidate_shuffle_h5_response",
                prefix="candidate_shuffle_h5",
            ),
            **_pool_signal_features(
                candidates,
                signal="time_shuffle_h5_response",
                prefix="time_shuffle_h5",
            ),
        }
        result.append(row)
    return result


def summarize(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = build_cell_rows(payload)
    regression_targets = (
        "probe_induced_true_regret",
        "adverse_probe_induced_true_regret",
        "absolute_probe_induced_true_regret",
    )
    return {
        "source": str(path),
        "task": payload["metadata"]["task"],
        "checkpoint": payload["metadata"]["checkpoint"],
        "role": (
            "exploratory bounded T1/T3 bridge; not closed-loop evidence"
        ),
        "flip_models": {
            name: _grouped_logistic_predictions(
                rows,
                feature_names=features,
                target_name="flip",
            )
            for name, features in FEATURE_SETS.items()
        },
        "true_regret_models": {
            target: {
                name: _grouped_ridge_predictions(
                    rows,
                    feature_names=features,
                    target_name=target,
                )
                for name, features in FEATURE_SETS.items()
            }
            for target in regression_targets
        },
        "cell_rows": rows,
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
            "analysis_status": "exploratory_after_predictive_target_reveal",
        },
        "artifacts": [summarize(path) for path in args.inputs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "artifact_count": len(result["artifacts"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
