#!/usr/bin/env python3
"""Validate and summarize the frozen ACPC-to-planner stability panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _mean(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(array.mean()) if array.size else float("nan")


def _median(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(np.median(array)) if array.size else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_results(
    addendum_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    addendum = _load_json(addendum_path)
    protocol_path = ROOT / addendum["parent_protocol"]["path"]
    if _sha256(protocol_path) != addendum["parent_protocol"]["sha256"]:
        raise RuntimeError("parent protocol hash mismatch")
    if addendum.get("status") != "frozen_pre_execution":
        raise RuntimeError("execution addendum is not frozen_pre_execution")

    loaded: list[dict[str, Any]] = []
    for shard in addendum["authorized_shards"]:
        result_path = ROOT / shard["output_path"]
        if not result_path.is_file():
            raise FileNotFoundError(f"missing authorized result: {result_path}")
        payload = _load_json(result_path)
        metadata = payload.get("metadata", {})
        if metadata.get("status") != "complete" or payload.get("errors"):
            raise RuntimeError(f"incomplete shard: {shard['shard_id']}")
        if metadata.get("checkpoint_sha256") != shard["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mismatch: {shard['shard_id']}")
        runner = shard["runner"]
        if metadata.get("script_sha256") != addendum["source_hashes"][runner]:
            raise RuntimeError(f"runner hash mismatch: {shard['shard_id']}")
        expected_rows = int(metadata["expected_rows"])
        if len(payload.get("rows", [])) != expected_rows:
            raise RuntimeError(f"row-count mismatch: {shard['shard_id']}")
        loaded.append({"shard": shard, "payload": payload})
    return addendum, loaded


def _join_reduced(
    loaded: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixed: dict[tuple[Any, ...], dict[str, Any]] = {}
    adaptive: dict[tuple[Any, ...], dict[str, Any]] = {}

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["task"],
            row["checkpoint_role"],
            row["trajectory_block_id"],
            float(row["severity"]),
            int(row["draw_index"]),
        )

    for item in loaded:
        role = item["shard"]["analysis_role"]
        if role not in {"fixed_reduced", "adaptive_reduced"}:
            continue
        target = fixed if role == "fixed_reduced" else adaptive
        for row in item["payload"]["rows"]:
            row_key = key(row)
            if row_key in target:
                raise RuntimeError(f"duplicate reduced row: {row_key}")
            target[row_key] = dict(row)
    if set(fixed) != set(adaptive):
        missing_fixed = len(set(adaptive) - set(fixed))
        missing_adaptive = len(set(fixed) - set(adaptive))
        raise RuntimeError(
            f"fixed/adaptive join mismatch: {missing_fixed=} {missing_adaptive=}"
        )
    return [
        {"fixed": fixed[row_key], "adaptive": adaptive[row_key]}
        for row_key in sorted(fixed, key=str)
    ]


def _design_row(pair: Mapping[str, Mapping[str, Any]]) -> dict[str, float | str]:
    fixed = pair["fixed"]
    adaptive = pair["adaptive"]
    return {
        "task": str(fixed["task"]),
        "checkpoint_role": str(fixed["checkpoint_role"]),
        "severity": float(fixed["severity"]),
        "h1": float(fixed["q90_candidate_h1_acpc_l2"]),
        "h5": float(fixed["q90_cost_space_acpc_final_l2"]),
        "margin": float(fixed["nominal_margin"]),
        "cost_drift": float(fixed["max_absolute_cost_drift"]),
        "first_action_rms": float(adaptive["first_action_rms"]),
        "positive_clean_regret": float(adaptive["positive_clean_decision_regret"]),
    }


def _feature_matrix(
    rows: Sequence[Mapping[str, float | str]],
    *,
    include_h5: bool,
) -> np.ndarray:
    columns = []
    for row in rows:
        values = [
            float(row["severity"]),
            1.0 if row["checkpoint_role"] == "endpoint" else 0.0,
            math.log1p(max(0.0, float(row["h1"]))),
            math.log1p(max(0.0, float(row["margin"]))),
        ]
        if include_h5:
            values.append(math.log1p(max(0.0, float(row["h5"]))))
        columns.append(values)
    return np.asarray(columns, dtype=np.float64)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    train_design = np.column_stack([np.ones(len(train_z)), train_z])
    test_design = np.column_stack([np.ones(len(test_z)), test_z])
    penalty = np.eye(train_design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_y,
    )
    return test_design @ coefficients


def _lobo_ridge(
    rows: Sequence[Mapping[str, float | str]],
    *,
    response: str,
    alpha: float = 1.0,
) -> dict[str, Any]:
    nonidentity = [row for row in rows if float(row["severity"]) > 0.0]
    tasks = sorted({str(row["task"]) for row in nonidentity})
    per_task: dict[str, Any] = {}
    for task in tasks:
        train = [row for row in nonidentity if row["task"] != task]
        test = [row for row in nonidentity if row["task"] == task]
        y_train = np.log1p(
            np.asarray([max(0.0, float(row[response])) for row in train])
        )
        y_test = np.log1p(
            np.asarray([max(0.0, float(row[response])) for row in test])
        )
        base_pred = _ridge_predict(
            _feature_matrix(train, include_h5=False),
            y_train,
            _feature_matrix(test, include_h5=False),
            alpha=alpha,
        )
        full_pred = _ridge_predict(
            _feature_matrix(train, include_h5=True),
            y_train,
            _feature_matrix(test, include_h5=True),
            alpha=alpha,
        )
        base_mae = float(np.mean(np.abs(base_pred - y_test)))
        full_mae = float(np.mean(np.abs(full_pred - y_test)))
        per_task[task] = {
            "n_test": len(test),
            "baseline_log1p_mae": base_mae,
            "plus_h5_log1p_mae": full_mae,
            "relative_mae_reduction": (
                (base_mae - full_mae) / base_mae if base_mae > 0.0 else 0.0
            ),
        }
    base_equal = _mean(
        value["baseline_log1p_mae"] for value in per_task.values()
    )
    full_equal = _mean(
        value["plus_h5_log1p_mae"] for value in per_task.values()
    )
    return {
        "response": response,
        "alpha": alpha,
        "baseline_features": [
            "severity",
            "endpoint_indicator",
            "log1p(candidate_H1_ACPC_q90)",
            "log1p(nominal_top1_margin)",
        ],
        "added_feature": "log1p(candidate_H5_ACPC_q90)",
        "per_task": per_task,
        "equal_task_baseline_log1p_mae": base_equal,
        "equal_task_plus_h5_log1p_mae": full_equal,
        "equal_task_relative_mae_reduction": (
            (base_equal - full_equal) / base_equal if base_equal > 0.0 else 0.0
        ),
        "tasks_improved": sum(
            value["plus_h5_log1p_mae"] < value["baseline_log1p_mae"]
            for value in per_task.values()
        ),
        "task_count": len(per_task),
    }


def _task_spearman(
    rows: Sequence[Mapping[str, float | str]], response: str
) -> dict[str, Any]:
    nonidentity = [row for row in rows if float(row["severity"]) > 0.0]
    result = {}
    for task in sorted({str(row["task"]) for row in nonidentity}):
        selected = [row for row in nonidentity if row["task"] == task]
        y = [float(row[response]) for row in selected]
        h1 = spearmanr([float(row["h1"]) for row in selected], y).statistic
        h5 = spearmanr([float(row["h5"]) for row in selected], y).statistic
        result[task] = {"h1": float(h1), "h5": float(h5), "n": len(selected)}
    return {
        "per_task": result,
        "equal_task_h1": _mean(value["h1"] for value in result.values()),
        "equal_task_h5": _mean(value["h5"] for value in result.values()),
    }


def _group_rows(joined: Sequence[Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[Mapping[str, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for pair in joined:
        fixed = pair["fixed"]
        groups[(fixed["task"], fixed["checkpoint_role"], float(fixed["severity"]))].append(pair)
    output = []
    for (task, role, severity), pairs in sorted(groups.items()):
        fixed = [pair["fixed"] for pair in pairs]
        adaptive = [pair["adaptive"] for pair in pairs]
        output.append(
            {
                "task": task,
                "checkpoint_role": role,
                "severity": severity,
                "n": len(pairs),
                "h1_acpc_q90_mean": _mean(
                    row["q90_candidate_h1_acpc_l2"] for row in fixed
                ),
                "h5_acpc_q90_mean": _mean(
                    row["q90_cost_space_acpc_final_l2"] for row in fixed
                ),
                "max_cost_drift_mean": _mean(
                    row["max_absolute_cost_drift"] for row in fixed
                ),
                "top1_stability_rate": _mean(row["exact_stable"] for row in fixed),
                "top1_certificate_coverage": _mean(
                    row["acpc_top1_certificate"] for row in fixed
                ),
                "elite_certificate_coverage": _mean(
                    row["acpc_elite_certificate"] for row in fixed
                ),
                "first_action_rms_mean": _mean(
                    row["first_action_rms"] for row in adaptive
                ),
                "first_action_rms_median": _median(
                    row["first_action_rms"] for row in adaptive
                ),
                "first_action_stability_rate": _mean(
                    row["first_action_stable"] for row in adaptive
                ),
                "positive_clean_regret_mean": _mean(
                    row["positive_clean_decision_regret"] for row in adaptive
                ),
            }
        )
    return output


def _full_budget_summary(
    loaded: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for item in loaded:
        if item["shard"]["analysis_role"] != "adaptive_full":
            continue
        rows = [row for row in item["payload"]["rows"] if row["severity"] > 0.0]
        output.append(
            {
                "task": item["shard"]["arguments"]["task"],
                "checkpoint_role": item["shard"]["arguments"]["checkpoint_role"],
                "n": len(rows),
                "candidate_count": item["shard"]["arguments"]["candidate_count"],
                "n_steps": item["shard"]["arguments"]["n_steps"],
                "first_action_rms_mean": _mean(row["first_action_rms"] for row in rows),
                "first_action_stability_rate": _mean(
                    row["first_action_stable"] for row in rows
                ),
                "positive_clean_regret_mean": _mean(
                    row["positive_clean_decision_regret"] for row in rows
                ),
            }
        )
    return sorted(output, key=lambda row: (row["task"], row["checkpoint_role"]))


def summarize(addendum_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    addendum, loaded = _validate_results(addendum_path)
    joined = _join_reduced(loaded)
    design_rows = [_design_row(pair) for pair in joined]
    fixed_rows = [pair["fixed"] for pair in joined]
    adaptive_rows = [pair["adaptive"] for pair in joined]

    identity_fixed = [row for row in fixed_rows if row["severity"] == 0.0]
    identity_adaptive = [row for row in adaptive_rows if row["severity"] == 0.0]
    bound_violations = sum(not row["all_mse_bounds_hold"] for row in fixed_rows)
    top1_false_certificates = sum(
        row["acpc_top1_certificate"] and not row["exact_stable"]
        for row in fixed_rows
    )
    elite_false_certificates = sum(
        row["acpc_elite_certificate"] and not row["exact_elite_set_stable"]
        for row in fixed_rows
    )
    identity_max_acpc = max(
        (row["max_cost_space_acpc_final_l2"] for row in identity_fixed),
        default=float("nan"),
    )
    identity_max_action = max(
        (row["first_action_rms"] for row in identity_adaptive),
        default=float("nan"),
    )

    analyses = {
        response: {
            "lobo_ridge": _lobo_ridge(design_rows, response=response),
            "spearman": _task_spearman(design_rows, response),
        }
        for response in (
            "cost_drift",
            "first_action_rms",
            "positive_clean_regret",
        )
    }
    summary = {
        "schema_version": "paper1-acpc-planner-stability-summary-1.0",
        "protocol_id": addendum["protocol_id"],
        "execution_addendum_sha256": _sha256(addendum_path),
        "authorized_shard_count": len(addendum["authorized_shards"]),
        "validated_shard_count": len(loaded),
        "joined_reduced_row_count": len(joined),
        "invariants": {
            "mse_cost_bound_violation_count": bound_violations,
            "top1_false_certificate_count": top1_false_certificates,
            "elite_false_certificate_count": elite_false_certificates,
            "identity_max_h5_acpc": identity_max_acpc,
            "identity_max_first_action_rms": identity_max_action,
            "identity_all_adaptive_updates_aligned": all(
                row["all_steps_distribution_aligned"] for row in identity_adaptive
            ),
            "pass": (
                bound_violations == 0
                and top1_false_certificates == 0
                and elite_false_certificates == 0
                and identity_max_acpc <= 1e-5
                and identity_max_action <= 1e-6
                and all(
                    row["all_steps_distribution_aligned"]
                    for row in identity_adaptive
                )
            ),
        },
        "predeclared_incremental_analyses": analyses,
        "group_summary": _group_rows(joined),
        "full_budget_summary": _full_budget_summary(loaded),
    }
    return _jsonable(summary), summary["group_summary"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary, groups = summarize(args.execution_addendum)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(groups[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(groups)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    return 0 if summary["invariants"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
