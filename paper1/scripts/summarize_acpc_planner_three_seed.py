#!/usr/bin/env python3
"""Validate and summarize the three-seed ACPC planner-stability panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from paper1.scripts.summarize_acpc_planner_stability import (
    _lobo_ridge,
    _task_spearman,
    _validate_results,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_SEEDS = (3072, 3073, 3074)
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")


def _sha256(path: Path) -> str:
    import hashlib

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


def _sample_sd(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


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


def _validate_results_strict(
    addendum_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    addendum, loaded = _validate_results(addendum_path)
    addendum_hash = _sha256(addendum_path)
    protocol_path = ROOT / addendum["parent_protocol"]["path"]
    protocol_hash = _sha256(protocol_path)
    for item in loaded:
        shard = item["shard"]
        metadata = item["payload"]["metadata"]
        if metadata.get("execution_addendum_sha256") != addendum_hash:
            raise RuntimeError(
                f"execution-addendum mismatch: {shard['shard_id']}"
            )
        if metadata.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"protocol mismatch: {shard['shard_id']}")
        if int(metadata.get("training_seed", -1)) != int(
            shard["arguments"]["training_seed"]
        ):
            raise RuntimeError(f"training-seed mismatch: {shard['shard_id']}")
    return addendum, loaded


def _validate_three_seed_contract(
    protocol_path: Path,
    extension_addendum_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = _load_json(protocol_path)
    if protocol.get("status") != "frozen_pre_execution":
        raise RuntimeError("three-seed protocol is not frozen_pre_execution")
    if protocol.get("immutable") is not True:
        raise RuntimeError("three-seed protocol is not immutable")
    if tuple(protocol["frozen_panel"]["formal_training_seeds"]) != FORMAL_SEEDS:
        raise RuntimeError("formal training-seed set changed")

    extension = _load_json(extension_addendum_path)
    if extension["parent_protocol"]["path"] != str(
        protocol_path.resolve().relative_to(ROOT.resolve())
    ):
        raise RuntimeError("extension parent-protocol path mismatch")
    if extension["parent_protocol"]["sha256"] != _sha256(protocol_path):
        raise RuntimeError("extension parent-protocol hash mismatch")

    reference = protocol["reference_seed3074_panel"]
    reference_protocol = ROOT / reference["protocol"]["path"]
    reference_execution = ROOT / reference["execution"]["path"]
    reference_summary = ROOT / reference["summary"]["path"]
    for path, record in (
        (reference_protocol, reference["protocol"]),
        (reference_execution, reference["execution"]),
        (reference_summary, reference["summary"]),
    ):
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"bound reference changed: {path}")

    _, reference_loaded = _validate_results_strict(reference_execution)
    _, extension_loaded = _validate_results_strict(extension_addendum_path)
    loaded = [*reference_loaded, *extension_loaded]
    role_seed_counts = Counter(
        (
            item["shard"]["analysis_role"],
            int(item["shard"]["arguments"]["training_seed"]),
        )
        for item in loaded
    )
    expected_counts = Counter(
        (role, seed)
        for seed in FORMAL_SEEDS
        for role in ("fixed_reduced", "adaptive_reduced", "adaptive_full")
        for _ in range(8)
    )
    if role_seed_counts != expected_counts:
        raise RuntimeError(
            f"unexpected role/seed shard counts: {role_seed_counts}"
        )
    return loaded, {
        "reference_shards": len(reference_loaded),
        "extension_shards": len(extension_loaded),
        "reference_execution_sha256": _sha256(reference_execution),
        "extension_execution_sha256": _sha256(extension_addendum_path),
    }


def _join_reduced(
    loaded: Sequence[Mapping[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    fixed: dict[tuple[Any, ...], dict[str, Any]] = {}
    adaptive: dict[tuple[Any, ...], dict[str, Any]] = {}

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(row["training_seed"]),
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
        for raw_row in item["payload"]["rows"]:
            row = dict(raw_row)
            row_key = key(row)
            if row_key in target:
                raise RuntimeError(f"duplicate reduced row: {row_key}")
            target[row_key] = row
    if set(fixed) != set(adaptive):
        raise RuntimeError(
            "fixed/adaptive join mismatch: "
            f"missing_fixed={len(set(adaptive) - set(fixed))} "
            f"missing_adaptive={len(set(fixed) - set(adaptive))}"
        )
    return [
        {"fixed": fixed[row_key], "adaptive": adaptive[row_key]}
        for row_key in sorted(fixed, key=str)
    ]


def _design_row(pair: Mapping[str, Mapping[str, Any]]) -> dict[str, float | str]:
    fixed = pair["fixed"]
    adaptive = pair["adaptive"]
    if int(fixed["training_seed"]) != int(adaptive["training_seed"]):
        raise RuntimeError("joined rows disagree on training seed")
    return {
        "training_seed": int(fixed["training_seed"]),
        "task": str(fixed["task"]),
        "checkpoint_role": str(fixed["checkpoint_role"]),
        "severity": float(fixed["severity"]),
        "h1": float(fixed["q90_candidate_h1_acpc_l2"]),
        "h5": float(fixed["q90_cost_space_acpc_final_l2"]),
        "margin": float(fixed["nominal_margin"]),
        "cost_drift": float(fixed["max_absolute_cost_drift"]),
        "first_action_rms": float(adaptive["first_action_rms"]),
        "positive_clean_regret": float(
            adaptive["positive_clean_decision_regret"]
        ),
    }


def _three_seed_incremental_analyses(
    design_rows: Sequence[Mapping[str, float | str]],
) -> dict[str, Any]:
    if {int(row["training_seed"]) for row in design_rows} != set(FORMAL_SEEDS):
        raise RuntimeError("design rows do not contain the formal seed set")
    analyses: dict[str, Any] = {}
    for response in (
        "cost_drift",
        "first_action_rms",
        "positive_clean_regret",
    ):
        per_seed: dict[str, Any] = {}
        for seed in FORMAL_SEEDS:
            seed_rows = [
                row for row in design_rows if int(row["training_seed"]) == seed
            ]
            lobo = _lobo_ridge(seed_rows, response=response)
            spearman = _task_spearman(seed_rows, response)
            gate = bool(
                lobo["equal_task_relative_mae_reduction"] >= 0.05
                and lobo["tasks_improved"] >= 3
            )
            per_seed[str(seed)] = {
                "lobo_ridge": lobo,
                "spearman": spearman,
                "strong_association_gate": gate,
            }

        reductions = [
            per_seed[str(seed)]["lobo_ridge"][
                "equal_task_relative_mae_reduction"
            ]
            for seed in FORMAL_SEEDS
        ]
        h1_spearman = [
            per_seed[str(seed)]["spearman"]["equal_task_h1"]
            for seed in FORMAL_SEEDS
        ]
        h5_spearman = [
            per_seed[str(seed)]["spearman"]["equal_task_h5"]
            for seed in FORMAL_SEEDS
        ]
        task_cells_improved = sum(
            per_seed[str(seed)]["lobo_ridge"]["tasks_improved"]
            for seed in FORMAL_SEEDS
        )
        positive_seed_count = sum(value > 0.0 for value in reductions)
        gate_pass_count = sum(
            per_seed[str(seed)]["strong_association_gate"]
            for seed in FORMAL_SEEDS
        )
        three_seed = {
            "relative_mae_reduction_mean": _mean(reductions),
            "relative_mae_reduction_sample_sd": _sample_sd(reductions),
            "relative_mae_reduction_min": min(reductions),
            "relative_mae_reduction_max": max(reductions),
            "positive_seed_count": positive_seed_count,
            "seed_count": len(FORMAL_SEEDS),
            "gate_pass_count": gate_pass_count,
            "task_seed_cells_improved": task_cells_improved,
            "task_seed_cell_count": len(FORMAL_SEEDS) * len(TASKS),
            "directionally_consistent": positive_seed_count == len(FORMAL_SEEDS),
            "strongly_replicated": gate_pass_count == len(FORMAL_SEEDS),
            "partially_replicated": (
                gate_pass_count == 2 and positive_seed_count == len(FORMAL_SEEDS)
            ),
            "equal_task_h1_spearman_mean": _mean(h1_spearman),
            "equal_task_h1_spearman_sample_sd": _sample_sd(h1_spearman),
            "equal_task_h5_spearman_mean": _mean(h5_spearman),
            "equal_task_h5_spearman_sample_sd": _sample_sd(h5_spearman),
        }
        analyses[response] = {
            "per_seed": per_seed,
            "three_seed_summary": three_seed,
            # Compatibility keys are explicitly seed-macro summaries, not a
            # regression fitted to pooled candidate rows.
            "lobo_ridge": {
                "aggregation": "mean of three independently fit seed-level analyses",
                "equal_task_relative_mae_reduction": three_seed[
                    "relative_mae_reduction_mean"
                ],
                "relative_mae_reduction_sample_sd": three_seed[
                    "relative_mae_reduction_sample_sd"
                ],
                "tasks_improved": task_cells_improved,
                "task_count": len(FORMAL_SEEDS) * len(TASKS),
                "positive_seed_count": positive_seed_count,
                "gate_pass_count": gate_pass_count,
            },
            "spearman": {
                "aggregation": "mean of seed-level equal-task correlations",
                "equal_task_h1": three_seed["equal_task_h1_spearman_mean"],
                "equal_task_h5": three_seed["equal_task_h5_spearman_mean"],
                "equal_task_h1_sample_sd": three_seed[
                    "equal_task_h1_spearman_sample_sd"
                ],
                "equal_task_h5_sample_sd": three_seed[
                    "equal_task_h5_spearman_sample_sd"
                ],
            },
        }
    return analyses


def _group_seed_rows(
    joined: Sequence[Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[int, str, str, float],
        list[Mapping[str, Mapping[str, Any]]],
    ] = defaultdict(list)
    for pair in joined:
        fixed = pair["fixed"]
        groups[
            (
                int(fixed["training_seed"]),
                str(fixed["task"]),
                str(fixed["checkpoint_role"]),
                float(fixed["severity"]),
            )
        ].append(pair)

    output = []
    for (seed, task, role, severity), pairs in sorted(groups.items()):
        fixed = [pair["fixed"] for pair in pairs]
        adaptive = [pair["adaptive"] for pair in pairs]
        output.append(
            {
                "training_seed": seed,
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


def _aggregate_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    output = []
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        if {int(row["training_seed"]) for row in selected} != set(FORMAL_SEEDS):
            raise RuntimeError(f"aggregate group lacks three seeds: {key}")
        record = dict(zip(key_fields, key))
        record["seed_count"] = len(selected)
        if "n" in selected[0]:
            record["n_per_seed"] = int(selected[0]["n"])
        numeric_fields = [
            field
            for field, value in selected[0].items()
            if field not in {*key_fields, "training_seed", "n"}
            and isinstance(value, (int, float, bool))
        ]
        for field in numeric_fields:
            values = [float(row[field]) for row in selected]
            record[field] = _mean(values)
            record[f"{field}_sample_sd"] = _sample_sd(values)
        output.append(record)
    return output


def _full_budget_seed_summary(
    loaded: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for item in loaded:
        if item["shard"]["analysis_role"] != "adaptive_full":
            continue
        rows = [row for row in item["payload"]["rows"] if row["severity"] > 0.0]
        arguments = item["shard"]["arguments"]
        output.append(
            {
                "training_seed": int(arguments["training_seed"]),
                "task": arguments["task"],
                "checkpoint_role": arguments["checkpoint_role"],
                "n": len(rows),
                "candidate_count": arguments["candidate_count"],
                "n_steps": arguments["n_steps"],
                "first_action_rms_mean": _mean(
                    row["first_action_rms"] for row in rows
                ),
                "first_action_stability_rate": _mean(
                    row["first_action_stable"] for row in rows
                ),
                "positive_clean_regret_mean": _mean(
                    row["positive_clean_decision_regret"] for row in rows
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["training_seed"],
            row["task"],
            row["checkpoint_role"],
        ),
    )


def summarize(
    protocol_path: Path,
    extension_addendum_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _load_json(protocol_path)
    loaded, provenance = _validate_three_seed_contract(
        protocol_path, extension_addendum_path
    )
    joined = _join_reduced(loaded)
    if len(joined) != 9600:
        raise RuntimeError(f"expected 9,600 joined rows, found {len(joined)}")
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
        row["max_cost_space_acpc_final_l2"] for row in identity_fixed
    )
    identity_max_action = max(row["first_action_rms"] for row in identity_adaptive)
    identity_updates_aligned = all(
        row["all_steps_distribution_aligned"] for row in identity_adaptive
    )
    invariants_pass = bool(
        bound_violations == 0
        and top1_false_certificates == 0
        and elite_false_certificates == 0
        and identity_max_acpc <= 1e-5
        and identity_max_action <= 1e-6
        and identity_updates_aligned
    )

    group_seed_summary = _group_seed_rows(joined)
    group_summary = _aggregate_seed_rows(
        group_seed_summary,
        key_fields=("task", "checkpoint_role", "severity"),
    )
    full_seed_summary = _full_budget_seed_summary(loaded)
    full_summary = _aggregate_seed_rows(
        full_seed_summary,
        key_fields=("task", "checkpoint_role"),
    )
    analyses = _three_seed_incremental_analyses(design_rows)

    summary = {
        "schema_version": "paper1-acpc-planner-stability-summary-2.0",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(protocol_path),
        "training_seeds": list(FORMAL_SEEDS),
        "reference_shard_count": provenance["reference_shards"],
        "extension_shard_count": provenance["extension_shards"],
        "authorized_shard_count": len(loaded),
        "validated_shard_count": len(loaded),
        "joined_reduced_row_count": len(joined),
        "reference_execution_sha256": provenance["reference_execution_sha256"],
        "extension_execution_sha256": provenance["extension_execution_sha256"],
        "invariants": {
            "mse_cost_bound_violation_count": bound_violations,
            "top1_false_certificate_count": top1_false_certificates,
            "elite_false_certificate_count": elite_false_certificates,
            "identity_max_h5_acpc": identity_max_acpc,
            "identity_max_first_action_rms": identity_max_action,
            "identity_all_adaptive_updates_aligned": identity_updates_aligned,
            "pass": invariants_pass,
        },
        "predeclared_incremental_analyses": analyses,
        "group_seed_summary": group_seed_summary,
        "group_summary": group_summary,
        "full_budget_seed_summary": full_seed_summary,
        "full_budget_summary": full_summary,
    }
    return _jsonable(summary), group_seed_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary, groups = summarize(args.protocol, args.execution_addendum)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
