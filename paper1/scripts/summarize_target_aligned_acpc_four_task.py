#!/usr/bin/env python3
"""Meta-summarize target-aligned ACPC evidence without pooling task scales.

Inputs are artifacts produced by summarize_target_aligned_acpc_mve.py. Each
task keeps its own grouped-CV fit; this script meta-analyzes the frozen pass
decision and relative MAE reductions with equal task weight. It never refits
a model or drops a task.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "paper1-target-aligned-acpc-four-task-meta-0.3"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
PRIMARY_MODELS = (
    "plus_correct_h8",
    "plus_correct_h1",
    "plus_action_zero_h8_control",
    "plus_candidate_shuffle_h8_control",
    "plus_time_shuffle_h8_control",
)
PRIMARY_TARGETS = {
    "absolute": "correct_absolute_h8_error_drift",
    "adverse": "correct_adverse_h8_error",
}
GATES = {
    "candidate": {
        "signed": "signed_degradation_original_dev_gate",
        "adverse": "adverse_degradation_semantic_reanalysis",
        "absolute": "absolute_error_drift_ranking_secondary",
    },
    "logged": {
        "signed": "correct_excess_h8_error",
        "adverse": "correct_adverse_h8_error",
        "absolute": "correct_absolute_h8_error_drift",
    },
}


def _checkpoint_value(source: str | Mapping[str, Any]) -> str:
    if isinstance(source, Mapping):
        return str(source["checkpoint"])
    return str(source)


def _training_seed(source: str | Mapping[str, Any]) -> int:
    if isinstance(source, Mapping) and source.get("training_seed") is not None:
        return int(source["training_seed"])
    checkpoint = _checkpoint_value(source)
    match = re.search(r"seed(\d+)", checkpoint)
    if match is None:
        raise ValueError(f"checkpoint has no training seed: {checkpoint}")
    return int(match.group(1))


def _role(source: str | Mapping[str, Any]) -> str:
    if isinstance(source, Mapping) and source.get("checkpoint_role"):
        return str(source["checkpoint_role"])
    checkpoint = _checkpoint_value(source)
    if "baseline" in checkpoint:
        return "base"
    if "noise_0to008" in checkpoint:
        return "endpoint"
    return "other"


def _gate_record(
    *,
    artifact: dict[str, Any],
    track: str,
    target: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": artifact["source"],
        "task": str(artifact["task"]),
        "training_seed": _training_seed(artifact),
        "role": _role(artifact),
        "track": track,
        "target": target,
        "pass": bool(gate["pass"]),
        "relative_mae_reduction_vs_h1": float(
            gate["relative_mae_reduction"]["versus_shallow"]
        ),
        "relative_mae_reduction_vs_best_destroyed": float(
            gate["relative_mae_reduction"]["versus_best_destroyed"]
        ),
        "both_win_blocks": int(
            gate["block_direction"]["both_win_count"]
        ),
        "block_count": int(
            gate["block_direction"]["paired_group_count"]
        ),
        "best_destroyed_name": str(gate["best_destroyed_name"]),
        "correct_mae": float(gate["mae"]["correct"]),
        "h1_mae": float(gate["mae"]["shallow"]),
        "best_destroyed_mae": float(gate["mae"]["best_destroyed"]),
    }


def _leave_one_task_out(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for omitted in TASKS:
        kept = [row for row in records if row["task"] != omitted]
        if not kept:
            continue
        results.append(
            {
                "omitted_task": omitted,
                "task_count": len(kept),
                "pass_count": sum(bool(row["pass"]) for row in kept),
                "mean_reduction_vs_h1": float(
                    np.mean(
                        [
                            row["relative_mae_reduction_vs_h1"]
                            for row in kept
                        ]
                    )
                ),
                "mean_reduction_vs_best_destroyed": float(
                    np.mean(
                        [
                            row[
                                "relative_mae_reduction_vs_best_destroyed"
                            ]
                            for row in kept
                        ]
                    )
                ),
            }
        )
    return results


def _aggregate_cell(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {row["task"]: row for row in records}
    missing = sorted(set(TASKS) - set(by_task))
    pass_tasks = sorted(task for task, row in by_task.items() if row["pass"])
    fail_tasks = sorted(task for task, row in by_task.items() if not row["pass"])
    reductions_h1 = [
        row["relative_mae_reduction_vs_h1"] for row in records
    ]
    reductions_control = [
        row["relative_mae_reduction_vs_best_destroyed"]
        for row in records
    ]
    return {
        "task_count": len(by_task),
        "missing_tasks": missing,
        "pass_count": len(pass_tasks),
        "pass_tasks": pass_tasks,
        "fail_tasks": fail_tasks,
        "minimum_three_of_four_tasks_pass": (
            not missing and len(pass_tasks) >= 3
        ),
        "all_four_tasks_pass": not missing and len(pass_tasks) == 4,
        "equal_task_mean_reduction_vs_h1": float(
            np.mean(reductions_h1)
        ),
        "equal_task_minimum_reduction_vs_h1": float(
            np.min(reductions_h1)
        ),
        "equal_task_mean_reduction_vs_best_destroyed": float(
            np.mean(reductions_control)
        ),
        "equal_task_minimum_reduction_vs_best_destroyed": float(
            np.min(reductions_control)
        ),
        "leave_one_task_out": _leave_one_task_out(records),
        "per_task": [by_task[task] for task in TASKS if task in by_task],
    }


def _per_group_mae(model: dict[str, Any]) -> dict[int, float]:
    return {
        int(row["trajectory_block_index"]): float(row["mae"])
        for row in model["per_group"]
    }


def _primary_block_arrays(
    artifacts: dict[str, dict[str, Any]],
    *,
    target: str,
) -> dict[str, dict[int, dict[str, dict[int, float]]]]:
    target_name = PRIMARY_TARGETS[target]
    arrays: dict[str, dict[int, dict[str, dict[int, float]]]] = defaultdict(dict)
    for artifact in artifacts.values():
        if _role(artifact) != "base":
            continue
        task = str(artifact["task"])
        seed = _training_seed(artifact)
        models = artifact["logged_grouped_cv"][target_name]
        arrays[task][seed] = {
            name: _per_group_mae(models[name]) for name in PRIMARY_MODELS
        }
    missing_tasks = sorted(set(TASKS) - set(arrays))
    if missing_tasks:
        raise ValueError(f"primary uncertainty missing tasks: {missing_tasks}")
    seed_sets = {task: set(arrays[task]) for task in TASKS}
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        raise ValueError(f"primary uncertainty seed mismatch: {seed_sets}")
    for task in TASKS:
        for seed, models in arrays[task].items():
            block_sets = {name: set(rows) for name, rows in models.items()}
            if len({tuple(sorted(blocks)) for blocks in block_sets.values()}) != 1:
                raise ValueError(
                    f"primary uncertainty block mismatch: {task}/{seed}"
                )
    return arrays


def _relative_reductions(
    arrays: dict[str, dict[int, dict[str, dict[int, float]]]],
    sampled_blocks: dict[str, np.ndarray],
) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    controls = PRIMARY_MODELS[2:]
    for task in TASKS:
        blocks = sampled_blocks[task]
        seeds = sorted(arrays[task])

        def mean_mae(model: str) -> float:
            values = [
                arrays[task][seed][model][int(block)]
                for seed in seeds
                for block in blocks
            ]
            return float(np.mean(values))

        correct = mean_mae("plus_correct_h8")
        h1 = mean_mae("plus_correct_h1")
        control_maes = {name: mean_mae(name) for name in controls}
        best_control_name = min(control_maes, key=control_maes.get)
        best_control = control_maes[best_control_name]
        task_rows.append(
            {
                "task": task,
                "correct_mae": correct,
                "h1_mae": h1,
                "best_destroyed_name": best_control_name,
                "best_destroyed_mae": best_control,
                "relative_reduction_vs_h1": (h1 - correct) / max(h1, 1e-12),
                "relative_reduction_vs_best_destroyed": (
                    best_control - correct
                )
                / max(best_control, 1e-12),
            }
        )
    return {
        "per_task": task_rows,
        "equal_task_mean_reduction_vs_h1": float(
            np.mean([row["relative_reduction_vs_h1"] for row in task_rows])
        ),
        "equal_task_mean_reduction_vs_best_destroyed": float(
            np.mean(
                [
                    row["relative_reduction_vs_best_destroyed"]
                    for row in task_rows
                ]
            )
        ),
    }


def _one_sided_sign_p_value(successes: int, trials: int) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid sign-test counts")
    return float(
        sum(math.comb(trials, k) for k in range(successes, trials + 1))
        / (2**trials)
    )


def _primary_uncertainty(
    artifacts: dict[str, dict[str, Any]],
    *,
    target: str,
    repetitions: int = 5000,
    seed: int = 20260713,
) -> dict[str, Any]:
    arrays = _primary_block_arrays(artifacts, target=target)
    training_seeds = sorted(arrays[TASKS[0]])
    known_provenance = {
        3072: "retrospective completeness check",
        3073: "development-era held-out model seed",
        3074: "protocol-frozen replication",
        3075: "prospectively frozen training seed",
    }
    provenance = {
        str(training_seed): known_provenance.get(
            training_seed, "reported model seed"
        )
        for training_seed in training_seeds
    }
    canonical_blocks: dict[str, np.ndarray] = {}
    for task in TASKS:
        first_seed = min(arrays[task])
        canonical_blocks[task] = np.asarray(
            sorted(arrays[task][first_seed][PRIMARY_MODELS[0]]), dtype=int
        )
    observed = _relative_reductions(arrays, canonical_blocks)

    rng = np.random.default_rng(seed)
    samples_h1: list[float] = []
    samples_control: list[float] = []
    for _ in range(repetitions):
        sampled = {
            task: rng.choice(blocks, size=len(blocks), replace=True)
            for task, blocks in canonical_blocks.items()
        }
        value = _relative_reductions(arrays, sampled)
        samples_h1.append(value["equal_task_mean_reduction_vs_h1"])
        samples_control.append(
            value["equal_task_mean_reduction_vs_best_destroyed"]
        )

    both_win_count = 0
    total_clusters = 0
    for task in TASKS:
        seeds = sorted(arrays[task])
        for block in canonical_blocks[task]:
            correct = float(
                np.mean(
                    [
                        arrays[task][training_seed]["plus_correct_h8"][
                            int(block)
                        ]
                        for training_seed in seeds
                    ]
                )
            )
            competitors = [
                float(
                    np.mean(
                        [
                            arrays[task][training_seed][model][int(block)]
                            for training_seed in seeds
                        ]
                    )
                )
                for model in PRIMARY_MODELS[1:]
            ]
            both_win_count += int(correct < min(competitors))
            total_clusters += 1

    return {
        "target": target,
        "observed": observed,
        "cluster_bootstrap": {
            "repetitions": repetitions,
            "seed": seed,
            "cluster": (
                "task x trajectory-block index; all listed training seeds "
                "are retained inside each resampled cluster"
            ),
            "equal_task_mean_reduction_vs_h1_ci95": [
                float(np.quantile(samples_h1, 0.025)),
                float(np.quantile(samples_h1, 0.975)),
            ],
            "equal_task_mean_reduction_vs_best_destroyed_ci95": [
                float(np.quantile(samples_control, 0.025)),
                float(np.quantile(samples_control, 0.975)),
            ],
            "training_seeds": training_seeds,
            "training_seed_provenance": provenance,
            "training_seed_scope": (
                "conditional on the listed model seeds; the trajectory bootstrap "
                "is not a population CI over model-training randomness"
            ),
        },
        "paired_block_direction": {
            "both_win_count": both_win_count,
            "cluster_count": total_clusters,
            "one_sided_exact_sign_p_value": _one_sided_sign_p_value(
                both_win_count, total_clusters
            ),
            "criterion": (
                "seed-averaged correct H8 block MAE is below H1 and every "
                "destroyed-H8 control"
            ),
        },
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for artifact in payload["artifacts"]:
            source = str(artifact["source"])
            if source in artifacts:
                raise ValueError(f"duplicate source artifact: {source}")
            artifacts[source] = artifact

    records: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    for artifact in artifacts.values():
        for target, key in GATES["candidate"].items():
            records.append(
                _gate_record(
                    artifact=artifact,
                    track="candidate",
                    target=target,
                    gate=artifact["candidate_gates"][key],
                )
            )
        for target, key in GATES["logged"].items():
            records.append(
                _gate_record(
                    artifact=artifact,
                    track="logged",
                    target=target,
                    gate=artifact["logged_gates"][key],
                )
            )
        certificate_rows.append(
            {
                "task": artifact["task"],
                "training_seed": _training_seed(artifact),
                "role": _role(artifact),
                "candidate_minimum_h5_slack": float(
                    artifact["candidate_correct_action_certificate"][
                        "minimum_h5_slack"
                    ]
                ),
                "logged_minimum_h8_slack": float(
                    artifact["logged_correct_action_certificate"][
                        "minimum_h8_slack"
                    ]
                ),
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                row["training_seed"],
                row["role"],
                row["track"],
                row["target"],
            )
        ].append(row)
    cells = [
        {
            "training_seed": key[0],
            "role": key[1],
            "track": key[2],
            "target": key[3],
            **_aggregate_cell(rows),
        }
        for key, rows in sorted(grouped.items())
    ]

    primary = [
        cell
        for cell in cells
        if cell["role"] == "base"
        and cell["track"] == "logged"
        and cell["target"] in {"adverse", "absolute"}
    ]
    candidate_bridge = [
        cell
        for cell in cells
        if cell["role"] == "base"
        and cell["track"] == "candidate"
        and cell["target"] in {"adverse", "absolute"}
    ]
    return {
        "metadata": {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task_scale_pooling": False,
            "equal_task_weight": True,
            "minimum_tasks_for_empirical_increment": 3,
            "all_tasks_reported": True,
        },
        "certificate": {
            "rows": certificate_rows,
            "candidate_zero_violation_all_rows": all(
                row["candidate_minimum_h5_slack"] >= -1e-5
                for row in certificate_rows
            ),
            "logged_zero_violation_all_rows": all(
                row["logged_minimum_h8_slack"] >= -1e-5
                for row in certificate_rows
            ),
        },
        "cells": cells,
        "primary_logged_fragile_base": {
            "cells": primary,
            "uncertainty": {
                target: _primary_uncertainty(artifacts, target=target)
                for target in ("absolute", "adverse")
            },
            "all_available_seeds_meet_three_task_gate": bool(primary)
            and all(
                cell["minimum_three_of_four_tasks_pass"]
                for cell in primary
            ),
            "all_available_seeds_pass_all_four_tasks": bool(primary)
            and all(cell["all_four_tasks_pass"] for cell in primary),
        },
        "candidate_fragile_base_bridge": {
            "cells": candidate_bridge,
            "all_available_seeds_meet_three_task_gate": bool(
                candidate_bridge
            )
            and all(
                cell["minimum_three_of_four_tasks_pass"]
                for cell in candidate_bridge
            ),
            "boundary": (
                "candidate bridge is secondary; failure does not replace "
                "the four-task logged-future result"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "logged_base": {
                    "all_available_seeds_meet_three_task_gate": result[
                        "primary_logged_fragile_base"
                    ]["all_available_seeds_meet_three_task_gate"],
                    "all_available_seeds_pass_all_four_tasks": result[
                        "primary_logged_fragile_base"
                    ]["all_available_seeds_pass_all_four_tasks"],
                    "cells": [
                        {
                            "seed": cell["training_seed"],
                            "target": cell["target"],
                            "pass_tasks": cell["pass_tasks"],
                            "fail_tasks": cell["fail_tasks"],
                        }
                        for cell in result[
                            "primary_logged_fragile_base"
                        ]["cells"]
                    ],
                },
                "candidate_base": [
                    {
                        "seed": cell["training_seed"],
                        "target": cell["target"],
                        "pass_tasks": cell["pass_tasks"],
                        "fail_tasks": cell["fail_tasks"],
                    }
                    for cell in result["candidate_fragile_base_bridge"][
                        "cells"
                    ]
                ],
                "certificate": result["certificate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
