#!/usr/bin/env python3
"""Freeze the two-seed extension that completes the planner panel at three seeds."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll"
)
REFERENCE_PROTOCOL = ROOT / "paper1/config/acpc_planner_stability_protocol_v2.json"
REFERENCE_EXECUTION = ROOT / "paper1/config/acpc_planner_stability_execution_v2.json"
REFERENCE_SUMMARY = ROOT / "paper1/results/acpc_planner_stability_v2/summary.json"
PROTOCOL_PATH = ROOT / "paper1/config/acpc_planner_stability_protocol_v3.json"
EXECUTION_PATH = ROOT / "paper1/config/acpc_planner_stability_execution_v3.json"
RESULT_ROOT = "paper1/results/acpc_planner_stability_v3/formal"

FORMAL_SEEDS = (3072, 3073, 3074)
EXTENSION_SEEDS = (3072, 3073)

TASKS: dict[str, dict[str, Any]] = {
    "TwoRoom": {
        "slug": "tworoom",
        "root": "lewm-tworooms",
        "dataset": "tworoom",
        "checkpoints": {
            3072: {
                "base": (
                    "tworoom_lewm_20260430/"
                    "tworoom_lewm_20260430_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "tworoom_lewm_noise_0to008_p1/"
                    "tworoom_lewm_noise_0to008_p1_epoch_10_object.ckpt"
                ),
            },
            3073: {
                "base": (
                    "tworoom_lewm_baseline_seed3073/"
                    "tworoom_lewm_baseline_seed3073_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "tworoom_lewm_noise_0to008_p1_seed3073/"
                    "tworoom_lewm_noise_0to008_p1_seed3073_epoch_10_object.ckpt"
                ),
            },
        },
    },
    "PushT": {
        "slug": "pusht",
        "root": "lewm-pusht",
        "dataset": "pusht_expert_train",
        "checkpoints": {
            3072: {
                "base": (
                    "pusht_lewm_20260430/"
                    "pusht_lewm_20260430_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "pusht_lewm_noise_0to008_p1/"
                    "pusht_lewm_noise_0to008_p1_epoch_10_object.ckpt"
                ),
            },
            3073: {
                "base": (
                    "pusht_lewm_baseline_seed3073/"
                    "pusht_lewm_baseline_seed3073_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "pusht_lewm_noise_0to008_p1_seed3073/"
                    "pusht_lewm_noise_0to008_p1_seed3073_epoch_10_object.ckpt"
                ),
            },
        },
    },
    "Reacher": {
        "slug": "reacher",
        "root": "lewm-reacher",
        "dataset": "reacher",
        "checkpoints": {
            3072: {
                "base": (
                    "reacher_lewm_20260430/"
                    "reacher_lewm_20260430_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "reacher_lewm_noise_0to008_p1/"
                    "reacher_lewm_noise_0to008_p1_epoch_10_object.ckpt"
                ),
            },
            3073: {
                "base": (
                    "reacher_lewm_baseline_seed3073/"
                    "reacher_lewm_baseline_seed3073_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "reacher_lewm_noise_0to008_p1_seed3073/"
                    "reacher_lewm_noise_0to008_p1_seed3073_epoch_10_object.ckpt"
                ),
            },
        },
    },
    "Cube": {
        "slug": "cube",
        "root": "lewm-cube",
        "dataset": "ogbench/cube_single_expert",
        "checkpoints": {
            3072: {
                "base": (
                    "cube_lewm_20260430/"
                    "cube_lewm_20260430_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "cube_lewm_noise_0to008_p1/"
                    "cube_lewm_noise_0to008_p1_epoch_10_object.ckpt"
                ),
            },
            3073: {
                "base": (
                    "cube_lewm_baseline_seed3073/"
                    "cube_lewm_baseline_seed3073_epoch_10_object.ckpt"
                ),
                "endpoint": (
                    "cube_lewm_noise_0to008_p1_seed3073/"
                    "cube_lewm_noise_0to008_p1_seed3073_epoch_10_object.ckpt"
                ),
            },
        },
    },
}

SOURCE_PATHS = (
    "paper1/scripts/freeze_acpc_planner_three_seed_protocol.py",
    "paper1/scripts/run_acpc_planner_stability_shards.py",
    "paper1/scripts/summarize_acpc_planner_three_seed.py",
    "tests/test_paper1_acpc_planner_three_seed.py",
    "tools/paper1_acpc_planner_stability_audit_v2.py",
    "tools/paper1_acpc_adaptive_cem_audit.py",
    "tools/paper1_operational_protocol.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def checkpoint_path(task: dict[str, Any], seed: int, role: str) -> Path:
    return DATA_ROOT / task["root"] / "ckpt" / task["checkpoints"][seed][role]


def fixed_arguments(
    task_name: str, task: dict[str, Any], seed: int, role: str
) -> dict[str, Any]:
    return {
        "method": "LeWM",
        "task": task_name,
        "training_seed": seed,
        "checkpoint_role": role,
        "anonymous_checkpoint_id": f"seed{seed}_{task['slug']}_{role}",
        "dataset_name": task["dataset"],
        "probe_family": "gaussian_noise",
        "severities": [0.0, 0.02, 0.05, 0.08],
        "draws": 1,
        "pool_index": 0,
        "n_blocks": 100,
        "future_steps": 9,
        "frameskip": 5,
        "img_size": 224,
        "candidate_count": 64,
        "topk": 8,
        "batch_size": 2,
        "plan_horizon": 5,
        "action_block": 5,
        "trajectory_seed": 9101,
        "probe_seed": 20260713,
        "cem_seed": 1234,
        "invariant_atol": 1e-5,
    }


def adaptive_arguments(
    task_name: str,
    task: dict[str, Any],
    seed: int,
    role: str,
    *,
    full_budget: bool,
) -> dict[str, Any]:
    return {
        "method": "LeWM",
        "task": task_name,
        "training_seed": seed,
        "checkpoint_role": role,
        "anonymous_checkpoint_id": f"seed{seed}_{task['slug']}_{role}",
        "dataset_name": task["dataset"],
        "probe_family": "gaussian_noise",
        "severities": [0.0, 0.08] if full_budget else [0.0, 0.02, 0.05, 0.08],
        "draws": 1,
        "n_blocks": 16 if full_budget else 100,
        "future_steps": 9,
        "frameskip": 5,
        "img_size": 224,
        "candidate_count": 300 if full_budget else 64,
        "topk": 30 if full_budget else 8,
        "batch_size": 1 if full_budget else 2,
        "n_steps": 30 if full_budget else 8,
        "plan_horizon": 5,
        "action_block": 5,
        "var_scale": 1.0,
        "trajectory_seed": 9101,
        "probe_seed": 20260713,
        "cem_seed": 1234,
        "first_action_tolerance": 0.10,
        "alignment_atol": 1e-6,
        "alignment_rtol": 1e-5,
    }


def authorized_shards() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoints: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for task_name, task in TASKS.items():
        for seed in EXTENSION_SEEDS:
            for role in ("base", "endpoint"):
                checkpoint = checkpoint_path(task, seed, role)
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                checkpoint_hash = sha256(checkpoint)
                checkpoints.append(
                    {
                        "task": task_name,
                        "checkpoint_role": role,
                        "training_seed": seed,
                        "path": str(checkpoint),
                        "sha256": checkpoint_hash,
                        "provenance": (
                            "existing frozen LeWM checkpoint; trained before the "
                            "three-seed planner replication protocol"
                        ),
                    }
                )
                roles = (
                    (
                        "fixed_reduced",
                        "tools/paper1_acpc_planner_stability_audit_v2.py",
                        fixed_arguments(task_name, task, seed, role),
                    ),
                    (
                        "adaptive_reduced",
                        "tools/paper1_acpc_adaptive_cem_audit.py",
                        adaptive_arguments(
                            task_name, task, seed, role, full_budget=False
                        ),
                    ),
                    (
                        "adaptive_full",
                        "tools/paper1_acpc_adaptive_cem_audit.py",
                        adaptive_arguments(
                            task_name, task, seed, role, full_budget=True
                        ),
                    ),
                )
                for analysis_role, runner, arguments in roles:
                    shard_id = (
                        f"{analysis_role}_{task['slug']}_{role}_seed{seed}"
                    )
                    shards.append(
                        {
                            "shard_id": shard_id,
                            "analysis_role": analysis_role,
                            "runner": runner,
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": checkpoint_hash,
                            "output_path": f"{RESULT_ROOT}/{shard_id}.json",
                            "arguments": arguments,
                            "timeout_seconds": (
                                7200 if analysis_role == "adaptive_full" else 3600
                            ),
                        }
                    )
    return checkpoints, shards


def validate_reference_panel() -> dict[str, Any]:
    protocol = read_json(REFERENCE_PROTOCOL)
    execution = read_json(REFERENCE_EXECUTION)
    summary = read_json(REFERENCE_SUMMARY)
    if protocol.get("status") != "frozen_pre_execution":
        raise RuntimeError("v2 reference protocol is not frozen")
    if execution.get("status") != "frozen_pre_execution":
        raise RuntimeError("v2 reference execution is not frozen")
    if execution["parent_protocol"]["sha256"] != sha256(REFERENCE_PROTOCOL):
        raise RuntimeError("v2 reference protocol hash mismatch")
    if summary.get("validated_shard_count") != 24:
        raise RuntimeError("v2 reference summary is incomplete")
    if summary.get("joined_reduced_row_count") != 3200:
        raise RuntimeError("v2 reference row count changed")
    if not summary.get("invariants", {}).get("pass"):
        raise RuntimeError("v2 reference invariants failed")
    if {
        shard["arguments"]["training_seed"]
        for shard in execution["authorized_shards"]
    } != {3074}:
        raise RuntimeError("v2 reference is not the seed3074 panel")
    return {
        "training_seed": 3074,
        "protocol": {
            "path": str(REFERENCE_PROTOCOL.relative_to(ROOT)),
            "sha256": sha256(REFERENCE_PROTOCOL),
        },
        "execution": {
            "path": str(REFERENCE_EXECUTION.relative_to(ROOT)),
            "sha256": sha256(REFERENCE_EXECUTION),
        },
        "summary": {
            "path": str(REFERENCE_SUMMARY.relative_to(ROOT)),
            "sha256": sha256(REFERENCE_SUMMARY),
        },
        "authorized_shards": 24,
        "joined_reduced_rows": 3200,
    }


def main() -> int:
    if PROTOCOL_PATH.exists() or EXECUTION_PATH.exists():
        raise SystemExit("v3 protocol already exists; refuse post-freeze overwrite")
    if (ROOT / RESULT_ROOT).exists():
        raise SystemExit("v3 result root already exists; refuse post-execution freeze")

    reference = validate_reference_panel()
    checkpoints, shards = authorized_shards()
    if len(checkpoints) != 16 or len(shards) != 48:
        raise RuntimeError("unexpected replication checkpoint or shard count")
    source_hashes = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    now = datetime.now(timezone.utc).isoformat()

    protocol: dict[str, Any] = {
        "schema_version": "paper1-acpc-planner-stability-protocol-3.0",
        "protocol_id": "paper1-acpc-planner-stability-v3",
        "status": "frozen_pre_execution",
        "immutable": True,
        "execution_authorized": True,
        "frozen_at_utc": now,
        "analysis_commit_parent": git_head(),
        "reference_seed3074_panel": reference,
        "scientific_scope": {
            "central_claim": (
                "candidate-conditioned five-step ACPC has planner-cost and "
                "decision-regret relevance beyond the frozen one-step feature set, "
                "evaluated symmetrically over three independent training seeds"
            ),
            "excluded_claims": [
                "closed-loop robustness improvement",
                "training-run population inference",
                "causal effect of ACPC",
                "cross-architecture transfer",
            ],
        },
        "provenance": {
            "extension_role": (
                "prospectively frozen exact-protocol replication on seeds 3072 and "
                "3073 after the seed3074 panel was observed"
            ),
            "reference_role": (
                "the immutable v2 seed3074 panel is reused without recomputation"
            ),
            "all_authorized_outcomes_reported": True,
            "no_outcome_dependent_retuning": True,
        },
        "frozen_panel": {
            "tasks": list(TASKS),
            "formal_training_seeds": list(FORMAL_SEEDS),
            "new_execution_seeds": list(EXTENSION_SEEDS),
            "checkpoint_roles": ["base", "endpoint"],
            "probe_family": "additional Gaussian history noise",
            "severities": [0.0, 0.02, 0.05, 0.08],
            "reduced": {
                "n_episode_blocks": 100,
                "candidate_count": 64,
                "topk": 8,
                "adaptive_steps": 8,
            },
            "full_budget": {
                "n_episode_blocks": 16,
                "severities": [0.0, 0.08],
                "candidate_count": 300,
                "topk": 30,
                "adaptive_steps": 30,
            },
            "trajectory_seed": 9101,
            "probe_seed": 20260713,
            "cem_seed": 1234,
        },
        "predeclared_analysis": {
            "join_key": [
                "training_seed",
                "task",
                "checkpoint_role",
                "trajectory_block_id",
                "severity",
                "draw_index",
            ],
            "independent_model_replication_unit": "training seed",
            "nested_measurement_units": (
                "task, checkpoint role, trajectory episode, severity, and draw"
            ),
            "incremental_model": {
                "fit_separately_within_each_training_seed": True,
                "split": "leave-one-task-out",
                "ridge_alpha": 1.0,
                "responses": [
                    "max fixed-pool absolute cost drift",
                    "adaptive first-action RMS",
                    "positive clean decision regret",
                ],
                "baseline_features": [
                    "severity",
                    "endpoint indicator",
                    "log1p candidate-H1 ACPC q90",
                    "log1p nominal top1 margin",
                ],
                "added_feature": "log1p candidate-H5 ACPC q90",
                "metric": "equal-task mean held-out log1p MAE",
                "identity_rows_excluded_from_regression": True,
            },
            "aggregation": [
                "compute held-out MAE separately for each task within each seed",
                "average the four tasks equally within each seed",
                "report all three seed-level reductions",
                "summarize seed-level reductions by mean and sample SD",
            ],
            "per_seed_strong_association_gate": (
                "at least 5% equal-task MAE reduction and improvement on at "
                "least 3 of 4 held-out tasks"
            ),
            "across_seed_reporting": {
                "directionally_consistent": "positive reduction on all 3 seeds",
                "strongly_replicated": "the per-seed gate passes on all 3 seeds",
                "partially_replicated": (
                    "the per-seed gate passes on exactly 2 seeds and all 3 seed "
                    "reductions remain positive"
                ),
                "always_report": [
                    "each seed value",
                    "three-seed mean and sample SD",
                    "positive-seed count",
                    "gate-pass count",
                    "improved task-by-seed cell count",
                ],
            },
            "full_budget_role": (
                "descriptive transfer sensitivity, summarized first within seed "
                "and then across seeds"
            ),
        },
        "invariants": {
            "identity_h5_acpc_max": 1e-5,
            "identity_first_action_rms_max": 1e-6,
            "mse_bound_violation_count": 0,
            "certificate_false_positive_count": 0,
            "identity_all_adaptive_updates_aligned": True,
        },
        "claim_adjudication": {
            "report_regardless_of_direction": True,
            "do_not_pool_candidate_rows_as_training_runs": True,
            "closed_loop_robustness_claim": "forbidden",
            "population_claim": "forbidden",
        },
        "execution": {
            "reduced_timeout_seconds": 3600,
            "full_budget_timeout_seconds": 7200,
            "max_concurrent_jobs": 4,
            "one_process_per_gpu": True,
            "native_threads_per_job": 2,
            "retry_policy": (
                "retry only exact frozen arguments after infrastructure failure"
            ),
        },
        "checkpoints": checkpoints,
        "source_hashes": source_hashes,
    }
    write_json(PROTOCOL_PATH, protocol)

    execution = {
        "schema_version": "paper1-acpc-planner-stability-execution-3.0",
        "protocol_id": protocol["protocol_id"],
        "created_utc": now,
        "parent_protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "reference_execution": reference["execution"],
        "result_root": RESULT_ROOT,
        "status": "frozen_pre_execution",
        "immutable": True,
        "post_result_protocol_edits_forbidden": True,
        "all_results_must_be_retained": True,
        "authorized_shards": shards,
        "source_hashes": {
            path: source_hashes[path]
            for path in (
                "paper1/scripts/run_acpc_planner_stability_shards.py",
                "tools/paper1_acpc_planner_stability_audit_v2.py",
                "tools/paper1_acpc_adaptive_cem_audit.py",
            )
        },
    }
    write_json(EXECUTION_PATH, execution)
    print(f"wrote {PROTOCOL_PATH.relative_to(ROOT)}")
    print(f"wrote {EXECUTION_PATH.relative_to(ROOT)}")
    print("reference shards: 24")
    print(f"new authorized shards: {len(shards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
