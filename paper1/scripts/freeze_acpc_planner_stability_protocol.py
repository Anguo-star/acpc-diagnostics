#!/usr/bin/env python3
"""Freeze the Paper 1 ACPC-to-planner protocol and executable shard manifest."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _data_root_from_env() -> Path | None:
    for name in ("PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


DATA_ROOT = _data_root_from_env()
PROTOCOL_PATH = ROOT / "paper1/config/acpc_planner_stability_protocol_v1.json"
ADDENDUM_PATH = ROOT / "paper1/config/acpc_planner_stability_execution_v1.json"
RESULT_ROOT = "paper1/results/acpc_planner_stability_v1/formal"


TASKS = {
    "TwoRoom": {
        "slug": "tworoom",
        "root": "lewm-tworooms",
        "dataset": "tworoom",
        "base": (
            "tworoom_lewm_baseline_seed3074/"
            "tworoom_lewm_baseline_seed3074_epoch_10_object.ckpt"
        ),
        "endpoint": (
            "tworoom_lewm_noise_0to008_p1_seed3074/"
            "tworoom_lewm_noise_0to008_p1_seed3074_epoch_10_object.ckpt"
        ),
    },
    "PushT": {
        "slug": "pusht",
        "root": "lewm-pusht",
        "dataset": "pusht_expert_train",
        "base": (
            "pusht_lewm_baseline_seed3074/"
            "pusht_lewm_baseline_seed3074_epoch_10_object.ckpt"
        ),
        "endpoint": (
            "pusht_lewm_noise_0to008_p1_seed3074/"
            "pusht_lewm_noise_0to008_p1_seed3074_epoch_10_object.ckpt"
        ),
    },
    "Reacher": {
        "slug": "reacher",
        "root": "lewm-reacher",
        "dataset": "reacher",
        "base": (
            "reacher_lewm_baseline_seed3074/"
            "reacher_lewm_baseline_seed3074_epoch_10_object.ckpt"
        ),
        "endpoint": (
            "reacher_lewm_noise_0to008_p1_seed3074/"
            "reacher_lewm_noise_0to008_p1_seed3074_epoch_10_object.ckpt"
        ),
    },
    "Cube": {
        "slug": "cube",
        "root": "lewm-cube",
        "dataset": "ogbench/cube_single_expert",
        "base": (
            "cube_lewm_baseline_seed3074/"
            "cube_lewm_baseline_seed3074_epoch_10_object.ckpt"
        ),
        "endpoint": (
            "cube_lewm_noise_0to008_p1_seed3074/"
            "cube_lewm_noise_0to008_p1_seed3074_epoch_10_object.ckpt"
        ),
    },
}

SOURCE_PATHS = [
    "paper1/docs/PAPER1_ACPC_SCIENTIFIC_REMEDIATION_PLAN_20260713.md",
    "paper1/scripts/freeze_acpc_planner_stability_protocol.py",
    "paper1/scripts/summarize_acpc_planner_stability.py",
    "tools/paper1_acpc_planner_stability_audit.py",
    "tools/paper1_acpc_adaptive_cem_audit.py",
    "tools/paper1_operational_protocol.py",
    "tools/paper1_phase0_acpc.py",
    "tools/paper1_cem_trace_audit.py",
    "tests/test_paper1_acpc_planner_stability.py",
    "tests/test_paper1_acpc_adaptive_cem.py",
    "tests/test_paper1_acpc_planner_summary.py",
    "tests/test_paper1_acpc_frozen_protocols.py",
    "jepa.py",
    "utils.py",
    "config/eval/solver/cem.yaml",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def checkpoint_path(task: dict[str, str], role: str) -> Path:
    if DATA_ROOT is None:
        raise RuntimeError(
            "set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root"
        )
    return DATA_ROOT / task["root"] / "ckpt" / task[role]


def fixed_arguments(task_name: str, task: dict[str, str], role: str) -> dict[str, Any]:
    return {
        "method": "LeWM",
        "task": task_name,
        "training_seed": 3074,
        "checkpoint_role": role,
        "anonymous_checkpoint_id": f"seed3074_{task['slug']}_{role}",
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
    task: dict[str, str],
    role: str,
    *,
    full_budget: bool,
) -> dict[str, Any]:
    return {
        "method": "LeWM",
        "task": task_name,
        "training_seed": 3074,
        "checkpoint_role": role,
        "anonymous_checkpoint_id": f"seed3074_{task['slug']}_{role}",
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
    checkpoints = []
    shards = []
    for task_name, task in TASKS.items():
        for role in ("base", "endpoint"):
            checkpoint = checkpoint_path(task, role)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_hash = sha256(checkpoint)
            checkpoint_record = {
                "task": task_name,
                "checkpoint_role": role,
                "training_seed": 3074,
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "provenance": (
                    "existing seed3074 checkpoint; model behavior predates this "
                    "planner-stability freeze"
                ),
            }
            checkpoints.append(checkpoint_record)
            for analysis_role, runner, arguments in (
                (
                    "fixed_reduced",
                    "tools/paper1_acpc_planner_stability_audit.py",
                    fixed_arguments(task_name, task, role),
                ),
                (
                    "adaptive_reduced",
                    "tools/paper1_acpc_adaptive_cem_audit.py",
                    adaptive_arguments(task_name, task, role, full_budget=False),
                ),
                (
                    "adaptive_full",
                    "tools/paper1_acpc_adaptive_cem_audit.py",
                    adaptive_arguments(task_name, task, role, full_budget=True),
                ),
            ):
                shard_id = f"{analysis_role}_{task['slug']}_{role}_seed3074"
                shards.append(
                    {
                        "shard_id": shard_id,
                        "analysis_role": analysis_role,
                        "runner": runner,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "output_path": f"{RESULT_ROOT}/{shard_id}.json",
                        "arguments": arguments,
                        "timeout_seconds": 7200 if analysis_role == "adaptive_full" else 3600,
                    }
                )
    return checkpoints, shards


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_sidecar(path: Path) -> None:
    relative = path.relative_to(ROOT)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {relative}\n", encoding="utf-8"
    )


def main() -> int:
    if DATA_ROOT is None:
        raise SystemExit(
            "set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root"
        )
    frozen_at = datetime.now(timezone.utc).isoformat()
    source_hashes = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    checkpoints, shards = authorized_shards()
    protocol = {
        "schema_version": "paper1-acpc-planner-stability-protocol-1.0",
        "protocol_id": "paper1-acpc-planner-stability-v1",
        "status": "frozen_pre_execution",
        "immutable": True,
        "execution_authorized": True,
        "frozen_at_utc": frozen_at,
        "analysis_commit_parent": git_head(),
        "scientific_scope": {
            "central_claim": (
                "action-matched long-horizon ACPC adds held-out future-drift "
                "information beyond a same-pool H1 comparator and links that "
                "information to planner cost and decision stability"
            ),
            "eligible_outputs": [
                "exact ACPC-to-squared-distance cost bound",
                "top-1 and elite-set margin certificates",
                "same-pool candidate-conditioned H1 versus H5 ACPC",
                "adaptive-CEM first-action RMS and clean decision regret",
                "K300 x 30-step transfer sensitivity",
            ],
            "excluded_side_branches": [
                "failed repair attempts",
                "PLDM side branch",
                "JVP/local-sensitivity branch",
                "raw full-sweep row dump",
                "behavior claims",
            ],
        },
        "provenance": {
            "seed3074_role": (
                "protocol-frozen replication for the earlier P1 endpoint analysis; "
                "not relabeled as a newly trained independent confirmatory seed"
            ),
            "planner_panel_role": (
                "prospective for the new fixed-pool/adaptive-CEM estimands and "
                "analysis code, retrospective with respect to checkpoint existence "
                "and previously observed model behavior"
            ),
            "all_authorized_outcomes_reported": True,
        },
        "estimands": {
            "fixed_pool": (
                "candidate-wise H5 latent drift, exact squared-distance cost-drift "
                "bound, and top-1/elite stability on the same ordered K64 proposal"
            ),
            "short_horizon_comparator": (
                "candidate-conditioned H1 latent drift on that identical proposal"
            ),
            "adaptive": (
                "final first-action RMS and clean-history decision regret under "
                "common-random-number independently adapting CEM branches"
            ),
            "independent_unit": "trajectory episode",
            "goal_handling": "history-only probe; goal branch fixed",
        },
        "frozen_panel": {
            "tasks": list(TASKS),
            "training_seed": 3074,
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
        "mathematical_contract": {
            "planner_cost": "C(z,g)=||z-g||_2^2 at the final predicted latent",
            "exact_bound": (
                "|C(z,g)-C(z_tilde,g)| <= ||z-z_tilde||_2 "
                "(||z-g||_2+||z_tilde-g||_2)"
            ),
            "top1_certificate": (
                "every clean competitor gap exceeds the sum of the winner and "
                "competitor candidate-specific cost bounds"
            ),
            "elite_certificate": (
                "minimum bounded non-elite cost exceeds maximum bounded elite cost"
            ),
            "adaptive_induction": (
                "with a common initial proposal and common random numbers, equal "
                "elite membership preserves the numerical CEM update; once updates "
                "diverge, later indices are diagnostic rather than shared candidates"
            ),
        },
        "invariants": {
            "identity_h5_acpc_max": 1e-5,
            "identity_first_action_rms_max": 1e-6,
            "mse_bound_violation_count": 0,
            "certificate_false_positive_count": 0,
            "identity_all_adaptive_updates_aligned": True,
        },
        "predeclared_analysis": {
            "join_key": [
                "task",
                "checkpoint_role",
                "trajectory_block_id",
                "severity",
                "draw_index",
            ],
            "primary_validation": [
                "exact bound and certificate soundness on every row",
                "identity invariants",
                "descriptive task/role/severity curves",
            ],
            "incremental_model": {
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
                "strong_association_threshold": (
                    "at least 5% equal-task MAE reduction and improvement on at "
                    "least 3 of 4 held-out tasks"
                ),
                "report_regardless_of_threshold": True,
            },
            "full_budget_role": (
                "transfer sensitivity; no reduced/full equivalence is assumed"
            ),
        },
        "claim_adjudication": {
            "bound_claim": "allowed only if every bound/certificate invariant passes",
            "adaptive_association_claim": (
                "allowed only for responses meeting the frozen 5% and 3/4-task "
                "criterion; otherwise report the result and narrow the claim"
            ),
            "causal_claim": "forbidden",
            "closed_loop_robustness_claim": "forbidden",
        },
        "execution": {
            "max_concurrent_jobs": 4,
            "native_threads_per_job": 2,
            "one_process_per_gpu": True,
            "reduced_timeout_seconds": 3600,
            "full_budget_timeout_seconds": 7200,
            "retry_policy": "retry only exact frozen arguments after infrastructure failure",
        },
        "checkpoints": checkpoints,
        "source_hashes": source_hashes,
    }
    write_json(PROTOCOL_PATH, protocol)
    write_sidecar(PROTOCOL_PATH)

    addendum = {
        "schema_version": "paper1-acpc-planner-stability-execution-1.0",
        "protocol_id": protocol["protocol_id"],
        "status": "frozen_pre_execution",
        "immutable": True,
        "created_utc": frozen_at,
        "parent_protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "result_root": RESULT_ROOT,
        "source_hashes": source_hashes,
        "authorized_shards": shards,
        "all_results_must_be_retained": True,
        "post_result_protocol_edits_forbidden": True,
    }
    write_json(ADDENDUM_PATH, addendum)
    write_sidecar(ADDENDUM_PATH)
    print(f"wrote {PROTOCOL_PATH.relative_to(ROOT)}")
    print(f"wrote {ADDENDUM_PATH.relative_to(ROOT)}")
    print(f"authorized shards: {len(shards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
