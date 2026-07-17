#!/usr/bin/env python3
"""Freeze prospective seed3075 training and target-aligned P1 evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    "/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"
)
PROTOCOL_PATH = ROOT / "paper1/config/p1_prospective_seed3075_protocol_v1.json"
RESULT_ROOT = ROOT / "paper1/results/target_aligned_acpc_prospective_v1"


TASKS = {
    "TwoRoom": {
        "slug": "tworoom",
        "root": "lewm-tworooms",
        "hydra_data": "tworoom",
        "dataset_name": "tworoom",
        "gpu": 4,
        "eval_timeout_seconds": 900,
        "replay_cache": None,
    },
    "PushT": {
        "slug": "pusht",
        "root": "lewm-pusht",
        "hydra_data": "pusht",
        "dataset_name": "pusht_expert_train",
        "gpu": 5,
        "eval_timeout_seconds": 900,
        "replay_cache": None,
    },
    "Reacher": {
        "slug": "reacher",
        "root": "lewm-reacher",
        "hydra_data": "dmc",
        "dataset_name": "reacher",
        "gpu": 6,
        "eval_timeout_seconds": 1800,
        "replay_cache": (
            "paper1/results/target_aligned_acpc_dev/cache/"
            "reacher_seed3072_goal25_k16_v1.npz"
        ),
    },
    "Cube": {
        "slug": "cube",
        "root": "lewm-cube",
        "hydra_data": "ogb",
        "dataset_name": "ogbench/cube_single_expert",
        "gpu": 7,
        "eval_timeout_seconds": 3600,
        "replay_cache": (
            "paper1/results/target_aligned_acpc_dev/cache/"
            "cube_seed3074_goal25_k16_v1.npz"
        ),
    },
}

SOURCE_PATHS = [
    "paper1/docs/PAPER1_ACPC_SCIENTIFIC_REMEDIATION_PLAN_20260713.md",
    "paper1/scripts/freeze_p1_prospective_seed3075_protocol.py",
    "paper1/scripts/run_target_aligned_acpc_mve.py",
    "paper1/scripts/summarize_target_aligned_acpc_mve.py",
    "paper1/scripts/summarize_target_aligned_acpc_four_task.py",
    "tests/test_paper1_target_aligned_acpc_mve.py",
    "tests/test_paper1_target_aligned_acpc.py",
    "tests/test_paper1_acpc_frozen_protocols.py",
    "train.py",
    "jepa.py",
    "module.py",
    "acpc_flow.py",
    "utils.py",
    "config/train/lewm.yaml",
    "config/train/data/tworoom.yaml",
    "config/train/data/pusht.yaml",
    "config/train/data/dmc.yaml",
    "config/train/data/ogb.yaml",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_file_fingerprint(
    path: Path,
    *,
    sample_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Bind a very large immutable dataset without a 244-GB full scan."""

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        head = stream.read(sample_bytes)
        digest.update(head)
        if stat.st_size > sample_bytes:
            stream.seek(max(sample_bytes, stat.st_size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return {
        "algorithm": "sha256(first_16MiB || last_16MiB)",
        "sample_bytes_per_edge": sample_bytes,
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def task_paths(task: dict[str, Any]) -> dict[str, Path]:
    stablewm_home = DATA_ROOT / task["root"]
    name = f"{task['slug']}_lewm_baseline_seed3075"
    run_dir = stablewm_home / "ckpt" / name
    return {
        "stablewm_home": stablewm_home,
        "dataset": stablewm_home / f"{task['dataset_name']}.h5",
        "run_dir": run_dir,
        "checkpoint": run_dir / f"{name}_epoch_10_object.ckpt",
        "config": run_dir / "config.yaml",
    }


def training_command(task: dict[str, Any]) -> list[str]:
    name = f"{task['slug']}_lewm_baseline_seed3075"
    return [
        "python",
        "train.py",
        f"data={task['hydra_data']}",
        "seed=3075",
        f"output_model_name={name}",
        f"subdir=ckpt/{name}",
        "image_noise.std_min=0.0",
        "image_noise.std_max=0.0",
        "image_noise.noise_prob=0.0",
        "image_noise.apply_to_val=false",
        "trainer.max_epochs=10",
        "swanlab.enabled=false",
        "wandb.enabled=false",
        "hydra.job.chdir=false",
    ]


def evaluation_command(task_name: str, task: dict[str, Any]) -> list[str]:
    paths = task_paths(task)
    out = RESULT_ROOT / f"mve_{task['slug']}_seed3075_base_goal25_v1_16block.json"
    command = [
        "python",
        "-m",
        "paper1.scripts.run_target_aligned_acpc_mve",
        "--checkpoint",
        str(paths["checkpoint"]),
        "--training-seed",
        "3075",
        "--checkpoint-role",
        "base",
        "--dataset-name",
        task["dataset_name"],
        "--dataset-path",
        str(paths["dataset"]),
        "--task",
        task_name,
        "--n-blocks",
        "16",
        "--future-steps",
        "9",
        "--frameskip",
        "5",
        "--img-size",
        "224",
        "--plan-horizon",
        "5",
        "--logged-horizon",
        "8",
        "--action-block",
        "5",
        "--candidate-count",
        "16",
        "--topk",
        "4",
        "--batch-size",
        "1",
        "--target-encode-batch-size",
        "16",
        "--trajectory-seed",
        "9101",
        "--cem-seed",
        "1234",
        "--control-seed",
        "4701",
        "--replay-seed",
        "0",
        "--probe-family",
        "gaussian_noise",
        "--severities",
        "0.0",
        "0.02",
        "0.05",
        "0.08",
        "--draws",
        "2",
        "--probe-seed",
        "20260712",
        "--invariant-atol",
        "1e-5",
        "--minimum-true-goal-cost-std",
        "1e-3",
        "--minimum-informative-block-fraction",
        "0.75",
    ]
    if task["replay_cache"] is not None:
        command.extend(["--replay-cache", str(ROOT / task["replay_cache"])])
    command.extend(["--out", str(out)])
    return command


def main() -> int:
    frozen_at = datetime.now(timezone.utc).isoformat()
    source_hashes = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    reference_inputs = {}
    for path in (
        "paper1/results/target_aligned_acpc_dev/"
        "adjudication_four_task_seed3073_goal25_base_endpoint_v1.json",
        "paper1/results/target_aligned_acpc_dev/"
        "adjudication_four_task_seed3074_goal25_base_endpoint_v1.json",
    ):
        reference_inputs[path] = sha256(ROOT / path)

    task_records = []
    for task_name, task in TASKS.items():
        paths = task_paths(task)
        if paths["run_dir"].exists():
            raise FileExistsError(
                f"prospective output already exists before freeze: {paths['run_dir']}"
            )
        if not paths["dataset"].is_file():
            raise FileNotFoundError(paths["dataset"])
        replay_cache = None
        if task["replay_cache"] is not None:
            cache_path = ROOT / task["replay_cache"]
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            replay_cache = {
                "path": task["replay_cache"],
                "sha256": sha256(cache_path),
                "role": "model-independent strict-key simulator replay target cache",
            }
        reference_config = (
            DATA_ROOT
            / task["root"]
            / "ckpt"
            / f"{task['slug']}_lewm_baseline_seed3074"
            / "config.yaml"
        )
        task_records.append(
            {
                "task": task_name,
                "gpu": task["gpu"],
                "stablewm_home": str(paths["stablewm_home"]),
                "dataset_path": str(paths["dataset"]),
                "dataset_fingerprint": sampled_file_fingerprint(paths["dataset"]),
                "training_command": training_command(task),
                "training_timeout_seconds": 14400,
                "expected_run_dir": str(paths["run_dir"]),
                "expected_checkpoint": str(paths["checkpoint"]),
                "expected_resolved_config": str(paths["config"]),
                "reference_seed3074_config": str(reference_config),
                "reference_seed3074_config_sha256": sha256(reference_config),
                "evaluation_command": evaluation_command(task_name, task),
                "evaluation_timeout_seconds": task["eval_timeout_seconds"],
                "evaluation_output": str(
                    RESULT_ROOT
                    / f"mve_{task['slug']}_seed3075_base_goal25_v1_16block.json"
                ),
                "replay_cache": replay_cache,
            }
        )

    prospective_summary = (
        RESULT_ROOT / "adjudication_four_task_seed3075_goal25_base_v1.json"
    )
    combined_meta = (
        RESULT_ROOT
        / "meta_four_task_seeds3073_3074_3075_goal25_base_endpoint_v1.json"
    )
    raw_outputs = [record["evaluation_output"] for record in task_records]
    seed3073 = (
        ROOT
        / "paper1/results/target_aligned_acpc_dev/"
        "adjudication_four_task_seed3073_goal25_base_endpoint_v1.json"
    )
    seed3074 = (
        ROOT
        / "paper1/results/target_aligned_acpc_dev/"
        "adjudication_four_task_seed3074_goal25_base_endpoint_v1.json"
    )
    protocol = {
        "schema_version": "paper1-p1-prospective-training-protocol-1.0",
        "protocol_id": "paper1-p1-prospective-seed3075-v1",
        "status": "frozen_pre_execution",
        "immutable": True,
        "execution_authorized": True,
        "frozen_at_utc": frozen_at,
        "analysis_commit_parent": git_head(),
        "prospective_boundary": {
            "training_seed": 3075,
            "seed_selected_before_any_seed3075_training": True,
            "seed3075_output_absent_at_freeze": True,
            "all_four_tasks_required": True,
            "all_outcomes_reported": True,
            "no_endpoint_training_required": True,
            "reason": (
                "the primary logged fragile-base estimand is defined on the "
                "unperturbed base checkpoint"
            ),
        },
        "provenance": {
            "seed3073": "development-era held-out model seed",
            "seed3074": "protocol-frozen replication after earlier development",
            "seed3075": "fully prospectively frozen training seed",
            "seed3072": (
                "retrospective completeness check; reported separately and not "
                "relabeled as an independent confirmatory replication"
            ),
        },
        "training_contract": {
            "model": "LeWM base",
            "epochs": 10,
            "image_noise_std": [0.0, 0.0],
            "image_noise_probability": 0.0,
            "validation_noise": False,
            "logger_disabled": True,
            "optimization_difference_from_seed3074_reference": "seed only",
            "operational_difference": (
                "online experiment loggers are disabled; this does not enter the "
                "model, data, optimizer, or random-number path"
            ),
            "exact_resume_after_infrastructure_failure_allowed": True,
            "dataset_binding": (
                "canonical absolute path, byte size, mtime_ns, and SHA-256 over "
                "the first and last 16 MiB; full 244-GB dataset hashing is not "
                "part of the execution contract"
            ),
        },
        "evaluation_contract": {
            "track": "logged correct-action H8 future prediction-error drift",
            "short_comparator": "correct-action H1",
            "destroyed_controls": [
                "action-zero H8",
                "candidate-shuffled H8",
                "time-shuffled H8",
            ],
            "tasks": list(TASKS),
            "n_episode_blocks": 16,
            "severities": [0.0, 0.02, 0.05, 0.08],
            "draws": 2,
            "trajectory_seed": 9101,
            "probe_seed": 20260712,
            "goal_offset_low_level_steps": 25,
            "all_tasks_retained": True,
        },
        "adjudication": {
            "per_task_gate": (
                "correct H8 must achieve at least 5% relative MAE reduction "
                "versus H1 and the best destroyed H8 control, plus the frozen "
                "within-block direction/rank condition"
            ),
            "seed_level_gate": "at least 3 of 4 tasks pass; all 4 are reported",
            "failure_policy": (
                "retain and report seed3075; narrow the cross-seed claim rather "
                "than changing thresholds or excluding a task"
            ),
        },
        "execution": {
            "parallel_training_jobs": 4,
            "one_training_process_per_gpu": True,
            "native_threads_per_process": 2,
            "data_loader_workers_per_process": 6,
            "training_gpu_indices": [4, 5, 6, 7],
            "evaluation_after_training_only": True,
        },
        "tasks": task_records,
        "summary_commands": {
            "seed3075": [
                "python",
                "-m",
                "paper1.scripts.summarize_target_aligned_acpc_mve",
                *raw_outputs,
                "--out",
                str(prospective_summary),
            ],
            "combined_seed3073_3074_3075": [
                "python",
                "-m",
                "paper1.scripts.summarize_target_aligned_acpc_four_task",
                str(seed3073),
                str(seed3074),
                str(prospective_summary),
                "--out",
                str(combined_meta),
            ],
        },
        "reference_input_hashes": reference_inputs,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "lightning",
                "hydra-core",
                "stable-pretraining",
                "stable-worldmodel",
            )
        },
        "source_hashes": source_hashes,
    }
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar = PROTOCOL_PATH.with_suffix(PROTOCOL_PATH.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256(PROTOCOL_PATH)}  {PROTOCOL_PATH.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(f"wrote {PROTOCOL_PATH.relative_to(ROOT)}")
    print("prospective tasks: 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
