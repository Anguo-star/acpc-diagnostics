#!/usr/bin/env python3
"""Build the frozen fixed-rho blur/resize external-validation artifacts.

The consumer is deliberately behavior-blind until the immutable ATR/SMPR rows
have been loaded and scored with the frozen gate.  It then joins independently
produced closed-loop summaries without searching thresholds or severities.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
LEWM_SEEDS = (3072, 3073, 3074)
STD_KEYS = ("0.0", "0.08")
FROZEN_RHO = 0.08
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
STRESSORS = {
    "blur": {
        "artifact_dir": "gaussian_blur",
        "behavior_key": "gaussian_blur",
        "severity": 15.0,
        "severity_parameter": "kernel_size",
        "eval_group": "pixels_blur_ks15",
    },
    "resize": {
        "artifact_dir": "resize",
        "behavior_key": "resize",
        "severity": 0.25,
        "severity_parameter": "scale_factor",
        "eval_group": "pixels_rs_factor0.25",
    },
}
BASELINE_FIELDS = (
    "encoder_q90",
    "h1_q90",
    "action_shuffled_h8_q90",
    "action_zeroed_h8_q90",
    "time_shuffled_h8_q90",
)
PAIRED_CHANGE_DIAGNOSTICS = (
    ("encoder_q90", "Encoder q90", "delta_encoder_q90", -1.0),
    ("h1_q90", "H1 q90", "delta_h1_q90", -1.0),
    ("atr_h8_q90", "Correct-action H8 q90", "delta_atr", -1.0),
    (
        "action_shuffled_h8_q90",
        "Action-shuffled H8 q90",
        "delta_action_shuffled_h8",
        -1.0,
    ),
    (
        "action_zeroed_h8_q90",
        "Action-zeroed H8 q90",
        "delta_action_zeroed_h8",
        -1.0,
    ),
    (
        "time_shuffled_h8_q90",
        "Time-shuffled H8 q90",
        "delta_time_shuffled_h8",
        -1.0,
    ),
    ("smpr", "SMPR", "delta_smpr", 1.0),
    ("joint_score", "Joint score", "delta_joint_score", 1.0),
)
PAIRED_CHANGE_BOOTSTRAP_SEED = 20260711
PAIRED_CHANGE_BOOTSTRAP_REPETITIONS = 5000
PAIRED_CHANGE_INCREMENTAL_BOOTSTRAP_SEED = 20260712
FIXED_FIELDS = (
    "model_family",
    "training_seed_or_family_id",
    "training_seed",
    "training_seed_semantics",
    "evaluation_seeds",
    "evaluation_seed_semantics",
    "task",
    "stressor_family",
    "stressor_severity",
    "stressor_severity_parameter",
    "rho",
    "checkpoint_sha256",
    "clean_score",
    "stressed_score",
    "clean_score_by_evaluation_seed",
    "stressed_score_by_evaluation_seed",
    "retention",
    "base_clean_score",
    "base_stressed_score",
    "delta_behavior",
    "clean_score_drop",
    "delta_retention",
    "behavior_class",
    "positive_transfer_label",
    "encoder_q90",
    "h1_q90",
    "action_shuffled_h8_q90",
    "action_zeroed_h8_q90",
    "time_shuffled_h8_q90",
    "base_atr_h8_q90",
    "atr_h8_q90",
    "delta_atr",
    "base_smpr",
    "smpr",
    "delta_smpr",
    "base_joint_score",
    "joint_score",
    "delta_joint_score",
    "base_joint_gate_pass",
    "joint_gate_pass",
    "split_name",
    "protocol_hash",
    "diagnostics_sha256",
    "behavior_source_sha256",
    "baseline_diagnostics_status",
)
PAIR_FIELDS = (
    "model_family",
    "training_seed_or_family_id",
    "training_seed",
    "training_seed_semantics",
    "evaluation_seeds",
    "evaluation_seed_semantics",
    "task",
    "stressor_family",
    "stressor_severity",
    "stressor_severity_parameter",
    "base_rho",
    "endpoint_rho",
    "base_checkpoint_sha256",
    "endpoint_checkpoint_sha256",
    "base_clean_score",
    "endpoint_clean_score",
    "base_stressed_score",
    "endpoint_stressed_score",
    "base_retention",
    "endpoint_retention",
    "delta_behavior",
    "clean_score_drop",
    "delta_retention",
    "behavior_class",
    "positive_transfer_label",
    "base_encoder_q90",
    "endpoint_encoder_q90",
    "delta_encoder_q90",
    "base_h1_q90",
    "endpoint_h1_q90",
    "delta_h1_q90",
    "base_action_shuffled_h8",
    "endpoint_action_shuffled_h8",
    "delta_action_shuffled_h8",
    "base_action_zeroed_h8",
    "endpoint_action_zeroed_h8",
    "delta_action_zeroed_h8",
    "base_time_shuffled_h8",
    "endpoint_time_shuffled_h8",
    "delta_time_shuffled_h8",
    "base_atr",
    "endpoint_atr",
    "delta_atr",
    "base_smpr",
    "endpoint_smpr",
    "delta_smpr",
    "base_joint_score",
    "endpoint_joint_score",
    "delta_joint_score",
    "base_gate_pass",
    "endpoint_gate_pass",
    "split_name",
    "protocol_hash",
    "base_diagnostics_sha256",
    "endpoint_diagnostics_sha256",
    "behavior_source_sha256",
    "baseline_diagnostics_status",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
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
        raise ValueError(f"{name}: booleans are not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected finite numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: value is not finite")
    return result


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot compute an empty mean")
    return statistics.fmean(values)


def _close(left: float, right: float, *, atol: float = 2e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def _csv_values(value: Any, *, name: str) -> list[float]:
    if isinstance(value, str):
        parts = [item for item in value.split(";") if item != ""]
    elif isinstance(value, list):
        parts = value
    else:
        raise ValueError(f"{name}: expected semicolon string or list")
    result = [_finite(item, name=f"{name}[{index}]") for index, item in enumerate(parts)]
    _require(len(result) == 3, f"{name}: expected exactly three evaluation seeds")
    return result


def _score_block(block: Mapping[str, Any], *, name: str) -> tuple[float, list[float]]:
    values = _csv_values(block.get("values"), name=f"{name}/values")
    mean_value = _finite(block.get("mean"), name=f"{name}/mean")
    _require(_close(mean_value, _mean(values)), f"{name}: mean/value mismatch")
    n = int(_finite(block.get("n_seeds", block.get("n")), name=f"{name}/n"))
    _require(n == 3, f"{name}: expected three evaluation seeds")
    seeds = block.get("seeds")
    if seeds is not None:
        parsed = (
            [int(item) for item in seeds.split(",")]
            if isinstance(seeds, str)
            else [int(item) for item in seeds]
        )
        _require(parsed == [42, 43, 44], f"{name}: evaluation seed mismatch")
    return mean_value, values


def _load_protocol(path: Path) -> tuple[dict[str, Any], bytes, int]:
    original = path.read_bytes()
    payload = json.loads(original.decode("utf-8"), parse_constant=_reject_constant)
    protocol_hash = hashlib.sha256(original).hexdigest()
    _require(protocol_hash == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    _require(
        payload.get("schema_version") == "paper1-frozen-diagnostic-protocol-1.0",
        "unsupported frozen protocol schema",
    )
    _require(payload.get("status") == "frozen", "protocol is not frozen")
    _require(payload.get("immutable") is True, "protocol is not immutable")
    policy = payload.get("external_policy", {})
    _require(policy.get("threshold_search_allowed") is False, "external threshold search enabled")
    _require(policy.get("protocol_write_allowed") is False, "external protocol writes enabled")
    return payload, original, path.stat().st_mtime_ns


def _assert_protocol_unchanged(path: Path, original: bytes, mtime_ns: int) -> None:
    _require(path.read_bytes() == original, "consumer changed frozen protocol bytes")
    _require(path.stat().st_mtime_ns == mtime_ns, "consumer changed frozen protocol mtime")


def _joint(
    atr: float,
    smpr: float,
    *,
    tau_atr: float,
    tau_smpr: float,
) -> tuple[float, bool]:
    atr_margin = (tau_atr - atr) / (abs(tau_atr) + 1e-12)
    smpr_margin = (smpr - tau_smpr) / (abs(tau_smpr) + 1e-12)
    return min(atr_margin, smpr_margin), atr <= tau_atr and smpr >= tau_smpr


def classify_transfer(
    *,
    delta_behavior: float,
    clean_score_drop: float,
    positive_delta_pp: float,
    neutral_band_pp: float,
    max_clean_drop_pp: float,
) -> str:
    """Apply the pre-registered three-way label, including boundary semantics."""
    if delta_behavior >= positive_delta_pp and clean_score_drop <= max_clean_drop_pp:
        return "positive"
    if (
        -neutral_band_pp < delta_behavior < neutral_band_pp
        and clean_score_drop <= max_clean_drop_pp
    ):
        return "neutral"
    return "negative"


def _behavior_summary(
    condition: Mapping[str, Any],
    *,
    name: str,
    expected_group: str,
) -> tuple[float, list[float], float, list[float]]:
    primary_group = str(condition.get("primary_stress_group"))
    _require(primary_group == expected_group, f"{name}: primary stress group mismatch")
    clean_block = condition.get("success_rate", {}).get("origin")
    stressed_block = condition.get("primary_stress_success")
    _require(isinstance(clean_block, Mapping), f"{name}: clean score block missing")
    _require(isinstance(stressed_block, Mapping), f"{name}: stress score block missing")
    clean_mean, clean_values = _score_block(clean_block, name=f"{name}/clean")
    stressed_mean, stressed_values = _score_block(stressed_block, name=f"{name}/stress")
    return clean_mean, clean_values, stressed_mean, stressed_values


def _load_lewm_behavior(
    paths: Mapping[int, Path],
    *,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    result: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for seed in LEWM_SEEDS:
        path = paths[seed]
        payload = _load_json(path)
        metadata = payload.get("metadata", {})
        _require(
            metadata.get("schema_version") == "paper1-unseen-perturbation-pilot-1.0",
            f"{path}: schema mismatch",
        )
        _require(int(metadata.get("train_seed")) == seed, f"{path}: training seed mismatch")
        source_key = f"lewm_behavior_seed{seed}"
        source_paths[source_key] = str(path)
        source_hashes[source_key] = _sha256(path)
        results = payload.get("results", {})
        for task in TASKS:
            for std_key in STD_KEYS:
                for stressor, spec in STRESSORS.items():
                    name = f"LeWM/{seed}/{task}/{std_key}/{stressor}"
                    condition = (
                        results.get(task, {})
                        .get(std_key, {})
                        .get(spec["behavior_key"])
                    )
                    _require(isinstance(condition, Mapping), f"{name}: behavior row missing")
                    clean, clean_values, stressed, stressed_values = _behavior_summary(
                        condition,
                        name=name,
                        expected_group=str(spec["eval_group"]),
                    )
                    magnitudes = {
                        _finite(value, name=f"{name}/magnitude")
                        for value in condition.get("magnitudes", [])
                    }
                    _require(
                        any(_close(value, float(spec["severity"])) for value in magnitudes),
                        f"{name}: strongest severity missing",
                    )
                    result[("LeWM", seed, task, stressor, std_key)] = {
                        "clean": clean,
                        "clean_values": clean_values,
                        "stressed": stressed,
                        "stressed_values": stressed_values,
                        "checkpoint_rel": str(condition.get("checkpoint_rel")),
                        "subdir": str(condition.get("subdir")),
                        "source_path": str(path),
                        "source_sha256": source_hashes[source_key],
                    }
    _require(len(result) == 48, "LeWM behavior matrix must contain 48 checkpoint-stressor rows")
    return result


def _canonical_clean_index(
    path: Path,
    *,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load_json(path)
    metadata = payload.get("_metadata", {})
    _require(
        metadata.get("schema_version") == "paper1-pldm-canonical-eval-manifest-0.2",
        "PLDM canonical manifest schema mismatch",
    )
    _require(metadata.get("status") == "complete", "PLDM canonical manifest incomplete")
    _require(metadata.get("training_seed") == 3072, "PLDM training seed mismatch")
    _require(
        metadata.get("training_seed_semantics")
        == "one independently trained PLDM checkpoint family",
        "PLDM training-seed semantics mismatch",
    )
    source_paths["pldm_clean_manifest"] = str(path)
    source_hashes["pldm_clean_manifest"] = _sha256(path)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for task in TASKS:
        for std_key in STD_KEYS:
            entry = payload.get(task, {}).get(std_key)
            _require(isinstance(entry, Mapping), f"PLDM/{task}/{std_key}: canonical entry missing")
            block = entry.get("metrics", {}).get("clean")
            _require(isinstance(block, Mapping), f"PLDM/{task}/{std_key}: clean metric missing")
            mean_value, values = _score_block(block, name=f"PLDM/{task}/{std_key}/clean")
            result[(task, std_key)] = {
                "clean": mean_value,
                "clean_values": values,
                "subdir": str(entry.get("subdir")),
                "model_file": str(entry.get("model_file")),
            }
    return result


def _read_eval_summary(path: Path, *, expected_group: str, name: str) -> tuple[float, list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row
        for row in rows
        if row.get("group") == expected_group and row.get("metric") == "success_rate"
    ]
    _require(len(matches) == 1, f"{name}: expected exactly one success-rate row")
    return _score_block(matches[0], name=name)


def _job_index(path: Path, *, expected_stressor: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = _load_json(path)
    metadata = payload.get("metadata", {})
    _require(
        metadata.get("schema_version") == "paper1-unseen-eval-grid-manifest-1.1",
        f"{path}: behavior job schema mismatch",
    )
    _require(int(metadata.get("train_seed")) == 3072, f"{path}: train seed mismatch")
    jobs = payload.get("jobs", [])
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for job in jobs:
        task = str(job.get("task"))
        std_key = str(job.get("std_key"))
        _require(job.get("family") == expected_stressor, f"{path}: stressor mismatch")
        _require(task in TASKS and std_key in STD_KEYS, f"{path}: unexpected job key")
        key = (task, std_key)
        _require(key not in index, f"{path}: duplicate behavior job")
        index[key] = job
    return index


def _load_pldm_behavior(
    *,
    clean_manifest_path: Path,
    blur_baseline_path: Path,
    blur_jobs_path: Path,
    resize_jobs_path: Path,
    data_root: Path,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
) -> dict[tuple[str, int, str, str, str], dict[str, Any]]:
    clean = _canonical_clean_index(
        clean_manifest_path,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )
    blur_baselines = _load_json(blur_baseline_path)
    _require(
        blur_baselines.get("metadata", {}).get("schema_version") == "blur-baselines-1.0",
        "PLDM blur baseline schema mismatch",
    )
    source_paths["pldm_blur_baseline"] = str(blur_baseline_path)
    source_hashes["pldm_blur_baseline"] = _sha256(blur_baseline_path)
    blur_jobs = _job_index(blur_jobs_path, expected_stressor="gaussian_blur")
    resize_jobs = _job_index(resize_jobs_path, expected_stressor="resize")
    _require(set(blur_jobs) == {(task, "0.08") for task in TASKS}, "PLDM blur endpoint job coverage mismatch")
    _require(
        set(resize_jobs) == {(task, std_key) for task in TASKS for std_key in STD_KEYS},
        "PLDM resize job coverage mismatch",
    )
    for key, path in (
        ("pldm_blur_jobs", blur_jobs_path),
        ("pldm_resize_jobs", resize_jobs_path),
    ):
        source_paths[key] = str(path)
        source_hashes[key] = _sha256(path)

    result: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    family_id = 3072
    for task in TASKS:
        for std_key in STD_KEYS:
            clean_row = clean[(task, std_key)]
            if std_key == "0.0":
                baseline = blur_baselines.get("baselines", {}).get("PLDM", {}).get(task, {})
                blur_clean, blur_clean_values = _score_block(
                    baseline.get("clean", {}),
                    name=f"PLDM/{task}/blur/base-clean",
                )
                _require(_close(blur_clean, clean_row["clean"]), f"PLDM/{task}: blur clean mismatch")
                stressed, stressed_values = _score_block(
                    baseline.get("blur", {}).get("pixels_blur_ks15", {}),
                    name=f"PLDM/{task}/blur/base-stress",
                )
                behavior_path = blur_baseline_path
                behavior_hash = source_hashes["pldm_blur_baseline"]
            else:
                job = blur_jobs[(task, std_key)]
                eval_path = data_root / str(job.get("eval_summary_rel"))
                _require(eval_path.is_file(), f"PLDM/{task}/blur/{std_key}: eval summary missing")
                stressed, stressed_values = _read_eval_summary(
                    eval_path,
                    expected_group=str(STRESSORS["blur"]["eval_group"]),
                    name=f"PLDM/{task}/blur/{std_key}",
                )
                behavior_path = eval_path
                behavior_hash = _sha256(eval_path)
                source_key = f"pldm_blur_eval_{task}_{std_key}"
                source_paths[source_key] = str(eval_path)
                source_hashes[source_key] = behavior_hash
            result[("PLDM", family_id, task, "blur", std_key)] = {
                "clean": clean_row["clean"],
                "clean_values": clean_row["clean_values"],
                "stressed": stressed,
                "stressed_values": stressed_values,
                "checkpoint_rel": clean_row["model_file"],
                "subdir": clean_row["subdir"],
                "source_path": str(behavior_path),
                "source_sha256": behavior_hash,
            }

            job = resize_jobs[(task, std_key)]
            eval_path = data_root / str(job.get("eval_summary_rel"))
            _require(eval_path.is_file(), f"PLDM/{task}/resize/{std_key}: eval summary missing")
            stressed, stressed_values = _read_eval_summary(
                eval_path,
                expected_group=str(STRESSORS["resize"]["eval_group"]),
                name=f"PLDM/{task}/resize/{std_key}",
            )
            behavior_hash = _sha256(eval_path)
            source_key = f"pldm_resize_eval_{task}_{std_key}"
            source_paths[source_key] = str(eval_path)
            source_hashes[source_key] = behavior_hash
            result[("PLDM", family_id, task, "resize", std_key)] = {
                "clean": clean_row["clean"],
                "clean_values": clean_row["clean_values"],
                "stressed": stressed,
                "stressed_values": stressed_values,
                "checkpoint_rel": clean_row["model_file"],
                "subdir": clean_row["subdir"],
                "source_path": str(eval_path),
                "source_sha256": behavior_hash,
            }
    _require(len(result) == 16, "PLDM behavior matrix must contain 16 checkpoint-stressor rows")
    return result


def _validate_diagnostic_metadata(
    metadata: Mapping[str, Any],
    *,
    path: Path,
    model_family: str,
    family_id: str,
    stressor: str,
    artifact_role: str,
) -> None:
    _require(metadata.get("status") == "complete", f"{path}: diagnostic artifact incomplete")
    _require(metadata.get("errors") == [], f"{path}: diagnostic errors present")
    _require(metadata.get("missing_rows") == [], f"{path}: diagnostic rows missing")
    _require(metadata.get("behavior_blind") is True, f"{path}: diagnostic is not behavior blind")
    _require(metadata.get("threshold_search_allowed") is False, f"{path}: threshold search enabled")
    _require(metadata.get("protocol_hash") == FROZEN_PROTOCOL_SHA256, f"{path}: protocol hash mismatch")
    _require(metadata.get("model_family") == model_family, f"{path}: model family mismatch")
    _require(metadata.get("training_family_id") == family_id, f"{path}: family id mismatch")
    _require(metadata.get("stressor_family") == stressor, f"{path}: stressor mismatch")
    _require(metadata.get("artifact_role") == artifact_role, f"{path}: artifact role mismatch")
    _require(
        _close(
            _finite(metadata.get("severity"), name=f"{path}/severity"),
            float(STRESSORS[stressor]["severity"]),
        ),
        f"{path}: severity mismatch",
    )


def _load_diagnostics(
    *,
    cross_source_root: Path,
    protocol: Mapping[str, Any],
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
    baseline_path: Path | None,
) -> tuple[dict[tuple[str, int, str, str, str], dict[str, Any]], str]:
    tau_atr = _finite(protocol.get("tau_atr"), name="protocol/tau_atr")
    tau_smpr = _finite(protocol.get("tau_smpr"), name="protocol/tau_smpr")
    baseline_index: dict[tuple[str, int, str, str, str], Mapping[str, Any]] = {}
    baseline_status = "missing_not_substituted"
    if baseline_path is not None:
        baseline_payload = _load_json(baseline_path)
        baseline_metadata = baseline_payload.get("metadata", {})
        _require(
            baseline_metadata.get("schema_version")
            == "paper1-diagnostic-baseline-all-1.0",
            f"{baseline_path}: baseline schema mismatch",
        )
        _require(baseline_metadata.get("status") == "complete", f"{baseline_path}: baseline incomplete")
        _require(baseline_metadata.get("errors") == [], f"{baseline_path}: baseline errors present")
        _require(baseline_metadata.get("missing_rows") == [], f"{baseline_path}: baseline rows missing")
        _require(
            baseline_metadata.get("behavior_blind_rows") is True,
            f"{baseline_path}: baseline rows are not behavior blind",
        )
        _require(
            baseline_metadata.get("external_threshold_search_allowed") is False,
            f"{baseline_path}: external threshold search enabled",
        )
        _require(
            baseline_metadata.get("protocol_hash") == FROZEN_PROTOCOL_SHA256,
            f"{baseline_path}: baseline protocol mismatch",
        )
        rows = [
            row
            for row in baseline_payload.get("rows", [])
            if row.get("split_name") in {"E3-L", "E3-P"}
        ]
        _require(len(rows) == 64, f"{baseline_path}: expected 64 E3 baseline rows")
        for row in rows:
            expected_split = "E3-L" if row.get("model_family") == "LeWM" else "E3-P"
            _require(row.get("split_name") == expected_split, f"{baseline_path}: E3 split/model mismatch")
            _require(row.get("status") == "ok", f"{baseline_path}: baseline row not ok")
            _require(row.get("reference_atr_match") is True, f"{baseline_path}: baseline ATR mismatch")
            key = (
                str(row.get("model_family")),
                int(row.get("training_seed")),
                str(row.get("task")),
                str(row.get("stressor_family")),
                str(row.get("std_key")),
            )
            _require(key not in baseline_index, f"{baseline_path}: duplicate baseline diagnostic row")
            baseline_index[key] = row
        source_paths["baseline_diagnostics"] = str(baseline_path)
        source_hashes["baseline_diagnostics"] = _sha256(baseline_path)
        baseline_status = "complete"

    result: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    families = [
        ("LeWM", seed, f"lewm_seed{seed}", f"lewm_seed{seed}")
        for seed in LEWM_SEEDS
    ] + [("PLDM", 3072, "pldm_canonical_seed3072", "pldm_canonical")]
    for model_family, seed, family_id, directory in families:
        for stressor, spec in STRESSORS.items():
            base_dir = cross_source_root / directory / str(spec["artifact_dir"])
            atr_path = base_dir / "acpc_horizon_v2_checkpoint_bound.json"
            smpr_path = base_dir / "smpr_v2_checkpoint_bound.json"
            atr_payload = _load_json(atr_path)
            smpr_payload = _load_json(smpr_path)
            _validate_diagnostic_metadata(
                atr_payload.get("metadata", {}),
                path=atr_path,
                model_family=model_family,
                family_id=family_id,
                stressor=stressor,
                artifact_role="cross_stressor_external_atr",
            )
            _validate_diagnostic_metadata(
                smpr_payload.get("metadata", {}),
                path=smpr_path,
                model_family=model_family,
                family_id=family_id,
                stressor=stressor,
                artifact_role="cross_stressor_external_smpr",
            )
            source_prefix = f"{directory}_{stressor}"
            source_paths[f"{source_prefix}_atr"] = str(atr_path)
            source_hashes[f"{source_prefix}_atr"] = _sha256(atr_path)
            source_paths[f"{source_prefix}_smpr"] = str(smpr_path)
            source_hashes[f"{source_prefix}_smpr"] = _sha256(smpr_path)
            atr_rows = {
                (str(row.get("task")), str(row.get("std_key"))): row
                for row in atr_payload.get("rows", [])
            }
            smpr_rows = {
                (str(row.get("task")), str(row.get("std_key"))): row
                for row in smpr_payload.get("rows", [])
            }
            expected = {(task, std_key) for task in TASKS for std_key in STD_KEYS}
            _require(set(atr_rows) == expected, f"{atr_path}: row coverage mismatch")
            _require(set(smpr_rows) == expected, f"{smpr_path}: row coverage mismatch")
            for task, std_key in sorted(expected):
                atr_row = atr_rows[(task, std_key)]
                smpr_row = smpr_rows[(task, std_key)]
                key_name = f"{model_family}/{seed}/{task}/{stressor}/{std_key}"
                _require(atr_row.get("status") == "ok", f"{key_name}: ATR row not ok")
                _require(smpr_row.get("status") == "ok", f"{key_name}: SMPR row not ok")
                _require(
                    atr_row.get("checkpoint_sha256") == smpr_row.get("checkpoint_sha256"),
                    f"{key_name}: checkpoint hash mismatch",
                )
                atr = _finite(atr_row.get("atr_horizon_v2_q90"), name=f"{key_name}/atr")
                reference_atr = _finite(
                    smpr_row.get("reference_atr_horizon_v2_q90"),
                    name=f"{key_name}/smpr-reference-atr",
                )
                _require(_close(atr, reference_atr, atol=1e-9), f"{key_name}: SMPR/ATR mismatch")
                smpr = _finite(smpr_row.get("smpr"), name=f"{key_name}/smpr")
                joint_score, gate_pass = _joint(
                    atr,
                    smpr,
                    tau_atr=tau_atr,
                    tau_smpr=tau_smpr,
                )
                key = (model_family, seed, task, stressor, std_key)
                baseline = baseline_index.get(key)
                if baseline_path is not None:
                    _require(baseline is not None, f"{key_name}: baseline diagnostic row missing")
                    _require(
                        baseline.get("checkpoint_sha256") == atr_row.get("checkpoint_sha256"),
                        f"{key_name}: baseline checkpoint mismatch",
                    )
                values: dict[str, Any] = {}
                for field in BASELINE_FIELDS:
                    values[field] = (
                        _finite(baseline.get(field), name=f"{key_name}/{field}")
                        if baseline is not None
                        else ""
                    )
                diagnostics_hash = hashlib.sha256(
                    (
                        source_hashes[f"{source_prefix}_atr"]
                        + source_hashes[f"{source_prefix}_smpr"]
                        + str(atr_row.get("checkpoint_sha256"))
                    ).encode("ascii")
                ).hexdigest()
                result[key] = {
                    "atr": atr,
                    "smpr": smpr,
                    "joint_score": joint_score,
                    "gate_pass": gate_pass,
                    "checkpoint_sha256": str(atr_row.get("checkpoint_sha256")),
                    "model_file": str(atr_row.get("model_file")),
                    "split_name": str(atr_row.get("split_name")),
                    "diagnostics_sha256": diagnostics_hash,
                    **values,
                }
    _require(len(result) == 64, "diagnostic matrix must contain 64 checkpoint-stressor rows")
    if baseline_path is not None:
        _require(len(baseline_index) == 64, "baseline diagnostic matrix must contain 64 rows")
    return result, baseline_status


def _strongest_mapping(
    path: Path,
    *,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
) -> dict[str, str]:
    payload = _load_json(path)
    rows = payload.get("rows", [])
    observed: dict[str, set[str]] = defaultdict(set)
    per_seed: set[tuple[int, str, str]] = set()
    for row in rows:
        seed = int(row.get("seed"))
        task = str(row.get("task"))
        family = str(row.get("family"))
        stressor = "blur" if family == "gaussian_blur" else family
        _require(seed in LEWM_SEEDS and task in TASKS and stressor in STRESSORS, f"{path}: invalid mapping row")
        observed[task].add(stressor)
        per_seed.add((seed, task, stressor))
    _require(len(per_seed) == 12, f"{path}: expected 12 pre-existing strongest rows")
    _require(all(len(observed[task]) == 1 for task in TASKS), f"{path}: mapping varies by seed")
    mapping = {task: next(iter(observed[task])) for task in TASKS}
    _require(
        mapping
        == {
            "TwoRoom": "blur",
            "Reacher": "blur",
            "PushT": "resize",
            "Cube": "resize",
        },
        f"{path}: strongest mapping differs from the pre-existing audit",
    )
    source_paths["strongest_mapping"] = str(path)
    source_hashes["strongest_mapping"] = _sha256(path)
    return mapping


def _retention(clean: float, stressed: float) -> float:
    _require(clean > 0.0, "clean score must be positive for retention")
    return stressed / clean


def _delta(endpoint: Any, base: Any) -> Any:
    if endpoint == "" or base == "":
        return ""
    return float(endpoint) - float(base)


def _rows(
    *,
    lewm_behavior: Mapping[tuple[str, int, str, str, str], Mapping[str, Any]],
    pldm_behavior: Mapping[tuple[str, int, str, str, str], Mapping[str, Any]],
    diagnostics: Mapping[tuple[str, int, str, str, str], Mapping[str, Any]],
    strongest_mapping: Mapping[str, str],
    protocol: Mapping[str, Any],
    baseline_status: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    behavior = {**lewm_behavior, **pldm_behavior}
    label = protocol.get("external_behavior_label", {})
    positive_delta = _finite(label.get("positive_delta_pp"), name="positive_delta_pp")
    neutral_band = _finite(label.get("neutral_band_pp"), name="neutral_band_pp")
    max_clean_drop = _finite(label.get("max_clean_drop_pp"), name="max_clean_drop_pp")
    pair_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    families = [("LeWM", seed, f"lewm_seed{seed}") for seed in LEWM_SEEDS] + [
        ("PLDM", 3072, "pldm_canonical_seed3072")
    ]
    for model_family, seed, family_id in families:
        seed_semantics = (
            "independently trained LeWM checkpoint seed"
            if model_family == "LeWM"
            else "one independently trained PLDM checkpoint family"
        )
        for task in TASKS:
            for stressor, spec in STRESSORS.items():
                base_key = (model_family, seed, task, stressor, "0.0")
                endpoint_key = (model_family, seed, task, stressor, "0.08")
                base_behavior = behavior[base_key]
                endpoint_behavior = behavior[endpoint_key]
                base_diag = diagnostics[base_key]
                endpoint_diag = diagnostics[endpoint_key]
                for block, diag, std_key in (
                    (base_behavior, base_diag, "0.0"),
                    (endpoint_behavior, endpoint_diag, "0.08"),
                ):
                    expected_name = Path(str(block["checkpoint_rel"])).name
                    actual_name = Path(str(diag["model_file"])).name
                    _require(
                        expected_name == actual_name,
                        f"{model_family}/{seed}/{task}/{stressor}/{std_key}: checkpoint filename mismatch",
                    )
                base_clean = float(base_behavior["clean"])
                endpoint_clean = float(endpoint_behavior["clean"])
                base_stressed = float(base_behavior["stressed"])
                endpoint_stressed = float(endpoint_behavior["stressed"])
                base_retention = _retention(base_clean, base_stressed)
                endpoint_retention = _retention(endpoint_clean, endpoint_stressed)
                delta_behavior = endpoint_stressed - base_stressed
                clean_drop = base_clean - endpoint_clean
                behavior_class = classify_transfer(
                    delta_behavior=delta_behavior,
                    clean_score_drop=clean_drop,
                    positive_delta_pp=positive_delta,
                    neutral_band_pp=neutral_band,
                    max_clean_drop_pp=max_clean_drop,
                )
                behavior_hash = hashlib.sha256(
                    (
                        str(base_behavior["source_sha256"])
                        + str(endpoint_behavior["source_sha256"])
                    ).encode("ascii")
                ).hexdigest()
                common = {
                    "model_family": model_family,
                    "training_seed_or_family_id": family_id,
                    "training_seed": seed,
                    "training_seed_semantics": seed_semantics,
                    "evaluation_seeds": "42;43;44",
                    "evaluation_seed_semantics": (
                        "conditional closed-loop evaluation replicates, not training seeds"
                    ),
                    "task": task,
                    "stressor_family": stressor,
                    "stressor_severity": spec["severity"],
                    "stressor_severity_parameter": spec["severity_parameter"],
                    "behavior_class": behavior_class,
                    "positive_transfer_label": behavior_class == "positive",
                    "split_name": str(endpoint_diag["split_name"]),
                    "protocol_hash": FROZEN_PROTOCOL_SHA256,
                    "behavior_source_sha256": behavior_hash,
                    "baseline_diagnostics_status": baseline_status,
                }
                pair = {
                    **common,
                    "base_rho": 0.0,
                    "endpoint_rho": FROZEN_RHO,
                    "base_checkpoint_sha256": base_diag["checkpoint_sha256"],
                    "endpoint_checkpoint_sha256": endpoint_diag["checkpoint_sha256"],
                    "base_clean_score": base_clean,
                    "endpoint_clean_score": endpoint_clean,
                    "base_stressed_score": base_stressed,
                    "endpoint_stressed_score": endpoint_stressed,
                    "base_retention": base_retention,
                    "endpoint_retention": endpoint_retention,
                    "delta_behavior": delta_behavior,
                    "clean_score_drop": clean_drop,
                    "delta_retention": endpoint_retention - base_retention,
                    "base_encoder_q90": base_diag["encoder_q90"],
                    "endpoint_encoder_q90": endpoint_diag["encoder_q90"],
                    "delta_encoder_q90": _delta(endpoint_diag["encoder_q90"], base_diag["encoder_q90"]),
                    "base_h1_q90": base_diag["h1_q90"],
                    "endpoint_h1_q90": endpoint_diag["h1_q90"],
                    "delta_h1_q90": _delta(endpoint_diag["h1_q90"], base_diag["h1_q90"]),
                    "base_action_shuffled_h8": base_diag["action_shuffled_h8_q90"],
                    "endpoint_action_shuffled_h8": endpoint_diag["action_shuffled_h8_q90"],
                    "delta_action_shuffled_h8": _delta(
                        endpoint_diag["action_shuffled_h8_q90"],
                        base_diag["action_shuffled_h8_q90"],
                    ),
                    "base_action_zeroed_h8": base_diag["action_zeroed_h8_q90"],
                    "endpoint_action_zeroed_h8": endpoint_diag["action_zeroed_h8_q90"],
                    "delta_action_zeroed_h8": _delta(
                        endpoint_diag["action_zeroed_h8_q90"],
                        base_diag["action_zeroed_h8_q90"],
                    ),
                    "base_time_shuffled_h8": base_diag["time_shuffled_h8_q90"],
                    "endpoint_time_shuffled_h8": endpoint_diag["time_shuffled_h8_q90"],
                    "delta_time_shuffled_h8": _delta(
                        endpoint_diag["time_shuffled_h8_q90"],
                        base_diag["time_shuffled_h8_q90"],
                    ),
                    "base_atr": base_diag["atr"],
                    "endpoint_atr": endpoint_diag["atr"],
                    "delta_atr": endpoint_diag["atr"] - base_diag["atr"],
                    "base_smpr": base_diag["smpr"],
                    "endpoint_smpr": endpoint_diag["smpr"],
                    "delta_smpr": endpoint_diag["smpr"] - base_diag["smpr"],
                    "base_joint_score": base_diag["joint_score"],
                    "endpoint_joint_score": endpoint_diag["joint_score"],
                    "delta_joint_score": endpoint_diag["joint_score"] - base_diag["joint_score"],
                    "base_gate_pass": base_diag["gate_pass"],
                    "endpoint_gate_pass": endpoint_diag["gate_pass"],
                    "base_diagnostics_sha256": base_diag["diagnostics_sha256"],
                    "endpoint_diagnostics_sha256": endpoint_diag["diagnostics_sha256"],
                }
                pair_rows.append(pair)
                if strongest_mapping[task] == stressor:
                    fixed_rows.append(
                        {
                            **common,
                            "rho": FROZEN_RHO,
                            "checkpoint_sha256": endpoint_diag["checkpoint_sha256"],
                            "clean_score": endpoint_clean,
                            "stressed_score": endpoint_stressed,
                            "clean_score_by_evaluation_seed": ";".join(
                                f"{value:.12g}" for value in endpoint_behavior["clean_values"]
                            ),
                            "stressed_score_by_evaluation_seed": ";".join(
                                f"{value:.12g}" for value in endpoint_behavior["stressed_values"]
                            ),
                            "retention": endpoint_retention,
                            "base_clean_score": base_clean,
                            "base_stressed_score": base_stressed,
                            "delta_behavior": delta_behavior,
                            "clean_score_drop": clean_drop,
                            "delta_retention": endpoint_retention - base_retention,
                            "encoder_q90": endpoint_diag["encoder_q90"],
                            "h1_q90": endpoint_diag["h1_q90"],
                            "action_shuffled_h8_q90": endpoint_diag["action_shuffled_h8_q90"],
                            "action_zeroed_h8_q90": endpoint_diag["action_zeroed_h8_q90"],
                            "time_shuffled_h8_q90": endpoint_diag["time_shuffled_h8_q90"],
                            "base_atr_h8_q90": base_diag["atr"],
                            "atr_h8_q90": endpoint_diag["atr"],
                            "delta_atr": endpoint_diag["atr"] - base_diag["atr"],
                            "base_smpr": base_diag["smpr"],
                            "smpr": endpoint_diag["smpr"],
                            "delta_smpr": endpoint_diag["smpr"] - base_diag["smpr"],
                            "base_joint_score": base_diag["joint_score"],
                            "joint_score": endpoint_diag["joint_score"],
                            "delta_joint_score": (
                                endpoint_diag["joint_score"] - base_diag["joint_score"]
                            ),
                            "base_joint_gate_pass": base_diag["gate_pass"],
                            "joint_gate_pass": endpoint_diag["gate_pass"],
                            "diagnostics_sha256": endpoint_diag["diagnostics_sha256"],
                        }
                    )
    fixed_rows.sort(
        key=lambda row: (
            row["model_family"],
            str(row["training_seed_or_family_id"]),
            TASKS.index(str(row["task"])),
        )
    )
    pair_rows.sort(
        key=lambda row: (
            row["model_family"],
            str(row["training_seed_or_family_id"]),
            TASKS.index(str(row["task"])),
            str(row["stressor_family"]),
        )
    )
    _require(len(fixed_rows) == 16, "fixed-rho endpoint table must contain 12 LeWM + 4 PLDM rows")
    _require(
        sum(row["model_family"] == "LeWM" for row in fixed_rows) == 12,
        "fixed-rho LeWM row count mismatch",
    )
    _require(
        sum(row["model_family"] == "PLDM" for row in fixed_rows) == 4,
        "fixed-rho PLDM row count mismatch",
    )
    _require(len(pair_rows) == 32, "all-pairs table must contain 24 LeWM + 8 PLDM rows")
    _require(
        sum(row["model_family"] == "LeWM" for row in pair_rows) == 24,
        "all-pairs LeWM row count mismatch",
    )
    _require(
        sum(row["model_family"] == "PLDM" for row in pair_rows) == 8,
        "all-pairs PLDM row count mismatch",
    )
    _require(
        {float(row["rho"]) for row in fixed_rows} == {FROZEN_RHO},
        "fixed-rho endpoint table contains mixed rho",
    )
    return fixed_rows, pair_rows


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2.0 + 1.0
        for index in order[start:stop]:
            ranks[index] = average_rank
        start = stop
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    groups: dict[float, list[bool]] = defaultdict(list)
    for label, score in zip(labels, scores):
        groups[score].append(label)
    tp = 0
    fp = 0
    recall = 0.0
    area = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        tp += sum(group)
        fp += len(group) - sum(group)
        next_recall = tp / positives
        precision = tp / (tp + fp)
        area += (next_recall - recall) * precision
        recall = next_recall
    return area


def _classification(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    tp = sum(label and pred for label, pred in zip(labels, predictions))
    tn = sum((not label) and (not pred) for label, pred in zip(labels, predictions))
    fp = sum((not label) and pred for label, pred in zip(labels, predictions))
    fn = sum(label and (not pred) for label, pred in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return {
        "n": len(labels),
        "positive_n": sum(labels),
        "predicted_positive_n": sum(predictions),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "false_pass_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [bool(row["positive_transfer_label"]) for row in rows]
    predictions = [bool(row["endpoint_gate_pass"] if "endpoint_gate_pass" in row else row["joint_gate_pass"]) for row in rows]
    scores = [
        float(row["endpoint_joint_score"] if "endpoint_joint_score" in row else row["joint_score"])
        for row in rows
    ]
    behavior = [float(row["delta_behavior"]) for row in rows]
    delta_joint = [
        float(row["delta_joint_score"])
        if "delta_joint_score" in row
        else float(row["joint_score"])
        for row in rows
    ]
    comparable = [
        (x, y)
        for x, y in zip(behavior, delta_joint)
        if x != 0.0 and y != 0.0
    ]
    classification = _classification(labels, predictions)
    classification["auprc"] = _average_precision(labels, scores)
    return {
        **classification,
        "pearson_delta_behavior_vs_endpoint_joint_score": _pearson(behavior, scores),
        "spearman_delta_behavior_vs_endpoint_joint_score": _spearman(behavior, scores),
        "pearson_delta_behavior_vs_delta_joint_score": _pearson(behavior, delta_joint),
        "spearman_delta_behavior_vs_delta_joint_score": _spearman(behavior, delta_joint),
        "signed_agreement_delta_behavior_vs_delta_joint_score": (
            sum((x > 0) == (y > 0) for x, y in comparable) / len(comparable)
            if comparable
            else None
        ),
        "signed_agreement_comparable_n": len(comparable),
    }


def _delta_metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    direction: float,
) -> dict[str, Any]:
    """Evaluate a no-retuning zero threshold on a paired diagnostic change."""
    _require(direction in {-1.0, 1.0}, "paired-change direction must be +/-1")
    labels = [bool(row["positive_transfer_label"]) for row in rows]
    scores = [direction * float(row[field]) for row in rows]
    predictions = [score > 0.0 for score in scores]
    behavior = [float(row["delta_behavior"]) for row in rows]
    comparable = [
        (behavior_delta, diagnostic_delta)
        for behavior_delta, diagnostic_delta in zip(behavior, scores)
        if behavior_delta != 0.0 and diagnostic_delta != 0.0
    ]
    classification = _classification(labels, predictions)
    classification["auprc"] = _average_precision(labels, scores)
    return {
        **classification,
        "paired_change_field": field,
        "higher_is_better_direction": direction,
        "decision_rule": "oriented endpoint-minus-base diagnostic change > 0",
        "decision_threshold": 0.0,
        "pearson_delta_behavior_vs_oriented_delta_score": _pearson(
            behavior,
            scores,
        ),
        "spearman_delta_behavior_vs_oriented_delta_score": _spearman(
            behavior,
            scores,
        ),
        "signed_agreement_delta_behavior_vs_oriented_delta_score": (
            sum((left > 0.0) == (right > 0.0) for left, right in comparable)
            / len(comparable)
            if comparable
            else None
        ),
        "signed_agreement_comparable_n": len(comparable),
    }


def _paired_change_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, display_name, field, direction in PAIRED_CHANGE_DIAGNOSTICS:
        result[key] = {
            "display_name": display_name,
            **_delta_metric_summary(rows, field=field, direction=direction),
        }
    return result


def _paired_blocks(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    _require(bool(rows), "paired audit requires at least one row")
    _require(
        all(row["model_family"] == "LeWM" for row in rows),
        "paired block audit is defined on the LeWM replication set",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task"]), str(row["training_seed_or_family_id"]))].append(row)
    _require(
        all(
            len(group) == 2
            and {str(row["stressor_family"]) for row in group} == set(STRESSORS)
            for group in groups.values()
        ),
        "each paired block must contain exactly one blur and one resize row",
    )
    return groups


def _selection_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    direction: float,
) -> dict[str, Any]:
    """Evaluate reference-versus-endpoint checkpoint choice without fitting."""
    _require(bool(rows), "selection audit requires at least one row")
    _require(direction in {-1.0, 1.0}, "selection direction must be +/-1")
    scores = [direction * float(row[field]) for row in rows]
    choose_endpoint = [score > 0.0 for score in scores]
    behavior = [float(row["delta_behavior"]) for row in rows]
    regrets: list[float] = []
    selected_scores: list[float] = []
    oracle_scores: list[float] = []
    for row, endpoint_selected in zip(rows, choose_endpoint):
        base = float(row["base_stressed_score"])
        endpoint = float(row["endpoint_stressed_score"])
        selected = endpoint if endpoint_selected else base
        oracle = max(base, endpoint)
        selected_scores.append(selected)
        oracle_scores.append(oracle)
        regrets.append(oracle - selected)

    comparable = [
        (delta, selected)
        for delta, selected in zip(behavior, choose_endpoint)
        if delta != 0.0
    ]
    material = [
        (delta, selected)
        for delta, selected in comparable
        if abs(delta) >= 5.0
    ]

    def choice_accuracy(items: Sequence[tuple[float, bool]]) -> float | None:
        if not items:
            return None
        return sum((delta > 0.0) == selected for delta, selected in items) / len(items)

    return {
        "n": len(rows),
        "decision_rule": "select endpoint iff oriented diagnostic change > 0",
        "comparable_n": len(comparable),
        "choice_accuracy": choice_accuracy(comparable),
        "material_change_threshold_pp": 5.0,
        "material_comparable_n": len(material),
        "material_choice_accuracy": choice_accuracy(material),
        "zero_regret_rate": sum(regret == 0.0 for regret in regrets) / len(regrets),
        "mean_regret_pp": _mean(regrets),
        "median_regret_pp": statistics.median(regrets),
        "q90_regret_pp": _quantile(regrets, 0.90),
        "max_regret_pp": max(regrets),
        "selected_mean_stressed_score": _mean(selected_scores),
        "oracle_mean_stressed_score": _mean(oracle_scores),
    }


def _joint_failure_map(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        prediction = float(row["delta_joint_score"]) > 0.0
        truth = bool(row["positive_transfer_label"])
        if prediction == truth:
            continue
        base = float(row["base_stressed_score"])
        endpoint = float(row["endpoint_stressed_score"])
        selected = endpoint if prediction else base
        failures.append(
            {
                "error_type": "false_positive" if prediction else "false_negative",
                "task": row["task"],
                "training_seed_or_family_id": row["training_seed_or_family_id"],
                "stressor_family": row["stressor_family"],
                "behavior_class": row["behavior_class"],
                "delta_behavior": float(row["delta_behavior"]),
                "clean_score_drop": float(row["clean_score_drop"]),
                "delta_joint_score": float(row["delta_joint_score"]),
                "selection_regret_pp": max(base, endpoint) - selected,
            }
        )
    return failures


def _deletion_stability(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
) -> dict[str, Any]:
    """Report deletion sensitivity; no model or threshold is fitted in a fold."""

    def compact(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = _delta_metric_summary(
            group,
            field="delta_joint_score",
            direction=1.0,
        )
        selection = _selection_summary(
            group,
            field="delta_joint_score",
            direction=1.0,
        )
        return {
            "n": len(group),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "auprc": metrics["auprc"],
            "spearman": metrics[
                "spearman_delta_behavior_vs_oriented_delta_score"
            ],
            "signed_agreement": metrics[
                "signed_agreement_delta_behavior_vs_oriented_delta_score"
            ],
            "choice_accuracy": selection["choice_accuracy"],
            "mean_regret_pp": selection["mean_regret_pp"],
        }

    values = sorted({str(row[group_field]) for row in rows})
    folds: list[dict[str, Any]] = []
    for value in values:
        held_out = [row for row in rows if str(row[group_field]) == value]
        remaining = [row for row in rows if str(row[group_field]) != value]
        folds.append(
            {
                "held_out_value": value,
                "held_out": compact(held_out),
                "remaining_after_deletion": compact(remaining),
            }
        )

    range_fields = (
        "balanced_accuracy",
        "auprc",
        "spearman",
        "signed_agreement",
        "choice_accuracy",
        "mean_regret_pp",
    )
    ranges: dict[str, list[float] | None] = {}
    for field in range_fields:
        samples = [
            float(fold["remaining_after_deletion"][field])
            for fold in folds
            if fold["remaining_after_deletion"][field] is not None
        ]
        ranges[field] = [min(samples), max(samples)] if samples else None
    return {
        "interpretation": (
            "deletion stability of the fixed zero-threshold rule; not fitted "
            "cross-validation"
        ),
        "group_field": group_field,
        "folds": folds,
        "remaining_metric_range": ranges,
    }


def _exact_binomial_upper_tail(successes: int, trials: int) -> float:
    _require(0 <= successes <= trials, "invalid binomial counts")
    _require(trials > 0, "binomial test requires trials")
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (
        2**trials
    )


def _exact_randomization_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Exact block sign-flip test under a blockwise sign-symmetry null."""
    groups = _paired_blocks(rows)
    block_keys = sorted(groups)
    ordered_groups = [
        sorted(groups[key], key=lambda row: str(row["stressor_family"]))
        for key in block_keys
    ]
    behavior = [
        float(row["delta_behavior"])
        for group in ordered_groups
        for row in group
    ]
    oriented = [
        float(row["delta_joint_score"])
        for group in ordered_groups
        for row in group
    ]
    observed = _spearman(behavior, oriented)
    _require(observed is not None, "observed Spearman correlation is undefined")

    null_statistics: list[float] = []
    for mask in range(1 << len(block_keys)):
        randomized: list[float] = []
        for block_index, group in enumerate(ordered_groups):
            sign = -1.0 if mask & (1 << block_index) else 1.0
            randomized.extend(
                sign * float(row["delta_joint_score"]) for row in group
            )
        statistic = _spearman(behavior, randomized)
        _require(statistic is not None, "randomized Spearman correlation is undefined")
        null_statistics.append(float(statistic))

    tolerance = 1e-12
    comparable = [
        (left, right)
        for left, right in zip(behavior, oriented)
        if left != 0.0 and right != 0.0
    ]
    agreement_count = sum((left > 0.0) == (right > 0.0) for left, right in comparable)
    return {
        "primary_test": "exact task-x-training-seed block sign-flip test",
        "null_hypothesis": (
            "joint diagnostic-change signs are blockwise sign-symmetric relative "
            "to behavior changes"
        ),
        "block_count": len(block_keys),
        "enumerated_assignments": len(null_statistics),
        "observed_spearman": observed,
        "one_sided_p_value": sum(
            value >= observed - tolerance for value in null_statistics
        )
        / len(null_statistics),
        "two_sided_p_value": sum(
            abs(value) >= abs(observed) - tolerance for value in null_statistics
        )
        / len(null_statistics),
        "null_spearman_ci95": [
            _quantile(null_statistics, 0.025),
            _quantile(null_statistics, 0.975),
        ],
        "row_level_signed_agreement": {
            "successes": agreement_count,
            "trials": len(comparable),
            "one_sided_exact_binomial_p_value": _exact_binomial_upper_tail(
                agreement_count,
                len(comparable),
            ),
            "dependence_warning": (
                "row-level binomial calculation is descriptive; the block sign-"
                "flip test is primary"
            ),
        },
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "cannot compute an empty quantile")
    _require(0.0 <= probability <= 1.0, "quantile probability must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    observed: float | None,
    values: Sequence[float],
) -> dict[str, Any]:
    _require(observed is not None, "observed bootstrap metric must be defined")
    _require(bool(values), "bootstrap metric has no valid repetitions")
    return {
        "observed": float(observed),
        "bootstrap_median": _quantile(values, 0.50),
        "ci95": [_quantile(values, 0.025), _quantile(values, 0.975)],
        "valid_repetitions": len(values),
    }


def _paired_incremental_bootstrap(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Paired block-bootstrap contrasts between joint and component diagnostics."""
    _require(len(rows) == 24, "incremental audit expects 24 LeWM rows")
    groups = _paired_blocks(rows)
    _require(len(groups) == 12, "incremental audit expects 12 LeWM blocks")
    block_keys = sorted(groups)
    comparators = [
        (key, display_name, field, direction)
        for key, display_name, field, direction in PAIRED_CHANGE_DIAGNOSTICS
        if key != "joint_score"
    ]

    joint_observed = _delta_metric_summary(
        rows,
        field="delta_joint_score",
        direction=1.0,
    )
    joint_selection = _selection_summary(
        rows,
        field="delta_joint_score",
        direction=1.0,
    )
    observed: dict[str, dict[str, float | None]] = {}
    samples: dict[str, dict[str, list[float]]] = {}
    for key, _display_name, field, direction in comparators:
        comparator = _delta_metric_summary(rows, field=field, direction=direction)
        comparator_selection = _selection_summary(
            rows,
            field=field,
            direction=direction,
        )
        observed[key] = {
            "delta_spearman_joint_minus_comparator": (
                float(
                    joint_observed[
                        "spearman_delta_behavior_vs_oriented_delta_score"
                    ]
                )
                - float(
                    comparator[
                        "spearman_delta_behavior_vs_oriented_delta_score"
                    ]
                )
            ),
            "delta_balanced_accuracy_joint_minus_comparator": (
                None
                if joint_observed["balanced_accuracy"] is None
                or comparator["balanced_accuracy"] is None
                else float(joint_observed["balanced_accuracy"])
                - float(comparator["balanced_accuracy"])
            ),
            "delta_choice_accuracy_joint_minus_comparator": (
                None
                if joint_selection["choice_accuracy"] is None
                or comparator_selection["choice_accuracy"] is None
                else float(joint_selection["choice_accuracy"])
                - float(comparator_selection["choice_accuracy"])
            ),
            "mean_regret_reduction_pp_comparator_minus_joint": (
                float(comparator_selection["mean_regret_pp"])
                - float(joint_selection["mean_regret_pp"])
            ),
        }
        samples[key] = {metric: [] for metric in observed[key]}

    rng = random.Random(PAIRED_CHANGE_INCREMENTAL_BOOTSTRAP_SEED)
    for _ in range(PAIRED_CHANGE_BOOTSTRAP_REPETITIONS):
        sampled_rows: list[Mapping[str, Any]] = []
        for _block in block_keys:
            sampled_rows.extend(groups[block_keys[rng.randrange(len(block_keys))]])

        joint_metric = _delta_metric_summary(
            sampled_rows,
            field="delta_joint_score",
            direction=1.0,
        )
        joint_choice = _selection_summary(
            sampled_rows,
            field="delta_joint_score",
            direction=1.0,
        )
        for key, _display_name, field, direction in comparators:
            comparator_metric = _delta_metric_summary(
                sampled_rows,
                field=field,
                direction=direction,
            )
            comparator_choice = _selection_summary(
                sampled_rows,
                field=field,
                direction=direction,
            )
            bootstrap_values: dict[str, float | None] = {
                "delta_spearman_joint_minus_comparator": (
                    None
                    if joint_metric[
                        "spearman_delta_behavior_vs_oriented_delta_score"
                    ]
                    is None
                    or comparator_metric[
                        "spearman_delta_behavior_vs_oriented_delta_score"
                    ]
                    is None
                    else float(
                        joint_metric[
                            "spearman_delta_behavior_vs_oriented_delta_score"
                        ]
                    )
                    - float(
                        comparator_metric[
                            "spearman_delta_behavior_vs_oriented_delta_score"
                        ]
                    )
                ),
                "delta_balanced_accuracy_joint_minus_comparator": (
                    None
                    if joint_metric["balanced_accuracy"] is None
                    or comparator_metric["balanced_accuracy"] is None
                    else float(joint_metric["balanced_accuracy"])
                    - float(comparator_metric["balanced_accuracy"])
                ),
                "delta_choice_accuracy_joint_minus_comparator": (
                    None
                    if joint_choice["choice_accuracy"] is None
                    or comparator_choice["choice_accuracy"] is None
                    else float(joint_choice["choice_accuracy"])
                    - float(comparator_choice["choice_accuracy"])
                ),
                "mean_regret_reduction_pp_comparator_minus_joint": (
                    float(comparator_choice["mean_regret_pp"])
                    - float(joint_choice["mean_regret_pp"])
                ),
            }
            for metric, value in bootstrap_values.items():
                if value is not None:
                    samples[key][metric].append(float(value))

    return {
        "estimand": (
            "joint minus comparator for association/classification/choice; "
            "comparator minus joint for regret so positive favors joint"
        ),
        "resampling_unit": "task x independently trained LeWM checkpoint seed",
        "stressors_retained_within_block": sorted(STRESSORS),
        "block_count": len(block_keys),
        "repetitions": PAIRED_CHANGE_BOOTSTRAP_REPETITIONS,
        "seed": PAIRED_CHANGE_INCREMENTAL_BOOTSTRAP_SEED,
        "comparisons": {
            key: {
                "display_name": display_name,
                "metrics": {
                    metric: _bootstrap_interval(
                        observed[key][metric],
                        samples[key][metric],
                    )
                    for metric in observed[key]
                },
            }
            for key, display_name, _field, _direction in comparators
        },
    }


def _lewm_paired_change_bootstrap(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resample LeWM task/training-seed blocks while retaining both stressors."""
    _require(len(rows) == 24, "LeWM paired-change bootstrap expects 24 rows")
    _require(
        all(row["model_family"] == "LeWM" for row in rows),
        "LeWM paired-change bootstrap received another model family",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task"]), str(row["training_seed_or_family_id"]))].append(row)
    _require(len(groups) == 12, "LeWM paired-change bootstrap expects 12 blocks")
    _require(
        all(
            len(group) == 2
            and {str(row["stressor_family"]) for row in group} == set(STRESSORS)
            for group in groups.values()
        ),
        "each LeWM paired-change block must retain blur and resize",
    )
    block_keys = sorted(groups)
    observed = _delta_metric_summary(
        rows,
        field="delta_joint_score",
        direction=1.0,
    )
    metric_fields = (
        "balanced_accuracy",
        "auprc",
        "precision",
        "recall",
        "spearman_delta_behavior_vs_oriented_delta_score",
        "signed_agreement_delta_behavior_vs_oriented_delta_score",
    )
    samples: dict[str, list[float]] = {field: [] for field in metric_fields}
    rng = random.Random(PAIRED_CHANGE_BOOTSTRAP_SEED)
    for _ in range(PAIRED_CHANGE_BOOTSTRAP_REPETITIONS):
        sampled_rows: list[Mapping[str, Any]] = []
        for _ in block_keys:
            sampled_rows.extend(groups[block_keys[rng.randrange(len(block_keys))]])
        metrics = _delta_metric_summary(
            sampled_rows,
            field="delta_joint_score",
            direction=1.0,
        )
        for field in metric_fields:
            value = metrics[field]
            if value is not None:
                samples[field].append(float(value))
    return {
        "resampling_unit": "task x independently trained LeWM checkpoint seed",
        "stressors_retained_within_block": sorted(STRESSORS),
        "block_count": len(block_keys),
        "repetitions": PAIRED_CHANGE_BOOTSTRAP_REPETITIONS,
        "seed": PAIRED_CHANGE_BOOTSTRAP_SEED,
        "interval": "percentile 95% block-bootstrap interval",
        "metrics": {
            field: _bootstrap_interval(observed[field], samples[field])
            for field in metric_fields
        },
    }


def _paired_robustness_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(len(rows) == 24, "paired robustness audit expects 24 LeWM rows")
    _require(len(_paired_blocks(rows)) == 12, "paired robustness audit expects 12 blocks")
    selection_by_diagnostic = {
        key: {
            "display_name": display_name,
            **_selection_summary(rows, field=field, direction=direction),
        }
        for key, display_name, field, direction in PAIRED_CHANGE_DIAGNOSTICS
    }
    failures = _joint_failure_map(rows)
    return {
        "scope": (
            "fixed zero-threshold LeWM paired comparison across blur and resize; "
            "all analyses are post-freeze and fit no stressor-specific threshold"
        ),
        "row_count": len(rows),
        "block_count": 12,
        "exact_randomization": _exact_randomization_audit(rows),
        "deletion_stability": {
            "leave_one_task_out": _deletion_stability(
                rows,
                group_field="task",
            ),
            "leave_one_training_seed_out": _deletion_stability(
                rows,
                group_field="training_seed_or_family_id",
            ),
        },
        "joint_failure_map": {
            "count": len(failures),
            "rows": failures,
        },
        "selection_by_diagnostic": selection_by_diagnostic,
        "incremental_block_bootstrap": _paired_incremental_bootstrap(rows),
    }


def _paired_change_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_model_family: dict[str, Any] = {}
    for model_family in ("LeWM", "PLDM"):
        family_rows = [row for row in rows if row["model_family"] == model_family]
        by_model_family[model_family] = {
            "n": len(family_rows),
            "diagnostics": _paired_change_diagnostics(family_rows),
            "joint_by_stressor": {
                stressor: _delta_metric_summary(
                    [
                        row
                        for row in family_rows
                        if row["stressor_family"] == stressor
                    ],
                    field="delta_joint_score",
                    direction=1.0,
                )
                for stressor in sorted(STRESSORS)
            },
        }
    return {
        "estimand": (
            "within-family endpoint-minus-base diagnostic change versus the "
            "corresponding closed-loop stressed-success change"
        ),
        "scope": (
            "a relative checkpoint comparison requiring a reference checkpoint; "
            "not an absolute cross-family robustness scale"
        ),
        "threshold_contract": (
            "one natural zero threshold is shared across blur and resize; no "
            "stressor-specific threshold search"
        ),
        "all_rows": {
            "n": len(rows),
            "diagnostics": _paired_change_diagnostics(rows),
        },
        "by_model_family": by_model_family,
        "lewm_block_bootstrap": _lewm_paired_change_bootstrap(
            [row for row in rows if row["model_family"] == "LeWM"]
        ),
        "lewm_robustness_audit": _paired_robustness_audit(
            [row for row in rows if row["model_family"] == "LeWM"]
        ),
    }


def _strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "model_family",
        "task",
        "stressor_family",
        "training_seed_or_family_id",
    ):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[field] = [
            {"value": value, **_metric_summary(group)}
            for value, group in sorted(groups.items())
        ]
    return result


def _discordant(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        prediction = bool(
            row["endpoint_gate_pass"] if "endpoint_gate_pass" in row else row["joint_gate_pass"]
        )
        truth = bool(row["positive_transfer_label"])
        if prediction == truth:
            continue
        result.append(
            {
                "model_family": row["model_family"],
                "training_seed_or_family_id": row["training_seed_or_family_id"],
                "task": row["task"],
                "stressor_family": row["stressor_family"],
                "behavior_class": row["behavior_class"],
                "delta_behavior": row["delta_behavior"],
                "clean_score_drop": row["clean_score_drop"],
                "joint_score": (
                    row["endpoint_joint_score"]
                    if "endpoint_joint_score" in row
                    else row["joint_score"]
                ),
                "gate_pass": prediction,
                "discordance": "false_pass" if prediction else "false_negative",
            }
        )
    return result


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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tex_metric(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _write_paired_change_table(
    path: Path,
    *,
    absolute_lewm: Mapping[str, Any],
    paired_change: Mapping[str, Any],
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    family = paired_change["by_model_family"]
    lewm_joint = family["LeWM"]["diagnostics"]["joint_score"]
    lewm_blur = family["LeWM"]["joint_by_stressor"]["blur"]
    lewm_resize = family["LeWM"]["joint_by_stressor"]["resize"]
    pldm_joint = family["PLDM"]["diagnostics"]["joint_score"]

    def table_row(
        mode: str,
        scope: str,
        metrics: Mapping[str, Any],
        spearman_field: str,
    ) -> str:
        return (
            f"{mode} & {scope} & {int(metrics['n'])} & "
            f"{_tex_metric(metrics['balanced_accuracy'])} & "
            f"{_tex_metric(metrics['auprc'])} & "
            f"{_tex_metric(metrics['precision'])} & "
            f"{_tex_metric(metrics['recall'])} & "
            f"{_tex_metric(metrics[spearman_field])} \\\\"
        )

    paired_spearman = "spearman_delta_behavior_vs_oriented_delta_score"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Absolute screening and paired comparison are distinct uses. "
            r"The absolute row applies the LeWM Gaussian-frozen joint gate to all "
            r"LeWM blur/resize endpoints. Paired rows use one untuned rule, "
            r"$\Delta$joint score $>0$, for both stressors. PLDM contains one "
            r"training family and is exploratory. $\rho_s$ correlates continuous "
            r"diagnostic and stressed-success changes.}"
        ),
        r"\label{tab:cross-stressor-paired-change}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        "Use & scope & $n$ & BA & AUPRC & precision & recall & $\\rho_s$ \\\\",
        r"\midrule",
        table_row(
            "absolute",
            "LeWM blur+resize",
            absolute_lewm,
            "spearman_delta_behavior_vs_endpoint_joint_score",
        ),
        r"\midrule",
        table_row(
            r"paired $\Delta>0$",
            "LeWM blur+resize",
            lewm_joint,
            paired_spearman,
        ),
        table_row(r"paired $\Delta>0$", "LeWM blur", lewm_blur, paired_spearman),
        table_row(
            r"paired $\Delta>0$",
            "LeWM resize",
            lewm_resize,
            paired_spearman,
        ),
        table_row(
            r"paired $\Delta>0$",
            "PLDM blur+resize",
            pldm_joint,
            paired_spearman,
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_robustness_audit_table(
    path: Path,
    *,
    paired_change: Mapping[str, Any],
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    lewm_metrics = paired_change["by_model_family"]["LeWM"]["diagnostics"]
    audit = paired_change["lewm_robustness_audit"]
    selection = audit["selection_by_diagnostic"]
    exact = audit["exact_randomization"]
    task_range = audit["deletion_stability"]["leave_one_task_out"][
        "remaining_metric_range"
    ]["spearman"]
    seed_range = audit["deletion_stability"]["leave_one_training_seed_out"][
        "remaining_metric_range"
    ]["spearman"]

    def table_row(key: str) -> str:
        metrics = lewm_metrics[key]
        selected = selection[key]
        return (
            f"{metrics['display_name']} & "
            f"{_tex_metric(metrics['balanced_accuracy'])} & "
            f"{_tex_metric(metrics['auprc'])} & "
            f"{_tex_metric(metrics['spearman_delta_behavior_vs_oriented_delta_score'])} & "
            f"{_tex_metric(selected['choice_accuracy'])} & "
            f"{_tex_metric(selected['mean_regret_pp'])} \\\\"
        )

    task_text = (
        "--"
        if task_range is None
        else f"[{float(task_range[0]):.3f},{float(task_range[1]):.3f}]"
    )
    seed_text = (
        "--"
        if seed_range is None
        else f"[{float(seed_range[0]):.3f},{float(seed_range[1]):.3f}]"
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Cross-stressor component comparison on all 24 LeWM blur/resize "
            r"pairs. Choice accuracy selects the endpoint iff the oriented "
            r"diagnostic change is positive; regret is stressed-success loss "
            r"against the better of base and endpoint. The exact 12-block "
            f"sign-flip test gives one-sided $p={float(exact['one_sided_p_value']):.4f}$. "
            f"Remaining-set Spearman ranges after deleting one task and one "
            f"training seed are {task_text} and {seed_text}. "
            r"Competitive component rows show that paired ordering transfers "
            r"without establishing a unique action-specific mechanism.}"
        ),
        r"\label{tab:cross-stressor-robustness-audit}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Diagnostic & BA & AUPRC & $\rho_s$ & choice acc. & regret (pp) \\",
        r"\midrule",
        *[table_row(key) for key, _name, _field, _direction in PAIRED_CHANGE_DIAGNOSTICS],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, rows: Sequence[Mapping[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"blur": "#4c78a8", "resize": "#f28e2b"}
    markers = {"LeWM": "o", "PLDM": "s"}
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        axis.scatter(
            float(row["delta_joint_score"]),
            float(row["delta_behavior"]),
            color=colors[str(row["stressor_family"])],
            marker=markers[str(row["model_family"])],
            s=52,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
    axis.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axis.axhline(5.0, color="#777777", linewidth=0.8, linestyle=":")
    axis.axhline(-5.0, color="#777777", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Endpoint minus base joint diagnostic score")
    axis.set_ylabel("Endpoint minus base stressed success (pp)")
    axis.set_title("Paired diagnostic change across fixed blur/resize stressors")
    axis.grid(alpha=0.18)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[stressor], label=stressor)
        for stressor in ("blur", "resize")
    ] + [
        plt.Line2D([], [], marker=markers[family], linestyle="", color="#333333", label=family)
        for family in ("LeWM", "PLDM")
    ]
    axis.legend(handles=handles, frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _default_data_root() -> Path:
    for name in ("PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    candidates = (
        Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll"),
        Path("/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"),
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json",
    )
    parser.add_argument(
        "--cross-source-root",
        type=Path,
        default=ROOT / "paper1/results/remediation_phase2_external_sources/cross_stressor",
    )
    parser.add_argument(
        "--lewm-behavior-template",
        default="assets/paper1_data/unseen_origin_vs_std008_strongest_s{seed}.json",
    )
    parser.add_argument(
        "--pldm-clean-manifest",
        type=Path,
        default=ROOT / "assets/paper1_data/canonical_evals_pldm_v2.json",
    )
    parser.add_argument(
        "--pldm-blur-baseline",
        type=Path,
        default=ROOT / "assets/paper1_data/canonical_blur_baselines_20260523.json",
    )
    parser.add_argument(
        "--pldm-blur-jobs",
        type=Path,
        default=ROOT / (
            "paper1/results/remediation_phase2_external_sources/cross_stressor/"
            "pldm_canonical/gaussian_blur/behavior_eval_jobs.json"
        ),
    )
    parser.add_argument(
        "--pldm-resize-jobs",
        type=Path,
        default=ROOT / (
            "paper1/results/remediation_phase2_external_sources/cross_stressor/"
            "pldm_canonical/resize/behavior_eval_jobs.json"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument(
        "--strongest-mapping-source",
        type=Path,
        default=ROOT / "assets/paper1_data/unseen_atr_smpr_summary_20260707.json",
    )
    parser.add_argument(
        "--baseline-diagnostics",
        type=Path,
        default=None,
        help="Optional canonical Phase-3 baseline artifact; its 64 E3 rows are selected strictly.",
    )
    parser.add_argument(
        "--fixed-rows",
        type=Path,
        default=ROOT / "paper1/results/external_validation/cross_stressor_fixed_rho_rows.csv",
    )
    parser.add_argument(
        "--all-pairs",
        type=Path,
        default=ROOT / "paper1/results/external_validation/cross_stressor_all_pairs.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "paper1/results/external_validation/cross_stressor_fixed_rho_summary.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper1/tables/table_cross_stressor_paired_change.tex",
    )
    parser.add_argument(
        "--audit-table",
        type=Path,
        default=ROOT / "paper1/tables/table_cross_stressor_robustness_audit.tex",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "assets/paper1_figs/fig_cross_stressor_fixed_rho.png",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protocol, protocol_bytes, protocol_mtime_ns = _load_protocol(args.protocol)
    source_paths: dict[str, str] = {"protocol": str(args.protocol)}
    source_hashes: dict[str, str] = {"protocol": FROZEN_PROTOCOL_SHA256}
    lewm_paths = {
        seed: ROOT / args.lewm_behavior_template.format(seed=seed)
        for seed in LEWM_SEEDS
    }

    # Score diagnostics first. Behavior sources are not passed to this function.
    diagnostics, baseline_status = _load_diagnostics(
        cross_source_root=args.cross_source_root,
        protocol=protocol,
        source_paths=source_paths,
        source_hashes=source_hashes,
        baseline_path=args.baseline_diagnostics,
    )
    strongest_mapping = _strongest_mapping(
        args.strongest_mapping_source,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )
    lewm_behavior = _load_lewm_behavior(
        lewm_paths,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )
    pldm_behavior = _load_pldm_behavior(
        clean_manifest_path=args.pldm_clean_manifest,
        blur_baseline_path=args.pldm_blur_baseline,
        blur_jobs_path=args.pldm_blur_jobs,
        resize_jobs_path=args.pldm_resize_jobs,
        data_root=args.data_root,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )
    fixed_rows, pair_rows = _rows(
        lewm_behavior=lewm_behavior,
        pldm_behavior=pldm_behavior,
        diagnostics=diagnostics,
        strongest_mapping=strongest_mapping,
        protocol=protocol,
        baseline_status=baseline_status,
    )
    fixed_metrics = _metric_summary(fixed_rows)
    pair_metrics = _metric_summary(pair_rows)
    paired_change = _paired_change_summary(pair_rows)
    absolute_lewm = _metric_summary(
        [row for row in pair_rows if row["model_family"] == "LeWM"]
    )

    script_path = Path(__file__).resolve()
    source_paths["builder"] = str(script_path.relative_to(ROOT))
    source_hashes["builder"] = _sha256(script_path)
    summary = {
        "metadata": {
            "schema_version": "paper1-cross-stressor-fixed-rho-1.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "status": (
                "complete"
                if baseline_status == "complete"
                else "complete_core_atr_smpr_behavior__phase3_baselines_pending"
            ),
            "protocol_hash": FROZEN_PROTOCOL_SHA256,
            "threshold_search_allowed": False,
            "severity_search_allowed": False,
            "paired_change_threshold_search_allowed": False,
            "paired_change_zero_threshold": 0.0,
            "robustness_audit_post_freeze": True,
            "robustness_audit_threshold_search_allowed": False,
            "absolute_calibration_scope": "model-family-specific",
            "analysis_interface_portability": "LeWM and PLDM",
            "behavior_loaded_after_frozen_diagnostic_scoring": True,
            "rho": FROZEN_RHO,
            "rho_unique_values": [FROZEN_RHO],
            "evaluation_seeds": [42, 43, 44],
            "evaluation_seed_semantics": (
                "conditional closed-loop evaluation replicates, not training seeds"
            ),
            "training_seed_semantics": {
                "LeWM": "three independently trained checkpoint seeds: 3072, 3073, 3074",
                "PLDM": "one independently trained canonical checkpoint family; eval seeds are not training seeds",
            },
            "baseline_diagnostics_status": baseline_status,
            "missing_baseline_fields": (
                [] if baseline_status == "complete" else list(BASELINE_FIELDS)
            ),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "errors": [],
            "missing_rows": [],
        },
        "count_contract": {
            "fixed_endpoint_rows": {
                "expected_total": 16,
                "observed_total": len(fixed_rows),
                "expected_lewm": 12,
                "observed_lewm": sum(row["model_family"] == "LeWM" for row in fixed_rows),
                "expected_pldm": 4,
                "observed_pldm": sum(row["model_family"] == "PLDM" for row in fixed_rows),
            },
            "all_base_to_endpoint_pairs": {
                "expected_total": 32,
                "observed_total": len(pair_rows),
                "expected_lewm": 24,
                "observed_lewm": sum(row["model_family"] == "LeWM" for row in pair_rows),
                "expected_pldm": 8,
                "observed_pldm": sum(row["model_family"] == "PLDM" for row in pair_rows),
            },
        },
        "strongest_task_stressor_mapping": dict(strongest_mapping),
        "behavior_label_rule": {
            **protocol["external_behavior_label"],
            "positive": "delta_behavior >= +5pp and clean_score_drop <= 5pp",
            "neutral": "delta_behavior in (-5,+5)pp and clean_score_drop <= 5pp",
            "negative": "delta_behavior <= -5pp or clean_score_drop > 5pp",
            "classification_use": "compact summary only; continuous deltas are retained",
        },
        "fixed_endpoint_metrics": fixed_metrics,
        "all_pair_metrics": pair_metrics,
        "paired_change": paired_change,
        "fixed_endpoint_strata": _strata(fixed_rows),
        "all_pair_strata": _strata(pair_rows),
        "fixed_endpoint_discordant_rows": _discordant(fixed_rows),
        "all_pair_discordant_rows": _discordant(pair_rows),
    }
    _write_csv(args.fixed_rows, fixed_rows, FIXED_FIELDS, force=args.force)
    _write_csv(args.all_pairs, pair_rows, PAIR_FIELDS, force=args.force)
    _write_json(args.summary, summary, force=args.force)
    _write_paired_change_table(
        args.table,
        absolute_lewm=absolute_lewm,
        paired_change=paired_change,
        force=args.force,
    )
    _write_robustness_audit_table(
        args.audit_table,
        paired_change=paired_change,
        force=args.force,
    )
    _plot(args.figure, pair_rows, force=args.force)
    _assert_protocol_unchanged(args.protocol, protocol_bytes, protocol_mtime_ns)
    print(
        "[cross-stressor] wrote "
        f"{len(fixed_rows)} fixed endpoint rows and {len(pair_rows)} all-pairs rows"
    )
    print(f"[cross-stressor] baseline diagnostics: {baseline_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
