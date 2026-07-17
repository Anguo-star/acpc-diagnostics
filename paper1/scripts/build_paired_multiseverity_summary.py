#!/usr/bin/env python3
"""Build the frozen LeWM paired multi-severity behavior/ATR/SMPR analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.build_cross_stressor_external_validation import (
    _deletion_stability,
    _delta_metric_summary,
    _exact_binomial_upper_tail,
    _joint,
    _quantile,
    _selection_summary,
    _spearman,
    classify_transfer,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "paper1/config/paired_multiseverity_protocol_v1.json"
DEFAULT_ADDENDUM = (
    ROOT / "paper1/config/paired_multiseverity_execution_addendum_v2.json"
)
DEFAULT_LEGACY = (
    ROOT / "paper1/results/external_validation/cross_stressor_all_pairs.csv"
)
DEFAULT_RAW_ROOT = ROOT / "paper1/results/multiseverity_v1/raw"
LEGACY_STRONGEST_ROWS_SHA256 = "dbf26c84bb3670dd8dd00a3a535fd2beaf6157a7abdeeb23d0baea0579f07b02"
DEFAULT_REFERENCE_ROOT = ROOT / "paper1/results/multiseverity_v1/reference"
DEFAULT_MANIFEST_ROOT = ROOT / "paper1/results/multiseverity_v1/manifests"
DEFAULT_OUT = ROOT / "paper1/results/multiseverity_v1/paired_multiseverity_summary.json"
DEFAULT_ROWS_OUT = ROOT / "paper1/results/multiseverity_v1/paired_multiseverity_rows.csv"
TASK_SLUG = {
    "TwoRoom": "tworoom",
    "PushT": "pusht",
    "Reacher": "reacher",
    "Cube": "cube",
}
FAMILY_SPEC = {
    "gaussian_blur": {
        "row_name": "blur",
        "severity_parameter": "kernel_size",
        "group_prefix": "pixels_blur_ks",
    },
    "resize": {
        "row_name": "resize",
        "severity_parameter": "scale_factor",
        "group_prefix": "pixels_rs_factor",
    },
}
CSV_FIELDS = (
    "model_family",
    "training_seed_or_family_id",
    "training_seed",
    "task",
    "stressor_family",
    "stressor_severity",
    "stressor_severity_parameter",
    "base_clean_score",
    "endpoint_clean_score",
    "base_stressed_score",
    "endpoint_stressed_score",
    "delta_behavior",
    "clean_score_drop",
    "behavior_class",
    "positive_transfer_label",
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
    "selection_regret_pp",
    "source_stage",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    _require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def _finite(value: Any, name: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{name}: expected finite numeric value")
    return result


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() == "true":
        return True
    if str(value).lower() == "false":
        return False
    raise ValueError(f"{name}: expected boolean")


def _slug_number(value: float) -> str:
    return f"{value:g}"


def _severity_dir(family: str, severity: float) -> str:
    if family == "gaussian_blur":
        return f"gaussian_blur_ks{_slug_number(severity)}"
    return f"resize_factor{_slug_number(severity).replace('.', 'p')}"


def _portable(path: Path, data_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        pass
    try:
        return "$PAPER1_DATA_ROOT/" + str(resolved.relative_to(data_root.resolve()))
    except ValueError:
        return str(resolved)


def _verify_sidecar(path: Path) -> str:
    sidecar_path = path.with_suffix(".sha256")
    tokens = sidecar_path.read_text(encoding="utf-8").split()
    _require(len(tokens) == 2, f"{sidecar_path}: invalid sidecar")
    digest = _sha256(path)
    _require(tokens[0] == digest, f"{path}: sidecar hash mismatch")
    return digest


def _summary_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("metric") != "success_rate":
                continue
            group = str(raw["group"])
            _require(group not in rows, f"{path}: duplicate success-rate group {group}")
            values = [
                _finite(value, f"{path}/{group}/value")
                for value in str(raw["values"]).split(";")
                if value != ""
            ]
            _require(int(raw["n_seeds"]) == 3, f"{path}/{group}: expected three eval seeds")
            _require(raw["seeds"] == "42,43,44", f"{path}/{group}: eval seeds changed")
            _require(len(values) == 3, f"{path}/{group}: value count changed")
            rows[group] = {
                "mean": _finite(raw["mean"], f"{path}/{group}/mean"),
                "values": values,
            }
    _require("origin" in rows, f"{path}: origin row missing")
    return rows


def _manifest_condition(
    *,
    path: Path,
    data_root: Path,
    task: str,
    std_key: str,
    family: str,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
    source_key: str,
) -> dict[str, Any]:
    payload = _load(path)
    jobs = payload.get("jobs", [])
    _require(len(jobs) == 1, f"{path}: expected one behavior job")
    job = jobs[0]
    _require(job.get("complete") is True, f"{path}: behavior job is incomplete")
    _require(job.get("task") == task, f"{path}: task mismatch")
    _require(str(job.get("std_key")) == std_key, f"{path}: checkpoint mismatch")
    _require(job.get("family") == family, f"{path}: stressor mismatch")
    _require(job.get("eval_seeds") == 3, f"{path}: eval-seed count mismatch")
    _require(job.get("eval_base_seed") == 42, f"{path}: eval base seed mismatch")
    _require(job.get("num_eval") == 300, f"{path}: evaluation count mismatch")
    summary_path = data_root / str(job["eval_summary_rel"])
    _require(summary_path.is_file(), f"{path}: eval summary missing: {summary_path}")
    source_paths[f"{source_key}_manifest"] = _portable(path, data_root)
    source_hashes[f"{source_key}_manifest"] = _sha256(path)
    source_paths[f"{source_key}_eval_summary"] = _portable(summary_path, data_root)
    source_hashes[f"{source_key}_eval_summary"] = _sha256(summary_path)
    return {
        "job": job,
        "scores": _summary_index(summary_path),
        "summary_sha256": _sha256(summary_path),
    }


def _load_medium_row(
    *,
    training_seed: int,
    task: str,
    family: str,
    severity: float,
    protocol: Mapping[str, Any],
    base_protocol: Mapping[str, Any],
    data_root: Path,
    raw_root: Path,
    reference_root: Path,
    manifest_root: Path,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    task_slug = TASK_SLUG[task]
    family_spec = FAMILY_SPEC[family]
    family_row = str(family_spec["row_name"])
    severity_text = _slug_number(severity)
    condition: dict[str, dict[str, Any]] = {}
    for std_key in ("0.0", "0.08"):
        std_slug = std_key.replace(".", "p")
        manifest_path = (
            manifest_root
            / f"behavior_s{training_seed}_{task_slug}_std{std_slug}_{family}.json"
        )
        source_key = f"behavior_s{training_seed}_{task_slug}_{family}_std{std_slug}"
        condition[std_key] = _manifest_condition(
            path=manifest_path,
            data_root=data_root,
            task=task,
            std_key=std_key,
            family=family,
            source_paths=source_paths,
            source_hashes=source_hashes,
            source_key=source_key,
        )

    stress_group = str(family_spec["group_prefix"]) + severity_text
    for std_key, item in condition.items():
        _require(
            stress_group in item["scores"],
            f"{task}/{training_seed}/{family}/{severity}/{std_key}: stress row missing",
        )

    severity_dir = _severity_dir(family, severity)
    raw_dir = raw_root / f"lewm_seed{training_seed}" / severity_dir
    atr_path = raw_dir / f"acpc_{task_slug}_v2.json"
    smpr_path = raw_dir / f"smpr_{task_slug}_v2.json"
    reference_path = (
        reference_root
        / f"lewm_seed{training_seed}"
        / severity_dir
        / f"acpc_{task_slug}_horizon_v2_checkpoint_bound.json"
    )
    atr = _load(atr_path)
    smpr = _load(smpr_path)
    reference = _load(reference_path)
    key_prefix = f"diag_s{training_seed}_{task_slug}_{family}_{severity_dir}"
    for name, path in (
        ("atr", atr_path),
        ("smpr", smpr_path),
        ("reference", reference_path),
    ):
        source_paths[f"{key_prefix}_{name}"] = _portable(path, data_root)
        source_hashes[f"{key_prefix}_{name}"] = _sha256(path)

    _require(atr.get("metadata", {}).get("status_counts") == {"ok": 2}, f"{atr_path}: incomplete")
    smpr_meta = smpr.get("metadata", {})
    _require(smpr_meta.get("status") == "complete", f"{smpr_path}: incomplete")
    _require(smpr_meta.get("status_counts") == {"ok": 2}, f"{smpr_path}: row failures")
    _require(smpr_meta.get("missing_rows") == [], f"{smpr_path}: missing rows")
    _require(smpr_meta.get("errors") == [], f"{smpr_path}: errors")
    _require(
        smpr_meta.get("script_sha256") == protocol["source_hashes"]["smpr_runner"],
        f"{smpr_path}: SMPR implementation changed",
    )
    _require(
        reference.get("metadata", {}).get("protocol_sha256")
        == _sha256(DEFAULT_PROTOCOL),
        f"{reference_path}: protocol binding changed",
    )
    _require(
        smpr_meta.get("source_hashes", {}).get("reference_atr") == _sha256(reference_path),
        f"{smpr_path}: reference hash mismatch",
    )

    atr_by_std = {str(row.get("std_key")): row for row in atr.get("rows", [])}
    smpr_by_std = {str(row.get("std_key")): row for row in smpr.get("rows", [])}
    _require(set(atr_by_std) == {"0.0", "0.08"}, f"{atr_path}: endpoint pair mismatch")
    _require(set(smpr_by_std) == {"0.0", "0.08"}, f"{smpr_path}: endpoint pair mismatch")
    diagnostic: dict[str, dict[str, Any]] = {}
    for std_key in ("0.0", "0.08"):
        atr_row = atr_by_std[std_key]
        smpr_row = smpr_by_std[std_key]
        _require(atr_row.get("status") == "ok", f"{atr_path}/{std_key}: not ok")
        _require(smpr_row.get("status") == "ok", f"{smpr_path}/{std_key}: not ok")
        _require(smpr_row.get("atr_reference_match") is True, f"{smpr_path}/{std_key}: ATR mismatch")
        atr_value = _finite(atr_row["atr_horizon_v2_q90"], "ATR")
        _require(
            math.isclose(
                atr_value,
                _finite(smpr_row["same_state_tube_radius"], "SMPR tube radius"),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            f"{smpr_path}/{std_key}: ATR value mismatch",
        )
        smpr_value = _finite(smpr_row["smpr"], "SMPR")
        joint_score, gate_pass = _joint(
            atr_value,
            smpr_value,
            tau_atr=_finite(base_protocol["tau_atr"], "tau_atr"),
            tau_smpr=_finite(base_protocol["tau_smpr"], "tau_smpr"),
        )
        diagnostic[std_key] = {
            "atr": atr_value,
            "smpr": smpr_value,
            "joint_score": joint_score,
            "gate_pass": gate_pass,
        }

    base_clean = condition["0.0"]["scores"]["origin"]["mean"]
    endpoint_clean = condition["0.08"]["scores"]["origin"]["mean"]
    base_stressed = condition["0.0"]["scores"][stress_group]["mean"]
    endpoint_stressed = condition["0.08"]["scores"][stress_group]["mean"]
    delta_behavior = endpoint_stressed - base_stressed
    clean_score_drop = base_clean - endpoint_clean
    behavior_class = classify_transfer(
        delta_behavior=delta_behavior,
        clean_score_drop=clean_score_drop,
        positive_delta_pp=5.0,
        neutral_band_pp=5.0,
        max_clean_drop_pp=5.0,
    )
    delta_joint = diagnostic["0.08"]["joint_score"] - diagnostic["0.0"]["joint_score"]
    endpoint_selected = delta_joint > 0.0
    selected = endpoint_stressed if endpoint_selected else base_stressed
    return {
        "model_family": "LeWM",
        "training_seed_or_family_id": f"lewm_seed{training_seed}",
        "training_seed": training_seed,
        "training_seed_semantics": "independently trained LeWM checkpoint seed",
        "evaluation_seeds": [42, 43, 44],
        "evaluation_seed_semantics": "conditional closed-loop evaluation replicates, not training seeds",
        "task": task,
        "stressor_family": family_row,
        "stressor_severity": severity,
        "stressor_severity_parameter": family_spec["severity_parameter"],
        "severity_key": f"{family_row}:{severity_text}",
        "base_rho": 0.0,
        "endpoint_rho": 0.08,
        "base_checkpoint_sha256": condition["0.0"]["job"]["model_sha256"],
        "endpoint_checkpoint_sha256": condition["0.08"]["job"]["model_sha256"],
        "base_clean_score": base_clean,
        "endpoint_clean_score": endpoint_clean,
        "base_stressed_score": base_stressed,
        "endpoint_stressed_score": endpoint_stressed,
        "base_stressed_score_by_evaluation_seed": condition["0.0"]["scores"][stress_group]["values"],
        "endpoint_stressed_score_by_evaluation_seed": condition["0.08"]["scores"][stress_group]["values"],
        "base_retention": base_stressed / base_clean if base_clean else None,
        "endpoint_retention": endpoint_stressed / endpoint_clean if endpoint_clean else None,
        "delta_behavior": delta_behavior,
        "clean_score_drop": clean_score_drop,
        "behavior_class": behavior_class,
        "positive_transfer_label": behavior_class == "positive",
        "base_atr": diagnostic["0.0"]["atr"],
        "endpoint_atr": diagnostic["0.08"]["atr"],
        "delta_atr": diagnostic["0.08"]["atr"] - diagnostic["0.0"]["atr"],
        "base_smpr": diagnostic["0.0"]["smpr"],
        "endpoint_smpr": diagnostic["0.08"]["smpr"],
        "delta_smpr": diagnostic["0.08"]["smpr"] - diagnostic["0.0"]["smpr"],
        "base_joint_score": diagnostic["0.0"]["joint_score"],
        "endpoint_joint_score": diagnostic["0.08"]["joint_score"],
        "delta_joint_score": delta_joint,
        "base_gate_pass": diagnostic["0.0"]["gate_pass"],
        "endpoint_gate_pass": diagnostic["0.08"]["gate_pass"],
        "selection_regret_pp": max(base_stressed, endpoint_stressed) - selected,
        "source_stage": "prospective_medium_severity",
    }


def _load_legacy_rows(
    path: Path,
    *,
    source_paths: dict[str, str],
    source_hashes: dict[str, str],
    data_root: Path,
) -> list[dict[str, Any]]:
    source_paths["legacy_strongest_rows"] = _portable(path, data_root)
    source_hashes["legacy_strongest_rows"] = _sha256(path)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("model_family") != "LeWM":
                continue
            family = str(raw["stressor_family"])
            severity = _finite(raw["stressor_severity"], "legacy severity")
            expected = 15.0 if family == "blur" else 0.25
            _require(
                math.isclose(severity, expected, rel_tol=0.0, abs_tol=1e-12),
                "legacy strongest severity changed",
            )
            base_stressed = _finite(raw["base_stressed_score"], "legacy base stressed")
            endpoint_stressed = _finite(raw["endpoint_stressed_score"], "legacy endpoint stressed")
            delta_joint = _finite(raw["delta_joint_score"], "legacy delta joint")
            endpoint_selected = delta_joint > 0.0
            selected = endpoint_stressed if endpoint_selected else base_stressed
            row = {
                "model_family": "LeWM",
                "training_seed_or_family_id": raw["training_seed_or_family_id"],
                "training_seed": int(raw["training_seed"]),
                "training_seed_semantics": raw["training_seed_semantics"],
                "evaluation_seeds": [42, 43, 44],
                "evaluation_seed_semantics": raw["evaluation_seed_semantics"],
                "task": raw["task"],
                "stressor_family": family,
                "stressor_severity": severity,
                "stressor_severity_parameter": raw["stressor_severity_parameter"],
                "severity_key": f"{family}:{_slug_number(severity)}",
                "base_rho": _finite(raw["base_rho"], "legacy base rho"),
                "endpoint_rho": _finite(raw["endpoint_rho"], "legacy endpoint rho"),
                "base_checkpoint_sha256": raw["base_checkpoint_sha256"],
                "endpoint_checkpoint_sha256": raw["endpoint_checkpoint_sha256"],
                "base_clean_score": _finite(raw["base_clean_score"], "legacy base clean"),
                "endpoint_clean_score": _finite(raw["endpoint_clean_score"], "legacy endpoint clean"),
                "base_stressed_score": base_stressed,
                "endpoint_stressed_score": endpoint_stressed,
                "base_retention": _finite(raw["base_retention"], "legacy base retention"),
                "endpoint_retention": _finite(raw["endpoint_retention"], "legacy endpoint retention"),
                "delta_behavior": _finite(raw["delta_behavior"], "legacy behavior delta"),
                "clean_score_drop": _finite(raw["clean_score_drop"], "legacy clean drop"),
                "behavior_class": raw["behavior_class"],
                "positive_transfer_label": _bool(raw["positive_transfer_label"], "legacy label"),
                "base_atr": _finite(raw["base_atr"], "legacy base ATR"),
                "endpoint_atr": _finite(raw["endpoint_atr"], "legacy endpoint ATR"),
                "delta_atr": _finite(raw["delta_atr"], "legacy delta ATR"),
                "base_smpr": _finite(raw["base_smpr"], "legacy base SMPR"),
                "endpoint_smpr": _finite(raw["endpoint_smpr"], "legacy endpoint SMPR"),
                "delta_smpr": _finite(raw["delta_smpr"], "legacy delta SMPR"),
                "base_joint_score": _finite(raw["base_joint_score"], "legacy base joint"),
                "endpoint_joint_score": _finite(raw["endpoint_joint_score"], "legacy endpoint joint"),
                "delta_joint_score": delta_joint,
                "base_gate_pass": _bool(raw["base_gate_pass"], "legacy base gate"),
                "endpoint_gate_pass": _bool(raw["endpoint_gate_pass"], "legacy endpoint gate"),
                "selection_regret_pp": max(base_stressed, endpoint_stressed) - selected,
                "source_stage": "legacy_locked_strongest_e3",
            }
            rows.append(row)
    _require(len(rows) == 24, f"{path}: expected 24 locked LeWM strongest rows")
    return rows


def _blocks(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task"]), str(row["training_seed_or_family_id"]))].append(row)
    _require(len(groups) == 12, "expected 12 task-x-training-seed blocks")
    for key, group in groups.items():
        _require(len(group) == 6, f"{key}: expected six family-severity rows")
        counts = defaultdict(int)
        for row in group:
            counts[str(row["stressor_family"])] += 1
        _require(dict(counts) == {"blur": 3, "resize": 3}, f"{key}: block composition changed")
    return groups


def _exact_randomization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = _blocks(rows)
    keys = sorted(groups)
    behavior = [
        float(row["delta_behavior"])
        for key in keys
        for row in sorted(groups[key], key=lambda item: item["severity_key"])
    ]
    observed_scores = [
        float(row["delta_joint_score"])
        for key in keys
        for row in sorted(groups[key], key=lambda item: item["severity_key"])
    ]
    observed = _spearman(behavior, observed_scores)
    _require(observed is not None, "observed Spearman is undefined")
    null: list[float] = []
    for mask in range(1 << len(keys)):
        randomized: list[float] = []
        for index, key in enumerate(keys):
            sign = -1.0 if mask & (1 << index) else 1.0
            group = sorted(groups[key], key=lambda item: item["severity_key"])
            randomized.extend(sign * float(row["delta_joint_score"]) for row in group)
        value = _spearman(behavior, randomized)
        _require(value is not None, "randomized Spearman is undefined")
        null.append(float(value))
    tolerance = 1e-12
    comparable = [
        (left, right)
        for left, right in zip(behavior, observed_scores)
        if left != 0.0 and right != 0.0
    ]
    agreements = sum((left > 0.0) == (right > 0.0) for left, right in comparable)
    return {
        "primary_test": "exact task-x-training-seed block sign-flip test",
        "block_count": len(keys),
        "rows_retained_per_block": 6,
        "enumerated_assignments": len(null),
        "observed_spearman": observed,
        "one_sided_p_value": sum(value >= observed - tolerance for value in null) / len(null),
        "two_sided_p_value": sum(abs(value) >= abs(observed) - tolerance for value in null) / len(null),
        "null_spearman_ci95": [_quantile(null, 0.025), _quantile(null, 0.975)],
        "row_level_signed_agreement": {
            "successes": agreements,
            "trials": len(comparable),
            "descriptive_exact_binomial_one_sided_p_value": _exact_binomial_upper_tail(
                agreements,
                len(comparable),
            ),
        },
    }


def _compact_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    diagnostic = _delta_metric_summary(rows, field="delta_joint_score", direction=1.0)
    selection = _selection_summary(rows, field="delta_joint_score", direction=1.0)
    return {
        "balanced_accuracy": diagnostic["balanced_accuracy"],
        "auprc": diagnostic["auprc"],
        "spearman": diagnostic["spearman_delta_behavior_vs_oriented_delta_score"],
        "signed_agreement": diagnostic[
            "signed_agreement_delta_behavior_vs_oriented_delta_score"
        ],
        "choice_accuracy": selection["choice_accuracy"],
        "mean_regret_pp": selection["mean_regret_pp"],
    }


def _block_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    groups = _blocks(rows)
    keys = sorted(groups)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        replicate: list[Mapping[str, Any]] = []
        for key in (rng.choice(keys) for _ in keys):
            replicate.extend(groups[key])
        metrics = _compact_metrics(replicate)
        for name, value in metrics.items():
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    point = _compact_metrics(rows)
    return {
        "block_unit": "task x independently trained LeWM checkpoint seed",
        "block_count": len(keys),
        "families_and_severities_retained_within_block": True,
        "repetitions": repetitions,
        "seed": seed,
        "metrics": {
            name: {
                "point": value,
                "ci95": (
                    [_quantile(samples[name], 0.025), _quantile(samples[name], 0.975)]
                    if samples[name]
                    else None
                ),
                "finite_replicates": len(samples[name]),
            }
            for name, value in point.items()
        },
    }


def _dose_response(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    order = {"blur": [7.0, 11.0, 15.0], "resize": [0.75, 0.5, 0.25]}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["task"]),
                str(row["training_seed_or_family_id"]),
                str(row["stressor_family"]),
            )
        ].append(row)
    _require(len(grouped) == 24, "expected 24 block-family dose curves")
    curves: list[dict[str, Any]] = []
    for (task, family_id, family), group in sorted(grouped.items()):
        expected = order[family]
        by_index: list[Mapping[str, Any]] = []
        for severity in expected:
            matches = [
                row
                for row in group
                if math.isclose(
                    float(row["stressor_severity"]),
                    severity,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            _require(len(matches) == 1, f"{task}/{family_id}/{family}: dose row missing")
            by_index.append(matches[0])
        behavior = [float(row["delta_behavior"]) for row in by_index]
        diagnostic = [float(row["delta_joint_score"]) for row in by_index]
        strength = [1.0, 2.0, 3.0]
        curves.append(
            {
                "task": task,
                "training_seed_or_family_id": family_id,
                "stressor_family": family,
                "severities_weak_to_strong": expected,
                "delta_behavior": behavior,
                "delta_joint_score": diagnostic,
                "behavior_strength_spearman": _spearman(strength, behavior),
                "joint_strength_spearman": _spearman(strength, diagnostic),
                "behavior_joint_spearman": _spearman(behavior, diagnostic),
                "behavior_nondecreasing": all(
                    right >= left - 1e-12
                    for left, right in itertools.pairwise(behavior)
                ),
                "joint_nondecreasing": all(
                    right >= left - 1e-12
                    for left, right in itertools.pairwise(diagnostic)
                ),
            }
        )

    def fraction(field: str) -> float:
        return sum(bool(curve[field]) for curve in curves) / len(curves)

    return {
        "curve_count": len(curves),
        "no_monotonicity_filtering": True,
        "behavior_nondecreasing_fraction": fraction("behavior_nondecreasing"),
        "joint_nondecreasing_fraction": fraction("joint_nondecreasing"),
        "curves": curves,
    }


def build(
    *,
    protocol_path: Path,
    addendum_path: Path,
    legacy_path: Path,
    data_root: Path,
    raw_root: Path,
    reference_root: Path,
    manifest_root: Path,
) -> dict[str, Any]:
    protocol_sha = _verify_sidecar(protocol_path)
    addendum_sha = _verify_sidecar(addendum_path)
    protocol = _load(protocol_path)
    addendum = _load(addendum_path)
    _require(
        addendum.get("parent_protocol", {}).get("sha256") == protocol_sha,
        "execution addendum parent mismatch",
    )
    base_protocol_path = ROOT / protocol["diagnostic"]["base_protocol_path"]
    parent_addendum_path = ROOT / addendum["parent_execution_addendum"]["path"]
    parent_addendum_sha = _verify_sidecar(parent_addendum_path)
    _require(
        parent_addendum_sha
        == addendum["parent_execution_addendum"]["sha256"],
        "execution addendum v1 lineage mismatch",
    )
    _require(
        _sha256(base_protocol_path) == protocol["diagnostic"]["base_protocol_sha256"],
        "base protocol hash mismatch",
    )
    base_protocol = _load(base_protocol_path)
    _require(protocol["primary_analysis"]["expected_pairs"] == 72, "pair contract changed")
    _require(protocol["diagnostic"]["decision_threshold"] == 0, "zero rule changed")
    _require(protocol["diagnostic"]["threshold_search_allowed"] is False, "threshold search enabled")
    _require(protocol["diagnostic"]["severity_search_allowed"] is False, "severity search enabled")

    source_paths: dict[str, str] = {
        "protocol": _portable(protocol_path, data_root),
        "execution_addendum": _portable(addendum_path, data_root),
        "base_protocol": _portable(base_protocol_path, data_root),
    }
    source_hashes: dict[str, str] = {
        "protocol": protocol_sha,
        "execution_addendum": addendum_sha,
        "base_protocol": _sha256(base_protocol_path),
    }
    _require(
        _sha256(legacy_path) == LEGACY_STRONGEST_ROWS_SHA256,
        "locked E3 strongest-row source hash changed",
    )
    rows = _load_legacy_rows(
        legacy_path,
        source_paths=source_paths,
        source_hashes=source_hashes,
        data_root=data_root,
    )
    for training_seed in protocol["scope"]["training_seeds"]:
        for task in protocol["scope"]["tasks"]:
            for family in ("gaussian_blur", "resize"):
                for severity in protocol["stressors"][family]["prospective_nonidentity"]:
                    rows.append(
                        _load_medium_row(
                            training_seed=int(training_seed),
                            task=str(task),
                            family=family,
                            severity=float(severity),
                            protocol=protocol,
                            base_protocol=base_protocol,
                            data_root=data_root,
                            raw_root=raw_root,
                            reference_root=reference_root,
                            manifest_root=manifest_root,
                            source_paths=source_paths,
                            source_hashes=source_hashes,
                        )
                    )
    rows.sort(
        key=lambda row: (
            int(row["training_seed"]),
            str(row["task"]),
            str(row["stressor_family"]),
            float(row["stressor_severity"])
            if row["stressor_family"] == "blur"
            else -float(row["stressor_severity"]),
        )
    )
    _require(len(rows) == 72, "combined multi-severity rows are incomplete")
    keys = {
        (
            row["training_seed"],
            row["task"],
            row["stressor_family"],
            row["stressor_severity"],
        )
        for row in rows
    }
    _require(len(keys) == 72, "combined multi-severity row keys are not unique")
    _blocks(rows)

    primary = _delta_metric_summary(rows, field="delta_joint_score", direction=1.0)
    selection = _selection_summary(rows, field="delta_joint_score", direction=1.0)
    by_stressor = {
        family: {
            "diagnostic": _delta_metric_summary(
                [row for row in rows if row["stressor_family"] == family],
                field="delta_joint_score",
                direction=1.0,
            ),
            "selection": _selection_summary(
                [row for row in rows if row["stressor_family"] == family],
                field="delta_joint_score",
                direction=1.0,
            ),
        }
        for family in ("blur", "resize")
    }
    return {
        "metadata": {
            "schema_version": "paper1-paired-multiseverity-summary-1.0",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol_sha,
            "execution_addendum_sha256": addendum_sha,
            "threshold_search_allowed": False,
            "severity_search_allowed": False,
            "decision_threshold": 0.0,
            "primary_model_family": "LeWM",
            "evaluation_seed_semantics": "conditional measurement replicates, not training seeds",
            "component_boundary_scope": (
                "component and action/time-shuffle comparisons remain the locked "
                "24-row strongest-severity E3 audit; this extension tests joint-score "
                "dose response without introducing a new baseline implementation"
            ),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
        },
        "count_contract": {
            "expected_pairs": 72,
            "observed_pairs": len(rows),
            "legacy_locked_strongest_pairs": sum(
                row["source_stage"] == "legacy_locked_strongest_e3" for row in rows
            ),
            "prospective_medium_pairs": sum(
                row["source_stage"] == "prospective_medium_severity" for row in rows
            ),
            "block_count": len(_blocks(rows)),
            "rows_per_block": 6,
        },
        "primary_joint_change": primary,
        "selection_utility": selection,
        "by_stressor": by_stressor,
        "exact_randomization": _exact_randomization(rows),
        "block_bootstrap": _block_bootstrap(
            rows,
            repetitions=int(protocol["primary_analysis"]["block_bootstrap_repetitions"]),
            seed=int(protocol["primary_analysis"]["block_bootstrap_seed"]),
        ),
        "deletion_stability": {
            "leave_one_task_out": _deletion_stability(rows, group_field="task"),
            "leave_one_training_seed_out": _deletion_stability(
                rows,
                group_field="training_seed_or_family_id",
            ),
            "leave_one_severity_out": _deletion_stability(
                rows,
                group_field="severity_key",
            ),
        },
        "dose_response": _dose_response(rows),
        "rows": rows,
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"
        ),
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows-out", type=Path, default=DEFAULT_ROWS_OUT)
    args = parser.parse_args()
    payload = build(
        protocol_path=args.protocol,
        addendum_path=args.addendum,
        legacy_path=args.legacy,
        data_root=args.data_root,
        raw_root=args.raw_root,
        reference_root=args.reference_root,
        manifest_root=args.manifest_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_rows(args.rows_out, payload["rows"])
    print(f"wrote {args.out}")
    print(f"wrote {args.rows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
