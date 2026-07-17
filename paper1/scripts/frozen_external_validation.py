#!/usr/bin/env python3
"""Apply and score the immutable Paper 1 diagnostic protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHO_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
BLIND_FIELDS = (
    "model_family",
    "training_family_id",
    "training_seed",
    "task",
    "training_rho",
    "stressor_family",
    "stressor_severity",
    "atr_horizon_v2_q90",
    "smpr",
    "atr_threshold_margin",
    "smpr_threshold_margin",
    "joint_score",
    "frozen_gate_pass",
    "split_name",
    "protocol_sha256",
    "diagnostics_sha256",
)
FORBIDDEN_BLIND_FIELD_PARTS = (
    "score",
    "success",
    "return",
    "label",
    "recovery",
    "ground_truth",
)
STRICT_DIAGNOSTIC_FIELDS = (
    "status",
    "model_family",
    "training_family_id",
    "training_seed",
    "task",
    "training_rho",
    "stressor_family",
    "stressor_severity",
    "atr_horizon_v2_q90",
    "smpr",
    "split_name",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_strict(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _protocol(path: Path) -> tuple[dict[str, Any], str, bytes, int]:
    original = path.read_bytes()
    mtime_ns = path.stat().st_mtime_ns
    payload = json.loads(
        original.decode("utf-8"),
        parse_constant=_reject_constant,
    )
    if payload.get("schema_version") != "paper1-frozen-diagnostic-protocol-1.0":
        raise ValueError("unsupported frozen protocol schema")
    if payload.get("status") != "frozen" or payload.get("immutable") is not True:
        raise ValueError("protocol is not marked frozen and immutable")
    external = payload.get("external_policy", {})
    if external.get("threshold_search_allowed") is not False:
        raise ValueError("protocol permits external threshold search")
    if external.get("protocol_write_allowed") is not False:
        raise ValueError("protocol permits consumer writes")
    return payload, hashlib.sha256(original).hexdigest(), original, mtime_ns


def _assert_protocol_unchanged(
    path: Path,
    *,
    original: bytes,
    mtime_ns: int,
) -> None:
    if path.read_bytes() != original or path.stat().st_mtime_ns != mtime_ns:
        raise RuntimeError("external consumer changed the frozen protocol")


def _write_csv_exclusive(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _joint_values(
    *,
    atr: float,
    smpr: float,
    tau_atr: float,
    tau_smpr: float,
) -> tuple[float, float, float, bool]:
    atr_margin = (tau_atr - atr) / (abs(tau_atr) + 1e-12)
    smpr_margin = (smpr - tau_smpr) / (abs(tau_smpr) + 1e-12)
    joint_score = min(atr_margin, smpr_margin)
    gate_pass = atr <= tau_atr and smpr >= tau_smpr
    return atr_margin, smpr_margin, joint_score, gate_pass


def _validate_strict_diagnostics(
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    contract = metadata.get("strict_external_contract")
    if contract is None:
        return
    if metadata.get("behavior_blind") is not True:
        raise ValueError("strict diagnostic input is not behavior blind")
    if metadata.get("threshold_search_available") is not False:
        raise ValueError("strict diagnostic input exposes threshold search")
    if any(set(row) != set(STRICT_DIAGNOSTIC_FIELDS) for row in rows):
        raise ValueError("strict diagnostic row field allowlist mismatch")
    expected_rhos = set(RHO_GRID)
    if contract == "lewm_heldout_gaussian_v1":
        expected_family = "LeWM"
        expected_seeds = {
            int(seed)
            for seed in protocol["external_policy"]["heldout_lewm_training_seeds"]
        }
        expected_split = "TEST"
        expected_count = len(expected_seeds) * len(TASKS) * len(RHO_GRID)
    elif contract == "pldm_canonical_gaussian_e2_v1":
        expected_family = "PLDM"
        expected_seeds = {int(metadata["training_seed"])}
        expected_split = "E2"
        expected_count = len(TASKS) * len(RHO_GRID)
    else:
        raise ValueError(f"unsupported strict external contract: {contract}")
    if len(rows) != expected_count:
        raise ValueError("strict diagnostic row count mismatch")
    expected_keys = {
        (seed, task, rho)
        for seed in expected_seeds
        for task in TASKS
        for rho in expected_rhos
    }
    observed_keys: set[tuple[int, str, float]] = set()
    for row in rows:
        seed = int(row["training_seed"])
        task = str(row["task"])
        rho = _finite(row["training_rho"], name="strict training_rho")
        if row.get("status") != "ok":
            raise ValueError("strict diagnostic row is not ok")
        if row.get("model_family") != expected_family:
            raise ValueError("strict diagnostic model family mismatch")
        expected_family_id = (
            f"lewm_seed{seed}"
            if expected_family == "LeWM"
            else f"pldm_canonical_seed{seed}"
        )
        if row.get("training_family_id") != expected_family_id:
            raise ValueError("strict diagnostic family id/seed mismatch")
        if row.get("split_name") != expected_split:
            raise ValueError("strict diagnostic split mismatch")
        if row.get("stressor_family") != "gaussian":
            raise ValueError("strict diagnostic stressor family mismatch")
        if not math.isclose(
            _finite(row["stressor_severity"], name="strict stressor severity"),
            float(protocol["diagnostic_sampling"]["evaluation_noise_std"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("strict diagnostic stressor severity mismatch")
        if _finite(row["atr_horizon_v2_q90"], name="strict ATR") < 0.0:
            raise ValueError("strict diagnostic ATR is negative")
        smpr = _finite(row["smpr"], name="strict SMPR")
        if not 0.0 <= smpr <= 1.0:
            raise ValueError("strict diagnostic SMPR is outside [0, 1]")
        observed_keys.add((seed, task, rho))
    if observed_keys != expected_keys:
        raise ValueError("strict diagnostic seed/task/rho coverage mismatch")


def apply_protocol(
    *,
    protocol_path: Path,
    diagnostics_path: Path,
    out_path: Path,
    created_utc: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol, protocol_sha, original, mtime_ns = _protocol(protocol_path)
    if out_path.resolve() == protocol_path.resolve():
        raise ValueError("external output cannot be the protocol path")
    sidecar = Path(str(out_path) + ".metadata.json")
    if out_path.exists() or sidecar.exists():
        raise FileExistsError(out_path if out_path.exists() else sidecar)
    diagnostics = _load_strict(diagnostics_path)
    metadata = diagnostics.get("metadata", {})
    if metadata.get("schema_version") != "paper1-frozen-diagnostic-input-1.0":
        raise ValueError("unsupported frozen diagnostic input schema")
    if metadata.get("status") != "complete":
        raise ValueError("external diagnostic input is incomplete")
    if metadata.get("protocol_sha256") != protocol_sha:
        raise ValueError("diagnostic input protocol hash mismatch")
    if metadata.get("missing_rows") != [] or metadata.get("errors") != []:
        raise ValueError("external diagnostic input records missing rows or errors")
    rows = diagnostics.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("external diagnostic input has no rows")
    _validate_strict_diagnostics(
        metadata=metadata,
        rows=rows,
        protocol=protocol,
    )

    tau_atr = _finite(protocol["tau_atr"], name="tau_atr")
    tau_smpr = _finite(protocol["tau_smpr"], name="tau_smpr")
    diagnostics_sha = _sha256(diagnostics_path)
    blind: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("status", "ok") != "ok":
            raise ValueError("external diagnostic row is not ok")
        atr = _finite(row.get("atr_horizon_v2_q90"), name="external ATR")
        smpr = _finite(row.get("smpr"), name="external SMPR")
        training_rho = _finite(row.get("training_rho"), name="training_rho")
        key = (
            row.get("model_family"),
            row.get("training_family_id"),
            row.get("training_seed"),
            row.get("task"),
            training_rho,
            row.get("stressor_family", "gaussian"),
            row.get("stressor_severity", 0.08),
        )
        if key in seen:
            raise ValueError(f"duplicate external diagnostic row: {key}")
        seen.add(key)
        atr_margin, smpr_margin, joint_score, gate_pass = _joint_values(
            atr=atr,
            smpr=smpr,
            tau_atr=tau_atr,
            tau_smpr=tau_smpr,
        )
        blind.append(
            {
                "model_family": row.get("model_family"),
                "training_family_id": row.get("training_family_id"),
                "training_seed": row.get("training_seed"),
                "task": row.get("task"),
                "training_rho": training_rho,
                "stressor_family": row.get("stressor_family", "gaussian"),
                "stressor_severity": row.get("stressor_severity", 0.08),
                "atr_horizon_v2_q90": atr,
                "smpr": smpr,
                "atr_threshold_margin": atr_margin,
                "smpr_threshold_margin": smpr_margin,
                "joint_score": joint_score,
                "frozen_gate_pass": _bool_text(gate_pass),
                "split_name": row.get("split_name", "external"),
                "protocol_sha256": protocol_sha,
                "diagnostics_sha256": diagnostics_sha,
            }
        )
    for field in BLIND_FIELDS:
        if any(part in field.lower() for part in FORBIDDEN_BLIND_FIELD_PARTS):
            if field != "joint_score":
                raise AssertionError(f"blind schema leaks behavior field: {field}")
    _write_csv_exclusive(out_path, blind, BLIND_FIELDS)
    script_path = Path(__file__).resolve()
    created = created_utc or datetime.now(timezone.utc).isoformat()
    sidecar_payload = {
        "metadata": {
            "schema_version": "paper1-frozen-external-blind-predictions-1.0",
            "created_utc": created,
            "code_commit": protocol["code_commit"],
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": {
                "protocol": str(protocol_path),
                "diagnostics": str(diagnostics_path),
                "blind_rows": str(out_path),
            },
            "source_hashes": {
                "protocol": protocol_sha,
                "diagnostics": diagnostics_sha,
                "blind_rows": _sha256(out_path),
            },
            "protocol_hash": protocol_sha,
            "model_family": sorted(
                {str(row["model_family"]) for row in blind}
            ),
            "training_seed_semantics": metadata.get("training_seed_semantics"),
            "evaluation_seed_semantics": metadata.get("evaluation_seed_semantics"),
            "status": "complete",
            "status_counts": {"ok": len(blind)},
            "missing_rows": [],
            "errors": [],
            "behavior_blind": True,
            "strict_external_contract": metadata.get("strict_external_contract"),
            "operator_blinding": metadata.get("operator_blinding"),
            "threshold_search_available": False,
        },
        "row_count": len(blind),
        "fields": list(BLIND_FIELDS),
    }
    _write_json_exclusive(sidecar, sidecar_payload)
    _assert_protocol_unchanged(
        protocol_path,
        original=original,
        mtime_ns=mtime_ns,
    )
    return blind, sidecar_payload


def _metric_mean(entry: Mapping[str, Any], name: str) -> float:
    return _finite(
        entry.get("metrics", {}).get(name, {}).get("mean"),
        name=f"behavior {name}",
    )


def _metric_values(entry: Mapping[str, Any], name: str) -> list[float]:
    values = entry.get("metrics", {}).get(name, {}).get("values")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"behavior {name} must contain three eval-seed values")
    return [_finite(value, name=f"behavior {name} value") for value in values]


def _eval_summary(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("metric") != "success_rate":
                continue
            group = str(row.get("group"))
            seeds = [int(value) for value in str(row.get("seeds", "")).split(",")]
            values = [
                _finite(value, name=f"{path}/{group}/value")
                for value in str(row.get("values", "")).split(";")
                if value
            ]
            rows[group] = {
                "n": int(row["n_seeds"]),
                "seeds": seeds,
                "mean": _finite(row["mean"], name=f"{path}/{group}/mean"),
                "values": values,
            }
    return rows


def _confusion(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> dict[str, Any]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth and pred)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if not truth and not pred)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if not truth and pred)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth and not pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "num_rows": len(y_true),
        "actual_positive": sum(1 for value in y_true if value),
        "predicted_positive": sum(1 for value in y_pred if value),
    }


def _auprc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """Return tie-aware stepwise Average Precision.

    All rows with an identical score enter the retrieval set together, so the
    result is invariant to row order within a tied score group.
    """
    ranked = sorted(
        zip(y_true, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    positives = sum(1 for truth, _score in ranked if truth)
    if positives == 0:
        raise ValueError("AUPRC requires positives")
    tp = 0
    fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(ranked):
        score = float(ranked[index][1])
        group_end = index
        group_tp = 0
        group_fp = 0
        while group_end < len(ranked) and float(ranked[group_end][1]) == score:
            if ranked[group_end][0]:
                group_tp += 1
            else:
                group_fp += 1
            group_end += 1
        tp += group_tp
        fp += group_fp
        recall = tp / positives
        precision = tp / (tp + fp)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        index = group_end
    return average_precision


def _heldout_behavior(
    manifests: Sequence[Path],
    protocol: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, str, float], dict[str, Any]],
    dict[str, str],
    dict[str, str],
]:
    expected_seeds = set(
        int(seed)
        for seed in protocol["external_policy"]["heldout_lewm_training_seeds"]
    )
    payloads: dict[int, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for path in manifests:
        payload = _load_strict(path)
        manifest_metadata = payload.get("_metadata", {})
        if (
            manifest_metadata.get("schema_version")
            != "paper1-training-seed-eval-manifest-0.1"
        ):
            raise ValueError("strict heldout manifest schema mismatch")
        if manifest_metadata.get("tasks") != list(TASKS):
            raise ValueError("strict heldout manifest task coverage mismatch")
        if tuple(manifest_metadata.get("std_keys", [])) != tuple(
            str(rho) for rho in RHO_GRID
        ):
            raise ValueError("strict heldout manifest rho coverage mismatch")
        seed = manifest_metadata.get("training_seed")
        if seed not in expected_seeds:
            raise ValueError(f"strict heldout mode rejects training seed {seed}")
        if seed in payloads:
            raise ValueError(f"duplicate heldout training seed {seed}")
        payloads[int(seed)] = payload
        source_key = f"lewm_seed{seed}"
        paths[source_key] = str(path)
        hashes[source_key] = _sha256(path)
    if set(payloads) != expected_seeds:
        raise ValueError(
            f"strict heldout mode requires seeds {sorted(expected_seeds)}, "
            f"got {sorted(payloads)}"
        )

    recovery_fraction = float(
        protocol["gaussian_behavior_label"]["recovery_fraction"]
    )
    clean_tolerance = float(
        protocol["gaussian_behavior_label"]["clean_tolerance_pp"]
    )
    behavior: dict[tuple[int, str, float], dict[str, Any]] = {}
    evaluation_seeds = [
        int(seed)
        for seed in protocol["behavior_evaluation"]["evaluation_seeds"]
    ]
    for seed, payload in payloads.items():
        for task in TASKS:
            entries = {
                rho: payload[task][str(rho)]
                for rho in RHO_GRID
            }
            base_clean = _metric_mean(entries[0.0], "clean")
            base_stress = _metric_mean(entries[0.0], "pixels_std0.08")
            best_stress = max(
                _metric_mean(entry, "pixels_std0.08")
                for entry in entries.values()
            )
            threshold = base_stress + recovery_fraction * (
                best_stress - base_stress
            )
            denom = max(best_stress - base_stress, 1e-12)
            for rho, entry in entries.items():
                for metric_name in ("clean", "pixels_std0.08"):
                    metric = entry.get("metrics", {}).get(metric_name, {})
                    if metric.get("seeds") != evaluation_seeds:
                        raise ValueError(
                            f"{seed}/{task}/{rho}: {metric_name} eval seeds mismatch"
                        )
                    values = _metric_values(entry, metric_name)
                    if int(metric.get("n", -1)) != len(evaluation_seeds):
                        raise ValueError(
                            f"{seed}/{task}/{rho}: {metric_name} n mismatch"
                        )
                    if not math.isclose(
                        _metric_mean(entry, metric_name),
                        sum(values) / len(values),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(
                            f"{seed}/{task}/{rho}: {metric_name} mean/value mismatch"
                        )
                eval_summary_path = (
                    Path(str(entry["path"])) / "eval_results" / "eval_summary.csv"
                )
                if not eval_summary_path.is_file():
                    raise ValueError(f"missing held-out eval summary: {eval_summary_path}")
                summary = _eval_summary(eval_summary_path)
                for metric_name, group in (
                    ("clean", "origin"),
                    ("pixels_std0.08", "pixels_std0.08"),
                ):
                    metric = entry["metrics"][metric_name]
                    source_metric = summary.get(group)
                    if source_metric is None:
                        raise ValueError(f"{eval_summary_path}: missing {group}")
                    if source_metric["seeds"] != evaluation_seeds:
                        raise ValueError(f"{eval_summary_path}: {group} seed mismatch")
                    if source_metric["n"] != len(evaluation_seeds):
                        raise ValueError(f"{eval_summary_path}: {group} n mismatch")
                    if len(source_metric["values"]) != len(evaluation_seeds):
                        raise ValueError(
                            f"{eval_summary_path}: {group} value coverage mismatch"
                        )
                    if any(
                        not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
                        for left, right in zip(source_metric["values"], metric["values"])
                    ):
                        raise ValueError(f"{eval_summary_path}: {group} values mismatch")
                    if not math.isclose(
                        source_metric["mean"],
                        float(metric["mean"]),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(f"{eval_summary_path}: {group} mean mismatch")
                rho_key = f"{rho:.2f}".replace(".", "p")
                source_key = f"eval_summary_lewm_seed{seed}_{task}_{rho_key}"
                paths[source_key] = str(eval_summary_path)
                hashes[source_key] = _sha256(eval_summary_path)
                clean = _metric_mean(entry, "clean")
                stress = _metric_mean(entry, "pixels_std0.08")
                label = (
                    stress >= threshold
                    and clean >= base_clean - clean_tolerance
                )
                behavior[(seed, task, rho)] = {
                    "clean_score": clean,
                    "stress_score": stress,
                    "clean_score_by_evaluation_seed": _metric_values(
                        entry,
                        "clean",
                    ),
                    "stress_score_by_evaluation_seed": _metric_values(
                        entry,
                        "pixels_std0.08",
                    ),
                    "base_clean_score": base_clean,
                    "base_stress_score": base_stress,
                    "best_stress_score": best_stress,
                    "recovery_score_threshold": threshold,
                    "normalized_recovery": (stress - base_stress) / denom,
                    "clean_constraint_pass": clean
                    >= base_clean - clean_tolerance,
                    "behavior_label": label,
                }
    if set(paths) != set(hashes):
        raise AssertionError("held-out manifest provenance keys differ")
    return behavior, paths, hashes


def score_heldout(
    *,
    protocol_path: Path,
    predictions_path: Path,
    eval_manifests: Sequence[Path],
    out_rows_path: Path,
    out_summary_path: Path,
    out_blocks_path: Path,
    created_utc: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    protocol, protocol_sha, original, mtime_ns = _protocol(protocol_path)
    for output in (out_rows_path, out_summary_path, out_blocks_path):
        if output.resolve() == protocol_path.resolve():
            raise ValueError("score output cannot be the protocol path")
        if output.exists():
            raise FileExistsError(output)
    blind = _read_csv(predictions_path)
    if not blind:
        raise ValueError("blind prediction file has no rows")
    if set(blind[0]) != set(BLIND_FIELDS):
        raise ValueError("blind prediction field schema mismatch")
    if any(row.get("protocol_sha256") != protocol_sha for row in blind):
        raise ValueError("blind prediction protocol hash mismatch")
    sidecar_path = Path(str(predictions_path) + ".metadata.json")
    sidecar = _load_strict(sidecar_path)
    sidecar_metadata = sidecar.get("metadata", {})
    if sidecar_metadata.get("protocol_hash") != protocol_sha:
        raise ValueError("blind prediction sidecar protocol hash mismatch")
    if sidecar_metadata.get("behavior_blind") is not True:
        raise ValueError("prediction artifact is not behavior blind")
    script_path = Path(__file__).resolve()
    if sidecar_metadata.get("script_sha256") != _sha256(script_path):
        raise ValueError("blind prediction producer hash differs from scorer")
    prediction_sha = _sha256(predictions_path)
    sidecar_hashes = sidecar_metadata.get("source_hashes", {})
    sidecar_paths = sidecar_metadata.get("source_paths", {})
    if set(sidecar_paths) != set(sidecar_hashes):
        raise ValueError("blind prediction sidecar source provenance keys differ")
    if sidecar_hashes.get("blind_rows") != prediction_sha:
        raise ValueError("blind prediction sidecar rows hash mismatch")
    if sidecar_hashes.get("protocol") != protocol_sha:
        raise ValueError("blind prediction sidecar source protocol hash mismatch")
    if sidecar.get("row_count") != len(blind):
        raise ValueError("blind prediction sidecar row count mismatch")
    if sidecar.get("fields") != list(BLIND_FIELDS):
        raise ValueError("blind prediction sidecar field schema mismatch")
    diagnostics_hashes = {row.get("diagnostics_sha256") for row in blind}
    if diagnostics_hashes != {sidecar_hashes.get("diagnostics")}:
        raise ValueError("blind prediction diagnostics hash mismatch")
    diagnostics_path = Path(str(sidecar_paths.get("diagnostics")))
    if not diagnostics_path.is_absolute():
        diagnostics_path = ROOT / diagnostics_path
    if not diagnostics_path.is_file():
        raise ValueError("blind prediction diagnostic source is missing")
    if _sha256(diagnostics_path) != sidecar_hashes.get("diagnostics"):
        raise ValueError("blind prediction diagnostic source hash mismatch")
    tau_atr = _finite(protocol["tau_atr"], name="tau_atr")
    tau_smpr = _finite(protocol["tau_smpr"], name="tau_smpr")
    for row in blind:
        expected = _joint_values(
            atr=_finite(row.get("atr_horizon_v2_q90"), name="blind ATR"),
            smpr=_finite(row.get("smpr"), name="blind SMPR"),
            tau_atr=tau_atr,
            tau_smpr=tau_smpr,
        )
        for field, value in zip(
            ("atr_threshold_margin", "smpr_threshold_margin", "joint_score"),
            expected[:3],
        ):
            if not math.isclose(
                _finite(row.get(field), name=f"blind {field}"),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"blind prediction {field} does not match frozen gate")
        if _as_bool(row.get("frozen_gate_pass")) is not expected[3]:
            raise ValueError("blind prediction gate decision does not match frozen gate")

    behavior, manifest_paths, manifest_hashes = _heldout_behavior(
        eval_manifests,
        protocol,
    )
    expected_keys = set(behavior)
    blind_index: dict[tuple[int, str, float], dict[str, str]] = {}
    for row in blind:
        if row.get("model_family") != "LeWM":
            raise ValueError("strict LeWM heldout score rejects another model family")
        seed = int(float(row["training_seed"]))
        if row.get("training_family_id") != f"lewm_seed{seed}":
            raise ValueError("strict LeWM heldout score rejects family id/seed mismatch")
        task = row["task"]
        rho = float(row["training_rho"])
        if row.get("split_name") != "TEST":
            raise ValueError("strict LeWM heldout score rejects non-TEST row")
        if row.get("stressor_family") != "gaussian":
            raise ValueError("strict LeWM heldout score rejects stressor mismatch")
        if not math.isclose(
            float(row["stressor_severity"]),
            float(protocol["diagnostic_sampling"]["evaluation_noise_std"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("strict LeWM heldout score rejects severity mismatch")
        key = (seed, task, rho)
        if key in blind_index:
            raise ValueError(f"duplicate blind row {key}")
        blind_index[key] = row
    if set(blind_index) != expected_keys:
        missing = expected_keys - set(blind_index)
        extra = set(blind_index) - expected_keys
        raise ValueError(
            f"strict heldout row coverage mismatch; missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    scored: list[dict[str, Any]] = []
    for key in sorted(
        expected_keys,
        key=lambda value: (value[0], TASKS.index(value[1]), value[2]),
    ):
        seed, task, rho = key
        pred = blind_index[key]
        truth = behavior[key]
        scored.append(
            {
                "model_family": "LeWM",
                "training_seed": seed,
                "task": task,
                "training_rho": rho,
                "stressor_family": pred["stressor_family"],
                "stressor_severity": float(pred["stressor_severity"]),
                "atr_horizon_v2_q90": float(pred["atr_horizon_v2_q90"]),
                "smpr": float(pred["smpr"]),
                "joint_score": float(pred["joint_score"]),
                "frozen_gate_pass": pred["frozen_gate_pass"],
                **{
                    name: (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, list)
                        else _bool_text(value)
                        if isinstance(value, bool)
                        else value
                    )
                    for name, value in truth.items()
                },
                "split_name": "TEST",
                "protocol_sha256": protocol_sha,
                "diagnostics_sha256": pred["diagnostics_sha256"],
            }
        )

    y_true = [_as_bool(row["behavior_label"]) for row in scored]
    y_pred = [_as_bool(row["frozen_gate_pass"]) for row in scored]
    joint_scores = [float(row["joint_score"]) for row in scored]
    blocks: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(int(row["training_seed"]), str(row["task"]))].append(row)
    for (seed, task), block in sorted(grouped.items()):
        block.sort(key=lambda row: float(row["training_rho"]))
        true_rhos = [
            float(row["training_rho"])
            for row in block
            if _as_bool(row["behavior_label"])
        ]
        pred_rhos = [
            float(row["training_rho"])
            for row in block
            if _as_bool(row["frozen_gate_pass"])
        ]
        true_onset = min(true_rhos) if true_rhos else None
        pred_onset = min(pred_rhos) if pred_rhos else None
        if true_onset is None and pred_onset is None:
            error = 0.0
        elif true_onset is None:
            error = -1.0
        elif pred_onset is None:
            error = 1.0
        else:
            error = pred_onset - true_onset
        block_truth = [_as_bool(row["behavior_label"]) for row in block]
        block_pred = [_as_bool(row["frozen_gate_pass"]) for row in block]
        blocks.append(
            {
                "model_family": "LeWM",
                "training_seed": seed,
                "task": task,
                "behavioral_onset": true_onset,
                "predicted_onset": pred_onset,
                "onset_error": error,
                "false_early": sum(
                    1
                    for row in block
                    if _as_bool(row["frozen_gate_pass"])
                    and true_onset is not None
                    and float(row["training_rho"]) < true_onset
                ),
                "false_late": sum(
                    1
                    for row in block
                    if _as_bool(row["behavior_label"])
                    and not _as_bool(row["frozen_gate_pass"])
                ),
                **_confusion(block_truth, block_pred),
                "protocol_sha256": protocol_sha,
            }
        )
    confusion = _confusion(y_true, y_pred)
    metrics = {
        **confusion,
        "auprc": _auprc(y_true, joint_scores),
        "mean_absolute_onset_error": sum(
            abs(float(row["onset_error"])) for row in blocks
        )
        / len(blocks),
        "max_absolute_onset_error": max(
            abs(float(row["onset_error"])) for row in blocks
        ),
        "false_early": sum(int(row["false_early"]) for row in blocks),
        "false_late": sum(int(row["false_late"]) for row in blocks),
    }
    created = created_utc or datetime.now(timezone.utc).isoformat()
    summary = {
        "metadata": {
            "schema_version": "paper1-frozen-external-validation-summary-1.1",
            "created_utc": created,
            "code_commit": protocol["code_commit"],
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": {
                "protocol": str(protocol_path),
                "predictions": str(predictions_path),
                "predictions_sidecar": str(sidecar_path),
                "diagnostics": str(diagnostics_path),
                **manifest_paths,
            },
            "source_hashes": {
                "protocol": protocol_sha,
                "predictions": prediction_sha,
                "predictions_sidecar": _sha256(sidecar_path),
                "diagnostics": _sha256(diagnostics_path),
                **manifest_hashes,
            },
            "protocol_hash": protocol_sha,
            "model_family": "LeWM",
            "training_seeds": sorted(
                int(seed)
                for seed in protocol["external_policy"][
                    "heldout_lewm_training_seeds"
                ]
            ),
            "training_seed_semantics": "two independent held-out LeWM training runs",
            "evaluation_seeds": [
                int(seed)
                for seed in protocol["behavior_evaluation"]["evaluation_seeds"]
            ],
            "evaluation_seed_semantics": (
                "conditional evaluation variability, not training replication"
            ),
            "status": "complete",
            "status_counts": {"ok": len(scored)},
            "missing_rows": [],
            "errors": [],
            "threshold_search_available": False,
            "strict_external_contract": sidecar_metadata.get(
                "strict_external_contract"
            ),
            "operator_blinding": sidecar_metadata.get("operator_blinding"),
            "auprc_definition": (
                "tie-aware stepwise Average Precision; exact joint_score ties "
                "enter each retrieval set together"
            ),
        },
        "metrics": metrics,
        "raw_confusion": {
            key: confusion[key]
            for key in ("tp", "tn", "fp", "fn")
        },
        "blocks": blocks,
    }
    _write_csv_exclusive(out_rows_path, scored)
    _write_csv_exclusive(out_blocks_path, blocks)
    _write_json_exclusive(out_summary_path, summary)
    _assert_protocol_unchanged(
        protocol_path,
        original=original,
        mtime_ns=mtime_ns,
    )
    return scored, summary, blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--protocol", type=Path, required=True)
    apply_parser.add_argument("--diagnostics", type=Path, required=True)
    apply_parser.add_argument("--out", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--protocol", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument(
        "--eval-manifest",
        type=Path,
        action="append",
        required=True,
    )
    score_parser.add_argument(
        "--out-rows",
        type=Path,
        default=ROOT
        / "paper1"
        / "results"
        / "frozen_external_validation_rows.csv",
    )
    score_parser.add_argument(
        "--out-summary",
        type=Path,
        default=ROOT
        / "paper1"
        / "results"
        / "frozen_external_validation_summary.json",
    )
    score_parser.add_argument(
        "--out-blocks",
        type=Path,
        default=ROOT
        / "paper1"
        / "results"
        / "frozen_external_validation_summary.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "apply":
        rows, _metadata = apply_protocol(
            protocol_path=args.protocol,
            diagnostics_path=args.diagnostics,
            out_path=args.out,
        )
        print(f"wrote {args.out} ({len(rows)} blind rows)")
        print(f"wrote {args.out}.metadata.json")
        return 0
    scored, summary, blocks = score_heldout(
        protocol_path=args.protocol,
        predictions_path=args.predictions,
        eval_manifests=args.eval_manifest,
        out_rows_path=args.out_rows,
        out_summary_path=args.out_summary,
        out_blocks_path=args.out_blocks,
    )
    print(f"wrote {args.out_rows} ({len(scored)} scored rows)")
    print(f"wrote {args.out_blocks} ({len(blocks)} block rows)")
    print(
        f"wrote {args.out_summary} "
        f"(AUPRC={summary['metrics']['auprc']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
