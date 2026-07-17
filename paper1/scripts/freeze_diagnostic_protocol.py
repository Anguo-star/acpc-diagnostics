#!/usr/bin/env python3
"""Freeze the Paper 1 public-v1 diagnostic protocol from seed3072 CAL only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHO_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
CALIBRATION_SEED = 3072
RECOVERY_FRACTION = 0.80
CLEAN_TOLERANCE_PP = 5.0
EPS = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    commit = proc.stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"unexpected git commit: {commit!r}")
    return commit


def _git_dirty() -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return bool(proc.stdout.strip())


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


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _metric(entry: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = entry.get("metrics", {}).get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing behavior metric {name}")
    return value


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
        "predicted_positive": sum(1 for value in y_pred if value),
        "actual_positive": sum(1 for value in y_true if value),
        "num_rows": len(y_true),
    }


def _auprc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    ranked = sorted(
        zip(y_true, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    positives = sum(1 for truth, _ in ranked if truth)
    if positives == 0:
        raise ValueError("AUPRC requires at least one positive CAL row")
    tp = 0
    precision_at_positive: list[float] = []
    for rank, (truth, _score) in enumerate(ranked, start=1):
        if truth:
            tp += 1
            precision_at_positive.append(tp / rank)
    return sum(precision_at_positive) / positives


def _joint_score(atr: float, smpr: float, tau_atr: float, tau_smpr: float) -> float:
    atr_margin = (tau_atr - atr) / (abs(tau_atr) + EPS)
    smpr_margin = (smpr - tau_smpr) / (abs(tau_smpr) + EPS)
    return min(atr_margin, smpr_margin)


def build_calibration_rows(
    *,
    atr_payload: Mapping[str, Any],
    smpr_payload: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    atr_meta = atr_payload.get("metadata", {})
    smpr_meta = smpr_payload.get("metadata", {})
    if atr_meta.get("schema_version") != "paper1-acpc-horizon-v2-1.0":
        raise ValueError("CAL ATR artifact has the wrong schema")
    if smpr_meta.get("schema_version") != "paper1-smpr-v2-merged-1.0":
        raise ValueError("CAL SMPR artifact has the wrong schema")
    if atr_meta.get("training_seed") != CALIBRATION_SEED:
        raise ValueError("CAL ATR must contain only LeWM seed3072")
    if smpr_meta.get("training_seed") != CALIBRATION_SEED:
        raise ValueError("CAL SMPR must contain only LeWM seed3072")
    if atr_meta.get("model_family") != "LeWM" or smpr_meta.get("model_family") != "LeWM":
        raise ValueError("CAL artifacts must use LeWM")
    if eval_payload.get("_metadata", {}).get("training_seed") != CALIBRATION_SEED:
        raise ValueError("CAL behavior manifest must be LeWM seed3072")
    if atr_meta.get("status") != "complete" or smpr_meta.get("status") != "complete":
        raise ValueError("CAL artifacts must be complete")

    atr_rows = {
        (str(row["task"]), float(row["training_rho"])): row
        for row in atr_payload.get("rows", [])
        if row.get("status") == "ok"
    }
    smpr_rows = {
        (str(row["task"]), float(row["training_rho"])): row
        for row in smpr_payload.get("rows", [])
        if row.get("status") == "ok"
    }
    expected = {(task, rho) for task in TASKS for rho in RHO_GRID}
    if set(atr_rows) != expected or set(smpr_rows) != expected:
        raise ValueError("CAL ATR/SMPR coverage must be exactly 4 tasks x 9 rho")

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        behavior = {
            rho: {
                "clean": _finite(
                    _metric(eval_payload[task][str(rho)], "clean").get("mean"),
                    name=f"{task}/{rho}/clean",
                ),
                "stress": _finite(
                    _metric(
                        eval_payload[task][str(rho)],
                        "pixels_std0.08",
                    ).get("mean"),
                    name=f"{task}/{rho}/stress",
                ),
            }
            for rho in RHO_GRID
        }
        base_clean = behavior[0.0]["clean"]
        base_stress = behavior[0.0]["stress"]
        best_stress = max(values["stress"] for values in behavior.values())
        threshold = base_stress + RECOVERY_FRACTION * (best_stress - base_stress)
        denom = max(best_stress - base_stress, EPS)
        for rho in RHO_GRID:
            key = (task, rho)
            atr_row = atr_rows[key]
            smpr_row = smpr_rows[key]
            atr = _finite(
                atr_row.get("atr_horizon_v2_q90"),
                name=f"{key}/ATR",
            )
            smpr = _finite(smpr_row.get("smpr"), name=f"{key}/SMPR")
            tube = _finite(
                smpr_row.get("same_state_tube_radius"),
                name=f"{key}/SMPR tube",
            )
            if not math.isclose(atr, tube, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"{key}: ATR and SMPR tube are not identical")
            clean_score = behavior[rho]["clean"]
            stress_score = behavior[rho]["stress"]
            clean_pass = clean_score >= base_clean - CLEAN_TOLERANCE_PP
            label = stress_score >= threshold and clean_pass
            rows.append(
                {
                    "model_family": "LeWM",
                    "training_seed": CALIBRATION_SEED,
                    "task": task,
                    "training_rho": rho,
                    "split_name": "CAL",
                    "clean_score": clean_score,
                    "stress_score": stress_score,
                    "base_clean_score": base_clean,
                    "base_stress_score": base_stress,
                    "best_stress_score": best_stress,
                    "recovery_score_threshold": threshold,
                    "normalized_recovery": (stress_score - base_stress) / denom,
                    "clean_constraint_pass": clean_pass,
                    "behavior_label": label,
                    "atr_horizon_v2_q90": atr,
                    "smpr": smpr,
                    "semantic_pair_count": int(smpr_row["semantic_pair_count"]),
                    "semantic_skipped_anchor_count": int(
                        smpr_row["semantic_skipped_anchor_count"]
                    ),
                }
            )
    if len(rows) != 36:
        raise ValueError(f"CAL row count mismatch: {len(rows)}")
    return rows


def calibrate_global_gate(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    atr_candidates = sorted(
        {_finite(row["atr_horizon_v2_q90"], name="ATR") for row in rows}
    )
    smpr_candidates = sorted({_finite(row["smpr"], name="SMPR") for row in rows})
    y_true = [bool(row["behavior_label"]) for row in rows]
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int, float, float] | None = None
    for tau_atr in atr_candidates:
        for tau_smpr in smpr_candidates:
            pred = [
                float(row["atr_horizon_v2_q90"]) <= tau_atr
                and float(row["smpr"]) >= tau_smpr
                for row in rows
            ]
            metrics = _confusion(y_true, pred)
            selection_key = (
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
                metrics["predicted_positive"],
                -tau_atr,
                tau_smpr,
            )
            candidate = {
                "tau_atr": tau_atr,
                "tau_smpr": tau_smpr,
                **metrics,
                "selection_key": list(selection_key),
            }
            candidates.append(candidate)
            if best is None or selection_key > best_key:
                best = candidate
                best_key = selection_key
    if best is None:
        raise ValueError("no CAL threshold candidates")
    return best, candidates


def calibration_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    tau_atr: float,
    tau_smpr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true = [bool(row["behavior_label"]) for row in rows]
    y_pred = [
        float(row["atr_horizon_v2_q90"]) <= tau_atr
        and float(row["smpr"]) >= tau_smpr
        for row in rows
    ]
    scores = [
        _joint_score(
            float(row["atr_horizon_v2_q90"]),
            float(row["smpr"]),
            tau_atr,
            tau_smpr,
        )
        for row in rows
    ]
    blocks: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        task_pred = [
            float(row["atr_horizon_v2_q90"]) <= tau_atr
            and float(row["smpr"]) >= tau_smpr
            for row in task_rows
        ]
        true_rho = [
            float(row["training_rho"])
            for row in task_rows
            if bool(row["behavior_label"])
        ]
        pred_rho = [
            float(row["training_rho"])
            for row, pred in zip(task_rows, task_pred)
            if pred
        ]
        true_onset = min(true_rho) if true_rho else None
        pred_onset = min(pred_rho) if pred_rho else None
        if true_onset is None and pred_onset is None:
            onset_error = 0.0
        elif true_onset is None:
            onset_error = -1.0
        elif pred_onset is None:
            onset_error = 1.0
        else:
            onset_error = pred_onset - true_onset
        blocks.append(
            {
                "task": task,
                "behavioral_onset": true_onset,
                "predicted_onset": pred_onset,
                "onset_error": onset_error,
                "false_early": sum(
                    1
                    for row, pred in zip(task_rows, task_pred)
                    if pred
                    and true_onset is not None
                    and float(row["training_rho"]) < true_onset
                ),
                "false_late": sum(
                    1
                    for row, pred in zip(task_rows, task_pred)
                    if bool(row["behavior_label"]) and not pred
                ),
            }
        )
    confusion = _confusion(y_true, y_pred)
    return (
        {
            **confusion,
            "auprc": _auprc(y_true, scores),
            "mean_absolute_onset_error": sum(
                abs(float(block["onset_error"])) for block in blocks
            )
            / len(blocks),
            "max_absolute_onset_error": max(
                abs(float(block["onset_error"])) for block in blocks
            ),
            "false_early": sum(int(block["false_early"]) for block in blocks),
            "false_late": sum(int(block["false_late"]) for block in blocks),
        },
        blocks,
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def freeze(
    *,
    calibration_atr_path: Path,
    calibration_smpr_path: Path,
    calibration_evals_path: Path,
    schema_path: Path,
    out_path: Path,
    audit_path: Path,
    frozen_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (
        calibration_atr_path,
        calibration_smpr_path,
        calibration_evals_path,
        schema_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if out_path.exists():
        raise FileExistsError(f"frozen protocol already exists: {out_path}")
    if audit_path.exists():
        raise FileExistsError(f"calibration audit already exists: {audit_path}")

    atr_payload = _load_strict(calibration_atr_path)
    smpr_payload = _load_strict(calibration_smpr_path)
    eval_payload = _load_strict(calibration_evals_path)
    schema = _load_strict(schema_path)
    rows = build_calibration_rows(
        atr_payload=atr_payload,
        smpr_payload=smpr_payload,
        eval_payload=eval_payload,
    )
    selected, candidates = calibrate_global_gate(rows)
    tau_atr = float(selected["tau_atr"])
    tau_smpr = float(selected["tau_smpr"])
    metrics, block_rows = calibration_summary(
        rows,
        tau_atr=tau_atr,
        tau_smpr=tau_smpr,
    )
    created = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    commit = _git_commit()
    dirty = _git_dirty()
    script_path = Path(__file__).resolve()
    implementation_paths = {
        "calibration_atr": calibration_atr_path,
        "calibration_smpr": calibration_smpr_path,
        "calibration_evals": calibration_evals_path,
        "protocol_schema": schema_path,
        "freeze_builder": script_path,
        "canonical_metric": ROOT / "tools" / "paper1_acpc_metrics.py",
        "semantic_margin": ROOT / "tools" / "paper1_semantic_margin.py",
        "smpr_runner": ROOT / "paper1" / "scripts" / "smpr_sensitivity.py",
    }
    source_paths = {
        name: _relative(path)
        for name, path in implementation_paths.items()
    }
    source_hashes = {
        name: _sha256(path)
        for name, path in implementation_paths.items()
    }
    audit = {
        "metadata": {
            "schema_version": "paper1-frozen-diagnostic-protocol-calibration-1.0",
            "created_utc": created,
            "code_commit": commit,
            "calibration_worktree_dirty": dirty,
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "protocol_hash": None,
            "protocol_hash_status": "protocol is written after this audit",
            "model_family": "LeWM",
            "training_seed": CALIBRATION_SEED,
            "training_seed_semantics": "one independently trained CAL family",
            "evaluation_seeds": [42, 43, 44],
            "evaluation_seed_semantics": (
                "conditional evaluation variability, not training replication"
            ),
            "status": "complete",
            "missing_rows": [],
            "errors": [],
        },
        "behavior_label_rule": {
            "recovery_fraction": RECOVERY_FRACTION,
            "clean_tolerance_pp": CLEAN_TOLERANCE_PP,
        },
        "threshold_selection": {
            "scope": "global_absolute_canonical_metrics",
            "candidate_rule": "sorted_unique_observed_CAL_values",
            "objective_lexicographic": [
                "calibration_f1_max",
                "precision_max",
                "recall_max",
                "predicted_positive_count_max",
                "tau_atr_min",
                "tau_smpr_max",
            ],
            "comparators": {
                "atr": "atr_horizon_v2_q90 <= tau_atr",
                "smpr": "smpr >= tau_smpr",
                "joint": "atr AND smpr",
            },
            "tie_break_rationale": (
                "when CAL classification metrics are identical, prefer the "
                "stricter lower ATR boundary and stricter higher SMPR boundary"
            ),
            "candidate_count": len(candidates),
        },
        "selected_thresholds": {
            "tau_atr": tau_atr,
            "tau_smpr": tau_smpr,
            "joint_rule": "atr_and_smpr",
            "selected_candidate": selected,
        },
        "calibration_metrics": metrics,
        "calibration_blocks": block_rows,
        "calibration_rows": rows,
        "candidate_audit": candidates,
    }
    _write_exclusive(audit_path, audit)
    audit_sha = _sha256(audit_path)

    protocol = {
        "protocol_id": "paper1-public-v1",
        "schema_version": "paper1-frozen-diagnostic-protocol-1.0",
        "immutable": True,
        "created_utc": created,
        "code_commit": commit,
        "model_family": "LeWM",
        "training_seed_semantics": (
            "CAL uses one independently trained LeWM checkpoint family at seed3072"
        ),
        "evaluation_seed_semantics": (
            "seeds42/43/44 are conditional evaluation replicates, not training seeds"
        ),
        "status": "frozen",
        "missing_rows": [],
        "errors": [],
        "protocol_hash_status": (
            "computed from final file bytes and recorded by every consumer"
        ),
        "calibration_source": "LeWM seed3072 Gaussian full sweep",
        "radius_metric": "horizon_weighted_stacked_l2_v2",
        "rollout_horizon": 8,
        "horizon_weights": "uniform_1_over_H",
        "atr_quantile": 0.90,
        "normalization": (
            "per_anchor_observed_clean_transition_l2_q50_"
            "including_history_future_boundary"
        ),
        "noise_draw_aggregation": "per_anchor_mean_then_checkpoint_quantile",
        "smpr_pair_rule": "task_grounded_near_boundary_v2",
        "smpr_local_quantile": 0.35,
        "smpr_margin_delta_normalized": 0.10,
        "smpr_radius_quantile": 0.90,
        "tau_atr": tau_atr,
        "tau_smpr": tau_smpr,
        "joint_rule": "atr_and_smpr",
        "calibration": {
            "model_family": "LeWM",
            "training_seeds": [CALIBRATION_SEED],
            "training_stressor": "Gaussian input noise",
            "tasks": list(TASKS),
            "rho_grid": list(RHO_GRID),
            "expected_rows": 36,
            "row_key": [
                "model_family",
                "training_seed",
                "task",
                "training_rho",
            ],
            "split_name": "CAL",
        },
        "behavior_evaluation": {
            "evaluation_seeds": [42, 43, 44],
            "trajectories_per_evaluation_seed": 100,
            "evaluation_seed_semantics": (
                "conditional evaluation variability, not training-run replication"
            ),
        },
        "diagnostic_sampling": {
            "n_anchors": 100,
            "num_noise_draws": 5,
            "anchor_seed": 9101,
            "noise_draw_seed_rule": "anchor_seed+1009+7919*draw_index",
            "evaluation_noise_std": 0.08,
            "corrupt_goal": False,
            "embedding_space_policy": "checkpoint_inference_cost_space",
        },
        "fixed_pool": {
            "candidate_count": 65,
            "expert_candidate_count": 1,
            "random_candidate_count": 64,
            "candidate_seed_rule": "anchor_seed+2027",
            "pool_pairing": (
                "same ordered candidate pool for clean and corrupted branches"
            ),
        },
        "threshold_selection": {
            "scope": "global_absolute_canonical_metrics",
            "candidate_rule": "sorted_unique_observed_CAL_values",
            "objective_lexicographic": [
                "calibration_f1_max",
                "precision_max",
                "recall_max",
                "predicted_positive_count_max",
                "tau_atr_min",
                "tau_smpr_max",
            ],
            "comparators": {
                "atr": "atr_horizon_v2_q90 <= tau_atr",
                "smpr": "smpr >= tau_smpr",
                "joint": "atr AND smpr",
            },
            "tie_break_rationale": (
                "when CAL classification metrics are identical, prefer the "
                "stricter lower ATR boundary and stricter higher SMPR boundary"
            ),
            "external_recalibration": False,
        },
        "joint_score_definition": {
            "name": "minimum_normalized_threshold_margin",
            "formula": (
                "min((tau_atr-atr)/(abs(tau_atr)+1e-12),"
                "(smpr-tau_smpr)/(abs(tau_smpr)+1e-12))"
            ),
            "higher_is_better": True,
        },
        "gaussian_behavior_label": {
            "recovery_fraction": RECOVERY_FRACTION,
            "clean_tolerance_pp": CLEAN_TOLERANCE_PP,
        },
        "external_behavior_label": {
            "positive_delta_pp": 5.0,
            "neutral_band_pp": 5.0,
            "max_clean_drop_pp": 5.0,
        },
        "external_severities": {
            "blur": {
                "implementation_parameter": "kernel_size",
                "supported_nonidentity": [3, 7, 11, 15],
                "v1_strongest": 15,
                "v2_mild_medium_strong": [3, 7, 15],
                "selection_basis": "physical kernel width, fixed before external results",
            },
            "resize": {
                "implementation_parameter": "scale_factor",
                "supported_nonidentity": [0.75, 0.50, 0.25],
                "v1_strongest": 0.25,
                "v2_mild_medium_strong": [0.75, 0.50, 0.25],
                "selection_basis": "physical downsampling factor, fixed before external results",
            },
        },
        "external_policy": {
            "threshold_search_allowed": False,
            "protocol_write_allowed": False,
            "forbidden_calibration_sources": [
                "LeWM training seeds 3073/3074",
                "PLDM all checkpoints and evaluation results",
                "blur/resize behavior or diagnostics",
                "target-view/heteroscedastic/robust-CEM results",
                "prospective PLDM training seeds",
            ],
            "heldout_lewm_training_seeds": [3073, 3074],
        },
        "calibration_metrics": metrics,
        "calibration_commit": commit,
        "calibration_worktree_dirty": dirty,
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "calibration_audit_path": _relative(audit_path),
        "calibration_audit_sha256": audit_sha,
        "frozen_at_utc": created,
    }
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(protocol), key=lambda error: list(error.path))
    if errors:
        joined = "\n".join(
            f"{'/'.join(str(value) for value in error.path)}: {error.message}"
            for error in errors
        )
        raise ValueError(f"protocol schema validation failed:\n{joined}")
    _write_exclusive(out_path, protocol)
    return protocol, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-atr", type=Path, required=True)
    parser.add_argument("--calibration-smpr", type=Path, required=True)
    parser.add_argument("--calibration-evals", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT
        / "paper1"
        / "config"
        / "frozen_diagnostic_protocol_v1.schema.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "paper1"
        / "config"
        / "frozen_diagnostic_protocol_v1.json",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=ROOT
        / "paper1"
        / "results"
        / "frozen_diagnostic_protocol_calibration.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    protocol, audit = freeze(
        calibration_atr_path=args.calibration_atr,
        calibration_smpr_path=args.calibration_smpr,
        calibration_evals_path=args.calibration_evals,
        schema_path=args.schema,
        out_path=args.out,
        audit_path=args.audit_out,
    )
    print(
        f"wrote {args.out} with tau_atr={protocol['tau_atr']:.9g}, "
        f"tau_smpr={protocol['tau_smpr']:.9g}"
    )
    print(
        f"wrote {args.audit_out} "
        f"({len(audit['candidate_audit'])} candidates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
