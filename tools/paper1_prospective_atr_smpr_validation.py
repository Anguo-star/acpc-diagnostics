#!/usr/bin/env python3
"""Prospective ATR/SMPR robust-interval validation for Paper 1.

This script is deliberately evaluation-free. It consumes previously computed
closed-loop scores only in the calibration and validation stages, and writes the
held-out prediction file before joining held-out scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_PHASE0 = DATA_DIR / "acpc_phase0_lewm_three_seed.json"
DEFAULT_SMPR = DATA_DIR / "semantic_task_grounded_margin_lewm_full_sweep_20260708.json"
DEFAULT_MANIFEST_DIR = DATA_DIR / "training_seed_eval_manifests"
DEFAULT_OUT_DIR = ROOT / "paper1" / "results" / "prospective_diagnostic"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
HELDOUT_SEEDS = (3073, 3074)
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
EPS = 1e-12
FORBIDDEN_PREDICTION_COLUMNS = {"score", "return", "success", "eval_score", "closed_loop_score", "true_robust"}


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _fmt_std(value: Any) -> str:
    return f"{float(value):.2f}"


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_mean(entry: Mapping[str, Any], key: str) -> float:
    metric = entry.get("metrics", {}).get(key)
    if metric is None:
        raise KeyError(f"missing metric {key}")
    return float(metric["mean"])


def _score_map(manifest_dir: Path, eval_metric: str) -> dict[tuple[str, int, str], dict[str, float]]:
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    for seed in SEEDS:
        manifest = _load_json(manifest_dir / f"lewm_seed{seed}_evals.json")
        for task in TASKS:
            for std_key in STD_KEYS:
                entry = manifest[task][std_key]
                out[(task, seed, std_key)] = {
                    "closed_loop_score": _metric_mean(entry, eval_metric),
                    "clean_score": _metric_mean(entry, "clean"),
                }
    return out


def _phase0_map(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    payload = _load_json(path)
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in payload.get("rows", []):
        if row.get("status") != "ok":
            continue
        key = (str(row["task"]), int(row["training_seed"]), str(row["std_key"]))
        trans = _as_float(row.get("clean_transition_l2_median"))
        acpc_q90 = _as_float(row.get("acpc_h_l2_p90"))
        atr = acpc_q90 / trans if trans > EPS else float("nan")
        out[key] = {
            "atr_q90": atr,
            "same_radius_q90": acpc_q90,
            "clean_transition_l2_median": trans,
            "model_file": row.get("model_file", ""),
            "phase0_status": row.get("status", ""),
        }
    return out


def _smpr_map(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    payload = _load_json(path)
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in payload.get("rows", []):
        key = (str(row["task"]), int(row["training_seed"]), str(row["std_key"]))
        out[key] = {
            "smpr": _as_float(row.get("semantic_margin_pass_rate")),
            "same_radius_median": _as_float(row.get("same_state_noisy_radius_median")),
            "semantic_diff_l2_median": _as_float(row.get("semantic_diff_l2_median")),
            "semantic_margin_median": _as_float(row.get("semantic_margin_median")),
            "semantic_pair_count": row.get("semantic_pair_count", ""),
            "smpr_status": row.get("status", ""),
            "smpr_pair_rule": row.get("semantic_pair_rule", ""),
        }
    return out


def _expected_keys() -> set[tuple[str, int, str]]:
    return {(task, seed, std_key) for task in TASKS for seed in SEEDS for std_key in STD_KEYS}


def build_diagnostics(phase0_path: Path, smpr_path: Path, out_csv: Path) -> list[dict[str, Any]]:
    phase0 = _phase0_map(phase0_path)
    smpr = _smpr_map(smpr_path)
    expected = _expected_keys()
    missing_phase0 = sorted(expected - set(phase0))
    missing_smpr = sorted(expected - set(smpr))
    if missing_phase0:
        raise ValueError(f"missing Phase-0 ATR rows, first missing={missing_phase0[:5]}")
    if missing_smpr:
        raise ValueError(
            "missing full-sweep SMPR rows; run tools.paper1_semantic_margin over all "
            f"std keys first, first missing={missing_smpr[:5]}"
        )
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for seed in SEEDS:
            for std_key in STD_KEYS:
                key = (task, seed, std_key)
                p = phase0[key]
                s = smpr[key]
                rows.append({
                    "task": task,
                    "train_seed": seed,
                    "stdmax": std_key,
                    "eval_noise_sigma": "0.08",
                    "diag_seed": 9101,
                    "num_probe_states": "100",
                    "horizon": 8,
                    "atr_q90": p["atr_q90"],
                    "same_radius_q90": p["same_radius_q90"],
                    "clean_transition_l2_median": p["clean_transition_l2_median"],
                    "smpr": s["smpr"],
                    "same_radius_median": s["same_radius_median"],
                    "semantic_diff_l2_median": s["semantic_diff_l2_median"],
                    "semantic_margin_median": s["semantic_margin_median"],
                    "semantic_pair_count": s["semantic_pair_count"],
                    "smpr_status": s["smpr_status"],
                    "smpr_pair_rule": s["smpr_pair_rule"],
                    "atr_source": str(phase0_path.relative_to(ROOT) if phase0_path.is_relative_to(ROOT) else phase0_path),
                    "smpr_source": str(smpr_path.relative_to(ROOT) if smpr_path.is_relative_to(ROOT) else smpr_path),
                })
    fieldnames = [
        "task", "train_seed", "stdmax", "eval_noise_sigma", "diag_seed", "num_probe_states", "horizon",
        "atr_q90", "same_radius_q90", "clean_transition_l2_median", "smpr", "same_radius_median",
        "semantic_diff_l2_median", "semantic_margin_median", "semantic_pair_count", "smpr_status",
        "smpr_pair_rule", "atr_source", "smpr_source",
    ]
    _csv_write(out_csv, rows, fieldnames)
    return rows


def _diagnostic_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    for row in _csv_read(path):
        converted: dict[str, Any] = dict(row)
        converted["train_seed"] = int(converted["train_seed"])
        for key in ("atr_q90", "smpr", "same_radius_q90", "same_radius_median", "semantic_diff_l2_median", "semantic_margin_median"):
            converted[key] = _as_float(converted.get(key))
        out.append(converted)
    return out


def _add_atr_rel(rows: list[dict[str, Any]]) -> None:
    base: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["stdmax"] == "0.0":
            base[(row["task"], int(row["train_seed"]))] = float(row["atr_q90"])
    for row in rows:
        b = base.get((row["task"], int(row["train_seed"])), float("nan"))
        row["atr_rel"] = float(row["atr_q90"]) / b if b > EPS else float("nan")


def _label_map(score_rows: Mapping[tuple[str, int, str], dict[str, float]], rho: float, clean_guard_pp: float | None) -> dict[tuple[str, int, str], dict[str, float | bool]]:
    out: dict[tuple[str, int, str], dict[str, float | bool]] = {}
    for task in TASKS:
        for seed in SEEDS:
            block = [(std, score_rows[(task, seed, std)]) for std in STD_KEYS]
            base_score = float(score_rows[(task, seed, "0.0")]["closed_loop_score"])
            base_clean = float(score_rows[(task, seed, "0.0")]["clean_score"])
            best_score = max(float(v["closed_loop_score"]) for _, v in block)
            denom = best_score - base_score
            threshold = base_score + float(rho) * denom if denom > EPS else best_score
            for std, scores in block:
                score = float(scores["closed_loop_score"])
                recovery = (score - base_score) / denom if denom > EPS else (1.0 if score >= threshold - EPS else 0.0)
                robust = score >= threshold - EPS
                if clean_guard_pp is not None:
                    robust = robust and float(scores["clean_score"]) >= base_clean - float(clean_guard_pp) - EPS
                out[(task, seed, std)] = {
                    "closed_loop_score": score,
                    "clean_score": float(scores["clean_score"]),
                    "base_closed_loop_score": base_score,
                    "best_closed_loop_score": best_score,
                    "robust_threshold": threshold,
                    "normalized_recovery": recovery,
                    "true_robust": bool(robust),
                }
    return out


def _confusion(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Metrics:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y and p)
    fp = sum(1 for y, p in zip(y_true, y_pred) if not y and p)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y and not p)
    tn = sum(1 for y, p in zip(y_true, y_pred) if not y and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return Metrics(tp, fp, fn, tn, precision, recall, f1, 0.5 * (tpr + tnr))


def _rankdata(values: Sequence[float]) -> list[float]:
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for _, idx in pairs[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return float("nan")
    xs2, ys2 = zip(*vals)
    mx = sum(xs2) / len(xs2)
    my = sum(ys2) / len(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= EPS or vy <= EPS:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs2, ys2)) / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 2:
        return float("nan")
    xs2, ys2 = zip(*vals)
    return _pearson(_rankdata(xs2), _rankdata(ys2))


def _auroc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    vals = [(bool(y), float(s)) for y, s in zip(y_true, scores) if math.isfinite(float(s))]
    pos = [s for y, s in vals if y]
    neg = [s for y, s in vals if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _auprc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    vals = sorted(
        [(bool(y), float(s)) for y, s in zip(y_true, scores) if math.isfinite(float(s))],
        key=lambda item: item[1],
        reverse=True,
    )
    positives = sum(1 for y, _ in vals if y)
    if positives == 0:
        return float("nan")
    tp = 0
    precisions = []
    for i, (y, _) in enumerate(vals, start=1):
        if y:
            tp += 1
            precisions.append(tp / i)
    return sum(precisions) / positives if precisions else float("nan")


def _candidate_thresholds(values: Sequence[float], lower_is_better: bool) -> list[float]:
    vals = sorted({float(v) for v in values if math.isfinite(float(v))})
    if not vals:
        return []
    return vals if lower_is_better else vals


def _predict_row(row: Mapping[str, Any], rule: str, atr_key: str, theta_atr: float | None, theta_smpr: float | None) -> bool:
    atr_ok = True if theta_atr is None else float(row[atr_key]) <= float(theta_atr)
    smpr_ok = True if theta_smpr is None else float(row["smpr"]) >= float(theta_smpr)
    if rule == "atr_only":
        return atr_ok
    if rule == "smpr_only":
        return smpr_ok
    if rule == "atr_smpr":
        return atr_ok and smpr_ok
    raise ValueError(f"unknown rule {rule}")


def _diagnostic_score(row: Mapping[str, Any], rule: str, atr_key: str) -> float:
    if rule == "atr_only":
        return -float(row[atr_key])
    if rule == "smpr_only":
        return float(row["smpr"])
    return -float(row[atr_key]) + float(row["smpr"])


def _calibrate_one(rows: Sequence[dict[str, Any]], labels: Mapping[tuple[str, int, str], dict[str, float | bool]], rule: str, atr_key: str) -> dict[str, Any]:
    atr_values = _candidate_thresholds([r[atr_key] for r in rows], lower_is_better=True)
    smpr_values = _candidate_thresholds([r["smpr"] for r in rows], lower_is_better=False)
    theta_atr_values = [None] if rule == "smpr_only" else atr_values
    theta_smpr_values = [None] if rule == "atr_only" else smpr_values
    if not theta_atr_values or not theta_smpr_values:
        raise ValueError(f"cannot calibrate {rule}; missing diagnostic values")
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float, int, float, float] | None = None
    y_true = [bool(labels[(r["task"], int(r["train_seed"]), r["stdmax"])]["true_robust"]) for r in rows]
    for theta_atr in theta_atr_values:
        for theta_smpr in theta_smpr_values:
            pred = [_predict_row(r, rule, atr_key, theta_atr, theta_smpr) for r in rows]
            m = _confusion(y_true, pred)
            interval_width = sum(1 for p in pred if p)
            theta_atr_key = -float(theta_atr) if theta_atr is not None and math.isfinite(float(theta_atr)) else 0.0
            theta_smpr_key = -float(theta_smpr) if theta_smpr is not None and math.isfinite(float(theta_smpr)) else 0.0
            key = (m.f1, m.precision, m.recall, interval_width, theta_atr_key, theta_smpr_key)
            if best is None or key > best_key:
                best_key = key
                best = {
                    "theta_atr": theta_atr,
                    "theta_smpr": theta_smpr,
                    "calibration_tp": m.tp,
                    "calibration_fp": m.fp,
                    "calibration_fn": m.fn,
                    "calibration_tn": m.tn,
                    "calibration_precision": m.precision,
                    "calibration_recall": m.recall,
                    "calibration_f1": m.f1,
                    "calibration_balanced_accuracy": m.balanced_accuracy,
                    "calibration_predicted_positive": interval_width,
                }
    assert best is not None
    return best


def calibrate(
    diagnostics_csv: Path,
    manifest_dir: Path,
    out_json: Path,
    *,
    rho: float,
    eval_metric: str,
    clean_guard_pp: float | None,
) -> dict[str, Any]:
    rows = _diagnostic_rows(diagnostics_csv)
    _add_atr_rel(rows)
    scores = _score_map(manifest_dir, eval_metric)
    labels = _label_map(scores, rho, clean_guard_pp)
    thresholds: dict[str, Any] = {"per_task": {}, "global_normalized": {}}
    for task in TASKS:
        calibration_rows = [r for r in rows if r["task"] == task and int(r["train_seed"]) == 3072]
        thresholds["per_task"][task] = {}
        for rule in ("atr_only", "smpr_only", "atr_smpr"):
            thresholds["per_task"][task][rule] = _calibrate_one(calibration_rows, labels, rule, "atr_q90")
    global_rows = [r for r in rows if int(r["train_seed"]) == 3072]
    for rule in ("atr_only", "smpr_only", "atr_smpr"):
        thresholds["global_normalized"][rule] = _calibrate_one(global_rows, labels, rule, "atr_rel")
    payload = {
        "metadata": {
            "schema_version": "paper1-prospective-atr-smpr-thresholds-v1",
            "diagnostics_csv": str(diagnostics_csv.relative_to(ROOT) if diagnostics_csv.is_relative_to(ROOT) else diagnostics_csv),
            "score_manifest_dir": str(manifest_dir.relative_to(ROOT) if manifest_dir.is_relative_to(ROOT) else manifest_dir),
            "calibration_seed": 3072,
            "heldout_training_seeds": list(HELDOUT_SEEDS),
            "rho": float(rho),
            "robust_label": "normalized recovery against pixels_std0.08 full-sweep best; no held-out score used for prediction",
            "eval_metric": eval_metric,
            "clean_guard_pp": clean_guard_pp,
        },
        "thresholds": thresholds,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def predict(diagnostics_csv: Path, thresholds_json: Path, out_csv: Path) -> list[dict[str, Any]]:
    rows = _diagnostic_rows(diagnostics_csv)
    _add_atr_rel(rows)
    thresholds = _load_json(thresholds_json)["thresholds"]
    out: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["train_seed"])
        if seed not in HELDOUT_SEEDS:
            continue
        task = row["task"]
        for rule, spec in thresholds["per_task"][task].items():
            pred = _predict_row(row, rule, "atr_q90", spec.get("theta_atr"), spec.get("theta_smpr"))
            out.append({
                "protocol": "per_task",
                "rule": rule,
                "task": task,
                "train_seed": seed,
                "stdmax": row["stdmax"],
                "eval_noise_sigma": row["eval_noise_sigma"],
                "atr_q90": row["atr_q90"],
                "atr_rel": row["atr_rel"],
                "smpr": row["smpr"],
                "theta_atr": spec.get("theta_atr", ""),
                "theta_smpr": spec.get("theta_smpr", ""),
                "pred_robust": int(pred),
                "diagnostic_score": _diagnostic_score(row, rule, "atr_rel"),
                "pred_source": "seed3072_per_task_threshold",
            })
        for rule, spec in thresholds["global_normalized"].items():
            pred = _predict_row(row, rule, "atr_rel", spec.get("theta_atr"), spec.get("theta_smpr"))
            out.append({
                "protocol": "global_normalized",
                "rule": rule,
                "task": task,
                "train_seed": seed,
                "stdmax": row["stdmax"],
                "eval_noise_sigma": row["eval_noise_sigma"],
                "atr_q90": row["atr_q90"],
                "atr_rel": row["atr_rel"],
                "smpr": row["smpr"],
                "theta_atr": spec.get("theta_atr", ""),
                "theta_smpr": spec.get("theta_smpr", ""),
                "pred_robust": int(pred),
                "diagnostic_score": _diagnostic_score(row, rule, "atr_rel"),
                "pred_source": "seed3072_global_normalized_threshold",
            })
    fieldnames = [
        "protocol", "rule", "task", "train_seed", "stdmax", "eval_noise_sigma",
        "atr_q90", "atr_rel", "smpr", "theta_atr", "theta_smpr", "pred_robust",
        "diagnostic_score", "pred_source",
    ]
    bad = FORBIDDEN_PREDICTION_COLUMNS & set(fieldnames)
    if bad:
        raise AssertionError(f"prediction file would leak forbidden columns: {sorted(bad)}")
    _csv_write(out_csv, out, fieldnames)
    return out


def _interval_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pred = {float(r["stdmax"]) for r in rows if bool(int(r["pred_robust"]))}
    obs = {float(r["stdmax"]) for r in rows if bool(int(r["true_robust"]))}
    union = pred | obs
    inter = pred & obs
    iou = len(inter) / len(union) if union else float("nan")
    if pred and obs:
        left_err = min(pred) - min(obs)
        right_err = max(pred) - max(obs)
    else:
        left_err = float("nan")
        right_err = float("nan")
    return {
        "predicted_robust_stdmax_set": ";".join(f"{v:.2f}" for v in sorted(pred)) if pred else "none",
        "observed_robust_stdmax_set": ";".join(f"{v:.2f}" for v in sorted(obs)) if obs else "none",
        "interval_iou": iou,
        "left_endpoint_error": left_err,
        "right_endpoint_error": right_err,
    }


def _summary_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    y_true = [bool(int(r["true_robust"])) for r in rows]
    y_pred = [bool(int(r["pred_robust"])) for r in rows]
    m = _confusion(y_true, y_pred)
    score = [float(r["closed_loop_score"]) for r in rows]
    neg_atr = [-float(r["atr_rel"]) for r in rows]
    smpr = [float(r["smpr"]) for r in rows]
    combo = [float(r["diagnostic_score"]) for r in rows]
    return {
        "num_ckpts": len(rows),
        "true_positive": m.tp,
        "false_positive": m.fp,
        "false_negative": m.fn,
        "true_negative": m.tn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "balanced_accuracy": m.balanced_accuracy,
        "auroc_combo": _auroc(y_true, combo),
        "auprc_combo": _auprc(y_true, combo),
        "spearman_score_neg_atr_rel": _spearman(score, neg_atr),
        "spearman_score_smpr": _spearman(score, smpr),
        "spearman_score_combo": _spearman(score, combo),
    }


def validate(
    predictions_csv: Path,
    manifest_dir: Path,
    out_dir: Path,
    *,
    rho: float,
    eval_metric: str,
    clean_guard_pp: float | None,
) -> dict[str, Path]:
    predictions = _csv_read(predictions_csv)
    forbidden = FORBIDDEN_PREDICTION_COLUMNS & set(predictions[0].keys() if predictions else [])
    if forbidden:
        raise AssertionError(f"prediction file leaks score/label columns: {sorted(forbidden)}")
    scores = _score_map(manifest_dir, eval_metric)
    labels = _label_map(scores, rho, clean_guard_pp)
    joined: list[dict[str, Any]] = []
    for row in predictions:
        key = (row["task"], int(row["train_seed"]), row["stdmax"])
        label = labels[key]
        joined.append({
            **row,
            "closed_loop_score": label["closed_loop_score"],
            "clean_score": label["clean_score"],
            "base_closed_loop_score": label["base_closed_loop_score"],
            "best_closed_loop_score": label["best_closed_loop_score"],
            "robust_threshold": label["robust_threshold"],
            "normalized_recovery": label["normalized_recovery"],
            "true_robust": int(bool(label["true_robust"])),
        })
    validation_rows_csv = out_dir / "validation_rows_with_scores.csv"
    validation_fieldnames = list(joined[0].keys()) if joined else []
    _csv_write(validation_rows_csv, joined, validation_fieldnames)

    block_rows: list[dict[str, Any]] = []
    for protocol in sorted({r["protocol"] for r in joined}):
        for rule in sorted({r["rule"] for r in joined if r["protocol"] == protocol}):
            for task in TASKS:
                for seed in HELDOUT_SEEDS:
                    block = [r for r in joined if r["protocol"] == protocol and r["rule"] == rule and r["task"] == task and int(r["train_seed"]) == seed]
                    if not block:
                        continue
                    block_rows.append({
                        "protocol": protocol,
                        "rule": rule,
                        "task": task,
                        "train_seed": seed,
                        **_interval_metrics(block),
                    })
    interval_blocks_csv = out_dir / "validation_interval_blocks.csv"
    _csv_write(interval_blocks_csv, block_rows, [
        "protocol", "rule", "task", "train_seed", "predicted_robust_stdmax_set", "observed_robust_stdmax_set", "interval_iou", "left_endpoint_error", "right_endpoint_error",
    ])

    summary_rows: list[dict[str, Any]] = []
    by_task_rows: list[dict[str, Any]] = []
    for protocol in sorted({r["protocol"] for r in joined}):
        for rule in sorted({r["rule"] for r in joined if r["protocol"] == protocol}):
            group = [r for r in joined if r["protocol"] == protocol and r["rule"] == rule]
            relevant_blocks = [b for b in block_rows if b["protocol"] == protocol and b["rule"] == rule]
            summary = _summary_for(group)
            summary_rows.append({
                "split": "heldout_training_seeds_3073_3074",
                "protocol": protocol,
                "rule": rule,
                **summary,
                "mean_interval_iou": _safe_mean([_as_float(b["interval_iou"]) for b in relevant_blocks]),
                "mean_abs_left_endpoint_error": _safe_mean_abs([_as_float(b["left_endpoint_error"]) for b in relevant_blocks]),
                "mean_abs_right_endpoint_error": _safe_mean_abs([_as_float(b["right_endpoint_error"]) for b in relevant_blocks]),
            })
            for task in TASKS:
                task_group = [r for r in group if r["task"] == task]
                if not task_group:
                    continue
                task_blocks = [b for b in relevant_blocks if b["task"] == task]
                by_task_rows.append({
                    "split": "heldout_training_seeds_3073_3074",
                    "protocol": protocol,
                    "rule": rule,
                    "task": task,
                    **_summary_for(task_group),
                    "mean_interval_iou": _safe_mean([_as_float(b["interval_iou"]) for b in task_blocks]),
                    "mean_abs_left_endpoint_error": _safe_mean_abs([_as_float(b["left_endpoint_error"]) for b in task_blocks]),
                    "mean_abs_right_endpoint_error": _safe_mean_abs([_as_float(b["right_endpoint_error"]) for b in task_blocks]),
                })
    summary_csv = out_dir / "validation_summary.csv"
    by_task_csv = out_dir / "validation_summary_by_task.csv"
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    by_task_fields = list(by_task_rows[0].keys()) if by_task_rows else []
    _csv_write(summary_csv, summary_rows, summary_fields)
    _csv_write(by_task_csv, by_task_rows, by_task_fields)
    return {
        "validation_rows": validation_rows_csv,
        "interval_blocks": interval_blocks_csv,
        "summary": summary_csv,
        "by_task": by_task_csv,
    }


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _safe_mean_abs(values: Sequence[float]) -> float:
    vals = [abs(float(v)) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")



def plot_heldout(rows_csv: Path, out_dir: Path, *, protocol: str = "per_task", rule: str = "atr_smpr") -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting is optional for headless validation.
        print(f"skipping plots: {exc!r}")
        return []
    rows = _csv_read(rows_csv)
    rows = [r for r in rows if r["protocol"] == protocol and r["rule"] == rule]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        by_std: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in task_rows:
            by_std[row["stdmax"]].append(row)
        stds = [s for s in STD_KEYS if s in by_std]
        x = [float(s) for s in stds]
        score = [_safe_mean([_as_float(r["closed_loop_score"]) for r in by_std[s]]) for s in stds]
        atr_rel = [_safe_mean([_as_float(r["atr_rel"]) for r in by_std[s]]) for s in stds]
        smpr = [_safe_mean([_as_float(r["smpr"]) for r in by_std[s]]) for s in stds]
        pred_frac = [_safe_mean([float(int(r["pred_robust"])) for r in by_std[s]]) for s in stds]
        true_frac = [_safe_mean([float(int(r["true_robust"])) for r in by_std[s]]) for s in stds]
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for xx, pf, tf in zip(x, pred_frac, true_frac):
            if tf >= 0.5:
                ax.axvspan(xx - 0.0038, xx + 0.0038, color="#d9ead3", alpha=0.75, lw=0)
            if pf >= 0.5:
                ax.axvspan(xx - 0.0020, xx + 0.0020, color="#c9daf8", alpha=0.65, lw=0)
        l1 = ax.plot(x, score, marker="o", color="#1f4e79", lw=1.8, label="heldout score")[0]
        ax.set_ylim(0, 105)
        ax.set_xlabel("training noise stdmax")
        ax.set_ylabel("obs sigma 0.08 score")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        l2 = ax2.plot(x, atr_rel, marker="s", color="#a61c00", lw=1.5, label="ATR rel")[0]
        l3 = ax2.plot(x, smpr, marker="^", color="#38761d", lw=1.5, label="SMPR")[0]
        ax2.set_ylabel("diagnostic value")
        ax2.set_ylim(bottom=0)
        ax.set_title(f"{task}: heldout robust interval prediction ({protocol}, {rule})")
        ax.legend([l1, l2, l3], ["heldout score", "ATR rel", "SMPR"], loc="lower right", fontsize=8)
        fig.tight_layout()
        out = fig_dir / f"{task.lower()}_heldout_{protocol}_{rule}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        paths.append(out)
    return paths

def write_readme(out_dir: Path, smpr_path: Path, rho: float) -> Path:
    path = out_dir / "README.md"
    smpr_rel = smpr_path.relative_to(ROOT) if smpr_path.is_relative_to(ROOT) else smpr_path
    text = f"""# Prospective ATR/SMPR Diagnostic Validation

This directory is an internal Paper1 validation artifact. It tests whether thresholds calibrated on training seed 3072 predict robust intervals on held-out training seeds 3073/3074 before reading their closed-loop scores.

Full-sweep SMPR computation command:

```bash
python -m tools.paper1_semantic_margin \\
  --seeds 3072 3073 3074 \\
  --tasks TwoRoom PushT Reacher Cube \\
  --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \\
  --n-sequences 100 \\
  --device cuda \\
  --pair-rule task_grounded_near_boundary \\
  --out {smpr_rel}
```

Validation command after the SMPR artifact exists:

```bash
python -m tools.paper1_prospective_atr_smpr_validation \\
  --smpr {smpr_rel} \\
  --rho {rho:.2f} \\
  --out-dir paper1/results/prospective_diagnostic
```

Leakage rule: `predictions_heldout.csv` is written without score, return, success, eval_score, or true-label columns. Held-out closed-loop scores are joined only in `validation_rows_with_scores.csv` and the validation summary files.

`figures/` contains review plots for the default `per_task` + `atr_smpr` held-out rule; these are internal result-inspection figures, not paper figures by default.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_all(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_csv = args.out_dir / "diagnostics_all_ckpts.csv"
    thresholds_json = args.out_dir / "calibration_thresholds_seed3072.json"
    predictions_csv = args.out_dir / "predictions_heldout.csv"
    build_diagnostics(args.phase0, args.smpr, diagnostics_csv)
    calibrate(
        diagnostics_csv,
        args.score_manifest_dir,
        thresholds_json,
        rho=args.rho,
        eval_metric=args.eval_metric,
        clean_guard_pp=args.clean_guard_pp,
    )
    predict(diagnostics_csv, thresholds_json, predictions_csv)
    validation_paths = validate(
        predictions_csv,
        args.score_manifest_dir,
        args.out_dir,
        rho=args.rho,
        eval_metric=args.eval_metric,
        clean_guard_pp=args.clean_guard_pp,
    )
    plot_paths = plot_heldout(validation_paths["validation_rows"], args.out_dir)
    readme = write_readme(args.out_dir, args.smpr, args.rho)
    print(f"wrote {diagnostics_csv}")
    print(f"wrote {thresholds_json}")
    print(f"wrote {predictions_csv}")
    print(f"wrote {args.out_dir / 'validation_summary.csv'}")
    for plot_path in plot_paths:
        print(f"wrote {plot_path}")
    print(f"wrote {readme}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0", type=Path, default=DEFAULT_PHASE0)
    parser.add_argument("--smpr", type=Path, default=DEFAULT_SMPR)
    parser.add_argument("--score-manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--eval-metric", default="pixels_std0.08")
    parser.add_argument("--clean-guard-pp", type=float, default=None)
    args = parser.parse_args(argv)
    run_all(args)


if __name__ == "__main__":
    main()
