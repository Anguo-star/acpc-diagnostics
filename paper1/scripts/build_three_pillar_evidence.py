#!/usr/bin/env python3
"""Build the claim-aligned P1/P2/P3 evidence bundle and LaTeX tables.

This script performs no threshold search and no model evaluation. It consumes
the frozen target-aligned, cross-seed, and cross-stressor artifacts, verifies
their contracts, adds block-level uncertainty/deletion audits, and renders the
three compact tables used by the rewritten Paper1 mainline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta as beta_distribution
from sklearn.metrics import average_precision_score


SCHEMA = "paper1-three-pillar-evidence-1.0"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
TEST_SEEDS = (3073, 3074)
PROTOCOL_HASH = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value!r}")
    return normalized == "true"


def _classification_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        raise ValueError("classification metrics require at least one row")
    truth = np.asarray([_bool(row["behavior_label"]) for row in selected])
    pred = np.asarray([_bool(row["frozen_gate_pass"]) for row in selected])
    score = np.asarray([float(row["joint_score"]) for row in selected])
    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    auprc = (
        float(average_precision_score(truth.astype(int), score))
        if np.any(truth) and np.any(~truth)
        else None
    )
    return {
        "n": len(selected),
        "positive_n": int(np.sum(truth)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "balanced_accuracy": balanced_accuracy,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def _percentile_ci(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("cannot form an interval from no values")
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _p2_bootstrap(
    rows: list[dict[str, Any]],
    *,
    repetitions: int = 5000,
    seed: int = 20260713,
) -> dict[str, Any]:
    blocks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        blocks[(str(row["task"]), int(row["training_seed"]))].append(row)
    expected = {(task, seed_id) for task in TASKS for seed_id in TEST_SEEDS}
    if set(blocks) != expected:
        raise ValueError(
            f"P2 block contract mismatch: expected {sorted(expected)}, "
            f"got {sorted(blocks)}"
        )
    if any(len(block) != 9 for block in blocks.values()):
        raise ValueError("P2 requires nine checkpoint rows per task/seed block")

    block_keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        indices = rng.integers(0, len(block_keys), size=len(block_keys))
        sampled_rows = [
            row for index in indices for row in blocks[block_keys[int(index)]]
        ]
        metrics = _classification_metrics(sampled_rows)
        for field in (
            "balanced_accuracy",
            "auprc",
            "precision",
            "recall",
            "specificity",
        ):
            value = metrics[field]
            if value is not None:
                samples[field].append(float(value))
    return {
        "block_count": len(block_keys),
        "repetitions": repetitions,
        "seed": seed,
        "block": "task x independently trained TEST seed; all nine rows retained",
        "metrics": {
            field: {
                "observed": _classification_metrics(rows)[field],
                "ci95": _percentile_ci(values),
                "valid_repetitions": len(values),
            }
            for field, values in sorted(samples.items())
        },
    }


def _range(values: Iterable[float | None]) -> list[float] | None:
    finite = [float(value) for value in values if value is not None]
    return [min(finite), max(finite)] if finite else None


def _p2_deletion(
    rows: list[dict[str, Any]],
    *,
    field: str,
    values: Iterable[str | int],
) -> dict[str, Any]:
    folds = []
    for value in values:
        remaining = [row for row in rows if str(row[field]) != str(value)]
        held_out = [row for row in rows if str(row[field]) == str(value)]
        folds.append(
            {
                "held_out_value": value,
                "remaining_after_deletion": _classification_metrics(remaining),
                "held_out": _classification_metrics(held_out),
            }
        )
    metric_fields = (
        "balanced_accuracy",
        "auprc",
        "precision",
        "recall",
        "specificity",
    )
    return {
        "group_field": field,
        "interpretation": (
            "deletion stability of the frozen seed3072 gate; no threshold or "
            "feature is fitted in a fold"
        ),
        "folds": folds,
        "remaining_metric_range": {
            metric: _range(
                fold["remaining_after_deletion"][metric] for fold in folds
            )
            for metric in metric_fields
        },
    }


def _p2_evidence(
    rows: list[dict[str, Any]],
    frozen_summary: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != 72:
        raise ValueError(f"P2 requires 72 TEST rows, got {len(rows)}")
    if {str(row["model_family"]) for row in rows} != {"LeWM"}:
        raise ValueError("P2 rows must contain only LeWM")
    if {int(row["training_seed"]) for row in rows} != set(TEST_SEEDS):
        raise ValueError("P2 rows must contain exactly seeds3073/3074")
    if {str(row["task"]) for row in rows} != set(TASKS):
        raise ValueError("P2 rows must contain exactly four tasks")
    if {str(row["protocol_sha256"]) for row in rows} != {PROTOCOL_HASH}:
        raise ValueError("P2 protocol hash mismatch")

    overall = _classification_metrics(rows)
    frozen_metrics = frozen_summary["metrics"]
    for field in ("balanced_accuracy", "auprc", "precision", "recall"):
        if not np.isclose(float(overall[field]), float(frozen_metrics[field])):
            raise ValueError(f"P2 metric mismatch for {field}")

    by_seed = {
        str(seed_id): _classification_metrics(
            row for row in rows if int(row["training_seed"]) == seed_id
        )
        for seed_id in TEST_SEEDS
    }
    by_task = {
        task: _classification_metrics(
            row for row in rows if str(row["task"]) == task
        )
        for task in TASKS
    }
    onset_errors = [
        abs(float(block["onset_error"])) for block in frozen_summary["blocks"]
    ]
    return {
        "calibration": "LeWM seed3072 only",
        "test_scope": "LeWM seeds3073/3074 x four tasks x nine checkpoints",
        "threshold_search_allowed": False,
        "protocol_sha256": PROTOCOL_HASH,
        "overall": overall,
        "by_training_seed": by_seed,
        "by_task": by_task,
        "onset_error": {
            "mean_absolute": float(np.mean(onset_errors)),
            "maximum_absolute": float(np.max(onset_errors)),
            "unit": "training-noise rho",
        },
        "block_bootstrap": _p2_bootstrap(rows),
        "deletion_stability": {
            "leave_one_task_out": _p2_deletion(
                rows, field="task", values=TASKS
            ),
            "leave_one_training_seed_out": _p2_deletion(
                rows, field="training_seed", values=TEST_SEEDS
            ),
        },
    }


def _clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(
            beta_distribution.ppf(
                alpha / 2.0, successes, trials - successes + 1
            )
        )
    )
    upper = (
        1.0
        if successes == trials
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0, successes + 1, trials - successes
            )
        )
    )
    return [lower, upper]


def _one_sided_binomial_upper(
    events: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> float:
    if trials < 1 or not 0 <= events <= trials:
        raise ValueError("invalid binomial counts")
    if events == trials:
        return 1.0
    return float(
        beta_distribution.ppf(confidence, events + 1, trials - events)
    )


def _p3_absolute_uncertainty(
    rows: list[dict[str, Any]],
    *,
    repetitions: int = 5000,
    seed: int = 20260713,
) -> dict[str, Any]:
    selected = [row for row in rows if str(row["model_family"]) == "LeWM"]
    if len(selected) != 24:
        raise ValueError(f"P3 absolute audit requires 24 LeWM rows, got {len(selected)}")
    normalized = [
        {
            **row,
            "behavior_label": _bool(row["positive_transfer_label"]),
            "frozen_gate_pass": _bool(row["endpoint_gate_pass"]),
            "joint_score": float(row["endpoint_joint_score"]),
        }
        for row in selected
    ]
    blocks: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        blocks[(str(row["task"]), int(row["training_seed"]))].append(row)
    expected = {
        (task, seed_id) for task in TASKS for seed_id in (3072, 3073, 3074)
    }
    if set(blocks) != expected:
        raise ValueError("P3 absolute task/training-seed block contract mismatch")
    if any(
        len(block) != 2
        or {str(row["stressor_family"]) for row in block} != {"blur", "resize"}
        for block in blocks.values()
    ):
        raise ValueError("P3 absolute blocks must retain one blur and one resize row")

    observed = _classification_metrics(normalized)
    pass_count = observed["tp"] + observed["fp"]
    coverage = pass_count / observed["n"]
    false_pass_risk = observed["fp"] / pass_count if pass_count else None

    block_keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        indices = rng.integers(0, len(block_keys), size=len(block_keys))
        sampled_rows = [
            row for index in indices for row in blocks[block_keys[int(index)]]
        ]
        metrics = _classification_metrics(sampled_rows)
        for field in (
            "balanced_accuracy",
            "auprc",
            "precision",
            "recall",
            "specificity",
        ):
            value = metrics[field]
            if value is not None:
                samples[field].append(float(value))
        sampled_pass_count = metrics["tp"] + metrics["fp"]
        samples["coverage"].append(sampled_pass_count / metrics["n"])
        if sampled_pass_count:
            samples["selective_false_pass_risk"].append(
                metrics["fp"] / sampled_pass_count
            )

    return {
        "coverage": {
            "observed": coverage,
            "pass_count": pass_count,
            "n": observed["n"],
            "exact_two_sided_95_ci": _clopper_pearson_interval(
                pass_count, observed["n"]
            ),
        },
        "selective_false_pass_risk": {
            "observed": false_pass_risk,
            "false_pass_count": observed["fp"],
            "pass_count": pass_count,
            "exact_one_sided_95_upper": _one_sided_binomial_upper(
                observed["fp"], pass_count
            ),
            "conditioning": "among absolute-gate passes",
        },
        "block_bootstrap": {
            "block_count": len(block_keys),
            "repetitions": repetitions,
            "seed": seed,
            "block": (
                "task x independently trained LeWM seed; blur and resize "
                "retained inside each resampled block"
            ),
            "metrics": {
                field: {
                    "observed": (
                        coverage
                        if field == "coverage"
                        else false_pass_risk
                        if field == "selective_false_pass_risk"
                        else observed[field]
                    ),
                    "ci95": _percentile_ci(values),
                    "valid_repetitions": len(values),
                }
                for field, values in sorted(samples.items())
            },
        },
    }


def _p3_evidence(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = summary["metadata"]
    if metadata["protocol_hash"] != PROTOCOL_HASH:
        raise ValueError("P3 protocol hash mismatch")
    if metadata["threshold_search_allowed"]:
        raise ValueError("P3 unexpectedly permits threshold search")
    absolute = next(
        row
        for row in summary["all_pair_strata"]["model_family"]
        if row["value"] == "LeWM"
    )
    paired = summary["paired_change"]["by_model_family"]["LeWM"]
    audit = summary["paired_change"]["lewm_robustness_audit"]
    absolute_uncertainty = _p3_absolute_uncertainty(rows)
    return {
        "calibration": "Gaussian only; unchanged LeWM gate",
        "test_scope": "three LeWM seeds x four tasks x blur/resize",
        "threshold_search_allowed": False,
        "absolute_single_checkpoint_screen": {
            **absolute,
            **absolute_uncertainty,
            "evaluation_target": (
                "pair-derived positive transfer: endpoint stressed success "
                "improves by at least 5pp over base while clean drop is at "
                "most 5pp"
            ),
            "scoring_input": (
                "endpoint checkpoint only; no base score is used by the "
                "frozen gate"
            ),
        },
        "paired_reference_rule": {
            "overall": paired["diagnostics"]["joint_score"],
            "by_stressor": paired["joint_by_stressor"],
            "block_bootstrap": summary["paired_change"][
                "lewm_block_bootstrap"
            ],
            "exact_randomization": audit["exact_randomization"],
            "deletion_stability": audit["deletion_stability"],
            "selection": audit["selection_by_diagnostic"]["joint_score"],
        },
        "boundary": (
            "the endpoint gate is reference-free at scoring time but is "
            "evaluated against a pair-derived improvement label, not an "
            "absolute stability label; the paired rule requires a reference "
            "checkpoint"
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _render_p1_table(p1: dict[str, Any]) -> str:
    uncertainty = p1["primary_logged_fragile_base"]["uncertainty"]
    abs_u = uncertainty["absolute"]
    adv_u = uncertainty["adverse"]
    abs_tasks = {row["task"]: row for row in abs_u["observed"]["per_task"]}
    adv_tasks = {row["task"]: row for row in adv_u["observed"]["per_task"]}
    lines = []
    for task in TASKS:
        a = abs_tasks[task]
        d = adv_tasks[task]
        lines.append(
            f"{task:<7} & {_pct(a['relative_reduction_vs_h1'])} & "
            f"{_pct(a['relative_reduction_vs_best_destroyed'])} & "
            f"{_pct(d['relative_reduction_vs_h1'])} & "
            f"{_pct(d['relative_reduction_vs_best_destroyed'])} \\\\"
        )
    cells = p1["primary_logged_fragile_base"]["cells"]
    seed_ids = sorted(
        {
            int(cell["training_seed"])
            for cell in cells
            if cell["target"] == "absolute"
        }
    )
    seed_labels = {
        3073: "dev-era",
        3074: "frozen repl.",
        3075: "prospective",
    }
    for seed_id in seed_ids:
        absolute = next(
            cell
            for cell in cells
            if cell["training_seed"] == seed_id and cell["target"] == "absolute"
        )
        adverse = next(
            cell
            for cell in cells
            if cell["training_seed"] == seed_id and cell["target"] == "adverse"
        )
        provenance = seed_labels.get(seed_id, "reported")
        lines.append(
            f"Seed {seed_id} ({provenance}) & "
            f"{_pct(absolute['equal_task_mean_reduction_vs_h1'])} & "
            f"{_pct(absolute['equal_task_mean_reduction_vs_best_destroyed'])} & "
            f"{_pct(adverse['equal_task_mean_reduction_vs_h1'])} & "
            f"{_pct(adverse['equal_task_mean_reduction_vs_best_destroyed'])} \\\\"
        )
    abs_h1_ci = abs_u["cluster_bootstrap"][
        "equal_task_mean_reduction_vs_h1_ci95"
    ]
    abs_control_ci = abs_u["cluster_bootstrap"][
        "equal_task_mean_reduction_vs_best_destroyed_ci95"
    ]
    adv_h1_ci = adv_u["cluster_bootstrap"][
        "equal_task_mean_reduction_vs_h1_ci95"
    ]
    adv_control_ci = adv_u["cluster_bootstrap"][
        "equal_task_mean_reduction_vs_best_destroyed_ci95"
    ]
    lines.append(
        f"{len(seed_ids)}-run conditional mean [95\\% CI] & "
        f"{_pct(abs_u['observed']['equal_task_mean_reduction_vs_h1'])} "
        f"[{_pct(abs_h1_ci[0])},{_pct(abs_h1_ci[1])}] & "
        f"{_pct(abs_u['observed']['equal_task_mean_reduction_vs_best_destroyed'])} "
        f"[{_pct(abs_control_ci[0])},{_pct(abs_control_ci[1])}] & "
        f"{_pct(adv_u['observed']['equal_task_mean_reduction_vs_h1'])} "
        f"[{_pct(adv_h1_ci[0])},{_pct(adv_h1_ci[1])}] & "
        f"{_pct(adv_u['observed']['equal_task_mean_reduction_vs_best_destroyed'])} "
        f"[{_pct(adv_control_ci[0])},{_pct(adv_control_ci[1])}] \\\\"
    )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Held-out future-drift prediction on no-noise LeWM base checkpoints under the frozen visual-probe grid. Entries are positive MAE reductions (\\%) after adding correct-action H8 to common encoder+H1 covariates; destroyed H8 is the best action/time-destroyed variant. Seed 3075 is fully prospective, seed 3074 is a protocol-frozen replication, and seed 3073 has development-era provenance. Intervals resample task$\\times$trajectory blocks conditional on the listed trained checkpoints, not a population of training runs.}",
            "\\label{tab:target-aligned-acpc}",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Task / aggregation & \\multicolumn{2}{c}{Absolute drift} & \\multicolumn{2}{c}{Adverse drift} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "& vs H1 & vs destroyed H8 & vs H1 & vs destroyed H8 \\\\",
            "\\midrule",
            *lines[:4],
            "\\midrule",
            *lines[4:],
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _render_p1_absolute_table(p1: dict[str, Any]) -> str:
    cells = p1["primary_logged_fragile_base"]["cells"]
    absolute_cells = {
        int(cell["training_seed"]): cell
        for cell in cells
        if cell["target"] == "absolute"
    }
    control_codes = {
        "plus_action_zero_h8_control": "AZ",
        "plus_candidate_shuffle_h8_control": "CS",
        "plus_time_shuffle_h8_control": "TS",
    }
    rows: list[str] = []
    for task_index, task in enumerate(TASKS):
        if task_index:
            rows.append("\\addlinespace[1pt]")
        for seed_id, cell in sorted(absolute_cells.items()):
            row = next(item for item in cell["per_task"] if item["task"] == task)
            code = control_codes.get(row["best_destroyed_name"])
            if code is None:
                raise ValueError(
                    f"unknown destroyed-control name: {row['best_destroyed_name']}"
                )
            correct = f"{row['correct_mae']:.3f}"
            if row["pass"]:
                correct = f"\\textbf{{{correct}}}"
            rows.append(
                f"{task} & {seed_id} & {correct} & {row['h1_mae']:.3f} & "
                f"{row['best_destroyed_mae']:.3f} ({code}) & "
                f"{row['both_win_blocks']}/{row['block_count']} & "
                f"{'yes' if row['pass'] else 'no'} \\\\"
            )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Absolute held-out MAE for the primary logged-future absolute-drift target. Each task$\\times$seed row contains 16 trajectory blocks; lower is better, and bold marks correct-action H8 when it passes the frozen within-row gate against H1 and every destroyed H8 variant. The destroyed column reports the strongest of action-zeroed (AZ), candidate-shuffled (CS), and time-shuffled (TS) H8. Seed 3075 is prospective, seed 3074 is the protocol-frozen replication, and seed 3073 is development-era.}",
            "\\label{tab:target-aligned-acpc-absolute}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3.2pt}",
            "\\begin{tabular}{lrrrrcc}",
            "\\toprule",
            "Task & seed & correct H8 & H1 & best destroyed H8 & win blocks & gate \\\\ ",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _render_p2_table(p2: dict[str, Any]) -> str:
    overall = p2["overall"]
    rows = []
    for label, metrics in [
        ("Both test seeds", overall),
        ("Seed 3073", p2["by_training_seed"]["3073"]),
        ("Seed 3074", p2["by_training_seed"]["3074"]),
    ]:
        rows.append(
            f"{label} & {metrics['n']} & {_fmt(metrics['balanced_accuracy'])} & "
            f"{_fmt(metrics['auprc'])} & {_fmt(metrics['precision'])} & "
            f"{_fmt(metrics['recall'])} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Held-out transfer of a LeWM calibration fit on seed 3072 and applied without refitting to independently trained seeds 3073/3074. BA denotes balanced accuracy; AUPRC denotes area under the precision--recall curve.}",
            "\\label{tab:seed-transfer-audit}",
            "\\small",
            "\\setlength{\\tabcolsep}{5pt}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Evaluation set & rows & BA & AUPRC & precision & recall \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def _render_p3_table(p3: dict[str, Any]) -> str:
    absolute = p3["absolute_single_checkpoint_screen"]
    paired = p3["paired_reference_rule"]
    rows = [
        (
            "Single-checkpoint region",
            absolute,
            "--",
        ),
        (
            "Paired change ($\\Delta S>0$)",
            paired["overall"],
            _fmt(paired["overall"][
                "spearman_delta_behavior_vs_oriented_delta_score"
            ]),
        ),
    ]
    rendered = [
        f"{label} & {_fmt(metrics['balanced_accuracy'])} & {_fmt(metrics['auprc'])} & "
        f"{_fmt(metrics['precision'])} & "
        f"{_fmt(metrics['recall'])} & {rho} \\\\"
        for label, metrics, rho in rows
    ]
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Transfer to all 24 fixed-severity LeWM blur/resize pairs without stressor-specific retuning. BA is balanced accuracy and AUPRC is area under the precision--recall curve. The single-checkpoint region accepts 7/24 endpoints (selective false-pass-risk U95: 0.348); paired change requires the no-noise reference.}",
            "\\label{tab:cross-stressor-transfer}",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Rule & BA & AUPRC & precision & recall & $\\rho_s$ \\\\",
            "\\midrule",
            *rendered,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def build(
    *,
    p1_meta_path: Path,
    p2_rows_path: Path,
    p2_summary_path: Path,
    p3_summary_path: Path,
    p3_rows_path: Path,
) -> dict[str, Any]:
    p1 = _load_json(p1_meta_path)
    p1_gate_pass = bool(
        p1["primary_logged_fragile_base"][
            "all_available_seeds_meet_three_task_gate"
        ]
    )
    p2 = _p2_evidence(_load_csv(p2_rows_path), _load_json(p2_summary_path))
    p3 = _p3_evidence(
        _load_json(p3_summary_path),
        _load_csv(p3_rows_path),
    )
    return {
        "metadata": {
            "schema_version": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "threshold_search_allowed": False,
            "model_evaluation_performed": False,
            "p1_prospective_gate_pass": p1_gate_pass,
            "claims": [
                (
                    "P1 target-aligned ACPC theory-evidence closure"
                    if p1_gate_pass
                    else "P1 prospective result reported; cross-run claim must be narrowed"
                ),
                "P2 within-family cross-seed frozen threshold transfer",
                "P3 Gaussian-to-blur/resize frozen transfer",
            ],
        },
        "P1": p1,
        "P2": p2,
        "P3": p3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-meta",
        type=Path,
        default=Path(
            "paper1/results/target_aligned_acpc_prospective_v1/"
            "meta_four_task_seeds3073_3074_3075_goal25_base_endpoint_v1.json"
        ),
    )
    parser.add_argument(
        "--p2-rows",
        type=Path,
        default=Path("paper1/results/frozen_external_validation_rows_v2.csv"),
    )
    parser.add_argument(
        "--p2-summary",
        type=Path,
        default=Path("paper1/results/frozen_external_validation_summary_v3.json"),
    )
    parser.add_argument(
        "--p3-summary",
        type=Path,
        default=Path(
            "paper1/results/external_validation/"
            "cross_stressor_fixed_rho_summary.json"
        ),
    )
    parser.add_argument(
        "--p3-rows",
        type=Path,
        default=Path(
            "paper1/results/external_validation/"
            "cross_stressor_all_pairs.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper1/results/three_pillar_evidence_summary.json"),
    )
    parser.add_argument(
        "--p1-table",
        type=Path,
        default=Path("paper1/tables/table_target_aligned_acpc.tex"),
    )
    parser.add_argument(
        "--p1-absolute-table",
        type=Path,
        default=Path("paper1/tables/table_target_aligned_acpc_absolute.tex"),
    )
    parser.add_argument(
        "--p2-table",
        type=Path,
        default=Path("paper1/tables/table_seed_transfer_audit.tex"),
    )
    parser.add_argument(
        "--p3-table",
        type=Path,
        default=Path("paper1/tables/table_cross_stressor_transfer.tex"),
    )
    args = parser.parse_args()
    result = build(
        p1_meta_path=args.p1_meta,
        p2_rows_path=args.p2_rows,
        p2_summary_path=args.p2_summary,
        p3_summary_path=args.p3_summary,
        p3_rows_path=args.p3_rows,
    )
    for path in (
        args.out,
        args.p1_table,
        args.p1_absolute_table,
        args.p2_table,
        args.p3_table,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.p1_table.write_text(_render_p1_table(result["P1"]), encoding="utf-8")
    args.p1_absolute_table.write_text(
        _render_p1_absolute_table(result["P1"]), encoding="utf-8"
    )
    args.p2_table.write_text(_render_p2_table(result["P2"]), encoding="utf-8")
    args.p3_table.write_text(_render_p3_table(result["P3"]), encoding="utf-8")
    print(
        json.dumps(
            {
                "P1_absolute_reduction_vs_h1": result["P1"][
                    "primary_logged_fragile_base"
                ]["uncertainty"]["absolute"]["observed"][
                    "equal_task_mean_reduction_vs_h1"
                ],
                "P2_balanced_accuracy": result["P2"]["overall"][
                    "balanced_accuracy"
                ],
                "P3_absolute_precision": result["P3"][
                    "absolute_single_checkpoint_screen"
                ]["precision"],
                "P3_paired_balanced_accuracy": result["P3"][
                    "paired_reference_rule"
                ]["overall"]["balanced_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
