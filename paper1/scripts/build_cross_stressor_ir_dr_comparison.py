#!/usr/bin/env python3
"""Compare IR--DR score changes with blur/resize planning changes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .ir_dr_compat import to_ir_dr
from .utils_paper1_io import ROOT, SEEDS, TASKS, write_csv


DEFAULT_PAIRS = ROOT / "paper1/results/external_validation/cross_stressor_all_pairs.csv"
DEFAULT_P2 = ROOT / "paper1/results/cross_task_ir_dr_all_subsets_summary_v1.json"
DEFAULT_ROWS = ROOT / "paper1/results/external_validation/cross_stressor_three_source_ir_dr_v1.csv"
DEFAULT_SUMMARY = ROOT / "paper1/results/external_validation/cross_stressor_three_source_ir_dr_v1.json"
DEFAULT_TABLE = ROOT / "paper1/tables/table_cross_stressor_ir_dr_summary_v1.tex"
DEFAULT_ALL_PAIRS = ROOT / "paper1/tables/table_cross_stressor_ir_dr_all_pairs_v1.tex"
DEFAULT_FIGURE = ROOT / "assets/paper1_figs/fig_cross_stressor_ir_dr_comparison_v1.pdf"

FIELDS = [
    "task",
    "training_seed",
    "stressor",
    "ir_threshold",
    "dr_threshold",
    "base_stressed_score",
    "endpoint_stressed_score",
    "delta_behavior",
    "behavior_class",
    "observed_positive",
    "base_ir_relative",
    "endpoint_ir_relative",
    "base_dr",
    "endpoint_dr",
    "base_ir_dr_score",
    "endpoint_ir_dr_score",
    "delta_ir_dr_score",
    "predicted_positive",
    "discordance",
]

TASK_MARKER = {"TwoRoom": "o", "PushT": "s", "Reacher": "^", "Cube": "D"}
STRESSOR_STYLE = {
    "blur": {"color": "#0072B2", "filled": True, "label": "Blur"},
    "resize": {"color": "#E69F00", "filled": False, "label": "Resize"},
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _score(
    ir_relative: float,
    dr: float,
    ir_threshold: float,
    dr_threshold: float,
) -> float:
    radius_margin = (ir_threshold - ir_relative) / (abs(ir_threshold) + 1e-12)
    separation_margin = (dr - dr_threshold) / (abs(dr_threshold) + 1e-12)
    return min(radius_margin, separation_margin)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = 0.5 * ((position + 1) + end)
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else math.nan


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["observed_positive"] and row["predicted_positive"] for row in rows)
    tn = sum((not row["observed_positive"]) and (not row["predicted_positive"]) for row in rows)
    fp = sum((not row["observed_positive"]) and row["predicted_positive"] for row in rows)
    fn = sum(row["observed_positive"] and (not row["predicted_positive"]) for row in rows)
    recall = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    if math.isfinite(recall) and math.isfinite(specificity):
        balanced_accuracy = 0.5 * (recall + specificity)
    elif math.isfinite(recall):
        balanced_accuracy = recall
    elif math.isfinite(specificity):
        balanced_accuracy = specificity
    else:
        balanced_accuracy = math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall
        else math.nan
    )
    return {
        "n": len(rows),
        "positive_n": sum(row["observed_positive"] for row in rows),
        "predicted_positive_n": sum(row["predicted_positive"] for row in rows),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "spearman_delta_behavior_vs_delta_ir_dr_score": _spearman(
            [row["delta_behavior"] for row in rows],
            [row["delta_ir_dr_score"] for row in rows],
        ),
        "signed_order_agreement": sum(
            (row["delta_behavior"] > 0) == (row["delta_ir_dr_score"] > 0)
            for row in rows
        )
        / len(rows),
        "discordant_n": sum(row["discordance"].startswith("false_") for row in rows),
    }


def _task_thresholds(p2_summary: dict[str, Any]) -> dict[str, tuple[float, float]]:
    thresholds: dict[str, tuple[float, float]] = {}
    for item in p2_summary["partitions"]:
        if item["source_coverage"] != 3 or len(item["evaluation_tasks"]) != 1:
            continue
        task = item["evaluation_tasks"][0]
        thresholds[task] = (
            float(item["ir_threshold"]),
            float(item["dr_threshold"]),
        )
    if set(thresholds) != set(TASKS):
        raise ValueError(f"missing three-source thresholds: {sorted(set(TASKS) - set(thresholds))}")
    return thresholds


def build(
    pair_rows: list[dict[str, str]], p2_summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_rows = to_ir_dr(pair_rows)
    p2_summary = to_ir_dr(p2_summary)
    lewm = [row for row in pair_rows if row["model_family"] == "LeWM"]
    if len(lewm) != 24:
        raise ValueError(f"expected 24 LeWM pairs, found {len(lewm)}")
    expected = {(task, seed, stressor) for task in TASKS for seed in SEEDS for stressor in ("blur", "resize")}
    observed = {
        (row["task"], int(row["training_seed"]), row["stressor_family"])
        for row in lewm
    }
    if observed != expected:
        raise ValueError("cross-stressor pair coverage mismatch")

    thresholds = _task_thresholds(p2_summary)
    output: list[dict[str, Any]] = []
    for row in lewm:
        task = row["task"]
        ir_threshold, dr_threshold = thresholds[task]
        base_ir_raw = float(row["base_ir_raw"])
        endpoint_ir_raw = float(row["endpoint_ir_raw"])
        if base_ir_raw <= 0:
            raise ValueError(f"{task}: non-positive cross-stressor IR reference")
        base_ir_relative = 1.0
        endpoint_ir_relative = endpoint_ir_raw / base_ir_raw
        base_dr = float(row["base_dr"])
        endpoint_dr = float(row["endpoint_dr"])
        base_score = _score(
            base_ir_relative,
            base_dr,
            ir_threshold,
            dr_threshold,
        )
        endpoint_score = _score(
            endpoint_ir_relative,
            endpoint_dr,
            ir_threshold,
            dr_threshold,
        )
        delta_score = endpoint_score - base_score
        observed_positive = _bool(row["positive_transfer_label"])
        predicted_positive = delta_score > 0
        if observed_positive and predicted_positive:
            discordance = "concordant_positive"
        elif (not observed_positive) and (not predicted_positive):
            discordance = "concordant_nonpositive"
        elif predicted_positive:
            discordance = "false_positive"
        else:
            discordance = "false_negative"
        output.append(
            {
                "task": task,
                "training_seed": int(row["training_seed"]),
                "stressor": row["stressor_family"],
                "ir_threshold": ir_threshold,
                "dr_threshold": dr_threshold,
                "base_stressed_score": float(row["base_stressed_score"]),
                "endpoint_stressed_score": float(row["endpoint_stressed_score"]),
                "delta_behavior": float(row["delta_behavior"]),
                "behavior_class": row["behavior_class"],
                "observed_positive": observed_positive,
                "base_ir_relative": base_ir_relative,
                "endpoint_ir_relative": endpoint_ir_relative,
                "base_dr": base_dr,
                "endpoint_dr": endpoint_dr,
                "base_ir_dr_score": base_score,
                "endpoint_ir_dr_score": endpoint_score,
                "delta_ir_dr_score": delta_score,
                "predicted_positive": predicted_positive,
                "discordance": discordance,
            }
        )
    output.sort(key=lambda item: (TASKS.index(item["task"]), item["training_seed"], item["stressor"]))

    by_stressor = {
        stressor: _metrics([row for row in output if row["stressor"] == stressor])
        for stressor in ("blur", "resize")
    }
    by_task = {
        task: _metrics([row for row in output if row["task"] == task])
        for task in TASKS
    }
    summary = {
        "schema_version": "paper1-cross-stressor-ir-dr-comparison-1.0",
        "threshold_source": "three Gaussian source tasks for each evaluation task",
        "threshold_search_on_blur_or_resize": False,
        "training_seeds": SEEDS,
        "task_thresholds": {
            task: {"ir_threshold": value[0], "dr_threshold": value[1]}
            for task, value in thresholds.items()
        },
        "overall": _metrics(output),
        "by_stressor": by_stressor,
        "by_task": by_task,
        "discordant_rows": [row for row in output if row["discordance"].startswith("false_")],
    }
    return output, summary


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_summary_table(summary: dict[str, Any], out: Path) -> None:
    rows = [("All pairs", summary["overall"])] + [
        (label, summary["by_stressor"][key]) for key, label in (("blur", "Blur"), ("resize", "Resize"))
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{IR--DR scores under blur and resize. For each task, thresholds are chosen on the other three Gaussian-noise tasks and then fixed; each pair compares the $\stdmax{}=0.08$ checkpoint with its unaugmented counterpart. ``Discordant'' counts pairs whose score-change sign disagrees with the predefined success criterion (\Cref{sec:exp-cross-stressor}).}",
        r"\label{tab:cross-stressor-ir-dr-summary}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Visual shift & Pairs & BA & Precision / recall & Spearman & Discordant \\",
        r"\midrule",
    ]
    for label, metrics in rows:
        lines.append(
            f"{label} & {metrics['n']} & {metrics['balanced_accuracy']:.3f} & "
            f"{metrics['precision']:.3f} / {metrics['recall']:.3f} & "
            f"{metrics['spearman_delta_behavior_vs_delta_ir_dr_score']:.3f} & "
            f"{metrics['discordant_n']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_pairs_table(rows: list[dict[str, Any]], out: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{All 24 LeWM checkpoint pairs evaluated under blur and resize. IR$_{\rm rel}$ and DR are measured for the checkpoint trained with Gaussian-noise augmentation ($\stdmax{}=0.08$); $\Delta P$ and $\Delta S$ are its success-rate and IR--DR score changes relative to the unaugmented checkpoint. A pair is positive when $\Delta P\geq5$ percentage points with at most a five-point clean loss. Daggers mark the two discordant pairs.}",
        r"\label{tab:cross-stressor-ir-dr-all-pairs}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrlrrrrrl}",
        r"\toprule",
        r"Task & Seed & Shift & IR$_{\rm rel}$ & DR & $\Delta P$ & $\Delta S$ & Outcome (success / IR--DR) \\",
        r"\midrule",
    ]
    labels = {
        "concordant_positive": "positive / positive",
        "concordant_nonpositive": "nonpositive / nonpositive",
        "false_positive": "nonpositive / positive",
        "false_negative": "positive / nonpositive",
    }
    for row in rows:
        lines.append(
            f"{row['task']} & {row['training_seed']} & {row['stressor']} & "
            f"{row['endpoint_ir_relative']:.3f} & {row['endpoint_dr']:.3f} & "
            f"{row['delta_behavior']:.1f} & {row['delta_ir_dr_score']:.3f} & "
            f"{labels[row['discordance']]}"
            + ("$^\\dag$" if row["discordance"].startswith("false_") else "")
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, Any]], out: Path) -> None:
    """Two aligned descriptive rows per task and shift, averaged over the
    three training runs: the observed success change under the shift (top)
    and the IR--DR score change (bottom). No thresholds or verdicts are
    drawn; per-pair values are in the appendix table."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stressor_colors = {"blur": "#4477AA", "resize": "#EE6677"}
    stressors = ("blur", "resize")
    with plt.rc_context(STYLE):
        fig, (behavior_ax, score_ax) = plt.subplots(
            1,
            2,
            figsize=(6.7, 2.2),
            gridspec_kw={"wspace": 0.26},
        )

        behavior_means: list[float] = []
        score_means: list[float] = []
        positions: list[float] = []
        bar_stressors: list[str] = []
        tick_positions: list[float] = []
        for task_index, task in enumerate(TASKS):
            base = task_index * 3.0
            tick_positions.append(base + 0.5)
            for offset, stressor in enumerate(stressors):
                cell = [
                    row
                    for row in rows
                    if row["task"] == task and row["stressor"] == stressor
                ]
                if len(cell) != len(SEEDS):
                    raise ValueError(f"{task}/{stressor}: expected one row per run")
                positions.append(base + offset)
                bar_stressors.append(stressor)
                behavior_means.append(mean(row["delta_behavior"] for row in cell))
                score_means.append(
                    mean(row["delta_ir_dr_score"] for row in cell)
                )

        for position, stressor, behavior, score in zip(
            positions, bar_stressors, behavior_means, score_means
        ):
            color = stressor_colors[stressor]
            bar_kwargs = {
                "width": 0.85,
                "color": color,
                "hatch": "///" if stressor == "resize" else None,
                "edgecolor": "white" if stressor == "resize" else color,
                "linewidth": 0.4,
                "zorder": 2,
            }
            behavior_ax.bar(position, behavior, **bar_kwargs)
            score_ax.bar(position, score, **bar_kwargs)
            behavior_ax.annotate(
                f"{behavior:.0f}",
                (position, behavior),
                textcoords="offset points",
                xytext=(0, 1.5 if behavior >= 0 else -7),
                ha="center",
                fontsize=5.8,
                color="#333333",
            )
            score_ax.annotate(
                f"{score:.1f}",
                (position, score),
                textcoords="offset points",
                xytext=(0, 1.5 if score >= 0 else -7),
                ha="center",
                fontsize=5.8,
                color="#333333",
            )

        behavior_ax.axhline(0.0, color="#333333", linewidth=0.9, zorder=3)
        behavior_ax.set_ylim(min(behavior_means) - 4.0, max(behavior_means) + 13.0)
        behavior_ax.set_ylabel("Success-rate change\nunder the shift (pp)")
        behavior_ax.set_title("(a) Planning gain", loc="left", fontweight="semibold")

        score_ax.axhline(0.0, color="#333333", linewidth=0.9, zorder=3)
        score_ax.set_ylim(min(score_means) - 0.4, max(score_means) + 0.9)
        score_ax.set_ylabel("IR--DR score\nchange $\\Delta S$")
        score_ax.set_title("(b) Diagnostic gain", loc="left", fontweight="semibold")

        for axis in (behavior_ax, score_ax):
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(list(TASKS))
            axis.tick_params(axis="x", length=0)
            axis.set_xlim(-0.9, positions[-1] + 0.9)
        for axis in (behavior_ax, score_ax):
            axis.grid(True, axis="y", color="#B0B0B0", alpha=0.22, linewidth=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        legend_handles = [
            Patch(facecolor=stressor_colors["blur"], label="Blur"),
            Patch(
                facecolor=stressor_colors["resize"],
                hatch="///",
                edgecolor="white",
                label="Resize",
            ),
        ]
        for axis in (behavior_ax, score_ax):
            axis.legend(
                handles=legend_handles,
                loc="upper left",
                ncol=2,
                frameon=False,
                fontsize=6.8,
                handlelength=1.5,
                columnspacing=0.9,
                handletextpad=0.5,
                borderaxespad=0.2,
            )
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.86, wspace=0.30)
        fig.savefig(out)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--p2-summary", type=Path, default=DEFAULT_P2)
    parser.add_argument("--rows-out", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--all-pairs-table-out", type=Path, default=DEFAULT_ALL_PAIRS)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    p2_summary = json.loads(args.p2_summary.read_text(encoding="utf-8"))
    pair_rows = _read_csv(args.pairs)
    rows, summary = build(pair_rows, p2_summary)
    write_csv(args.rows_out, rows, FIELDS)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_table(summary, args.table_out)
    write_all_pairs_table(rows, args.all_pairs_table_out)
    plot(rows, args.figure_out)
    print(f"wrote {args.rows_out} ({len(rows)} pairs)")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.table_out}")
    print(f"wrote {args.all_pairs_table_out}")
    print(f"wrote {args.figure_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
