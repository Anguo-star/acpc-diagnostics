#!/usr/bin/env python3
"""Build the reader-facing three-seed future-drift display from adjudication JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .utils_paper1_io import ROOT, TASKS, write_csv


SOURCE_DIR = ROOT / "paper1/results/target_aligned_acpc_dev"
OUT_CSV = ROOT / "paper1/results/future_drift_three_seed_v1.csv"
OUT_SUMMARY = ROOT / "paper1/results/future_drift_three_seed_summary_v1.json"
OUT_TABLE = ROOT / "paper1/tables/table_target_aligned_acpc.tex"
OUT_ABSOLUTE = ROOT / "paper1/tables/table_target_aligned_acpc_absolute.tex"
OUT_FIGURE = ROOT / "assets/paper1_figs/fig_future_drift_three_seed_v1.pdf"

# Reader-facing control names (Sec. 5.3): internal CSV names -> manuscript names.
CONTROL_DISPLAY = {
    "zero actions": "zeroed actions",
    "batch-permuted actions": "swapped actions",
    "time-shuffled actions": "shuffled actions",
}

SEEDS = (3072, 3073, 3074)
CONTROL_NAMES = {
    "plus_action_zero_h8_control": "zero actions",
    "plus_candidate_shuffle_h8_control": "batch-permuted actions",
    "plus_time_shuffle_h8_control": "time-shuffled actions",
}


def _seed_rows(seed: int) -> list[dict[str, Any]]:
    if seed == 3072:
        path = SOURCE_DIR / "adjudication_four_task_seed3072_goal25_base_endpoint_retrospective_v1.json"
        payload = json.loads(path.read_text())
        cells = payload["primary_logged_fragile_base"]["cells"]
        absolute = next(cell for cell in cells if cell["target"] == "absolute")
        rows = []
        for item in absolute["per_task"]:
            rows.append(
                {
                    "task": item["task"],
                    "training_seed": seed,
                    "one_step_mae": item["h1_mae"],
                    "best_control_mae": item["best_destroyed_mae"],
                    "eight_step_mae": item["correct_mae"],
                    "best_control": CONTROL_NAMES[item["best_destroyed_name"]],
                    "reduction_vs_one_step": item["relative_mae_reduction_vs_h1"],
                    "reduction_vs_control": item["relative_mae_reduction_vs_best_destroyed"],
                    "win_blocks": item["both_win_blocks"],
                    "block_count": item["block_count"],
                    "pass": item["pass"],
                }
            )
        return rows

    path = SOURCE_DIR / f"adjudication_four_task_seed{seed}_goal25_base_endpoint_v1.json"
    payload = json.loads(path.read_text())
    rows = []
    for artifact in payload["artifacts"]:
        if f"baseline_seed{seed}" not in os.path.basename(artifact["checkpoint"]):
            continue
        gate = artifact["logged_gates"]["correct_absolute_h8_error_drift"]
        rows.append(
            {
                "task": artifact["task"],
                "training_seed": seed,
                "one_step_mae": gate["mae"]["shallow"],
                "best_control_mae": gate["mae"]["best_destroyed"],
                "eight_step_mae": gate["mae"]["correct"],
                "best_control": CONTROL_NAMES[gate["best_destroyed_name"]],
                "reduction_vs_one_step": gate["relative_mae_reduction"]["versus_shallow"],
                "reduction_vs_control": gate["relative_mae_reduction"]["versus_best_destroyed"],
                "win_blocks": gate["block_direction"]["both_win_count"],
                "block_count": gate["block_direction"]["paired_group_count"],
                "pass": gate["pass"],
            }
        )
    return rows


def _seed_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    selected = [row for row in rows if row["training_seed"] == seed]
    if {row["task"] for row in selected} != set(TASKS):
        raise ValueError(f"seed {seed} does not contain all four tasks")
    return {
        "training_seed": seed,
        "reduction_vs_one_step": mean(row["reduction_vs_one_step"] for row in selected),
        "reduction_vs_control": mean(row["reduction_vs_control"] for row in selected),
        "tasks_passed": sum(bool(row["pass"]) for row in selected),
        "task_count": len(selected),
    }


def _write_summary_table(summaries: list[dict[str, Any]]) -> None:
    a = [100 * row["reduction_vs_one_step"] for row in summaries]
    b = [100 * row["reduction_vs_control"] for row in summaries]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Relative reduction in out-of-group MAE from Base+ACPC$_8$ when predicting the error drift $d$ on the unaugmented checkpoints (protocol and model definitions in \Cref{sec:exp-target-aligned}). Base+Control$_8$ uses the per-cell oracle control; higher is better. The last column counts task--run cells in which Base+ACPC$_8$ is best.}",
        r"\label{tab:target-aligned-acpc}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Training run (seed) & vs. Base & vs. Base+Control$_8$ & Task--run cells improved \\",
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{row['training_seed']} & {100*row['reduction_vs_one_step']:.1f}\\% & "
            f"{100*row['reduction_vs_control']:.1f}\\% & {row['tasks_passed']}/{row['task_count']} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"mean $\\pm$ SD & {mean(a):.1f} $\\pm$ {stdev(a):.1f}\\% & "
            f"{mean(b):.1f} $\\pm$ {stdev(b):.1f}\\% & 12/12 \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    OUT_TABLE.write_text("\n".join(lines) + "\n")


def _write_absolute_table(rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Out-of-group MAE (16-fold leave-one-trajectory-group-out) for predicting the error drift $d$ on the unaugmented checkpoints; lower is better, model definitions in \Cref{sec:exp-target-aligned}. ``Selected control'' is the destroyed-action control chosen by the per-cell oracle; ``paired wins'' counts the folds (of 16) in which Base+ACPC$_8$ beats Base+Control$_8$.}",
        r"\label{tab:target-aligned-acpc-absolute}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lrrrrlc}",
        r"\toprule",
        r"Task & \shortstack{run\\(seed)} & Base & \shortstack{Base\\$+$Control$_8$} & \shortstack{Base\\$+$ACPC$_8$} & \shortstack{selected\\control} & \shortstack{paired\\wins} \\",
        r"\midrule",
    ]
    ordered = sorted(rows, key=lambda row: (TASKS.index(row["task"]), row["training_seed"]))
    for index, row in enumerate(ordered):
        lines.append(
            f"{row['task']} & {row['training_seed']} & {row['one_step_mae']:.3f} & "
            f"{row['best_control_mae']:.3f} & {row['eight_step_mae']:.3f} & "
            f"{CONTROL_DISPLAY[row['best_control']]} & {row['win_blocks']}/{row['block_count']} \\\\"
        )
        if index + 1 < len(ordered) and ordered[index + 1]["task"] != row["task"]:
            lines.append(r"\addlinespace[1pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    OUT_ABSOLUTE.write_text("\n".join(lines) + "\n")


def _plot(rows: list[dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6.8, 2.6))
    methods = (
        ("one_step_mae", "1-step\nbaseline", "#9A9A9A"),
        (
            "best_control_mae",
            "+ 8-step\noracle",
            "#5F5F5F",
        ),
        (
            "eight_step_mae",
            "+ 8-step\nrecorded",
            "#0072B2",
        ),
    )
    run_offsets = (-0.055, 0.0, 0.055)
    for panel_index, (ax, task) in enumerate(zip(axes, TASKS)):
        task_rows = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: int(row["training_seed"]),
        )
        if len(task_rows) != len(SEEDS):
            raise ValueError(f"{task}: expected one row per training run")

        for run_offset, row in zip(run_offsets, task_rows):
            run_values = [float(row[key]) for key, _, _ in methods]
            ax.plot(
                [index + run_offset for index in range(len(methods))],
                run_values,
                color="#B8B8B8",
                linewidth=0.65,
                alpha=0.85,
                zorder=1,
            )
            for method_index, ((_, _, color), value) in enumerate(
                zip(methods, run_values)
            ):
                ax.scatter(
                    method_index + run_offset,
                    value,
                    s=18,
                    color=color,
                    alpha=0.72,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=2,
                )

        for method_index, (key, _, color) in enumerate(methods):
            values = [float(row[key]) for row in task_rows]
            ax.scatter(
                method_index,
                mean(values),
                marker="D",
                s=48,
                color=color,
                edgecolor="#222222",
                linewidth=0.55,
                zorder=3,
            )

        task_max = max(float(row[key]) for row in task_rows for key, _, _ in methods)
        ax.set_ylim(0, task_max * 1.16)
        ax.set_xlim(-0.24, len(methods) - 0.76)
        ax.set_xticks(
            range(len(methods)),
            [label for _, label, _ in methods],
        )
        ax.set_title(
            task,
            loc="left",
            fontsize=8.5,
            fontweight="semibold",
            pad=5,
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", direction="out", length=2.5, labelsize=6.4)
        ax.tick_params(axis="y", direction="out", length=2.5, labelsize=7.0)
        ax.locator_params(axis="y", nbins=4)

    axes[0].set_ylabel(
        r"Regression MAE ($\downarrow$)",
        fontsize=7.6,
        labelpad=5,
    )
    legend_labels = (
        r"baseline = recorded-action ACPC$_1$",
        r"oracle control = $+$ best-of-three ACPC$_8$ control",
        r"recorded = $+$ recorded-action ACPC$_8$",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor="#222222",
            markeredgewidth=0.45,
        )
        for _, _, color in methods
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=6.8,
        handletextpad=0.35,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.17, top=0.82, wspace=0.25)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE)
    plt.close(fig)


def main() -> None:
    rows = [row for seed in SEEDS for row in _seed_rows(seed)]
    summaries = [_seed_summary(rows, seed) for seed in SEEDS]
    write_csv(OUT_CSV, rows, list(rows[0]))
    a = [row["reduction_vs_one_step"] for row in summaries]
    b = [row["reduction_vs_control"] for row in summaries]
    payload = {
        "target": (
            "absolute difference between nominal and perturbed eight-step "
            "latent prediction errors"
        ),
        "training_seeds": list(SEEDS),
        "seed_summaries": summaries,
        "mean_reduction_vs_one_step": mean(a),
        "sample_sd_reduction_vs_one_step": stdev(a),
        "mean_reduction_vs_same_horizon_control": mean(b),
        "sample_sd_reduction_vs_same_horizon_control": stdev(b),
        "all_seed_task_cells_pass": all(row["pass"] for row in rows),
    }
    OUT_SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_summary_table(summaries)
    _write_absolute_table(rows)
    _plot(rows)


if __name__ == "__main__":
    main()
