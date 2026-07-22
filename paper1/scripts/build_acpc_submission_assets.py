#!/usr/bin/env python3
"""Build submission-facing ACPC planner figure and compact tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .cross_task_selective_rule import run_all_subsets
    from .ir_sr_compat import to_ir_sr
except ImportError:  # Support the historical direct-script entry point.
    from paper1.scripts.cross_task_selective_rule import run_all_subsets
    from paper1.scripts.ir_sr_compat import to_ir_sr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER = ROOT / "paper1/results/acpc_planner_stability_v4/summary.json"
DEFAULT_SWEEP = ROOT / "paper1/results/full_sweep_diagnostics_summary.csv"
DEFAULT_PLANNER_FIG = ROOT / "assets/paper1_figs/fig_acpc_planner_evidence.pdf"
DEFAULT_INCREMENT_TABLE = ROOT / "paper1/tables/table_acpc_planner_increment.tex"
DEFAULT_ABSOLUTE_TABLE = ROOT / "paper1/tables/table_acpc_planner_absolute.tex"
DEFAULT_SWEEP_TABLE = ROOT / "paper1/tables/table_full_sweep_compact_ir_sr_v2.tex"
DEFAULT_PLDM_ROWS = ROOT / "paper1/results/external_validation/pldm_frozen_rows_v2.csv"
DEFAULT_CROSS_TASK_SUMMARY = ROOT / "paper1/results/cross_task_ir_sr_all_subsets_summary_v2.json"
DEFAULT_PLDM_TABLE = ROOT / "paper1/tables/table_pldm_architecture_portability_ir_sr_v2.tex"

TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
TASK_MARKERS = {"TwoRoom": "o", "PushT": "s", "Reacher": "^", "Cube": "D"}
TASK_COLORS = {
    "TwoRoom": "#4E79A7",
    "PushT": "#F28E2B",
    "Reacher": "#59A14F",
    "Cube": "#E15759",
}
GRID_COLOR = "#A7A7A7"
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _polish(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.24, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _group_index(summary: dict[str, Any]) -> dict[tuple[str, str, float], dict[str, Any]]:
    return {
        (row["task"], row["checkpoint_role"], float(row["severity"])): row
        for row in summary["group_summary"]
    }


def plot_planner(summary: dict[str, Any], out: Path) -> None:
    analysis = summary["predeclared_incremental_analyses"]["positive_clean_regret"]
    seeds = tuple(int(seed) for seed in summary["training_seeds"])
    seed_offsets = {
        seed: offset for seed, offset in zip(seeds, (-0.045, 0.0, 0.045))
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        for task in TASKS:
            color = TASK_COLORS[task]
            marker = TASK_MARKERS[task]
            for seed in seeds:
                task_result = analysis["per_seed"][str(seed)]["lobo_ridge"][
                    "per_task"
                ][task]
                offset = seed_offsets[seed]
                xs = [offset, 1.0 + offset]
                ys = [
                    task_result["baseline_log1p_mae"],
                    task_result["plus_h5_log1p_mae"],
                ]
                ax.plot(xs, ys, color=color, lw=0.8, alpha=0.45, zorder=1)
                ax.scatter(
                    xs,
                    ys,
                    marker=marker,
                    s=24,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.82,
                    zorder=2,
                )

        base_run_means = [
            analysis["per_seed"][str(seed)]["lobo_ridge"][
                "equal_task_baseline_log1p_mae"
            ]
            for seed in seeds
        ]
        h5_run_means = [
            analysis["per_seed"][str(seed)]["lobo_ridge"][
                "equal_task_plus_h5_log1p_mae"
            ]
            for seed in seeds
        ]
        equal_task_means = [
            sum(base_run_means) / len(base_run_means),
            sum(h5_run_means) / len(h5_run_means),
        ]
        ax.plot(
            [0.0, 1.0],
            equal_task_means,
            color="#111111",
            lw=2.5,
            marker="D",
            markersize=5.6,
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=1.1,
            zorder=4,
        )
        aggregate = analysis["three_seed_summary"]
        reduction = 100.0 * aggregate["relative_mae_reduction_mean"]
        reduction_sd = 100.0 * aggregate["relative_mae_reduction_sample_sd"]
        improved = aggregate["task_seed_cells_improved"]
        total = aggregate["task_seed_cell_count"]
        ax.text(
            0.97,
            0.96,
            f"{reduction:.1f} ± {reduction_sd:.1f}% lower MAE\n"
            f"{improved}/{total} cross-task tests improve",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            fontweight="semibold",
            linespacing=1.25,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
                "pad": 1.8,
            },
        )
        ax.set_xlim(-0.20, 1.20)
        ax.set_xticks(
            [0.0, 1.0],
            [
                "Baseline factors",
                "Baseline +\nplanner-horizon ACPC",
            ],
        )
        ax.set_ylabel(r"Cross-task test MAE ($\downarrow$)")
        _polish(ax)

        task_handles = [
            Line2D(
                [],
                [],
                marker=TASK_MARKERS[task],
                color=TASK_COLORS[task],
                markerfacecolor=TASK_COLORS[task],
                ls="-",
                lw=0.9,
                label=task,
                ms=4.2,
            )
            for task in TASKS
        ]
        task_handles.append(
            Line2D(
                [],
                [],
                marker="D",
                color="#111111",
                markerfacecolor="white",
                lw=2.2,
                label="Equal-task mean",
                ms=4.6,
            )
        )
        fig.legend(
            handles=task_handles,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.995),
            frameon=False,
            handlelength=1.5,
            columnspacing=1.2,
        )
        fig.subplots_adjust(
            left=0.15,
            right=0.985,
            bottom=0.20,
            top=0.82,
        )
        fig.savefig(out, dpi=240)
        plt.close(fig)


def build_increment_table(summary: dict[str, Any]) -> str:
    analyses = summary["predeclared_incremental_analyses"]
    specs = (
        ("cost_drift", "Largest candidate-cost change in the shared pool"),
        ("positive_clean_regret", "Adaptive-CEM decision regret"),
        ("first_action_rms", "RMS change in the first planned action"),
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Reduction in prediction MAE after adding candidate-specific five-step ACPC. Models are fitted on three tasks and evaluated on the remaining task; higher is better.}",
        r"\label{tab:acpc-planner-increment}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabularx}{\linewidth}{Xcc}",
        r"\toprule",
        r"Prediction target & \shortstack{MAE reduction (\%) $\uparrow$\\mean $\pm$ run-to-run std. dev.} & \shortstack{Task--run evaluations\\improved (/12)} \\",
        r"\midrule",
    ]
    for key, label in specs:
        aggregate = analyses[key]["three_seed_summary"]
        mean = 100 * aggregate["relative_mae_reduction_mean"]
        sd = 100 * aggregate["relative_mae_reduction_sample_sd"]
        mean_display = f"{mean:.1f} $\\pm$ {sd:.1f}"
        lines.append(
            f"{label} & {mean_display} & "
            f"{aggregate['task_seed_cells_improved']}/12 \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\vspace{2pt}",
            "",
            r"\parbox{0.98\linewidth}{\scriptsize MAE is computed on the $\log(1+\mathrm{target})$ scale. The four evaluation tasks are weighted equally within each training run; entries then report the mean and standard deviation across three runs. The final column counts the 12 task--run evaluations separately.}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_absolute_table(summary: dict[str, Any]) -> str:
    index = _group_index(summary)
    full = {(row["task"], row["checkpoint_role"]): row for row in summary["full_budget_summary"]}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{CEM decision statistics at Gaussian perturbation severity 0.08. Values are mean $\pm$ standard deviation across training runs after averaging histories within each run. Reduced-budget rows use 100 histories, $K=64$, top-8 elites, and eight CEM iterations; full-budget rows use the same 16 histories per task with $K=300$, top-30 elites, and 30 iterations. Regret uses the model's squared latent goal cost, not an environment success rate.}",
        r"\label{tab:acpc-planner-absolute}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Best candidate unchanged} & \multicolumn{2}{c}{Reduced-budget regret} & \multicolumn{2}{c}{Full-budget regret} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Task & No aug. & $\sigma_{\max}{=}0.08$ & No aug. & $\sigma_{\max}{=}0.08$ & No aug. & $\sigma_{\max}{=}0.08$ \\",
        r"\midrule",
    ]
    def cell(record: dict[str, Any], field: str) -> str:
        return (
            f"{record[field]:.2f}$\\pm$"
            f"{record[field + '_sample_sd']:.2f}"
        )

    for task in TASKS:
        cells = [
            cell(index[(task, "base", 0.08)], "top1_stability_rate"),
            cell(index[(task, "endpoint", 0.08)], "top1_stability_rate"),
            cell(index[(task, "base", 0.08)], "positive_clean_regret_mean"),
            cell(index[(task, "endpoint", 0.08)], "positive_clean_regret_mean"),
            cell(full[(task, "base")], "positive_clean_regret_mean"),
            cell(full[(task, "endpoint")], "positive_clean_regret_mean"),
        ]
        lines.append(f"{task} & {' & '.join(cells)} \\\\")

    group_seed_rows = summary["group_seed_summary"]
    full_seed_rows = summary["full_budget_seed_summary"]

    def seed_macro(
        rows: list[dict[str, Any]],
        *,
        role: str,
        field: str,
        severity: float | None = None,
    ) -> str:
        values = []
        for seed in summary["training_seeds"]:
            selected = [
                row
                for row in rows
                if int(row["training_seed"]) == int(seed)
                and row["checkpoint_role"] == role
                and (severity is None or float(row["severity"]) == severity)
            ]
            if len(selected) != len(TASKS):
                raise ValueError("planner seed macro lacks four tasks")
            values.append(sum(float(row[field]) for row in selected) / len(TASKS))
        mean = sum(values) / len(values)
        sd = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        return f"{mean:.2f}$\\pm${sd:.2f}"

    means = [
        seed_macro(
            group_seed_rows,
            role="base",
            field="top1_stability_rate",
            severity=0.08,
        ),
        seed_macro(
            group_seed_rows,
            role="endpoint",
            field="top1_stability_rate",
            severity=0.08,
        ),
        seed_macro(
            group_seed_rows,
            role="base",
            field="positive_clean_regret_mean",
            severity=0.08,
        ),
        seed_macro(
            group_seed_rows,
            role="endpoint",
            field="positive_clean_regret_mean",
            severity=0.08,
        ),
        seed_macro(full_seed_rows, role="base", field="positive_clean_regret_mean"),
        seed_macro(
            full_seed_rows,
            role="endpoint",
            field="positive_clean_regret_mean",
        ),
    ]
    lines.extend(
        [
            r"\midrule",
            "Equal-task mean & " + " & ".join(means) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_sweep_table(rows: list[dict[str, str]]) -> str:
    rows = to_ir_sr(rows)
    by_task = {task: [row for row in rows if row["task"] == task] for task in TASKS}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Summary of the Gaussian-noise training sweep (nine checkpoints per task: no augmentation plus eight noise levels; \Cref{fig:full-sweep-diagnostics} shows every level). ``Best'' is the level with the highest mean planning success at evaluation noise $\sigma=0.08$; arrows give the change from the unaugmented checkpoint to that level. Relative IR is lower-is-better, and SR is higher-is-better. The last column lists levels meeting the success-rate criterion (\Cref{sec:bg}).}",
        r"\label{tab:full-sweep-compact}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Task & No-aug. success (\%) & Best success (\%) & Best $\sigma_{\max}$ & Relative IR & SR & Levels meeting criterion \\",
        r"\midrule",
    ]
    for task in TASKS:
        task_rows = sorted(by_task[task], key=lambda row: float(row["rho"]))
        base = next(row for row in task_rows if abs(float(row["rho"])) < 1e-12)
        best = max(task_rows, key=lambda row: float(row["obs_sigma_008_score_mean"]))
        recovered = [float(row["rho"]) for row in task_rows if float(row["recovery_label_rate"]) >= 0.5]
        recovery_text = "--" if not recovered else f"{min(recovered):.2f}--{max(recovered):.2f}"
        lines.append(
            f"{task} & {float(base['obs_sigma_008_score_mean']):.1f} & "
            f"{float(best['obs_sigma_008_score_mean']):.1f} & {float(best['rho']):.2f} & "
            f"{float(base['ir_relative_q90_mean']):.2f}$\\to${float(best['ir_relative_q90_mean']):.2f} & "
            f"{float(base['sr_delta010_mean']):.2f}$\\to${float(best['sr_delta010_mean']):.2f} & "
            f"{recovery_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_pldm_table(
    frozen_rows: list[dict[str, str]],
    lewm_cross_task_summary: dict[str, Any],
) -> str:
    """Compare PLDM-local calibration with score-aligned LeWM transfer."""
    frozen_rows = to_ir_sr(frozen_rows)
    lewm_cross_task_summary = to_ir_sr(lewm_cross_task_summary)
    if len(frozen_rows) != 36 or {
        row.get("model_family") for row in frozen_rows
    } != {"PLDM"}:
        raise ValueError("PLDM frozen validation is incomplete")

    seeds = sorted({int(float(row["training_seed"])) for row in frozen_rows})
    observed = {
        (row["task"], int(round(100 * float(row["training_rho"]))))
        for row in frozen_rows
    }
    expected = {(task, rho) for task in TASKS for rho in range(9)}
    if len(seeds) != 1 or observed != expected:
        raise ValueError("PLDM sweep must contain one complete four-task grid")

    base_ir: dict[str, float] = {}
    for task in TASKS:
        base = [
            row
            for row in frozen_rows
            if row["task"] == task and abs(float(row["training_rho"])) < 1e-12
        ]
        if len(base) != 1 or float(base[0]["ir_raw_q90"]) <= 0:
            raise ValueError(f"invalid PLDM IR reference for {task}")
        base_ir[task] = float(base[0]["ir_raw_q90"])

    local_rows = [
        {
            "task": row["task"],
            "training_seed": seeds[0],
            "rho": float(row["training_rho"]),
            "ir_relative_q90": (
                float(row["ir_raw_q90"]) / base_ir[row["task"]]
            ),
            "sr_delta010": float(row["sr"]),
            "recovery_label": row["behavior_label"],
        }
        for row in frozen_rows
    ]
    details, _, local_summary = run_all_subsets(
        local_rows, expected_seeds=seeds
    )
    local_eval = [row for row in details if row["source_coverage"] == 3]
    if len(local_eval) != len(TASKS):
        raise ValueError("PLDM leave-one-task-out analysis is incomplete")

    local_confusion = {
        key: sum(int(row[key]) for row in local_eval)
        for key in ("tp", "tn", "fp", "fn")
    }

    def pooled(confusion: dict[str, int]) -> tuple[float, float, float]:
        tp, tn = confusion["tp"], confusion["tn"]
        fp, fn = confusion["fp"], confusion["fn"]
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        return 0.5 * (recall + specificity), precision, recall

    local_ba, local_precision, local_recall = pooled(local_confusion)

    lewm_thresholds: dict[str, tuple[float, float]] = {}
    for partition in lewm_cross_task_summary.get("partitions", []):
        if (
            int(partition.get("source_coverage", -1)) == 3
            and len(partition.get("evaluation_tasks", [])) == 1
        ):
            task = str(partition["evaluation_tasks"][0])
            lewm_thresholds[task] = (
                float(partition["ir_threshold"]),
                float(partition["sr_threshold"]),
            )
    if set(lewm_thresholds) != set(TASKS):
        raise ValueError("LeWM three-source thresholds are incomplete")

    relative_lewm_confusion = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    for row in local_rows:
        ir_threshold, sr_threshold = lewm_thresholds[str(row["task"])]
        truth = str(row["recovery_label"]).lower() == "true"
        pred = (
            float(row["ir_relative_q90"]) <= ir_threshold
            and float(row["sr_delta010"]) >= sr_threshold
        )
        key = "tp" if truth and pred else "fn" if truth else "fp" if pred else "tn"
        relative_lewm_confusion[key] += 1
    relative_ba, relative_precision, relative_recall = pooled(
        relative_lewm_confusion
    )

    if not any(
        item["source_coverage"] == 3 for item in local_summary["coverage"]
    ):
        raise ValueError("PLDM three-source summary is missing")

    lewm_task_ba: dict[str, float] = {}
    for partition in lewm_cross_task_summary.get("partitions", []):
        if (
            int(partition.get("source_coverage", -1)) == 3
            and len(partition.get("evaluation_tasks", [])) == 1
        ):
            lewm_task_ba[str(partition["evaluation_tasks"][0])] = float(
                partition["balanced_accuracy"]
            )
    if set(lewm_task_ba) != set(TASKS):
        raise ValueError("LeWM three-source balanced accuracies are incomplete")

    pldm_task_ba = {
        str(row["task"]): float(row["balanced_accuracy"]) for row in local_eval
    }
    if set(pldm_task_ba) != set(TASKS):
        raise ValueError("PLDM three-source balanced accuracies are incomplete")

    if relative_lewm_confusion != local_confusion:
        raise ValueError(
            "score-aligned LeWM thresholds no longer reproduce the PLDM decisions"
        )
    if (local_ba, local_precision, local_recall) != (
        relative_ba,
        relative_precision,
        relative_recall,
    ):
        raise ValueError("PLDM pooled metrics diverge between calibrations")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The same leave-task-out IR--SR screen in two model families: for each evaluation task, thresholds are selected on that family's other three tasks and applied unchanged. Chance is $0.5$; LeWM values average three training runs, whereas PLDM has one run per setting. Applying the LeWM threshold pair to PLDM after within-task IR normalization yields identical PLDM decisions.}",
        r"\label{tab:pldm-architecture-portability}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Balanced accuracy} \\",
        r"\cmidrule(lr){2-3}",
        r"Evaluation task & LeWM & PLDM \\",
        r"\midrule",
    ]
    for task in TASKS:
        lines.append(
            f"{task} & {lewm_task_ba[task]:.3f} & {pldm_task_ba[task]:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-summary", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--full-sweep-summary", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--planner-figure", type=Path, default=DEFAULT_PLANNER_FIG)
    parser.add_argument("--increment-table", type=Path, default=DEFAULT_INCREMENT_TABLE)
    parser.add_argument("--absolute-table", type=Path, default=DEFAULT_ABSOLUTE_TABLE)
    parser.add_argument("--sweep-table", type=Path, default=DEFAULT_SWEEP_TABLE)
    parser.add_argument("--pldm-rows", type=Path, default=DEFAULT_PLDM_ROWS)
    parser.add_argument(
        "--cross-task-summary", type=Path, default=DEFAULT_CROSS_TASK_SUMMARY
    )
    parser.add_argument("--pldm-table", type=Path, default=DEFAULT_PLDM_TABLE)
    args = parser.parse_args()

    planner = _load_json(args.planner_summary)
    if (
        planner["validated_shard_count"] != 72
        or planner.get("training_seeds") != [3072, 3073, 3074]
        or not planner["invariants"]["pass"]
    ):
        raise SystemExit("planner summary is incomplete or failed invariants")
    plot_planner(planner, args.planner_figure)
    _write(args.increment_table, build_increment_table(planner))
    _write(args.absolute_table, build_absolute_table(planner))
    sweep_rows = to_ir_sr(_read_csv(args.full_sweep_summary))
    pldm_rows = to_ir_sr(_read_csv(args.pldm_rows))
    cross_task_summary = to_ir_sr(_load_json(args.cross_task_summary))
    _write(args.sweep_table, build_sweep_table(sweep_rows))
    _write(
        args.pldm_table,
        build_pldm_table(
            pldm_rows,
            cross_task_summary,
        ),
    )
    for path in (
        args.planner_figure,
        args.increment_table,
        args.absolute_table,
        args.sweep_table,
        args.pldm_table,
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
