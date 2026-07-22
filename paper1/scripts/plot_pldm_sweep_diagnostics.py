"""Render the PLDM analogue of the LeWM full-sweep diagnostics figure.

The layout, colors, and dotted common-threshold lines mirror
``plot_full_sweep_diagnostics.py`` so that readers can compare the two model
families panel by panel. PLDM has one training run per setting; which levels
meet the success-rate criterion is reported in the text and tables rather
than drawn on the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .ir_dr_compat import to_ir_dr
from .utils_paper1_io import ROOT, TASKS

DEFAULT_ROWS = ROOT / "paper1/results/external_validation/pldm_frozen_rows_v2.csv"
DEFAULT_FIG = ROOT / "assets/paper1_figs/fig_pldm_sweep_diagnostics.pdf"

# Shared with the LeWM sweep figure.
COMMON_TI = 0.30
COMMON_TD = 0.95
RECOVERY_COLOR = "#d9ead3"

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def _read_rows(path: Path) -> dict[str, list[dict[str, float | bool]]]:
    by_task: dict[str, list[dict[str, float | bool]]] = {task: [] for task in TASKS}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = to_ir_dr(list(csv.DictReader(stream)))
    for row in rows:
        task = row["task"]
        if task not in by_task:
            raise ValueError(f"unexpected task {task!r}")
        by_task[task].append(
            {
                "rho": float(row["training_rho"]),
                "ir_raw_q90": float(row["ir_raw_q90"]),
                "dr": float(row["dr"]),
                "stress_score": float(row["stress_score"]),
                "stress_seed_scores": json.loads(
                    row["stress_score_by_evaluation_seed"]
                ),
                "base_clean": float(row["base_clean_score"]),
                "recovered": str(row["behavior_label"]).lower() == "true",
            }
        )
    for task, rows in by_task.items():
        if len(rows) != 9:
            raise ValueError(f"{task}: expected nine sweep levels")
        rows.sort(key=lambda item: item["rho"])
    return by_task


def _recovery_spans(x: list[float], recovered: list[bool]) -> list[tuple[float, float]]:
    edges = [x[0] - (x[1] - x[0]) / 2]
    edges.extend((left + right) / 2 for left, right in zip(x, x[1:]))
    edges.append(x[-1] + (x[-1] - x[-2]) / 2)
    spans: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, flag in enumerate(recovered):
        if flag and run_start is None:
            run_start = index
        if run_start is not None and (not flag or index == len(recovered) - 1):
            run_end = index if flag else index - 1
            spans.append((edges[run_start], edges[run_end + 1]))
            run_start = None
    return spans


def plot(by_task: dict[str, list[dict[str, float | bool]]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(6.7, 2.75))
        outer = fig.add_gridspec(
            1, 4, left=0.075, right=0.995, bottom=0.17, top=0.84, wspace=0.34
        )

        for index, task in enumerate(TASKS):
            block = outer[0, index].subgridspec(
                2, 1, height_ratios=(0.92, 1.08), hspace=0.08
            )
            score_ax = fig.add_subplot(block[0])
            diagnostic_ax = fig.add_subplot(block[1], sharex=score_ax)
            rows = by_task[task]
            x = [row["rho"] for row in rows]
            score = [row["stress_score"] for row in rows]
            base_ir_raw = rows[0]["ir_raw_q90"]
            ir_relative = [row["ir_raw_q90"] / base_ir_raw for row in rows]
            dr = [row["dr"] for row in rows]

            clean_base = rows[0]["base_clean"]
            score_lo = [min(row["stress_seed_scores"]) for row in rows]
            score_hi = [max(row["stress_seed_scores"]) for row in rows]
            score_ax.axhline(clean_base, color="#888888", ls="--", lw=0.9, zorder=1.5)
            score_ax.errorbar(
                x,
                score,
                yerr=[
                    [m - l for m, l in zip(score, score_lo)],
                    [h - m for m, h in zip(score, score_hi)],
                ],
                color="#222222",
                marker="o",
                lw=1.6,
                ms=3.6,
                elinewidth=0.9,
                capsize=1.6,
                zorder=2,
            )
            diagnostic_ax.plot(
                x,
                ir_relative,
                color="#d95f02",
                marker="s",
                lw=1.35,
                ms=3.4,
                zorder=2,
            )
            diagnostic_ax.plot(
                x,
                dr,
                color="#7570b3",
                marker="^",
                lw=1.35,
                ms=3.5,
                ls="--",
                zorder=2,
            )
            diagnostic_ax.axhline(COMMON_TI, color="#d95f02", ls=":", lw=1.0, zorder=1.6)
            diagnostic_ax.axhline(COMMON_TD, color="#7570b3", ls=":", lw=1.0, zorder=1.6)
            if index == 0:
                diagnostic_ax.annotate(
                    r"$t_I{=}0.3$",
                    xy=(0.081, COMMON_TI),
                    xytext=(0, 1.6),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=6.0,
                    color="#d95f02",
                )
                diagnostic_ax.annotate(
                    r"$t_D{=}0.95$",
                    xy=(0.081, COMMON_TD),
                    xytext=(0, -1.6),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=6.0,
                    color="#7570b3",
                )

            score_ax.set_title(f"({chr(97 + index)}) {task}", loc="left", fontweight="semibold")
            if index % 4 == 0:
                score_ax.set_ylabel("Planning\nsuccess rate (%)")
                diagnostic_ax.set_ylabel("Relative IR\n/ DR")
            score_span = [*score_lo, *score_hi, clean_base]
            score_ax.set_ylim(min(score_span) - 5.0, max(score_span) + 5.0)
            score_ax.tick_params(axis="x", labelbottom=False, length=0)
            max_ir_relative = max(ir_relative)
            diagnostic_ax.set_ylim(-0.05, max(1.03, max_ir_relative + 0.1))
            diagnostic_ax.set_yticks(
                [0, 0.5, 1.0] + ([1.5] if max_ir_relative > 1.25 else [])
            )
            diagnostic_ax.set_xlim(-0.003, 0.083)
            diagnostic_ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08])
            diagnostic_ax.set_xlabel(r"$\sigma_{\max}^{\mathrm{train}}$", labelpad=1.5)
            for axis in (score_ax, diagnostic_ax):
                axis.grid(True, axis="y", color="#b0b0b0", alpha=0.22, lw=0.6)
                axis.tick_params(axis="both", which="major", direction="out", length=3.0)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)

        legend_handles = [
            Line2D(
                [],
                [],
                color="#222222",
                marker="o",
                lw=1.6,
                ms=3.6,
                label=r"Success rate ($\sigma_{\rm eval}=0.08$)",
            ),
            Line2D([], [], color="#d95f02", marker="s", lw=1.35, ms=3.4, label=r"Relative IR ($\downarrow$)"),
            Line2D([], [], color="#7570b3", marker="^", lw=1.35, ms=3.5, ls="--", label=r"DR ($\uparrow$)"),
            Line2D([], [], color="#888888", ls="--", lw=0.9, label="Unaugmented baseline"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=5,
            frameon=False,
            columnspacing=1.1,
            handletextpad=0.35,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.savefig(out_fig, dpi=230)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot(_read_rows(args.rows), args.out_fig)
    print(f"wrote {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
