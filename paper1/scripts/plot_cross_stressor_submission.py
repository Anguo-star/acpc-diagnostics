#!/usr/bin/env python3
"""Render the submission-facing LeWM cross-stressor paired-change figure."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .utils_paper1_io import ROOT

DEFAULT_INPUT = ROOT / "paper1" / "results" / "external_validation" / "cross_stressor_all_pairs.csv"
DEFAULT_OUTPUT = ROOT / "assets" / "paper1_figs" / "fig_cross_stressor_submission.pdf"

STRESSOR_STYLE = {
    "blur": {"color": "#4477AA", "filled": True, "label": "Blur"},
    "resize": {"color": "#EE6677", "filled": False, "label": "Resize"},
}
TASK_MARKER = {"TwoRoom": "o", "PushT": "s", "Reacher": "^", "Cube": "D"}

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
}


def read_lewm_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model_family"] == "LeWM"]
    if len(rows) != 24:
        raise ValueError(f"expected 24 LeWM blur/resize pairs, found {len(rows)}")
    return rows


def plot(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.45, 3.35))
        ax.axhspan(-5.0, 5.0, color="#D9D9D9", alpha=0.30, linewidth=0, zorder=0)
        for row in rows:
            stressor = row["stressor_family"]
            task = row["task"]
            style = STRESSOR_STYLE[stressor]
            color = style["color"]
            facecolor = color if style["filled"] else "white"
            ax.scatter(
                float(row["delta_joint_score"]),
                float(row["delta_behavior"]),
                marker=TASK_MARKER[task],
                s=43,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.88,
                zorder=3,
            )

        ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
        for threshold in (-5.0, 5.0):
            ax.axhline(threshold, color="#888888", linewidth=0.7, linestyle=":", zorder=1)
        ax.set_xlabel(r"Paired diagnostic change, $\Delta S$")
        ax.set_ylabel("Stressed-success change (pp)")
        ax.grid(True, color="#B0B0B0", alpha=0.22, linewidth=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        stressor_handles = [
            Line2D(
                [], [], marker="o", linestyle="none", markersize=5.5,
                markerfacecolor=(style["color"] if style["filled"] else "white"),
                markeredgecolor=style["color"], markeredgewidth=1.0,
                label=style["label"],
            )
            for style in STRESSOR_STYLE.values()
        ]
        task_handles = [
            Line2D(
                [], [], marker=marker, linestyle="none", markersize=5.2,
                markerfacecolor="#777777", markeredgecolor="#777777", label=task,
            )
            for task, marker in TASK_MARKER.items()
        ]
        stressor_legend = ax.legend(
            handles=stressor_handles,
            title="Stressor",
            loc="upper left",
            ncol=2,
            fontsize=6.7,
            title_fontsize=6.9,
            frameon=False,
            handletextpad=0.35,
            columnspacing=0.8,
        )
        ax.add_artist(stressor_legend)
        ax.legend(
            handles=task_handles,
            title="Task",
            loc="lower right",
            ncol=2,
            fontsize=6.7,
            title_fontsize=6.9,
            frameon=False,
            handletextpad=0.35,
            columnspacing=0.8,
        )
        fig.tight_layout(pad=0.5)
        fig.savefig(output, dpi=240, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(read_lewm_rows(args.input), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
