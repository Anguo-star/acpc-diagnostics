#!/usr/bin/env python3
"""Plot fixed-pool event-rate Wilson intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .utils_paper1_io import ROOT, fnum, read_csv

DEFAULT_CI = ROOT / "paper1" / "results" / "sample_level_event_rate_wilson_ci.csv"
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_fixed_pool_event_rates.png"
TASK_ORDER = ["TwoRoom", "PushT", "Reacher", "Cube", "ALL"]
MAIN_METRICS = [
    ("cert-pass", "(a) Cert-pass rate", "higher is better", "right"),
    ("top-1 flip", "(b) Top-1 flip rate", "lower is better", "left"),
]


def _lookup(rows: list[dict[str, str]], task: str, split: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["split"] == split and row["metric"] == metric:
            return row
    raise KeyError((task, split, metric))


def _direction_cue(ax: plt.Axes, direction: str) -> None:
    """Place the better-direction cue outside the data region."""
    text = "← better" if direction == "left" else "better →"
    ax.text(
        0.99, 1.025, text, transform=ax.transAxes, ha="right", va="bottom",
        color="#4d4d4d", fontsize=8.0, clip_on=False,
    )


def plot(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    colors = {"fragile": "#777777", "recovered": "#0072B2"}
    markers = {"fragile": "o", "recovered": "s"}
    offsets = {"fragile": 0.12, "recovered": -0.12}
    y_base = list(range(len(TASK_ORDER)))
    style = {
        "font.size": 8.25,
        "axes.titlesize": 9.25,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.25,
    }

    with plt.rc_context(style):
        # Native full-text width avoids shrinking labels in LaTeX.
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.05), sharey=True)
        for ax, (metric, title, _direction_text, direction) in zip(axes, MAIN_METRICS):
            all_idx = TASK_ORDER.index("ALL")
            ax.axhspan(all_idx - 0.42, all_idx + 0.42, color="#eaf2f8", alpha=0.8, zorder=0)
            for task_idx, task in enumerate(TASK_ORDER):
                split_rows = {split: _lookup(rows, task, split, metric) for split in ("fragile", "recovered")}
                fragile_rate = fnum(split_rows["fragile"]["rate"])
                recovered_rate = fnum(split_rows["recovered"]["rate"])
                ax.plot(
                    [fragile_rate, recovered_rate],
                    [task_idx + offsets["fragile"], task_idx + offsets["recovered"]],
                    color="#c4c4c4", lw=1.05, zorder=1,
                )
                for split in ("fragile", "recovered"):
                    row = split_rows[split]
                    rate = fnum(row["rate"])
                    low = fnum(row["wilson_low"])
                    high = fnum(row["wilson_high"])
                    y = task_idx + offsets[split]
                    ax.errorbar(
                        rate, y,
                        xerr=[[max(rate - low, 0.0)], [max(high - rate, 0.0)]],
                        fmt=markers[split], color=colors[split], ecolor=colors[split],
                        markeredgecolor="#ffffff", markeredgewidth=0.65,
                        elinewidth=1.05, capsize=2.0, capthick=1.0,
                        ms=6.0 if task == "ALL" else 5.4, zorder=3,
                        label=split if task_idx == 0 else None,
                    )
            ax.set_title(title, loc="left", pad=8, fontweight="semibold")
            ax.set_xlim(-0.025, 1.025)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.grid(True, axis="x", color="#d9d9d9", lw=0.65, alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", length=3.0, color="#666666")
            ax.set_xlabel("Event rate")
            _direction_cue(ax, direction)

        axes[0].set_yticks(y_base)
        axes[0].set_yticklabels(TASK_ORDER)
        axes[0].set_ylim(len(TASK_ORDER) - 0.55, -0.55)
        for label in axes[0].get_yticklabels():
            if label.get_text() == "ALL":
                label.set_fontweight("bold")
        axes[1].tick_params(axis="y", labelleft=False)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=2, frameon=False,
            handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, 0.985),
        )
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.19, top=0.78, wspace=0.22)
        fig.savefig(out_fig, dpi=300, facecolor="white")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", type=Path, default=DEFAULT_CI)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot(read_csv(args.ci), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
