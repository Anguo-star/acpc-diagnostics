#!/usr/bin/env python3
"""Plot endpoint ATR/SMPR selective-ACPC diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_paper1_io import ROOT, TASKS, fnum, read_csv, safe_mean, safe_pstdev

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_endpoint_atr_smpr.png"


def _endpoint_stats(rows: list[dict[str, str]], task: str, rho: str, key: str) -> tuple[float, float]:
    vals = [fnum(r[key]) for r in rows if r["task"] == task and f"{fnum(r['rho']):.2f}" == rho]
    return safe_mean(vals), safe_pstdev(vals)


def _direction_cue(ax: plt.Axes, direction: str) -> None:
    """Place the better-direction cue outside the data region."""
    text = "← better" if direction == "left" else "better →"
    ax.text(
        0.99,
        1.025,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#4d4d4d",
        fontsize=8.0,
        clip_on=False,
    )


def plot(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    y = list(range(len(TASKS)))
    specs = [
        ("atr_q90", "(a) Radius tail (ATR q90)", "log", "left"),
        ("smpr_delta0", "(b) Guard pass rate (SMPR)", "linear", "right"),
    ]
    colors = {"0.00": "#777777", "0.08": "#0072B2"}
    labels = {"0.00": "base", "0.08": "noise-trained"}
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
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9), sharey=True)
        for ax, (key, title, scale, direction) in zip(axes, specs):
            for i, task in enumerate(TASKS):
                base_mean, base_std = _endpoint_stats(rows, task, "0.00", key)
                end_mean, end_std = _endpoint_stats(rows, task, "0.08", key)
                ax.annotate(
                    "",
                    xy=(end_mean, i),
                    xytext=(base_mean, i),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#b9b9b9", lw=1.25,
                        mutation_scale=8.0, shrinkA=5.5, shrinkB=5.5,
                    ),
                    zorder=1,
                )
                for rho, mean, std, marker, zorder in (
                    ("0.00", base_mean, base_std, "o", 2),
                    ("0.08", end_mean, end_std, "s", 3),
                ):
                    ax.errorbar(
                        mean, i, xerr=std, fmt=marker,
                        color=colors[rho], ecolor=colors[rho],
                        markeredgecolor="#ffffff", markeredgewidth=0.65,
                        elinewidth=1.05, capsize=2.1, capthick=1.0, ms=6.0,
                        zorder=zorder, label=labels[rho] if i == 0 else None,
                    )

            ax.set_title(title, loc="left", pad=8, fontweight="semibold")
            ax.grid(True, axis="x", color="#d9d9d9", lw=0.65, alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", length=3.0, color="#666666")
            _direction_cue(ax, direction)
            if scale == "log":
                ax.set_xscale("log")
                ax.set_xlim(0.05, 5.0)
                ax.set_xticks([0.05, 0.1, 0.3, 1.0, 3.0])
                ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3"])
                ax.set_xlabel("ATR q90 (log scale)")
            else:
                ax.set_xlim(0.0, 1.035)
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xlabel("SMPR pass rate")

        axes[0].set_yticks(y)
        axes[0].set_yticklabels(TASKS)
        axes[0].set_ylim(len(TASKS) - 0.55, -0.55)
        axes[1].tick_params(axis="y", labelleft=False)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", ncol=2, frameon=False,
            handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, 0.985),
        )
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.20, top=0.77, wspace=0.22)
        fig.savefig(out_fig, dpi=300, facecolor="white")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    plot(read_csv(args.diagnostics), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
