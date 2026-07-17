#!/usr/bin/env python3
"""Plot Paper1 full-sweep diagnostic dynamics."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .utils_paper1_io import ROOT, TASKS, fnum, read_csv, safe_mean

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_FIG = ROOT / "assets" / "paper1_figs" / "fig_full_sweep_diagnostics.pdf"
DEFAULT_REGION_FIG = ROOT / "assets" / "paper1_figs" / "fig_full_sweep_diagnostic_region.pdf"
DEFAULT_PLANNER_FIG = ROOT / "assets" / "paper1_figs" / "fig_full_sweep_planner_guard.pdf"

# Common threshold pair selected by 13 of the 14 cross-task partitions
# (paper Sec. "A common threshold range across tasks"); drawn as dotted
# reference lines on every diagnostic panel.
COMMON_TR = 0.30
COMMON_TM = 0.95

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}
RECOVERY_COLOR = "#d9ead3"


def _by_task(rows: list[dict[str, str]]):
    out = defaultdict(list)
    for row in rows:
        out[row["task"]].append(row)
    return out


def _task_rhos(task_rows: list[dict[str, str]]) -> list[float]:
    return sorted({fnum(r["rho"]) for r in task_rows})


def _mean_by_rho(task_rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for rho in _task_rhos(task_rows):
        vals = [r[key] for r in task_rows if abs(fnum(r["rho"]) - rho) < 1e-12]
        out.append(safe_mean(vals))
    return out


def _range_by_rho(
    task_rows: list[dict[str, str]], key: str
) -> tuple[list[float], list[float]]:
    lower, upper = [], []
    for rho in _task_rhos(task_rows):
        values = [
            fnum(row[key])
            for row in task_rows
            if abs(fnum(row["rho"]) - rho) < 1e-12
        ]
        values = [value for value in values if math.isfinite(value)]
        lower.append(min(values) if values else math.nan)
        upper.append(max(values) if values else math.nan)
    return lower, upper


def _rate_by_rho(task_rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for rho in _task_rhos(task_rows):
        vals = []
        for r in task_rows:
            if abs(fnum(r["rho"]) - rho) < 1e-12:
                vals.append(1.0 if str(r.get(key, "")).lower() == "true" else 0.0)
        out.append(safe_mean(vals))
    return out


def _proxy_positive_by_rho(task_rows: list[dict[str, str]]) -> list[float]:
    out = []
    for rho in _task_rhos(task_rows):
        vals = [1.0 if fnum(r["proxy_gap_q50q90"]) > 0 else 0.0 for r in task_rows if abs(fnum(r["rho"]) - rho) < 1e-12]
        out.append(safe_mean(vals))
    return out


def _recovery_spans(
    x: list[float], recovery: list[float], threshold: float = 0.5
) -> list[tuple[float, float]]:
    """Return cell-aligned spans for contiguous majority-recovered grid runs."""
    if len(x) != len(recovery):
        raise ValueError("x and recovery must have the same length")
    if not x:
        return []

    edges = [x[0] - (x[1] - x[0]) / 2] if len(x) > 1 else [x[0] - 0.5]
    edges.extend((left + right) / 2 for left, right in zip(x, x[1:]))
    edges.append(x[-1] + (x[-1] - x[-2]) / 2 if len(x) > 1 else x[-1] + 0.5)

    spans: list[tuple[float, float]] = []
    run_start: int | None = None
    for index, value in enumerate(recovery):
        recovered = math.isfinite(value) and value >= threshold
        if recovered and run_start is None:
            run_start = index
        if run_start is not None and (not recovered or index == len(recovery) - 1):
            run_end = index if recovered else index - 1
            spans.append((edges[run_start], edges[run_end + 1]))
            run_start = None
    return spans


def _shade_recovery(ax: plt.Axes, x: list[float], recovery: list[float]) -> None:
    for start, end in _recovery_spans(x, recovery):
        ax.axvspan(start, end, color=RECOVERY_COLOR, alpha=0.50, lw=0, zorder=0)


def _polish_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#b0b0b0", alpha=0.22, lw=0.6)
    ax.tick_params(axis="both", which="major", direction="out", length=3.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_dynamics(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    compact_style = {
        **PLOT_STYLE,
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
    }
    with plt.rc_context(compact_style):
        # Arrange the four task blocks left to right; each block retains its
        # success-rate axis above the joint ATR--SMPR axis.
        fig = plt.figure(figsize=(6.7, 2.75))
        outer = fig.add_gridspec(
            1, 4, left=0.075, right=0.985, bottom=0.17, top=0.84, wspace=0.34
        )
        by_task = _by_task(rows)

        for index, task in enumerate(TASKS):
            block = outer[0, index].subgridspec(
                2, 1, height_ratios=(0.92, 1.08), hspace=0.08
            )
            score_ax = fig.add_subplot(block[0])
            diagnostic_ax = fig.add_subplot(block[1], sharex=score_ax)
            trs = by_task[task]
            x = _task_rhos(trs)
            score = _mean_by_rho(trs, "obs_sigma_008_score")
            atr = _mean_by_rho(trs, "atr_normalized_q90")
            smpr = _mean_by_rho(trs, "smpr_delta010")
            score_lo, score_hi = _range_by_rho(trs, "obs_sigma_008_score")
            atr_lo, atr_hi = _range_by_rho(trs, "atr_normalized_q90")
            smpr_lo, smpr_hi = _range_by_rho(trs, "smpr_delta010")

            clean_base = safe_mean(
                [fnum(tr.get("base_clean_score")) for tr in trs]
            )
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
            diagnostic_ax.plot(x, atr, color="#d95f02", marker="s", lw=1.35, ms=3.4, zorder=2)
            diagnostic_ax.plot(
                x, smpr, color="#7570b3", marker="^", lw=1.35, ms=3.5, ls="--", zorder=2
            )
            diagnostic_ax.axhline(COMMON_TR, color="#d95f02", ls=":", lw=1.0, zorder=1.6)
            diagnostic_ax.axhline(COMMON_TM, color="#7570b3", ls=":", lw=1.0, zorder=1.6)
            if index == 0:
                diagnostic_ax.annotate(
                    r"$t_R{=}0.3$",
                    xy=(0.081, COMMON_TR),
                    xytext=(0, 1.6),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=6.0,
                    color="#d95f02",
                )
                diagnostic_ax.annotate(
                    r"$t_M{=}0.95$",
                    xy=(0.081, COMMON_TM),
                    xytext=(0, -1.6),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=6.0,
                    color="#7570b3",
                )

            score_ax.set_title(f"({chr(97 + index)}) {task}", loc="left", fontweight="semibold")
            if index == 0:
                score_ax.set_ylabel("Planning\nsuccess rate (%)")
                diagnostic_ax.set_ylabel("Relative ATR\n/ SMPR")
            else:
                # Score panels keep their tick labels: y-ranges adapt per task,
                # so hiding them would leave panels (b)-(d) unreadable. The
                # diagnostic row shares one fixed range, so labels stay on the
                # leftmost panel only.
                diagnostic_ax.tick_params(axis="y", labelleft=False)
            score_span = [*score_lo, *score_hi, clean_base]
            score_ax.set_ylim(min(score_span) - 5.0, max(score_span) + 5.0)
            score_ax.tick_params(axis="x", labelbottom=False, length=0)
            diagnostic_ax.set_ylim(-0.03, 1.03)
            diagnostic_ax.set_yticks([0, 0.5, 1.0])
            diagnostic_ax.set_yticks([0.25, 0.75], minor=True)
            diagnostic_ax.set_xlim(-0.003, 0.083)
            diagnostic_ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08])
            diagnostic_ax.set_xlabel(r"$\sigma_{\max}^{\mathrm{train}}$", labelpad=1.5)
            _polish_axis(score_ax)
            _polish_axis(diagnostic_ax)
            diagnostic_ax.grid(
                True,
                axis="y",
                which="minor",
                color="#b0b0b0",
                alpha=0.12,
                lw=0.45,
            )
            diagnostic_ax.tick_params(
                axis="y", which="minor", direction="out", length=2.0
            )

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
            Line2D([], [], color="#d95f02", marker="s", lw=1.35, ms=3.4, label=r"Relative ATR ($\downarrow$)"),
            Line2D([], [], color="#7570b3", marker="^", lw=1.35, ms=3.5, ls="--", label=r"SMPR ($\uparrow$)"),
            Line2D([], [], color="#888888", ls="--", lw=0.9, label="Unaugmented baseline"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=5,
            frameon=False,
            columnspacing=0.7,
            handletextpad=0.35,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.savefig(out_fig, dpi=230)
        plt.close(fig)


def plot_planner_guard(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    compact_style = {
        **PLOT_STYLE,
        "font.size": 7.25,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.25,
        "savefig.bbox": None,
    }
    with plt.rc_context(compact_style):
        fig, axes = plt.subplots(1, 4, figsize=(6.7, 2.35), sharex=True, sharey=True)
        by_task = _by_task(rows)

        for index, (ax, task) in enumerate(zip(axes, TASKS)):
            trs = by_task[task]
            x = _task_rhos(trs)
            top1_fail = [
                1.0 - value if math.isfinite(value) else math.nan
                for value in _mean_by_rho(trs, "top1_agree")
            ]
            proxy_pos = _proxy_positive_by_rho(trs)
            recovery = _rate_by_rho(trs, "recovery_label")
            _shade_recovery(ax, x, recovery)
            ax.plot(x, top1_fail, color="#b2182b", marker="d", lw=1.35, ms=3.2, zorder=2)
            ax.plot(x, proxy_pos, color="#2166ac", marker="s", lw=1.2, ms=3.1, ls="--", zorder=2)
            ax.set_title(f"({chr(97 + index)}) {task}", loc="left", fontweight="semibold")
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlim(-0.003, 0.083)
            ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08])
            ax.tick_params(length=2.5, pad=2.0)
            _polish_axis(ax)

        legend_handles = [
            Line2D([], [], color="#b2182b", marker="d", lw=1.35, ms=3.2, label="Top-1 flip"),
            Line2D([], [], color="#2166ac", marker="s", lw=1.2, ms=3.1, ls="--", label="Proxy gap > 0"),
            Patch(facecolor=RECOVERY_COLOR, edgecolor="none", alpha=0.50, label="Majority recovered"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            handletextpad=0.5,
            columnspacing=1.2,
        )
        fig.supxlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$", y=0.035)
        fig.supylabel("Event rate", x=0.012)
        fig.subplots_adjust(left=0.075, right=0.992, bottom=0.26, top=0.76, wspace=0.12)
        fig.savefig(out_fig, dpi=230)
        plt.close(fig)


def plot_region(rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.3), sharex=True, sharey=True)
    axes = axes.ravel()
    by_task = _by_task(rows)
    for ax, task in zip(axes, TASKS):
        for row in by_task[task]:
            recovered = str(row.get("recovery_label", "")).lower() == "true"
            color = "#1f77b4" if recovered else "#8c8c8c"
            marker = "o" if int(float(row["training_seed"])) == 3072 else "s" if int(float(row["training_seed"])) == 3073 else "^"
            ax.scatter(fnum(row["atr_normalized_q90"]), fnum(row["smpr_delta010"]), s=26, alpha=0.78, c=color, marker=marker)
        ax.set_title(task, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.02, 1.04)
    for ax in axes[2:]:
        ax.set_xlabel("normalized ATR q90")
    for ax in axes[::2]:
        ax.set_ylabel("SMPR")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--region-out", type=Path, default=DEFAULT_REGION_FIG)
    parser.add_argument("--planner-out", type=Path, default=DEFAULT_PLANNER_FIG)
    args = parser.parse_args()
    rows = read_csv(args.diagnostics)
    plot_dynamics(rows, args.out)
    plot_region(rows, args.region_out)
    plot_planner_guard(rows, args.planner_out)
    print(f"wrote {args.out}")
    print(f"wrote {args.region_out}")
    print(f"wrote {args.planner_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
