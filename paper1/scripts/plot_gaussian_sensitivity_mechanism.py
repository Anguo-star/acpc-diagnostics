#!/usr/bin/env python3
"""Plot local Gaussian sensitivity mechanism diagnostics."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .utils_paper1_io import ROOT, TASKS, fnum, read_csv

DEFAULT_FD = ROOT / "paper1" / "results" / "gaussian_sensitivity_summary.csv"
DEFAULT_JVP = ROOT / "paper1" / "results" / "jvp_hutchinson_sensitivity_summary.csv"
DEFAULT_MAIN_FIG = ROOT / "assets" / "paper1_figs" / "fig_gaussian_sensitivity_main.png"
DEFAULT_DECOMP_FIG = ROOT / "assets" / "paper1_figs" / "fig_jvp_trace_decomposition_heatmap.png"


def _row(rows: list[dict[str, str]], task: str, checkpoint_type: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["checkpoint_type"] == checkpoint_type:
            return row
    raise KeyError((task, checkpoint_type))


def _log10_ratio(value: float) -> float:
    return math.log10(max(value, 1e-6))


def _contrast_text_color(image: object, value: float) -> str:
    """Return a readable annotation color for a scalar-mapped cell."""
    red, green, blue, _alpha = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.52 else "#1a1a1a"


def _kappa_relative_isotropic_vs_base(row: dict[str, str]) -> float:
    """Read the canonical kappa ratio, falling back to the legacy alias."""
    value = row.get("kappa_relative_isotropic_vs_base")
    if value is None or value == "":
        value = row.get("alignment_coefficient_vs_base")
    return fnum(value)


def plot_main(fd_rows: list[dict[str, str]], jvp_rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.75), sharey=True)
    y = list(range(len(TASKS)))
    panels = [
        (
            axes[0],
            [fnum(_row(fd_rows, task, "endpoint")["sensitivity_slope_vs_base"]) for task in TASKS],
            "Finite-difference slope",
        ),
        (
            axes[1],
            [fnum(_row(jvp_rows, task, "endpoint")["composed_trace_per_pixel_dim_vs_base"]) for task in TASKS],
            "JVP/Hutchinson raw composed trace",
        ),
    ]
    for ax, vals, title in panels:
        ax.hlines(y, vals, 1.0, color="#aab7c4", lw=1.4, zorder=1)
        ax.scatter([1.0] * len(y), y, s=24, facecolor="white", edgecolor="#7b8794", lw=0.9, zorder=2)
        ax.scatter(vals, y, s=42, color="#0072b2", edgecolor="white", lw=0.7, zorder=3)
        ax.axvline(1.0, color="#68737d", lw=1.0, ls=(0, (3, 2)), zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.35)
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
        ax.set_title(title, fontsize=9.2, pad=7)
        ax.set_xlabel("Endpoint / base ratio  (\u2193 less sensitivity)", fontsize=8.3, labelpad=5)
        ax.grid(True, axis="x", which="major", color="#d9dfe5", lw=0.7)
        ax.tick_params(axis="both", which="major", labelsize=8.1)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#a6adb4")
        ax.text(
            1.0, 0.76, "base = 1", transform=ax.get_xaxis_transform(), ha="right", va="center",
            fontsize=8.0, color="#59636d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
        )
        for yi, val in zip(y, vals):
            ax.annotate(
                f"{val:.3g}\u00d7", (val, yi), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.0, color="#174b6b",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
            )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(TASKS, fontsize=8.3)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    fig.tight_layout(pad=0.55, w_pad=0.75)
    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def plot_decomposition(jvp_rows: list[dict[str, str]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_heat, ax_kappa) = plt.subplots(
        1, 2, figsize=(6.7, 3.05), gridspec_kw={"width_ratios": [1.6, 1.0]}, constrained_layout=True
    )
    y = list(range(len(TASKS)))
    trace_cols = [
        ("encoder", "encoder_trace_per_pixel_dim_vs_base"),
        ("rollout", "rollout_trace_per_latent_dim_vs_base"),
        ("composed", "composed_trace_per_pixel_dim_vs_base"),
    ]
    matrix = []
    labels = []
    kappa_vals = []
    for task in TASKS:
        row = _row(jvp_rows, task, "endpoint")
        matrix.append([_log10_ratio(fnum(row[key])) for _label, key in trace_cols])
        labels.append([fnum(row[key]) for _label, key in trace_cols])
        kappa_vals.append(_kappa_relative_isotropic_vs_base(row))

    im = ax_heat.imshow(matrix, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=0.5), aspect="auto")
    ax_heat.set_xticks(range(len(trace_cols)))
    ax_heat.set_xticklabels([label.title() for label, _key in trace_cols], fontsize=8.2)
    ax_heat.set_yticks(y)
    ax_heat.set_yticklabels(TASKS, fontsize=8.2)
    ax_heat.set_title("(a) Raw trace decomposition ratios", fontsize=9.2, pad=7)
    ax_heat.tick_params(length=0)
    for i, row in enumerate(labels):
        for j, val in enumerate(row):
            mapped_value = matrix[i][j]
            ax_heat.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=8.1, color=_contrast_text_color(im, mapped_value))
    cbar = fig.colorbar(im, ax=ax_heat, orientation="horizontal", fraction=0.14, pad=0.12, aspect=24)
    cbar.set_ticks([-3.0, -2.0, -1.0, 0.0])
    cbar.set_ticklabels(["0.001", "0.01", "0.1", "1"])
    cbar.ax.tick_params(labelsize=8.0, length=2.5, pad=2)
    cbar.set_label("Endpoint / base raw-trace ratio (log color scale)", fontsize=8.0, labelpad=3)
    cbar.outline.set_linewidth(0.6)

    ax_kappa.hlines(y, 1.0, kappa_vals, color="#b4a8c7", lw=1.4, zorder=1)
    ax_kappa.scatter([1.0] * len(y), y, s=24, facecolor="white", edgecolor="#7b8794", lw=0.9, zorder=2)
    ax_kappa.scatter(kappa_vals, y, s=42, color="#6f5aa8", edgecolor="white", lw=0.7, zorder=3)
    ax_kappa.axvline(1.0, color="#68737d", lw=1.0, ls=(0, (3, 2)), zorder=0)
    ax_kappa.set_yticks(y)
    ax_kappa.set_yticklabels([])
    ax_kappa.set_ylim(len(TASKS) - 0.5, -0.5)
    ax_kappa.set_title("(b) Relative isotropic alignment gain", fontsize=9.2, pad=7)
    ax_kappa.set_xlabel(r"Endpoint / base $\kappa_{\rm rel}$ ratio (can exceed 1)", fontsize=8.1, labelpad=5)
    ax_kappa.grid(True, axis="x", color="#d9dfe5", lw=0.7)
    ax_kappa.tick_params(axis="x", labelsize=8.0)
    ax_kappa.tick_params(axis="y", length=0)
    ax_kappa.spines[["top", "right", "left"]].set_visible(False)
    ax_kappa.spines["bottom"].set_color("#a6adb4")
    ax_kappa.text(
        1.0, 0.76, "base = 1", transform=ax_kappa.get_xaxis_transform(), ha="right", va="center",
        fontsize=8.0, color="#59636d",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
    )
    for yi, val in zip(y, kappa_vals):
        x_offset = 5 if val >= 1.0 else -5
        ax_kappa.annotate(
            f"{val:.2f}\u00d7", (val, yi), xytext=(x_offset, 0), textcoords="offset points",
            ha="left" if val >= 1.0 else "right", va="center", fontsize=8.0, color="#493b70",
        )
    ax_kappa.set_xlim(0.4, max(2.45, max(kappa_vals) * 1.1))

    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finite-diff", type=Path, default=DEFAULT_FD)
    parser.add_argument("--jvp", type=Path, default=DEFAULT_JVP)
    parser.add_argument("--out-main", type=Path, default=DEFAULT_MAIN_FIG)
    parser.add_argument("--out-decomp", type=Path, default=DEFAULT_DECOMP_FIG)
    args = parser.parse_args()
    fd_rows = read_csv(args.finite_diff)
    jvp_rows = read_csv(args.jvp)
    plot_main(fd_rows, jvp_rows, args.out_main)
    plot_decomposition(jvp_rows, args.out_decomp)
    print(f"wrote {args.out_main}")
    print(f"wrote {args.out_decomp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
