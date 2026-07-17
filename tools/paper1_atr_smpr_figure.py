#!/usr/bin/env python3
"""Render the Paper1 compressed ATR/SMPR diagnostic-plane figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "paper1_data" / "compressed_metrics_summary_20260706.json"
OUT = ROOT / "assets" / "paper1_figs" / "fig_atr_smpr_plane.png"

TASK_ORDER = ["TwoRoom", "PushT", "Reacher", "Cube"]
COLORS = {
    "TwoRoom": "#2c7fb8",
    "PushT": "#d95f0e",
    "Reacher": "#31a354",
    "Cube": "#756bb1",
}


def _cell(row: dict, key: str) -> tuple[float, float]:
    block = row[key]
    return float(block["mean"]), float(block["pstdev"])


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    rows = {row["task"]: row for row in data["summary_rows"]}

    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=220)
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, color="#e3e3dd", linewidth=0.8, zorder=0)

    for task in TASK_ORDER:
        row = rows[task]
        atr_base, atr_base_sd = _cell(row, "ATR_q90_0.0")
        atr_rob, atr_rob_sd = _cell(row, "ATR_q90_0.08")
        smpr_base, smpr_base_sd = _cell(row, "SMPR_0.0")
        smpr_rob, smpr_rob_sd = _cell(row, "SMPR_0.08")
        color = COLORS[task]

        ax.errorbar(
            atr_base,
            smpr_base,
            xerr=atr_base_sd,
            yerr=smpr_base_sd,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=5.4,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=4,
        )
        ax.errorbar(
            atr_rob,
            smpr_rob,
            xerr=atr_rob_sd,
            yerr=smpr_rob_sd,
            fmt="s",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=5.4,
            markerfacecolor=color,
            markeredgewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            "",
            xy=(atr_rob, smpr_rob),
            xytext=(atr_base, smpr_base),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.7, shrinkA=5, shrinkB=5),
            zorder=3,
        )
        label_x = atr_rob + 0.05 if task != "Reacher" else atr_rob + 0.08
        label_y = smpr_rob - 0.035 if task in {"PushT", "Reacher", "Cube"} else smpr_rob + 0.02
        ax.text(label_x, label_y, task, color=color, fontsize=9.2, weight="semibold")

    # Direction cue only; it is not a decision boundary.
    ax.annotate(
        "lower ATR\nhigher SMPR",
        xy=(0.45, 0.90),
        xytext=(1.35, 0.72),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"),
        fontsize=8.8,
        color="#444444",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bbbbbb", linewidth=0.7),
    )

    ax.set_xlim(0.0, 4.05)
    ax.set_ylim(0.25, 1.04)
    ax.set_xlabel("ACPC Tail Risk (ATR, lower is better)")
    ax.set_ylabel("Selective Margin Pass Rate (SMPR, higher is better)")
    ax.set_title("Compressed selective-ACPC movement across training seeds", pad=9)

    base_handle = ax.scatter([], [], marker="o", facecolor="white", edgecolor="#555555", s=40, label="no-noise baseline")
    robust_handle = ax.scatter([], [], marker="s", facecolor="#555555", edgecolor="#555555", s=40, label="std0.08 checkpoint")
    ax.legend(handles=[base_handle, robust_handle], loc="lower right", frameon=True, framealpha=0.95, fontsize=8.5)

    fig.tight_layout(pad=0.7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
