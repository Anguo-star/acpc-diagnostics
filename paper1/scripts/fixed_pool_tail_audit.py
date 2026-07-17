#!/usr/bin/env python3
"""Build summary-level fixed-pool tail/top-1 agreement analysis for Paper1.

The available Paper1 summary stores q50/q90 margin/drift summaries and the
fixed-pool flip rate, but not sample-level candidate-cost traces. Therefore this
script reports observed fixed-pool top-1 agreement and proxy gaps over the full
sweep, while marking pool-level sufficient-event rates as unavailable.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_paper1_io import ROOT, RHO_GRID, TASKS, fnum, read_csv, safe_mean, safe_pstdev, write_csv

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_OUT = ROOT / "paper1" / "results" / "fixed_pool_tail_audit.csv"
DEFAULT_SUMMARY = ROOT / "paper1" / "results" / "fixed_pool_tail_audit_summary.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_fixed_pool_tail_audit.tex"
DEFAULT_FIG = ROOT / "paper1" / "figures" / "fig_fixed_pool_tail_audit.png"
DEFAULT_TOP1_FIG = ROOT / "paper1" / "figures" / "fig_top1_agreement_full_sweep.png"
DEFAULT_MISSING = ROOT / "paper1" / "results" / "MISSING_DATA_fixed_pool_tail_audit.md"

AUDIT_FIELDS = [
    "task", "training_seed", "rho", "anchor_id", "pool_id", "K", "clean_margin",
    "max_cost_drift", "pool_cert_gap", "cert_pass_pool", "top1_agree",
    "observed_top1_flip", "cost_drift_q50", "cost_drift_q90", "cost_drift_q95",
    "cost_drift_q99", "proxy_gap_q50q90", "data_level", "notes",
]
SUMMARY_FIELDS = [
    "task", "rho", "n_training_seeds", "cert_pass_rate_mean", "cert_pass_rate_std",
    "top1_agree_mean", "top1_agree_std", "max_drift_q95", "max_drift_q99",
    "margin_q10", "margin_q50", "margin_q90", "proxy_gap_q50q90_mean",
    "proxy_gap_positive_rate", "notes",
]


def build_audit_rows(rows):
    out = []
    for row in rows:
        top1 = fnum(row["top1_agree"])
        out.append({
            "task": row["task"],
            "training_seed": row["training_seed"],
            "rho": row["rho"],
            "anchor_id": "summary_only",
            "pool_id": "summary_level_fixed_pool_summary",
            "K": row["candidate_count"],
            "clean_margin": row["clean_margin_q50"],
            "max_cost_drift": "",
            "pool_cert_gap": "",
            "cert_pass_pool": "",
            "top1_agree": top1,
            "observed_top1_flip": 1.0 - top1 if math.isfinite(top1) else "",
            "cost_drift_q50": row["cost_drift_q50"],
            "cost_drift_q90": row["cost_drift_q90"],
            "cost_drift_q95": "",
            "cost_drift_q99": "",
            "proxy_gap_q50q90": row["proxy_gap_q50q90"],
            "data_level": "aggregate_summary_level",
            "notes": "sample-level max-cost-drift traces are unavailable; cert_pass_pool unavailable",
        })
    return out


def build_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["rho"])].append(row)
    out = []
    for task in TASKS:
        for rho in RHO_GRID:
            block = grouped[(task, rho)]
            out.append({
                "task": task,
                "rho": rho,
                "n_training_seeds": len(block),
                "cert_pass_rate_mean": "",
                "cert_pass_rate_std": "",
                "top1_agree_mean": safe_mean(r["top1_agree"] for r in block),
                "top1_agree_std": safe_pstdev(r["top1_agree"] for r in block),
                "max_drift_q95": "",
                "max_drift_q99": "",
                "margin_q10": "",
                "margin_q50": safe_mean(r["clean_margin"] for r in block),
                "margin_q90": "",
                "proxy_gap_q50q90_mean": safe_mean(r["proxy_gap_q50q90"] for r in block),
                "proxy_gap_positive_rate": safe_mean(1.0 if fnum(r["proxy_gap_q50q90"]) > 0 else 0.0 for r in block),
                "notes": "Top-1 agreement is observed from measured flip-rate summaries; cert-pass requires sample-level fixed-pool traces.",
            })
    return out


def write_table(summary, out: Path) -> None:
    selected = []
    for task in TASKS:
        task_rows = [r for r in summary if r["task"] == task]
        base = next(r for r in task_rows if r["rho"] == "0.00")
        endpoint = next(r for r in task_rows if r["rho"] == "0.08")
        onset_candidates = [r for r in task_rows if fnum(r["proxy_gap_positive_rate"]) >= 0.5]
        onset = onset_candidates[0] if onset_candidates else endpoint
        selected.extend([(task, "base", base), (task, "proxy onset", onset), (task, "endpoint", endpoint)])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Summary-level fixed-pool top-1 agreement rises as the q50/q90 proxy gap turns positive. Agreement is one minus the measured fixed-pool flip rate; cert-pass rates require sample-level maximum cost-drift traces.}",
        r"\label{tab:fixed-pool-tail-audit}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Task & row & $\sigma_{\max}^{\mathrm{train}}$ & top-1 agreement & proxy gap & proxy pass rate \\",
        r"\midrule",
    ]
    for task, role, row in selected:
        lines.append(
            f"{task} & {role} & ${fnum(row['rho']):.2f}$ & ${fnum(row['top1_agree_mean']):.3f}$ & "
            f"${fnum(row['proxy_gap_q50q90_mean']):.2f}$ & ${fnum(row['proxy_gap_positive_rate']):.2f}$"
            + " \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def plot(summary, out_fig: Path, top1_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        rows = [r for r in summary if r["task"] == task]
        x = [fnum(r["rho"]) for r in rows]
        gap = [fnum(r["proxy_gap_q50q90_mean"]) for r in rows]
        top1 = [fnum(r["top1_agree_mean"]) for r in rows]
        ax.axhline(0, color="#666666", lw=0.8)
        ax.plot(x, gap, color="#1f4e79", marker="o", lw=1.7, label="proxy gap")
        ax.set_title(task, fontsize=10)
        ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
        ax.set_ylabel("q50/q90 proxy gap")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(x, top1, color="#38761d", marker="s", lw=1.4, ls="--", label="top-1 agreement")
        ax2.set_ylim(0, 1.03)
        ax2.set_ylabel("top-1 agreement")
        if task == TASKS[0]:
            ax.legend(loc="lower right", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for task in TASKS:
        rows = [r for r in summary if r["task"] == task]
        ax.plot([fnum(r["rho"]) for r in rows], [fnum(r["top1_agree_mean"]) for r in rows], marker="o", lw=1.6, label=task)
    ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
    ax.set_ylabel("fixed-pool top-1 agreement")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(top1_fig, dpi=220)
    plt.close(fig)


def write_missing(path: Path) -> None:
    path.write_text(
        "# Missing raw fixed-pool tail data\n\n"
        "The available Paper1 summaries contain aggregate q50/q90 candidate-margin and paired-drift summaries plus fixed-pool flip rates. "
        "They do not contain sample-level candidate-cost traces, per-anchor maximum cost drift, q10 clean-margin tails, q95/q99 max-drift tails, or pool-level sufficient-event pass flags. "
        "Accordingly, the fixed-pool analysis reports observed top-1 agreement and q50/q90 proxy gaps, and leaves cert_pass_pool/cert_pass_rate fields empty rather than inferring them from summaries.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--fig-out", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--top1-fig-out", type=Path, default=DEFAULT_TOP1_FIG)
    parser.add_argument("--missing-out", type=Path, default=DEFAULT_MISSING)
    args = parser.parse_args()
    rows = read_csv(args.diagnostics)
    audit = build_audit_rows(rows)
    write_csv(args.out, audit, AUDIT_FIELDS)
    summary = build_summary(audit)
    write_csv(args.summary_out, summary, SUMMARY_FIELDS)
    write_table(summary, args.table_out)
    plot(summary, args.fig_out, args.top1_fig_out)
    write_missing(args.missing_out)
    print(f"wrote {args.out} ({len(audit)} rows)")
    print(f"wrote {args.summary_out} ({len(summary)} rows)")
    print(f"wrote {args.table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
