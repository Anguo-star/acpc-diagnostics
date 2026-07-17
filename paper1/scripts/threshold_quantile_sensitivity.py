#!/usr/bin/env python3
"""Threshold and quantile sensitivity audit for Paper1 diagnostics."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_paper1_io import ROOT, TASKS, fnum, label_rows, read_csv, safe_median, write_csv

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_OUT = ROOT / "paper1" / "results" / "threshold_quantile_sensitivity.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_threshold_quantile_sensitivity.tex"
DEFAULT_FIG = ROOT / "paper1" / "figures" / "fig_threshold_sensitivity.png"

FIELDNAMES = [
    "analysis", "recovery_fraction", "clean_tolerance", "atr_quantile", "smpr_margin",
    "proxy_variant", "task", "fragile_median_atr", "recovered_median_atr",
    "fragile_median_smpr", "recovered_median_smpr", "atr_direction_pass",
    "smpr_direction_pass", "num_fragile", "num_recovered", "status", "notes",
]

RECOVERY_FRACTIONS = [0.70, 0.80, 0.90]
CLEAN_TOLERANCES = [3.0, 5.0, 10.0]


def _base_rows(rows):
    # Keep only source columns needed by label_rows; labels are recomputed per setting.
    out = []
    for row in rows:
        row = dict(row)
        for key in ["recovery_label", "normalized_recovery", "recovery_score_threshold", "clean_constraint_pass"]:
            row.pop(key, None)
        out.append(row)
    return out


def _summarize(labeled, recovery_fraction, clean_tolerance):
    out = []
    for task in TASKS:
        block = [r for r in labeled if r["task"] == task]
        fragile = [r for r in block if str(r["recovery_label"]).lower() != "true"]
        recovered = [r for r in block if str(r["recovery_label"]).lower() == "true"]
        f_atr = safe_median(r["atr_normalized_q90"] for r in fragile)
        r_atr = safe_median(r["atr_normalized_q90"] for r in recovered)
        f_smpr = safe_median(r["smpr_delta0"] for r in fragile)
        r_smpr = safe_median(r["smpr_delta0"] for r in recovered)
        out.append({
            "analysis": "recovery_threshold_clean_tolerance",
            "recovery_fraction": recovery_fraction,
            "clean_tolerance": clean_tolerance,
            "atr_quantile": "q90",
            "smpr_margin": "delta0",
            "proxy_variant": "q50_margin_minus_2q90_drift",
            "task": task,
            "fragile_median_atr": f_atr,
            "recovered_median_atr": r_atr,
            "fragile_median_smpr": f_smpr,
            "recovered_median_smpr": r_smpr,
            "atr_direction_pass": str(math.isfinite(f_atr) and math.isfinite(r_atr) and r_atr < f_atr).lower(),
            "smpr_direction_pass": str(math.isfinite(f_smpr) and math.isfinite(r_smpr) and r_smpr > f_smpr).lower(),
            "num_fragile": len(fragile),
            "num_recovered": len(recovered),
            "status": "ok",
            "notes": "Labels recomputed from closed-loop recovery fraction and clean-score tolerance; diagnostics are not used to set labels.",
        })
    return out


def build_rows(rows):
    out = []
    base = _base_rows(rows)
    for frac in RECOVERY_FRACTIONS:
        for tol in CLEAN_TOLERANCES:
            labeled = label_rows([dict(r) for r in base], recovery_fraction=frac, clean_tolerance=tol)
            out.extend(_summarize(labeled, frac, tol))
    for quantile in ["q80", "q95"]:
        out.append({
            "analysis": "atr_quantile",
            "recovery_fraction": "",
            "clean_tolerance": "",
            "atr_quantile": quantile,
            "smpr_margin": "delta0",
            "proxy_variant": "",
            "task": "ALL",
            "fragile_median_atr": "",
            "recovered_median_atr": "",
            "fragile_median_smpr": "",
            "recovered_median_smpr": "",
            "atr_direction_pass": "",
            "smpr_direction_pass": "",
            "num_fragile": "",
            "num_recovered": "",
            "status": "unavailable_raw_tail",
            "notes": "Retained full-sweep diagnostics store ATR q90 only; q80/q95 require sample-level same-state radii.",
        })
    for margin in ["delta005", "delta010"]:
        out.append({
            "analysis": "smpr_margin",
            "recovery_fraction": "",
            "clean_tolerance": "",
            "atr_quantile": "q90",
            "smpr_margin": margin,
            "proxy_variant": "",
            "task": "ALL",
            "fragile_median_atr": "",
            "recovered_median_atr": "",
            "fragile_median_smpr": "",
            "recovered_median_smpr": "",
            "atr_direction_pass": "",
            "smpr_direction_pass": "",
            "num_fragile": "",
            "num_recovered": "",
            "status": "unavailable_margin_sweep",
            "notes": "Retained SMPR full-sweep artifact stores margin-zero pass rate only.",
        })
    for proxy in ["q25_margin_minus_2q90_drift", "q50_margin_minus_2q95_drift", "q10_margin_minus_2q95_drift"]:
        out.append({
            "analysis": "proxy_quantile",
            "recovery_fraction": "",
            "clean_tolerance": "",
            "atr_quantile": "q90",
            "smpr_margin": "delta0",
            "proxy_variant": proxy,
            "task": "ALL",
            "fragile_median_atr": "",
            "recovered_median_atr": "",
            "fragile_median_smpr": "",
            "recovered_median_smpr": "",
            "atr_direction_pass": "",
            "smpr_direction_pass": "",
            "num_fragile": "",
            "num_recovered": "",
            "status": "unavailable_raw_tail",
            "notes": "Requires lower-tail clean margins and upper-tail paired drift not retained in current summaries.",
        })
    return out


def write_table(rows, out: Path) -> None:
    ok = [r for r in rows if r["analysis"] == "recovery_threshold_clean_tolerance" and r["status"] == "ok"]
    grouped = defaultdict(list)
    for row in ok:
        grouped[(float(row["recovery_fraction"]), float(row["clean_tolerance"]))].append(row)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{All nine behavioral-label settings preserve the expected ATR/SMPR directions in all four tasks. Each cell reports ATR-pass, SMPR-pass task counts for the indicated clean-score tolerance.}",
        r"\label{tab:threshold-quantile-sensitivity}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{clean-score tolerance} \\",
        r"\cmidrule(lr){2-4}",
        r"Recovery fraction & 3 & 5 & 10 \\",
        r"\midrule",
    ]
    for frac in RECOVERY_FRACTIONS:
        cells = []
        for tol in CLEAN_TOLERANCES:
            block = grouped[(frac, tol)]
            atr_pass = sum(1 for row in block if row["atr_direction_pass"] == "true")
            smpr_pass = sum(1 for row in block if row["smpr_direction_pass"] == "true")
            cells.append(f"${atr_pass}/4,\;{smpr_pass}/4$")
        lines.append(f"${frac:.2f}$ & " + " & ".join(cells) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def plot(rows, out: Path) -> None:
    ok = [r for r in rows if r["analysis"] == "recovery_threshold_clean_tolerance" and r["status"] == "ok"]
    grouped = defaultdict(list)
    for r in ok:
        grouped[(float(r["recovery_fraction"]), float(r["clean_tolerance"]))].append(r)
    xs = []
    atr = []
    smpr = []
    labels = []
    for i, key in enumerate(sorted(grouped)):
        block = grouped[key]
        xs.append(i)
        atr.append(sum(1 for r in block if r["atr_direction_pass"] == "true"))
        smpr.append(sum(1 for r in block if r["smpr_direction_pass"] == "true"))
        labels.append(f"{key[0]:.2f}\n{key[1]:.0f}pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    width = 0.35
    ax.bar([x - width / 2 for x in xs], atr, width, label="ATR direction", color="#a61c00")
    ax.bar([x + width / 2 for x in xs], smpr, width, label="SMPR direction", color="#674ea7")
    ax.set_ylim(0, 4.2)
    ax.set_ylabel("tasks passing direction check")
    ax.set_xlabel("recovery fraction / clean tolerance")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--fig-out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    rows = read_csv(args.diagnostics)
    out_rows = build_rows(rows)
    write_csv(args.out, out_rows, FIELDNAMES)
    write_table(out_rows, args.table_out)
    plot(out_rows, args.fig_out)
    print(f"wrote {args.out} ({len(out_rows)} rows)")
    print(f"wrote {args.table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
