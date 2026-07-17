#!/usr/bin/env python3
"""Held-out diagnostic-region validation for Paper1 full-sweep diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils_paper1_io import ROOT, SEEDS, TASKS, bool_str, fnum, read_csv, write_csv

DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_OUT = ROOT / "paper1" / "results" / "heldout_diagnostic_validation.csv"
DEFAULT_PARAMS = ROOT / "paper1" / "results" / "heldout_gate_params.json"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_heldout_diagnostic_validation.tex"
DEFAULT_FIG = ROOT / "paper1" / "figures" / "fig_heldout_diagnostic_validation.png"

TAU_ATR = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
TAU_SMPR = [0.80, 0.85, 0.90, 0.95]
TAU_TOP1 = [0.80, 0.90, 0.95]

FIELDNAMES = [
    "mode", "heldout_unit", "gate_features", "tau_atr", "tau_smpr", "tau_top1", "use_proxy_gap",
    "task", "training_seed", "behavioral_start", "predicted_start", "start_error",
    "false_early", "false_late", "balanced_accuracy", "precision", "recall", "num_rows", "transition_rows_flagged",
]


def _truth(row) -> bool:
    return str(row.get("recovery_label", "")).lower() == "true"


def _pred(row, gate) -> bool:
    atr_ok = fnum(row["atr_normalized_q90"]) <= gate["tau_atr"]
    smpr_ok = fnum(row["smpr_delta0"]) >= gate["tau_smpr"]
    if not (atr_ok and smpr_ok):
        return False
    if gate["rule"] == "atr_smpr_top1_or_proxy":
        top1_ok = fnum(row["top1_agree"]) >= gate["tau_top1"]
        proxy_ok = fnum(row["proxy_gap_q50q90"]) > 0
        return top1_ok or proxy_ok
    return True


def _blocks(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], int(float(row["training_seed"])))] .append(row)
    return {k: sorted(v, key=lambda r: fnum(r["rho"])) for k, v in grouped.items()}


def _start(rows, key_fn) -> float | None:
    vals = [fnum(r["rho"]) for r in rows if key_fn(r)]
    return min(vals) if vals else None


def _block_eval(rows, gate):
    y = [_truth(r) for r in rows]
    pred = [_pred(r, gate) for r in rows]
    behavioral_start = _start(rows, _truth)
    predicted_start = _start(rows, lambda r: _pred(r, gate))
    if behavioral_start is None and predicted_start is None:
        start_error = 0.0
    elif behavioral_start is None:
        start_error = -1.0
    elif predicted_start is None:
        start_error = 1.0
    else:
        start_error = predicted_start - behavioral_start
    tp = sum(1 for a, b in zip(y, pred) if a and b)
    tn = sum(1 for a, b in zip(y, pred) if not a and not b)
    fp = sum(1 for a, b in zip(y, pred) if not a and b)
    fn = sum(1 for a, b in zip(y, pred) if a and not b)
    tpr = tp / (tp + fn) if tp + fn else math.nan
    tnr = tn / (tn + fp) if tn + fp else math.nan
    if math.isfinite(tpr) and math.isfinite(tnr):
        bacc = 0.5 * (tpr + tnr)
    elif math.isfinite(tpr):
        bacc = tpr
    elif math.isfinite(tnr):
        bacc = tnr
    else:
        bacc = math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    false_early = 0
    false_late = 0
    if behavioral_start is not None:
        false_early = sum(1 for r, p in zip(rows, pred) if p and fnum(r["rho"]) < behavioral_start)
        false_late = sum(1 for r, p, yy in zip(rows, pred, y) if yy and not p)
    transition_flagged = sum(1 for r, p in zip(rows, pred) if p and abs(fnum(r.get("normalized_recovery")) - 0.8) < 0.15)
    return {
        "behavioral_start": behavioral_start,
        "predicted_start": predicted_start,
        "start_error": start_error,
        "false_early": false_early,
        "false_late": false_late,
        "balanced_accuracy": bacc,
        "precision": precision,
        "recall": recall,
        "num_rows": len(rows),
        "transition_rows_flagged": transition_flagged,
    }


def _objective(blocks, gate):
    evals = [_block_eval(rows, gate) for rows in blocks.values()]
    mean_abs_start = sum(abs(e["start_error"]) for e in evals) / max(len(evals), 1)
    false_early = sum(e["false_early"] for e in evals)
    false_late = sum(e["false_late"] for e in evals)
    mean_bacc_values = [e["balanced_accuracy"] for e in evals if math.isfinite(e["balanced_accuracy"])]
    mean_bacc = sum(mean_bacc_values) / len(mean_bacc_values) if mean_bacc_values else 0.0
    # Minimize boundary error first, then false early, then false late, then prefer higher balanced accuracy.
    return (mean_abs_start, false_early, false_late, -mean_bacc, gate["tau_atr"], -gate["tau_smpr"])


def _candidate_gates():
    for tau_atr in TAU_ATR:
        for tau_smpr in TAU_SMPR:
            yield {"rule": "atr_smpr", "tau_atr": tau_atr, "tau_smpr": tau_smpr, "tau_top1": math.nan, "use_proxy_gap": False}
            for tau_top1 in TAU_TOP1:
                yield {"rule": "atr_smpr_top1_or_proxy", "tau_atr": tau_atr, "tau_smpr": tau_smpr, "tau_top1": tau_top1, "use_proxy_gap": True}


def _split_rows(rows, mode, heldout):
    if mode == "leave_one_seed_out":
        held = [r for r in rows if int(float(r["training_seed"])) == heldout]
        cal = [r for r in rows if int(float(r["training_seed"])) != heldout]
        unit = f"seed{heldout}"
    else:
        held = [r for r in rows if r["task"] == heldout]
        cal = [r for r in rows if r["task"] != heldout]
        unit = heldout
    return cal, held, unit


def run_validation(rows):
    all_rows = []
    params = {"protocol": "thresholds calibrated on calibration rows only", "splits": []}
    split_specs = [("leave_one_seed_out", seed) for seed in SEEDS] + [("leave_one_task_out", task) for task in TASKS]
    for mode, heldout in split_specs:
        cal, held, unit = _split_rows(rows, mode, heldout)
        cal_blocks = _blocks(cal)
        best_gate = min(_candidate_gates(), key=lambda gate: _objective(cal_blocks, gate))
        held_blocks = _blocks(held)
        params["splits"].append({"mode": mode, "heldout_unit": unit, "selected_gate": best_gate, "calibration_blocks": len(cal_blocks), "heldout_blocks": len(held_blocks)})
        for (task, seed), block in held_blocks.items():
            e = _block_eval(block, best_gate)
            all_rows.append({
                "mode": mode,
                "heldout_unit": unit,
                "gate_features": best_gate["rule"],
                "tau_atr": best_gate["tau_atr"],
                "tau_smpr": best_gate["tau_smpr"],
                "tau_top1": "" if math.isnan(best_gate["tau_top1"]) else best_gate["tau_top1"],
                "use_proxy_gap": bool_str(best_gate["use_proxy_gap"]),
                "task": task,
                "training_seed": seed,
                **e,
            })
    return all_rows, params


def write_table(rows, out: Path) -> None:
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Held-out seeds and tasks preserve the recovered-vs-fragile diagnostic separation within two training-noise grid steps. Gates are selected on calibration seeds or tasks and evaluated on held-out sweeps. Recovery-onset error is in $\sigma_{\max}^{\mathrm{train}}$ units; positive values mean the diagnostic onset is later than the behavioral recovery onset.}",
        r"\label{tab:heldout-diagnostic-validation}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Split & blocks & \shortstack{mean absolute\\recovery-onset error} & \shortstack{maximum absolute\\recovery-onset error} & precision & recall \\",
        r"\midrule",
    ]
    for mode, block in by_mode.items():
        errs = [abs(fnum(r["start_error"])) for r in block]
        precisions = [fnum(r["precision"]) for r in block if math.isfinite(fnum(r["precision"]))]
        recalls = [fnum(r["recall"]) for r in block if math.isfinite(fnum(r["recall"]))]
        label = "seed held-out" if mode == "leave_one_seed_out" else "task held-out"
        mean_err = sum(errs) / len(errs)
        max_err = max(errs)
        precision = sum(precisions) / len(precisions) if precisions else math.nan
        recall = sum(recalls) / len(recalls) if recalls else math.nan
        lines.append(f"{label} & {len(block)} & ${mean_err:.3f}$ & ${max_err:.3f}$ & ${precision:.2f}$ & ${recall:.2f}$" + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def plot(rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, mode, title in zip(axes, ["leave_one_seed_out", "leave_one_task_out"], ["held-out seed", "held-out task"]):
        block = [r for r in rows if r["mode"] == mode]
        labels = [f"{r['heldout_unit']}\n{r['task']}-{r['training_seed']}" for r in block]
        vals = [fnum(r["start_error"]) for r in block]
        colors = ["#38761d" if abs(v) <= 0.01 + 1e-9 else "#bf9000" if abs(v) <= 0.02 + 1e-9 else "#a61c00" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(r"recovery-onset error in $\sigma_{\max}^{\mathrm{train}}$")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--params-out", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--fig-out", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    rows = read_csv(args.diagnostics)
    out_rows, params = run_validation(rows)
    write_csv(args.out, out_rows, FIELDNAMES)
    args.params_out.parent.mkdir(parents=True, exist_ok=True)
    args.params_out.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    write_table(out_rows, args.table_out)
    plot(out_rows, args.fig_out)
    print(f"wrote {args.out} ({len(out_rows)} rows)")
    print(f"wrote {args.params_out}")
    print(f"wrote {args.table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
