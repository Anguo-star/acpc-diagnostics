#!/usr/bin/env python3
"""Summarize full-sweep sample-level fixed-pool certificate calibration."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .utils_paper1_io import ROOT, RHO_GRID, TASKS, fnum, fmt_rho, read_csv, write_csv

DEFAULT_AUDIT = ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_audit.csv"
DEFAULT_FULL_SWEEP = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_OUT = ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_summary.csv"
DEFAULT_ALIGNMENT = ROOT / "paper1" / "results" / "sample_level_certificate_recovery_alignment.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_sample_level_certificate_full_sweep.tex"

SUMMARY_FIELDS = [
    "task",
    "rho",
    "n_training_seeds",
    "recovery_label_rate",
    "obs_sigma_008_score_mean",
    "cert_pass_rate_mean",
    "cert_pass_rate_pstdev",
    "top1_flip_rate_mean",
    "top1_flip_rate_pstdev",
    "gap_q10_q95_mean",
    "gap_q10_q95_pstdev",
    "gap_q50_q95_mean",
    "gap_q50_q95_pstdev",
    "max_drift_q95_mean",
    "clean_margin_q10_mean",
    "candidate_count_mean",
]

ALIGNMENT_FIELDS = [
    "task",
    "split",
    "n_rows",
    "cert_pass_rate_median",
    "cert_pass_rate_mean",
    "top1_flip_rate_median",
    "top1_flip_rate_mean",
    "gap_q10_q95_median",
    "gap_q50_q95_median",
    "obs_sigma_008_score_median",
]


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        x = fnum(value)
        if math.isfinite(x):
            out.append(x)
    return out


def _mean(values: list[Any]) -> float:
    xs = _finite(values)
    return mean(xs) if xs else math.nan


def _median(values: list[Any]) -> float:
    xs = _finite(values)
    return median(xs) if xs else math.nan


def _pstdev(values: list[Any]) -> float:
    xs = _finite(values)
    if not xs:
        return math.nan
    return pstdev(xs) if len(xs) > 1 else 0.0


def _fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _fmt_pair(a: float, b: float, digits: int = 2) -> str:
    return f"{_fmt(a, digits)} $\\to$ {_fmt(b, digits)}"


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    seed = int(fnum(row.get("training_seed", row.get("train_seed"))))
    return row["task"], seed, fmt_rho(row.get("rho", row.get("std_key", row.get("stdmax"))))


def _read_audit(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if row.get("status") != "ok":
            continue
        row = dict(row)
        row["rho"] = fmt_rho(row.get("std_key"))
        rows.append(row)
    return rows


def _join(audit_rows: list[dict[str, Any]], full_sweep_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    full_index = {_key(row): row for row in full_sweep_rows}
    out = []
    for row in audit_rows:
        joined = dict(row)
        sweep = full_index.get(_key(row), {})
        for col in (
            "recovery_label",
            "obs_sigma_008_score",
            "clean_eval_score",
            "atr_normalized_q90",
            "smpr_delta0",
        ):
            joined[col] = sweep.get(col, "")
        out.append(joined)
    return out


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["rho"])].append(row)
    out = []
    for task in TASKS:
        for rho in RHO_GRID:
            block = grouped[(task, rho)]
            if not block:
                continue
            out.append({
                "task": task,
                "rho": rho,
                "n_training_seeds": len(block),
                "recovery_label_rate": _mean([1.0 if str(r.get("recovery_label")).lower() == "true" else 0.0 for r in block]),
                "obs_sigma_008_score_mean": _mean([r.get("obs_sigma_008_score") for r in block]),
                "cert_pass_rate_mean": _mean([r.get("sample_cert_pass_rate") for r in block]),
                "cert_pass_rate_pstdev": _pstdev([r.get("sample_cert_pass_rate") for r in block]),
                "top1_flip_rate_mean": _mean([r.get("sample_top1_flip_rate") for r in block]),
                "top1_flip_rate_pstdev": _pstdev([r.get("sample_top1_flip_rate") for r in block]),
                "gap_q10_q95_mean": _mean([r.get("certificate_gap_q10_q95") for r in block]),
                "gap_q10_q95_pstdev": _pstdev([r.get("certificate_gap_q10_q95") for r in block]),
                "gap_q50_q95_mean": _mean([r.get("certificate_gap_q50_q95") for r in block]),
                "gap_q50_q95_pstdev": _pstdev([r.get("certificate_gap_q50_q95") for r in block]),
                "max_drift_q95_mean": _mean([r.get("sample_max_drift_q95") for r in block]),
                "clean_margin_q10_mean": _mean([r.get("clean_margin_q10") for r in block]),
                "candidate_count_mean": _mean([r.get("candidate_count") for r in block]),
            })
    return out


def build_alignment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task in [*TASKS, "ALL"]:
        task_rows = rows if task == "ALL" else [r for r in rows if r["task"] == task]
        for split, pred in (
            ("fragile", lambda r: str(r.get("recovery_label")).lower() != "true"),
            ("recovered", lambda r: str(r.get("recovery_label")).lower() == "true"),
        ):
            block = [r for r in task_rows if pred(r)]
            out.append({
                "task": task,
                "split": split,
                "n_rows": len(block),
                "cert_pass_rate_median": _median([r.get("sample_cert_pass_rate") for r in block]),
                "cert_pass_rate_mean": _mean([r.get("sample_cert_pass_rate") for r in block]),
                "top1_flip_rate_median": _median([r.get("sample_top1_flip_rate") for r in block]),
                "top1_flip_rate_mean": _mean([r.get("sample_top1_flip_rate") for r in block]),
                "gap_q10_q95_median": _median([r.get("certificate_gap_q10_q95") for r in block]),
                "gap_q50_q95_median": _median([r.get("certificate_gap_q50_q95") for r in block]),
                "obs_sigma_008_score_median": _median([r.get("obs_sigma_008_score") for r in block]),
            })
    return out


def write_table(path: Path, alignment_rows: list[dict[str, Any]]) -> None:
    idx = {(row["task"], row["split"]): row for row in alignment_rows}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Full-sweep sample-level fixed-pool event-rate calibration. Rows split all task--seed--\stdmax{} checkpoints by the closed-loop recovery-band label. Cert-pass is the fraction of sampled states satisfying the fixed-pool sufficient event; top-1 flip is the observed clean/noisy fixed-pool best-candidate disagreement. These event rates strengthen the mechanism analysis but are not calibrated probability bounds.}",
        r"\label{tab:sample-level-certificate-full-sweep}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Task & cert-pass fragile $\to$ recovered & top-1 flip fragile $\to$ recovered & q10/q95 gap fragile $\to$ recovered \\",
        r"\midrule",
    ]
    for task in [*TASKS, "ALL"]:
        fragile = idx.get((task, "fragile"), {})
        recovered = idx.get((task, "recovered"), {})
        lines.append(
            f"{task} & "
            f"{_fmt_pair(fnum(fragile.get('cert_pass_rate_median')), fnum(recovered.get('cert_pass_rate_median')))} & "
            f"{_fmt_pair(fnum(fragile.get('top1_flip_rate_median')), fnum(recovered.get('top1_flip_rate_median')))} & "
            f"{_fmt_pair(fnum(fragile.get('gap_q10_q95_median')), fnum(recovered.get('gap_q10_q95_median')), 1)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--full-sweep", type=Path, default=DEFAULT_FULL_SWEEP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()

    rows = _join(_read_audit(args.audit), read_csv(args.full_sweep))
    summary = build_summary(rows)
    alignment = build_alignment(rows)
    write_csv(args.out, summary, SUMMARY_FIELDS)
    write_csv(args.alignment, alignment, ALIGNMENT_FIELDS)
    write_table(args.table, alignment)
    print(f"wrote {args.out}")
    print(f"wrote {args.alignment}")
    print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
