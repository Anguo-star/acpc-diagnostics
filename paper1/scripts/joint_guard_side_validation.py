#!/usr/bin/env python3
"""Summarize Paper1 joint ATR plus guard-side validation.

SMPR and fixed-pool action-candidate stability are guard-side criteria. They are
not interpreted as standalone robustness metrics; this summary keeps them next
to the ATR radius term.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .utils_paper1_io import ROOT, TASKS, fnum, fmt_rho, read_csv, write_csv

DEFAULT_FULL_SWEEP = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_SAMPLE_AUDIT = ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_audit.csv"
DEFAULT_OUT = ROOT / "paper1" / "results" / "joint_guard_side_validation.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_joint_guard_side_validation.tex"

FIELDS = [
    "task",
    "split",
    "n_rows",
    "atr_normalized_q90_median",
    "atr_normalized_q90_mean",
    "smpr_delta0_median",
    "smpr_delta0_mean",
    "fixed_pool_top1_flip_median",
    "fixed_pool_top1_flip_mean",
    "cert_pass_rate_median",
    "cert_pass_rate_mean",
    "obs_sigma_008_score_median",
    "notes",
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


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    seed = int(fnum(row.get("training_seed", row.get("train_seed"))))
    rho = fmt_rho(row.get("rho", row.get("std_key", row.get("stdmax"))))
    return row["task"], seed, rho


def _join(full_rows: list[dict[str, str]], sample_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    sample_index = {
        _key(row): row
        for row in sample_rows
        if row.get("status") == "ok"
    }
    out = []
    for row in full_rows:
        joined = dict(row)
        sample = sample_index.get(_key(row), {})
        joined["sample_cert_pass_rate"] = sample.get("sample_cert_pass_rate", "")
        joined["sample_top1_flip_rate"] = sample.get("sample_top1_flip_rate", "")
        joined["certificate_gap_q10_q95"] = sample.get("certificate_gap_q10_q95", "")
        out.append(joined)
    return out


def build_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = "recovered" if str(row.get("recovery_label", "")).lower() == "true" else "fragile"
        grouped[(row["task"], split)].append(row)
        grouped[("ALL", split)].append(row)
    out = []
    for task in [*TASKS, "ALL"]:
        for split in ("fragile", "recovered"):
            block = grouped[(task, split)]
            out.append({
                "task": task,
                "split": split,
                "n_rows": len(block),
                "atr_normalized_q90_median": _median([r.get("atr_normalized_q90") for r in block]),
                "atr_normalized_q90_mean": _mean([r.get("atr_normalized_q90") for r in block]),
                "smpr_delta0_median": _median([r.get("smpr_delta0") for r in block]),
                "smpr_delta0_mean": _mean([r.get("smpr_delta0") for r in block]),
                "fixed_pool_top1_flip_median": _median([r.get("sample_top1_flip_rate") for r in block]),
                "fixed_pool_top1_flip_mean": _mean([r.get("sample_top1_flip_rate") for r in block]),
                "cert_pass_rate_median": _median([r.get("sample_cert_pass_rate") for r in block]),
                "cert_pass_rate_mean": _mean([r.get("sample_cert_pass_rate") for r in block]),
                "obs_sigma_008_score_median": _median([r.get("obs_sigma_008_score") for r in block]),
                "notes": "SMPR and fixed-pool top1 flip are guard-side criteria interpreted only jointly with ATR",
            })
    return out


def _fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _pair(a: float, b: float, digits: int = 2) -> str:
    return f"{_fmt(a, digits)} $\\to$ {_fmt(b, digits)}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    idx = {(row["task"], row["split"]): row for row in rows}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{ATR, proxy SMPR, and fixed-pool top-1 flip co-movement across the Gaussian sweep. Rows are split by the closed-loop recovery-band label. SMPR is a positive-margin state-proxy guard; controls do not establish incremental task-label or action relevance.}",
        r"\label{tab:joint-guard-side-validation}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Task & ATR fragile $\to$ recovered & SMPR fragile $\to$ recovered & top-1 flip fragile $\to$ recovered \\",
        r"\midrule",
    ]
    for task in [*TASKS, "ALL"]:
        fragile = idx.get((task, "fragile"), {})
        recovered = idx.get((task, "recovered"), {})
        lines.append(
            f"{task} & "
            f"{_pair(fnum(fragile.get('atr_normalized_q90_median')), fnum(recovered.get('atr_normalized_q90_median')))} & "
            f"{_pair(fnum(fragile.get('smpr_delta0_median')), fnum(recovered.get('smpr_delta0_median')))} & "
            f"{_pair(fnum(fragile.get('fixed_pool_top1_flip_median')), fnum(recovered.get('fixed_pool_top1_flip_median')))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sweep", type=Path, default=DEFAULT_FULL_SWEEP)
    parser.add_argument("--sample-audit", type=Path, default=DEFAULT_SAMPLE_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()

    rows = build_rows(_join(read_csv(args.full_sweep), read_csv(args.sample_audit)))
    write_csv(args.out, rows, FIELDS)
    write_table(args.table, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
