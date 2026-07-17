#!/usr/bin/env python3
"""Empirical fixed-pool risk calibration intervals for Paper1."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils_paper1_io import ROOT, TASKS, fnum, fmt_rho, read_csv, write_csv

DEFAULT_SAMPLES = ROOT / "paper1" / "results" / "sample_level_certificate_full_sweep_samples.csv"
DEFAULT_FULL_SWEEP = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_OUT = ROOT / "paper1" / "results" / "sample_level_event_rate_wilson_ci.csv"
DEFAULT_TABLE = ROOT / "paper1" / "tables" / "table_sample_level_event_rate_ci.tex"
Z_95 = 1.959963984540054

FIELDS = [
    "task",
    "split",
    "metric",
    "successes",
    "n",
    "rate",
    "wilson_low",
    "wilson_high",
    "confidence_level",
    "notes",
]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return row["task"], int(fnum(row.get("training_seed", row.get("train_seed")))), fmt_rho(row.get("rho", row.get("std_key", row.get("stdmax"))))


def _wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    if n <= 0:
        return math.nan, math.nan, math.nan
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def build_rows(samples: list[dict[str, str]], full_sweep: list[dict[str, str]]) -> list[dict[str, Any]]:
    recovery = {_key(row): str(row.get("recovery_label", "")).lower() == "true" for row in full_sweep}
    groups: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in samples:
        split = "recovered" if recovery.get(_key(row), False) else "fragile"
        cert_pass = _bool(row.get("cert_pass"))
        top1_flip = _bool(row.get("top1_flip"))
        for task in (row["task"], "ALL"):
            groups[(task, split, "cert-pass")].append(cert_pass)
            groups[(task, split, "top-1 flip")].append(top1_flip)
            if cert_pass:
                groups[(task, split, "top-1 flip | cert-pass")].append(top1_flip)
    out = []
    for task in [*TASKS, "ALL"]:
        for split in ("fragile", "recovered"):
            for metric in ("cert-pass", "top-1 flip", "top-1 flip | cert-pass"):
                vals = groups[(task, split, metric)]
                successes = sum(1 for v in vals if v)
                n = len(vals)
                rate, low, high = _wilson(successes, n)
                note = "Wilson interval over sampled fixed-pool audit anchors; not a calibrated theorem probability bound"
                if metric == "top-1 flip | cert-pass":
                    note = "Wilson interval over cert-pass anchors only; empirical conditional flip risk, not a theorem probability bound"
                out.append({
                    "task": task,
                    "split": split,
                    "metric": metric,
                    "successes": successes,
                    "n": n,
                    "rate": rate,
                    "wilson_low": low,
                    "wilson_high": high,
                    "confidence_level": 0.95,
                    "notes": note,
                })
    return out

def _fmt_interval(row: dict[str, Any]) -> str:
    rate = fnum(row.get("rate"))
    low = fnum(row.get("wilson_low"))
    high = fnum(row.get("wilson_high"))
    if not all(math.isfinite(x) for x in (rate, low, high)):
        return "--"
    return f"{rate:.2f} [{low:.2f},{high:.2f}]"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    idx = {(row["task"], row["split"], row["metric"]): row for row in rows}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Finite-sample empirical fixed-pool risk audit. Each cell reports fragile $\to$ recovered rate [95\% Wilson interval] over sampled fixed-pool anchors. The conditional column restricts the denominator to cert-pass anchors, directly evaluating whether the sufficient event is associated with low fixed-pool candidate-flip risk. These intervals quantify event-rate estimation uncertainty and are not calibrated theorem probability bounds.}",
        r"\label{tab:sample-level-event-rate-ci}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Task & cert-pass & top-1 flip & flip $\mid$ cert-pass \\",
        r"\midrule",
    ]
    for task in [*TASKS, "ALL"]:
        cert_f = idx.get((task, "fragile", "cert-pass"), {})
        cert_r = idx.get((task, "recovered", "cert-pass"), {})
        flip_f = idx.get((task, "fragile", "top-1 flip"), {})
        flip_r = idx.get((task, "recovered", "top-1 flip"), {})
        cond_f = idx.get((task, "fragile", "top-1 flip | cert-pass"), {})
        cond_r = idx.get((task, "recovered", "top-1 flip | cert-pass"), {})
        lines.append(
            f"{task} & {_fmt_interval(cert_f)} $\\to$ {_fmt_interval(cert_r)} & "
            f"{_fmt_interval(flip_f)} $\\to$ {_fmt_interval(flip_r)} & "
            f"{_fmt_interval(cond_f)} $\\to$ {_fmt_interval(cond_r)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--full-sweep", type=Path, default=DEFAULT_FULL_SWEEP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    rows = build_rows(read_csv(args.samples), read_csv(args.full_sweep))
    write_csv(args.out, rows, FIELDS)
    write_table(args.table, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
