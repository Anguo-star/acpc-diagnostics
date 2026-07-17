#!/usr/bin/env python3
"""Shared helpers for Paper1 diagnostic remediation scripts."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, median
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["TwoRoom", "PushT", "Reacher", "Cube"]
SEEDS = [3072, 3073, 3074]
RHO_GRID = [f"{i / 100:.2f}" for i in range(9)]
EPS = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_rho(value: Any) -> str:
    return f"{float(value):.2f}"


def finite(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        x = fnum(value)
        if math.isfinite(x):
            out.append(x)
    return out


def safe_mean(values: Iterable[Any]) -> float:
    xs = finite(values)
    return mean(xs) if xs else math.nan


def safe_pstdev(values: Iterable[Any]) -> float:
    xs = finite(values)
    if not xs:
        return math.nan
    return pstdev(xs) if len(xs) > 1 else 0.0


def safe_median(values: Iterable[Any]) -> float:
    xs = finite(values)
    return median(xs) if xs else math.nan


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def get_start(rows: list[dict[str, Any]], pred_key: str) -> float | None:
    positives = [fnum(row["rho"]) for row in rows if str(row.get(pred_key, "")).lower() in {"true", "1", "yes"}]
    return min(positives) if positives else None


def label_rows(rows: list[dict[str, Any]], recovery_fraction: float = 0.8, clean_tolerance: float = 5.0) -> list[dict[str, Any]]:
    by_block: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[(row["task"], int(row["training_seed"]))].append(row)
    labeled: list[dict[str, Any]] = []
    for (_task, _seed), block in by_block.items():
        block = sorted(block, key=lambda r: fnum(r["rho"]))
        base = next(r for r in block if fmt_rho(r["rho"]) == "0.00")
        base_obs = fnum(base["obs_sigma_008_score"])
        base_clean = fnum(base["clean_eval_score"])
        best_obs = max(fnum(r["obs_sigma_008_score"]) for r in block)
        threshold = base_obs + recovery_fraction * (best_obs - base_obs)
        for row in block:
            row = dict(row)
            clean_pass = fnum(row["clean_eval_score"]) >= base_clean - clean_tolerance
            recovered = fnum(row["obs_sigma_008_score"]) >= threshold and clean_pass
            denom = max(best_obs - base_obs, EPS)
            row["base_clean_score"] = base_clean
            row["base_obs_sigma_008_score"] = base_obs
            row["max_obs_sigma_008_score"] = best_obs
            row["recovery_score_threshold"] = threshold
            row["clean_constraint_pass"] = bool_str(clean_pass)
            row["recovery_label"] = bool_str(recovered)
            row["normalized_recovery"] = (fnum(row["obs_sigma_008_score"]) - base_obs) / denom
            labeled.append(row)
    return labeled
