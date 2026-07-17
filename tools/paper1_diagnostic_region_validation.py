#!/usr/bin/env python3
"""Diagnostic-region validation for Paper 1 ATR/SMPR sweeps.

This script is intentionally not a threshold-classifier audit. It uses existing
full-sweep diagnostics and closed-loop evaluation manifests to summarize three
paper-facing properties:

1. shared low-ATR/high-SMPR region for recovered checkpoints,
2. direction consistency within task--training-seed sweeps,
3. recovered-vs-fragile diagnostic separation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_DIAGNOSTICS = ROOT / "paper1" / "results" / "prospective_diagnostic" / "diagnostics_all_ckpts.csv"
DEFAULT_MANIFEST_DIR = DATA_DIR / "training_seed_eval_manifests"
DEFAULT_OUT_DIR = ROOT / "paper1" / "results" / "diagnostic_region"

TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
HELDOUT_SEEDS = (3073, 3074)
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
REGIMES = ("fragile", "transition", "robust")
EPS = 1e-12


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _metric_mean(entry: Mapping[str, Any], key: str) -> float:
    metric = entry.get("metrics", {}).get(key)
    if metric is None:
        raise KeyError(f"missing metric {key}")
    return float(metric["mean"])


def _std_key(value: Any) -> str:
    val = float(value)
    return "0.0" if abs(val) <= EPS else "{:.2f}".format(val)


def _score_map(manifest_dir: Path, eval_metric: str) -> dict[tuple[str, int, str], dict[str, float]]:
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    for seed in SEEDS:
        manifest = _load_json(manifest_dir / f"lewm_seed{seed}_evals.json")
        for task in TASKS:
            for std_key in STD_KEYS:
                entry = manifest[task][std_key]
                out[(task, seed, std_key)] = {
                    "closed_loop_score": _metric_mean(entry, eval_metric),
                    "clean_score": _metric_mean(entry, "clean"),
                }
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _median(values: Sequence[float]) -> float:
    return _percentile(values, 0.5)


def _split_for_seed(seed: int) -> str:
    return "calibration" if seed == 3072 else "heldout"


def _regime(recovery: float, fragile_rho: float, robust_rho: float) -> str:
    if recovery <= fragile_rho + EPS:
        return "fragile"
    if recovery >= robust_rho - EPS:
        return "robust"
    return "transition"


def _labels(
    score_rows: Mapping[tuple[str, int, str], dict[str, float]],
    *,
    fragile_rho: float,
    robust_rho: float,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for task in TASKS:
        for seed in SEEDS:
            base = float(score_rows[(task, seed, "0.0")]["closed_loop_score"])
            best = max(float(score_rows[(task, seed, std)]["closed_loop_score"]) for std in STD_KEYS)
            denom = best - base
            for std_key in STD_KEYS:
                score = float(score_rows[(task, seed, std_key)]["closed_loop_score"])
                recovery = (score - base) / denom if denom > EPS else 1.0
                out[(task, seed, std_key)] = {
                    "closed_loop_score": score,
                    "base_closed_loop_score": base,
                    "best_closed_loop_score": best,
                    "normalized_recovery": recovery,
                    "regime": _regime(recovery, fragile_rho, robust_rho),
                }
    return out


def build_rows(
    diagnostics_csv: Path,
    manifest_dir: Path,
    *,
    eval_metric: str,
    fragile_rho: float,
    robust_rho: float,
) -> list[dict[str, Any]]:
    diagnostics = _read_csv(diagnostics_csv)
    scores = _score_map(manifest_dir, eval_metric)
    labels = _labels(scores, fragile_rho=fragile_rho, robust_rho=robust_rho)
    base_atr: dict[tuple[str, int], float] = {}
    for row in diagnostics:
        seed = int(row["train_seed"])
        if row["stdmax"] == "0.0":
            base_atr[(row["task"], seed)] = float(row["atr_q90"])

    out: list[dict[str, Any]] = []
    for row in diagnostics:
        task = row["task"]
        seed = int(row["train_seed"])
        std_key = _std_key(row["stdmax"])
        label = labels[(task, seed, std_key)]
        base = base_atr[(task, seed)]
        atr_q90 = float(row["atr_q90"])
        smpr = float(row["smpr"])
        out.append({
            "task": task,
            "train_seed": seed,
            "split": _split_for_seed(seed),
            "stdmax": std_key,
            "eval_noise_sigma": row.get("eval_noise_sigma", "0.08"),
            "closed_loop_score": label["closed_loop_score"],
            "base_closed_loop_score": label["base_closed_loop_score"],
            "best_closed_loop_score": label["best_closed_loop_score"],
            "normalized_recovery": label["normalized_recovery"],
            "regime": label["regime"],
            "atr_q90": atr_q90,
            "atr_rel": atr_q90 / base if base > EPS else float("nan"),
            "smpr": smpr,
        })
    return out


def _rows_for_split(rows: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    if split == "all":
        return list(rows)
    return [row for row in rows if row["split"] == split]


def summarize_regions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("calibration", "heldout", "all"):
        split_rows = _rows_for_split(rows, split)
        for regime in REGIMES:
            group = [row for row in split_rows if row["regime"] == regime]
            out.append({
                "split": split,
                "regime": regime,
                "n": len(group),
                "recovery_q25": _percentile([row["normalized_recovery"] for row in group], 0.25),
                "recovery_q50": _percentile([row["normalized_recovery"] for row in group], 0.50),
                "recovery_q75": _percentile([row["normalized_recovery"] for row in group], 0.75),
                "atr_rel_q25": _percentile([row["atr_rel"] for row in group], 0.25),
                "atr_rel_q50": _percentile([row["atr_rel"] for row in group], 0.50),
                "atr_rel_q75": _percentile([row["atr_rel"] for row in group], 0.75),
                "smpr_q25": _percentile([row["smpr"] for row in group], 0.25),
                "smpr_q50": _percentile([row["smpr"] for row in group], 0.50),
                "smpr_q75": _percentile([row["smpr"] for row in group], 0.75),
            })
    return out


def summarize_direction(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    for split in ("calibration", "heldout", "all"):
        split_rows = _rows_for_split(rows, split)
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in split_rows:
            grouped[(str(row["task"]), int(row["train_seed"]))].append(row)
        for (task, seed), group in sorted(grouped.items()):
            fragile = [row for row in group if row["regime"] == "fragile"]
            robust = [row for row in group if row["regime"] == "robust"]
            if not fragile or not robust:
                continue
            fragile_atr = _median([row["atr_rel"] for row in fragile])
            robust_atr = _median([row["atr_rel"] for row in robust])
            fragile_smpr = _median([row["smpr"] for row in fragile])
            robust_smpr = _median([row["smpr"] for row in robust])
            detail.append({
                "split": split,
                "task": task,
                "train_seed": seed,
                "fragile_n": len(fragile),
                "robust_n": len(robust),
                "fragile_atr_rel_median": fragile_atr,
                "robust_atr_rel_median": robust_atr,
                "fragile_smpr_median": fragile_smpr,
                "robust_smpr_median": robust_smpr,
                "atr_direction_ok": int(robust_atr < fragile_atr),
                "smpr_direction_ok": int(robust_smpr > fragile_smpr),
                "joint_direction_ok": int(robust_atr < fragile_atr and robust_smpr > fragile_smpr),
            })

    summary: list[dict[str, Any]] = []
    for split in ("calibration", "heldout", "all"):
        group = [row for row in detail if row["split"] == split]
        summary.append({
            "split": split,
            "eligible_blocks": len(group),
            "atr_direction_ok": sum(int(row["atr_direction_ok"]) for row in group),
            "smpr_direction_ok": sum(int(row["smpr_direction_ok"]) for row in group),
            "joint_direction_ok": sum(int(row["joint_direction_ok"]) for row in group),
        })
    return detail, summary


def summarize_separation(region_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["split"], row["regime"]): row for row in region_rows}
    out: list[dict[str, Any]] = []
    for split in ("calibration", "heldout", "all"):
        fragile = by_key[(split, "fragile")]
        robust = by_key[(split, "robust")]
        out.append({
            "split": split,
            "fragile_n": fragile["n"],
            "robust_n": robust["n"],
            "fragile_atr_rel_median": fragile["atr_rel_q50"],
            "robust_atr_rel_median": robust["atr_rel_q50"],
            "atr_rel_median_gap": fragile["atr_rel_q50"] - robust["atr_rel_q50"],
            "robust_atr_q75_below_fragile_q25": int(robust["atr_rel_q75"] < fragile["atr_rel_q25"]),
            "fragile_smpr_median": fragile["smpr_q50"],
            "robust_smpr_median": robust["smpr_q50"],
            "smpr_median_gap": robust["smpr_q50"] - fragile["smpr_q50"],
            "robust_smpr_q25_above_fragile_q75": int(robust["smpr_q25"] > fragile["smpr_q75"]),
        })
    return out


def write_readme(out_dir: Path, metadata: Mapping[str, Any]) -> None:
    text = f"""# Paper1 Diagnostic Region Validation

This directory is a paper-facing validation artifact for the ATR/SMPR sweep. It
does not rank threshold classifiers by F1, precision, recall, or interval IoU.
Instead, it checks whether full-sweep checkpoints form:

- a shared low-ATR/high-SMPR diagnostic region for recovered checkpoints,
- consistent directions within task--training-seed sweeps,
- separation between recovered and fragile checkpoints.

Regime labels are derived from normalized closed-loop recovery under
`{metadata['eval_metric']}`. Fragile means recovery <= {metadata['fragile_rho']};
robust means recovery >= {metadata['robust_rho']}; transition checkpoints are
kept visible rather than forced into either endpoint.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rows = build_rows(
        args.diagnostics,
        args.manifest_dir,
        eval_metric=args.eval_metric,
        fragile_rho=args.fragile_rho,
        robust_rho=args.robust_rho,
    )
    region_rows = summarize_regions(rows)
    direction_detail, direction_summary = summarize_direction(rows)
    separation_rows = summarize_separation(region_rows)

    row_fields = [
        "task", "train_seed", "split", "stdmax", "eval_noise_sigma", "closed_loop_score",
        "base_closed_loop_score", "best_closed_loop_score", "normalized_recovery", "regime",
        "atr_q90", "atr_rel", "smpr",
    ]
    region_fields = [
        "split", "regime", "n", "recovery_q25", "recovery_q50", "recovery_q75",
        "atr_rel_q25", "atr_rel_q50", "atr_rel_q75", "smpr_q25", "smpr_q50", "smpr_q75",
    ]
    direction_fields = [
        "split", "task", "train_seed", "fragile_n", "robust_n",
        "fragile_atr_rel_median", "robust_atr_rel_median",
        "fragile_smpr_median", "robust_smpr_median",
        "atr_direction_ok", "smpr_direction_ok", "joint_direction_ok",
    ]
    direction_summary_fields = [
        "split", "eligible_blocks", "atr_direction_ok", "smpr_direction_ok", "joint_direction_ok",
    ]
    separation_fields = [
        "split", "fragile_n", "robust_n", "fragile_atr_rel_median", "robust_atr_rel_median",
        "atr_rel_median_gap", "robust_atr_q75_below_fragile_q25",
        "fragile_smpr_median", "robust_smpr_median", "smpr_median_gap",
        "robust_smpr_q25_above_fragile_q75",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "diagnostic_region_rows.csv", rows, row_fields)
    _write_csv(args.out_dir / "diagnostic_region_summary.csv", region_rows, region_fields)
    _write_csv(args.out_dir / "direction_consistency_by_block.csv", direction_detail, direction_fields)
    _write_csv(args.out_dir / "direction_consistency_summary.csv", direction_summary, direction_summary_fields)
    _write_csv(args.out_dir / "robust_fragile_separation.csv", separation_rows, separation_fields)
    write_readme(args.out_dir, {
        "eval_metric": args.eval_metric,
        "fragile_rho": args.fragile_rho,
        "robust_rho": args.robust_rho,
    })

    print(f"wrote {args.out_dir / 'diagnostic_region_summary.csv'}")
    print(f"wrote {args.out_dir / 'direction_consistency_summary.csv'}")
    print(f"wrote {args.out_dir / 'robust_fragile_separation.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--eval-metric", default="pixels_std0.08")
    parser.add_argument("--fragile-rho", type=float, default=0.2)
    parser.add_argument("--robust-rho", type=float, default=0.8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
