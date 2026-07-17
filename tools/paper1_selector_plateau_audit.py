"""Plateau-membership audit for Paper 1.

The audit treats std_max as a candidate label inside each task/training-seed
block, not as a continuous covariate or point-best target. It evaluates whether
diagnostic scores screen a candidate region enriched for checkpoints inside the
closed-loop robustness plateau.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_IN = DATA_DIR / "acpc_phase0_lewm_three_seed.json"
DEFAULT_OUT_JSON = DATA_DIR / "selector_plateau_audit_20260704.json"
DEFAULT_OUT_MD = DATA_DIR / "selector_plateau_audit_20260704.md"

TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
NONZERO_STD_KEYS = ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
RULE = (
    ("ACPC-H/trans.", "acpc_h_norm_by_transition", -1.0),
    ("PCC", "pcc_abs_median", -1.0),
    ("CRA", "cra_spearman_mean", 1.0),
    ("MAF", "maf_flip_rate", -1.0),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())



def _rankdata(values: Sequence[float], *, higher_better: bool) -> list[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1], reverse=higher_better)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _aggregate_scores(block: Sequence[dict]) -> list[float]:
    scores = [0.0] * len(block)
    for _, key, sign in RULE:
        signed = [sign * float(row[key]) for row in block]
        ranks = _rankdata(signed, higher_better=True)
        for i, rank in enumerate(ranks):
            scores[i] -= rank
    return scores


def _metric_scores(key: str, sign: float) -> Callable[[Sequence[dict]], list[float]]:
    return lambda block: [sign * float(row[key]) for row in block]


def _prepare_rows(data: dict) -> list[dict]:
    rows = [row for row in data["rows"] if row.get("status") == "ok" and str(row["std_key"]) != "0.0"]
    expected = {(task, seed, std) for task in TASKS for seed in SEEDS for std in NONZERO_STD_KEYS}
    got = {(row["task"], int(row["training_seed"]), str(row["std_key"])) for row in rows}
    if got != expected:
        raise ValueError(f"coverage mismatch missing={sorted(expected - got)[:5]} extra={sorted(got - expected)[:5]}")
    return sorted(rows, key=lambda r: (r["task"], int(r["training_seed"]), float(r["std_key"])))


def _blocks(rows: Sequence[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for task in TASKS:
        for seed in SEEDS:
            block = [row for row in rows if row["task"] == task and int(row["training_seed"]) == seed]
            out.append(sorted(block, key=lambda r: float(r["std_key"])))
    return out


def _membership_summary(
    rows: Sequence[dict],
    *,
    name: str,
    score_fn: Callable[[Sequence[dict]], list[float]],
    tolerance_pp: float,
    screen_size: int,
) -> dict:
    block_rows = []
    tp = fp = fn = tn = 0
    hit_count = 0
    true_total = 0
    screened_total = 0
    for block in _blocks(rows):
        values = [float(row["pixels_std0.08_success"]) for row in block]
        best = max(values)
        true_plateau = {i for i, value in enumerate(values) if best - value <= tolerance_pp + 1e-9}
        scores = score_fn(block)
        screened = set(sorted(range(len(block)), key=lambda i: scores[i], reverse=True)[:screen_size])
        all_indices = set(range(len(block)))
        block_tp = len(screened & true_plateau)
        block_fp = len(screened - true_plateau)
        block_fn = len(true_plateau - screened)
        block_tn = len(all_indices - screened - true_plateau)
        tp += block_tp
        fp += block_fp
        fn += block_fn
        tn += block_tn
        hit_count += block_tp > 0
        true_total += len(true_plateau)
        screened_total += len(screened)
        block_rows.append(
            {
                "task": block[0]["task"],
                "training_seed": int(block[0]["training_seed"]),
                "true_plateau_count": len(true_plateau),
                "screened_count": len(screened),
                "true_positive_count": block_tp,
                "false_positive_count": block_fp,
                "false_negative_count": block_fn,
                "true_negative_count": block_tn,
                "plateau_presence_hit": block_tp > 0,
                "screened_std_keys": [str(block[i]["std_key"]) for i in sorted(screened, key=lambda j: float(block[j]["std_key"]))],
            }
        )
    return {
        "rule": name,
        "n_blocks": len(block_rows),
        "screen_size_per_block": screen_size,
        "screened_rows": screened_total,
        "true_plateau_rows": true_total,
        "plateau_presence_hits": hit_count,
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "false_negative_rows": fn,
        "true_negative_rows": tn,
        "screen_precision": tp / (tp + fp) if tp + fp else None,
        "plateau_recall": tp / (tp + fn) if tp + fn else None,
        "block_rows": block_rows,
    }


def _random_tophalf_summary(rows: Sequence[dict], tolerance_pp: float, screen_size: int) -> dict:
    true_total = 0
    expected_hit = 0.0
    n_blocks = 0
    candidates_per_block = len(NONZERO_STD_KEYS)
    for block in _blocks(rows):
        values = [float(row["pixels_std0.08_success"]) for row in block]
        best = max(values)
        true_count = sum(1 for value in values if best - value <= tolerance_pp + 1e-9)
        true_total += true_count
        bad_count = candidates_per_block - true_count
        total_sets = math.comb(candidates_per_block, screen_size)
        miss_sets = math.comb(bad_count, screen_size) if bad_count >= screen_size else 0
        expected_hit += 1.0 - miss_sets / total_sets
        n_blocks += 1
    screened_total = n_blocks * screen_size
    expected_tp = screen_size / candidates_per_block * true_total
    expected_fp = screened_total - expected_tp
    expected_fn = true_total - expected_tp
    expected_tn = n_blocks * candidates_per_block - screened_total - expected_fn
    return {
        "rule": "Random top-half reference (exact expectation)",
        "n_blocks": n_blocks,
        "screen_size_per_block": screen_size,
        "screened_rows": screened_total,
        "true_plateau_rows": true_total,
        "plateau_presence_hits_expected": expected_hit,
        "true_positive_rows_expected": expected_tp,
        "false_positive_rows_expected": expected_fp,
        "false_negative_rows_expected": expected_fn,
        "true_negative_rows_expected": expected_tn,
        "screen_precision_expected": expected_tp / screened_total,
        "plateau_recall_expected": expected_tp / true_total if true_total else None,
    }


def build_audit(data: dict, tolerance_pp: float) -> dict:
    rows = _prepare_rows(data)
    screen_size = 4
    score_fns = {
        "Aggregate ACPC/PCC/CRA/MAF": _aggregate_scores,
        "ACPC only": _metric_scores("acpc_h_norm_by_transition", -1.0),
        "PCC only": _metric_scores("pcc_abs_median", -1.0),
        "CRA only": _metric_scores("cra_spearman_mean", 1.0),
        "MAF only": _metric_scores("maf_flip_rate", -1.0),
        "High-std top-half reference": _metric_scores("std_key", 1.0),
    }
    return {
        "metadata": {
            "schema_version": "paper1-plateau-membership-audit-20260704-v2",
            "source_artifact": str(DEFAULT_IN.relative_to(ROOT)),
            "row_unit": "task-training-seed block, nonzero Gaussian std candidates",
            "tolerance_pp": tolerance_pp,
            "screen_size_per_block": screen_size,
            "interpretation": "Plateau-membership screen; std_max is treated as a candidate label, not a continuous covariate or point-optimal selector target.",
        },
        "membership_summaries": [
            _membership_summary(rows, name=name, score_fn=fn, tolerance_pp=tolerance_pp, screen_size=screen_size)
            for name, fn in score_fns.items()
        ]
        + [_random_tophalf_summary(rows, tolerance_pp, screen_size)],
    }


def write_markdown(audit: dict, path: Path) -> None:
    meta = audit["metadata"]
    lines = [
        "# Plateau membership audit",
        "",
        f"- Tolerance: {meta['tolerance_pp']:.1f} pp from the block's closed-loop best.",
        f"- Screen: top {meta['screen_size_per_block']} of 8 nonzero Gaussian candidates per task-training-seed block.",
        "- Reading: plateau-entry enrichment, not point-optimal checkpoint ranking.",
        "",
        "| Rule | presence hit | screen precision | plateau recall | TP/FP/FN |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audit["membership_summaries"]:
        if "plateau_presence_hits" in row:
            hit = f"{row['plateau_presence_hits']}/{row['n_blocks']}"
            precision = row["screen_precision"]
            recall = row["plateau_recall"]
            counts = f"{row['true_positive_rows']}/{row['false_positive_rows']}/{row['false_negative_rows']}"
        else:
            hit = f"{row['plateau_presence_hits_expected']:.2f}/{row['n_blocks']}"
            precision = row["screen_precision_expected"]
            recall = row["plateau_recall_expected"]
            counts = (
                f"{row['true_positive_rows_expected']:.1f}/"
                f"{row['false_positive_rows_expected']:.1f}/"
                f"{row['false_negative_rows_expected']:.1f}"
            )
        lines.append(f"| {row['rule']} | {hit} | {100.0 * precision:.1f}% | {100.0 * recall:.1f}% | {counts} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--tolerance-pp", type=float, default=5.0)
    args = parser.parse_args()

    audit = build_audit(_load(args.input), args.tolerance_pp)
    args.out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_markdown(audit, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
