#!/usr/bin/env python3
"""Merge resumable fixed-pool shards and report sharp-certificate risk coverage."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_sample_level_certificate as certificate


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
RHO_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
NESTED_FIELDS = {"model_search_dirs", "epsilon_tail_rows", "coverage_by_K", "risk_coverage"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    _require(isinstance(payload, dict), f"{path}: top-level JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _block_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    coarse_pass = sum(_bool(row["coarse_cert_pass"]) for row in samples)
    sharp_pass = sum(_bool(row["sharp_cert_pass"]) for row in samples)
    flips = sum(_bool(row["top1_flip"]) for row in samples)
    flips_when_fail = sum(
        _bool(row["top1_flip"]) and not _bool(row["sharp_cert_pass"])
        for row in samples
    )
    fail_n = total - sharp_pass
    lower = certificate._wilson_lower_one_sided95(sharp_pass, total)
    return {
        "n": total,
        "coarse_cert_pass_n": coarse_pass,
        "coarse_cert_pass_rate": coarse_pass / total if total else None,
        "sharp_cert_pass_n": sharp_pass,
        "sharp_cert_pass_rate": sharp_pass / total if total else None,
        "sharp_cert_pass_lower95_wilson": lower if total else None,
        "flip_risk_upper95_from_sharp_coverage": 1.0 - lower if total else None,
        "observed_flip_n": flips,
        "observed_flip_rate": flips / total if total else None,
        "flip_when_sharp_cert_fail_rate": (
            flips_when_fail / fail_n if fail_n else None
        ),
        "sharp_cert_invariant_flip_count": sum(
            _bool(row["top1_flip"]) and _bool(row["sharp_cert_pass"])
            for row in samples
        ),
    }


def _block_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = len(samples)
    sharp_pass = sum(_bool(row["sharp_cert_pass"]) for row in samples)
    return {
        "n": total,
        "coarse_pass": sum(_bool(row["coarse_cert_pass"]) for row in samples),
        "sharp_pass": sharp_pass,
        "flip": sum(_bool(row["top1_flip"]) for row in samples),
        "sharp_fail": total - sharp_pass,
        "flip_when_sharp_fail": sum(
            _bool(row["top1_flip"]) and not _bool(row["sharp_cert_pass"])
            for row in samples
        ),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot take percentile of empty values")
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _block_bootstrap(
    blocks: Sequence[Mapping[str, int]],
    *,
    repetitions: int = 2000,
    seed: int = 9101,
) -> dict[str, Any]:
    _require(len(blocks) >= 2, "block bootstrap requires at least two blocks")
    _require(repetitions >= 1, "block bootstrap repetitions must be positive")
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {
        "coarse_cert_pass_rate": [],
        "sharp_cert_pass_rate": [],
        "observed_flip_rate": [],
        "flip_when_sharp_cert_fail_rate": [],
    }
    for _ in range(repetitions):
        selected = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        totals = {
            key: sum(int(block[key]) for block in selected)
            for key in (
                "n",
                "coarse_pass",
                "sharp_pass",
                "flip",
                "sharp_fail",
                "flip_when_sharp_fail",
            )
        }
        draws["coarse_cert_pass_rate"].append(totals["coarse_pass"] / totals["n"])
        draws["sharp_cert_pass_rate"].append(totals["sharp_pass"] / totals["n"])
        draws["observed_flip_rate"].append(totals["flip"] / totals["n"])
        draws["flip_when_sharp_cert_fail_rate"].append(
            totals["flip_when_sharp_fail"] / totals["sharp_fail"]
            if totals["sharp_fail"]
            else 0.0
        )
    return {
        "unit": "task_x_training_seed block",
        "block_count": len(blocks),
        "repetitions": repetitions,
        "seed": seed,
        "interval": "percentile_95",
        "metrics": {
            metric: {
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
            }
            for metric, values in draws.items()
        },
    }


def build_artifacts(inputs: Sequence[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _require(len(inputs) == 12, "exactly 12 task-seed shards are required")
    checkpoint_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    k_rows: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    source_metadata: dict[str, Any] = {}
    blocks: set[tuple[str, int]] = set()
    for path in inputs:
        _require(path.is_file(), f"missing certificate shard: {path}")
        payload = _load(path)
        metadata = payload.get("metadata", {})
        _require(metadata.get("schema_version") == certificate.SCHEMA_VERSION, f"{path}: schema mismatch")
        _require(metadata.get("status") == "complete", f"{path}: shard incomplete")
        _require(metadata.get("status_counts") == {"ok": 9}, f"{path}: row failure")
        _require(metadata.get("missing_rows") == [], f"{path}: missing rows")
        _require(metadata.get("errors") == [], f"{path}: errors")
        rows = payload.get("rows", [])
        samples = payload.get("sample_rows", [])
        _require(len(rows) == 9 and len(samples) == 900, f"{path}: shard count mismatch")
        tasks = {str(row.get("task")) for row in rows}
        seeds = {int(row.get("training_seed")) for row in rows}
        _require(len(tasks) == 1 and len(seeds) == 1, f"{path}: mixed shard block")
        task = next(iter(tasks))
        seed = next(iter(seeds))
        block = (task, seed)
        _require(task in TASKS and seed in SEEDS and block not in blocks, f"{path}: duplicate/unknown block")
        blocks.add(block)
        _require({str(row.get("std_key")) for row in rows} == set(RHO_KEYS), f"{path}: rho coverage mismatch")
        source_key = f"{task}_seed{seed}"
        source_paths[f"shard_{source_key}"] = str(path)
        source_hashes[f"shard_{source_key}"] = _sha256(path)
        source_metadata[source_key] = metadata
        for raw in rows:
            row = {key: value for key, value in raw.items() if key not in NESTED_FIELDS}
            model_file = Path(str(row.get("model_file"))).expanduser().resolve()
            _require(model_file.is_file(), f"{path}: checkpoint missing")
            checkpoint_hash = _sha256(model_file)
            row["model_file"] = str(model_file)
            row["checkpoint_sha256"] = checkpoint_hash
            checkpoint_key = f"checkpoint_{task}_seed{seed}_{raw['std_key']}"
            source_paths[checkpoint_key] = str(model_file)
            source_hashes[checkpoint_key] = checkpoint_hash
            _require(raw.get("sharp_cert_invariant_flip_count") == 0, f"{path}: sharp invariant violated")
            for k, values in raw.get("coverage_by_K", {}).items():
                k_rows.append(
                    {
                        "training_seed": seed,
                        "task": task,
                        "std_key": raw["std_key"],
                        **values,
                    }
                )
            checkpoint_rows.append(row)
        sample_rows.extend(dict(row) for row in samples)
    _require(
        blocks == {(task, seed) for task in TASKS for seed in SEEDS},
        "certificate task-seed coverage mismatch",
    )
    _require(len(checkpoint_rows) == 108 and len(sample_rows) == 10800, "full certificate count mismatch")
    _require(
        not any(
            _bool(row["top1_flip"]) and _bool(row["sharp_cert_pass"])
            for row in sample_rows
        ),
        "sharp certificate invariant violated in merged samples",
    )
    slacks = torch.tensor(
        [float(row["sharp_cert_slack"]) for row in sample_rows],
        dtype=torch.float64,
    )
    flips = torch.tensor(
        [_bool(row["top1_flip"]) for row in sample_rows],
        dtype=torch.bool,
    )
    global_risk = certificate._risk_coverage_rows(slacks, flips)
    risk_rows = [
        {"scope": "all_108_checkpoints", **row}
        for row in global_risk
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(str(row["task"]), int(row["training_seed"]))].append(row)
    block_summaries = [
        {
            "task": task,
            "training_seed": seed,
            **_block_summary(rows),
        }
        for (task, seed), rows in sorted(grouped.items())
    ]
    block_bootstrap = _block_bootstrap(
        [_block_counts(rows) for _, rows in sorted(grouped.items())]
    )
    script_path = Path(__file__).resolve()
    runner_path = ROOT / "tools/paper1_sample_level_certificate.py"
    source_paths["merge_builder"] = str(script_path)
    source_hashes["merge_builder"] = _sha256(script_path)
    source_paths["certificate_runner"] = str(runner_path)
    source_hashes["certificate_runner"] = _sha256(runner_path)
    overall = _block_summary(sample_rows)
    summary = {
        "metadata": {
            "schema_version": "paper1-fixed-pool-sharp-certificate-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "status": "complete",
            "status_counts": {"ok": 108},
            "missing_rows": [],
            "errors": [],
            "tasks": list(TASKS),
            "training_seeds": list(SEEDS),
            "training_seed_semantics": "independently trained checkpoint seeds",
            "n_sequences_per_checkpoint": 100,
            "candidate_count": 65,
            "k_values": list(certificate.K_VALUES),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "source_metadata": source_metadata,
            "claim_scope": (
                "deterministic fixed-pool implication; flip|cert-pass=0 is an "
                "invariant check, not independent statistical evidence"
            ),
            "adaptive_cem_or_closed_loop_guarantee": False,
        },
        "overall": overall,
        "block_summaries": block_summaries,
        "hierarchical_block_bootstrap": block_bootstrap,
        "risk_coverage": risk_rows,
        "k_sensitivity_rows": len(k_rows),
        "count_contract": {
            "checkpoint_rows": len(checkpoint_rows),
            "sample_rows": len(sample_rows),
            "risk_coverage_rows": len(risk_rows),
            "k_sensitivity_rows": len(k_rows),
        },
    }
    return summary, checkpoint_rows, risk_rows, k_rows


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Sharp fixed-pool certificate coverage and observed risk by task--training-seed block.}",
        r"\label{tab:fixed-pool-certificate-coverage}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Block & Coarse cov. & Sharp cov. & Flip rate & Flip $\mid$ sharp fail \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['task']}--{int(row['training_seed'])} & "
            f"{float(row['coarse_cert_pass_rate']):.3f} & "
            f"{float(row['sharp_cert_pass_rate']):.3f} & "
            f"{float(row['observed_flip_rate']):.3f} & "
            f"{float(row['flip_when_sharp_cert_fail_rate']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = summary["overall"]
    intervals = summary["hierarchical_block_bootstrap"]["metrics"]
    panels = (
        (
            "(a) Certificate coverage",
            ("Common drift", "Candidate-wise"),
            ("coarse_cert_pass_rate", "sharp_cert_pass_rate"),
        ),
        (
            "(b) Observed top-1 flips",
            ("All histories", "Condition fails"),
            ("observed_flip_rate", "flip_when_sharp_cert_fail_rate"),
        ),
    )
    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), sharey=True)
        for axis, (title, labels, keys) in zip(axes, panels):
            values = [float(overall[key]) for key in keys]
            errors = [
                [value - float(intervals[key]["lower"]) for value, key in zip(values, keys)],
                [float(intervals[key]["upper"]) - value for value, key in zip(values, keys)],
            ]
            bars = axis.bar(
                range(len(keys)),
                values,
                color=("#4477AA", "#EE6677"),
                edgecolor="#333333",
                linewidth=0.55,
                yerr=errors,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
                capsize=3,
                width=0.62,
                zorder=2,
            )
            for bar, hatch in zip(bars, ("", "///")):
                bar.set_hatch(hatch)
            axis.set_xticks(range(len(keys)), labels)
            axis.set_ylim(0.0, 1.0)
            axis.set_title(title, loc="left", fontweight="semibold")
            axis.grid(axis="y", color="#B0B0B0", alpha=0.28, linewidth=0.6, zorder=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            for index, value in enumerate(values):
                axis.text(index, min(value + 0.05, 0.94), f"{value:.3f}", ha="center", fontsize=7.5)
        axes[0].set_ylabel("Rate")
        figure.tight_layout(pad=0.5, w_pad=1.1)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=ROOT / "paper1/results/fixed_pool_candidatewise_certificate_summary.json",
    )
    parser.add_argument(
        "--out-checkpoints",
        type=Path,
        default=ROOT / "paper1/results/fixed_pool_candidatewise_certificate.csv",
    )
    parser.add_argument(
        "--out-risk-coverage",
        type=Path,
        default=ROOT / "paper1/results/fixed_pool_risk_coverage.csv",
    )
    parser.add_argument(
        "--out-k-sensitivity",
        type=Path,
        default=ROOT / "paper1/results/fixed_pool_k_sensitivity.csv",
    )
    parser.add_argument(
        "--out-blocks",
        type=Path,
        default=ROOT / "paper1/results/fixed_pool_certificate_coverage_by_block.csv",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper1/tables/table_fixed_pool_certificate_coverage.tex",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "assets/paper1_figs/fig_fixed_pool_certificate_calibration.png",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary, checkpoints, risk_rows, k_rows = build_artifacts(args.input)
    _write_csv(args.out_checkpoints, checkpoints)
    _write_csv(args.out_risk_coverage, risk_rows)
    _write_csv(args.out_k_sensitivity, k_rows)
    _write_csv(args.out_blocks, summary["block_summaries"])
    _write_table(args.table, summary["block_summaries"])
    _plot(args.figure, summary)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(checkpoints)} checkpoints, "
        f"{len(risk_rows)} risk rows, {len(k_rows)} K rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
