#!/usr/bin/env python3
"""Build the Paper1 three-training-seed Gaussian sweep artifact.

Each plotted or tabulated training-seed point first averages evaluation seeds
42/43/44. Summary rows then report mean and population std across LeWM training
seeds 3072/3073/3074. This script does not load models or datasets.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_OUT_JSON = DATA_DIR / "three_seed_gaussian_sweep_summary_20260706.json"
DEFAULT_OUT_MD = DATA_DIR / "three_seed_gaussian_sweep_summary_20260706.md"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
TRAINING_SEEDS = (3072, 3073, 3074)
SWEEP_STDS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
ARTIFACT_CREATED_UTC = "2026-07-06T00:00:00+00:00"

METRIC_KEYS = {
    "clean": "clean",
    "obs_sigma_0.03": "pixels_std0.03",
    "obs_sigma_0.05": "pixels_std0.05",
    "obs_sigma_0.08": "pixels_std0.08",
    "obs_goal_sigma_0.08": "pixels_goal_std0.08",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_mean(entry: dict[str, Any], source_key: str) -> float:
    metric = entry["metrics"].get(source_key)
    if metric is None:
        raise KeyError(f"missing metric {source_key}")
    return float(metric["mean"])


def _metric_values(entry: dict[str, Any], source_key: str) -> list[float]:
    metric = entry["metrics"].get(source_key)
    if metric is None:
        raise KeyError(f"missing metric {source_key}")
    return [float(value) for value in metric.get("values", [])]


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _fmt_pm(metric: dict[str, Any]) -> str:
    return f"{float(metric['mean']):.2f} +/- {float(metric['pstdev']):.2f}"


def build_payload() -> dict[str, Any]:
    manifests = {
        seed: _load_json(DATA_DIR / "training_seed_eval_manifests" / f"lewm_seed{seed}_evals.json")
        for seed in TRAINING_SEEDS
    }
    per_seed_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for std_key in SWEEP_STDS:
            for seed in TRAINING_SEEDS:
                entry = manifests[seed][task][std_key]
                row: dict[str, Any] = {
                    "task": task,
                    "stdmax": std_key,
                    "training_seed": seed,
                    "source": f"assets/paper1_data/training_seed_eval_manifests/lewm_seed{seed}_evals.json",
                    "metrics": {},
                }
                for out_key, source_key in METRIC_KEYS.items():
                    row["metrics"][out_key] = {
                        "mean_over_eval_seeds": _metric_mean(entry, source_key),
                        "eval_seed_values": _metric_values(entry, source_key),
                    }
                per_seed_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for std_key in SWEEP_STDS:
            rows = [
                row for row in per_seed_rows
                if row["task"] == task and row["stdmax"] == std_key
            ]
            if [row["training_seed"] for row in rows] != list(TRAINING_SEEDS):
                raise ValueError(f"unexpected seed coverage for {task}/{std_key}")
            summary: dict[str, Any] = {
                "task": task,
                "stdmax": std_key,
                "training_seeds": list(TRAINING_SEEDS),
                "n_training_seeds": len(TRAINING_SEEDS),
                "metrics": {},
            }
            for out_key in METRIC_KEYS:
                values = [float(row["metrics"][out_key]["mean_over_eval_seeds"]) for row in rows]
                summary["metrics"][out_key] = {
                    "mean": statistics.fmean(values),
                    "pstdev": statistics.pstdev(values),
                    "per_training_seed_means": values,
                }
            summary_rows.append(summary)

    return {
        "metadata": {
            "schema_version": "paper1-three-seed-gaussian-sweep-summary-20260706-v1",
            "created_utc": ARTIFACT_CREATED_UTC,
            "scope": (
                "LeWM Gaussian training-noise sweep over stdmax 0.0..0.08 for tasks "
                "TwoRoom/PushT/Reacher/Cube. Each training-seed point averages "
                "evaluation seeds 42/43/44 with 100 trajectories per eval seed; "
                "summary rows report mean and population std across training seeds "
                "3072/3073/3074."
            ),
            "metric_keys": METRIC_KEYS,
            "tasks": list(TASKS),
            "training_seeds": list(TRAINING_SEEDS),
            "sweep_stdmax": list(SWEEP_STDS),
            "source_artifacts": [
                f"assets/paper1_data/training_seed_eval_manifests/lewm_seed{seed}_evals.json"
                for seed in TRAINING_SEEDS
            ],
        },
        "summary_rows": summary_rows,
        "per_seed_rows": per_seed_rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["summary_rows"]
    lines = [
        "# Three-Training-Seed Gaussian Sweep Summary",
        "",
        "Reading: this artifact is the exact source for the three-seed main sweep figure and appendix Gaussian tables. Each training-seed value first averages evaluation seeds 42/43/44; table cells below report mean +/- population std across training seeds 3072/3073/3074.",
        "",
        "| Task | stdmax | clean | obs sigma=0.03 | obs sigma=0.05 | obs sigma=0.08 | obs+goal sigma=0.08 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {task} | {stdmax} | {clean} | {obs03} | {obs05} | {obs08} | {obsgoal08} |".format(
                task=row["task"],
                stdmax=row["stdmax"],
                clean=_fmt_pm(metrics["clean"]),
                obs03=_fmt_pm(metrics["obs_sigma_0.03"]),
                obs05=_fmt_pm(metrics["obs_sigma_0.05"]),
                obs08=_fmt_pm(metrics["obs_sigma_0.08"]),
                obsgoal08=_fmt_pm(metrics["obs_goal_sigma_0.08"]),
            )
        )

    lines.extend([
        "",
        "## Per-training-seed means used before cross-seed aggregation",
        "",
        "| Task | stdmax | seed | clean | obs sigma=0.08 | obs+goal sigma=0.08 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["per_seed_rows"]:
        metrics = row["metrics"]
        lines.append(
            "| {task} | {stdmax} | {seed} | {clean} | {obs08} | {obsgoal08} |".format(
                task=row["task"],
                stdmax=row["stdmax"],
                seed=row["training_seed"],
                clean=_fmt(metrics["clean"]["mean_over_eval_seeds"]),
                obs08=_fmt(metrics["obs_sigma_0.08"]["mean_over_eval_seeds"]),
                obsgoal08=_fmt(metrics["obs_goal_sigma_0.08"]["mean_over_eval_seeds"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)
    payload = build_payload()
    write_json(args.out_json, payload)
    write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
