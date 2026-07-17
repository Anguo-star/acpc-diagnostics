#!/usr/bin/env python3
"""Aggregate the Paper1 LeWM-base multi-std noise-cliff table."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_OUT_JSON = DATA_DIR / "base_noise_cliff_multistd_20260706.json"
DEFAULT_OUT_MD = DATA_DIR / "base_noise_cliff_multistd_20260706.md"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
TRAINING_SEEDS = (3072, 3073, 3074)
BASE_STD_KEY = "0.0"
METRIC_KEYS = {
    "eval_sigma_0": "clean",
    "eval_sigma_0.03": "pixels_std0.03",
    "eval_sigma_0.05": "pixels_std0.05",
    "eval_sigma_0.08": "pixels_std0.08",
    "eval_sigma_0.08_obs_goal": "pixels_goal_std0.08",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(mean: float, std: float) -> str:
    return f"${mean:.2f} \\pm {std:.2f}$"


def build_payload() -> dict[str, Any]:
    manifests = {
        seed: _load_json(DATA_DIR / "training_seed_eval_manifests" / f"lewm_seed{seed}_evals.json")
        for seed in TRAINING_SEEDS
    }
    per_seed_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for seed in TRAINING_SEEDS:
            entry = manifests[seed][task][BASE_STD_KEY]
            metrics = entry["metrics"]
            missing = [source for source in METRIC_KEYS.values() if source not in metrics]
            if missing:
                raise KeyError(f"{task}/seed{seed}/{BASE_STD_KEY} missing metrics: {missing}")
            row = {
                "task": task,
                "training_seed": seed,
                "source": f"assets/paper1_data/training_seed_eval_manifests/lewm_seed{seed}_evals.json",
            }
            for out_key, source_key in METRIC_KEYS.items():
                row[out_key] = float(metrics[source_key]["mean"])
            per_seed_rows.append(row)

    task_summary_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [row for row in per_seed_rows if row["task"] == task]
        summary: dict[str, Any] = {
            "task": task,
            "training_seeds": list(TRAINING_SEEDS),
            "n_training_seeds": len(TRAINING_SEEDS),
        }
        for out_key in METRIC_KEYS:
            values = [float(row[out_key]) for row in task_rows]
            summary[out_key] = {
                "mean": statistics.fmean(values),
                "pstdev": statistics.pstdev(values),
                "values": values,
            }
        clean = summary["eval_sigma_0"]["values"]
        obs008 = summary["eval_sigma_0.08"]["values"]
        losses = [a - b for a, b in zip(clean, obs008)]
        summary["eval_sigma_0_to_0.08_loss"] = {
            "mean": statistics.fmean(losses),
            "pstdev": statistics.pstdev(losses),
            "values": losses,
        }
        task_summary_rows.append(summary)

    return {
        "metadata": {
            "schema_version": "paper1-base-noise-cliff-multistd-20260706-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "LeWM-base, std_key=0.0, observation-only Gaussian evaluation noise "
                "at sigma 0/0.03/0.05/0.08 across training seeds 3072/3073/3074; "
                "each training-seed value is the mean over evaluation seeds 42/43/44."
            ),
            "metric_keys": METRIC_KEYS,
            "tasks": list(TASKS),
            "training_seeds": list(TRAINING_SEEDS),
        },
        "per_seed_rows": per_seed_rows,
        "task_summary_rows": task_summary_rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LeWM-base Multi-Std Observation Noise Cliff",
        "",
        "Reading: this is the exact table source for the main noise-cliff table. "
        "The main endpoint perturbs observation pixels only and keeps the goal image clean.",
        "",
        "| Task | eval sigma=0 | eval sigma=0.03 | eval sigma=0.05 | eval sigma=0.08 | obs+goal sigma=0.08 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["task_summary_rows"]:
        cells = []
        for key in (
            "eval_sigma_0",
            "eval_sigma_0.03",
            "eval_sigma_0.05",
            "eval_sigma_0.08",
            "eval_sigma_0.08_obs_goal",
        ):
            block = row[key]
            cells.append(_fmt(float(block["mean"]), float(block["pstdev"])))
        lines.append(f"| {row['task']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The obs+goal column is auxiliary appendix evidence; it is not a main-table endpoint.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    payload = build_payload()
    write_json(args.out_json, payload)
    write_markdown(args.out_md, payload)
    print(args.out_json.relative_to(ROOT))
    print(args.out_md.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
