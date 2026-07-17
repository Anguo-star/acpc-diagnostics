"""Training-seed Gaussian lockbox summary for Paper 1.

This script builds a structured release artifact from the completed LeWM
Gaussian sweep across three training seeds. Seed 3072 is the canonical Paper1
sweep stored as JSON; seeds 3073 and 3074 are parsed from the development
lockbox note after their full Gaussian sweeps were completed. The script does
not load models, checkpoints, or datasets.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "paper1_data"
CANONICAL_EVALS = DATA_DIR / "canonical_evals_20260517.json"
LOCKBOX_NOTE = ROOT / "paper1" / "LOCKBOX_RESULTS_20260703.md"
DEFAULT_OUT_JSON = DATA_DIR / "training_seed_gaussian_lockbox.json"
DEFAULT_OUT_MD = DATA_DIR / "training_seed_gaussian_lockbox.md"
TASKS = ["TwoRoom", "PushT", "Reacher", "Cube"]
CANONICAL_TRAINING_SEED = 3072
EVAL_ENDPOINT = "pixels_std0.08"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pstdev(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _parse_float(text: str) -> float:
    return float(text.strip().lstrip("+"))


def _canonical_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows: list[dict] = []
    for task in TASKS:
        task_rows = data[task]

        def px08(std_key: str) -> float:
            return float(task_rows[std_key]["metrics"][EVAL_ENDPOINT]["mean"])

        base_key = "0.0"
        best_std = max(task_rows, key=lambda key: px08(key))
        baseline = px08(base_key)
        best = px08(best_std)
        std08 = px08("0.08")
        rows.append(
            {
                "task": task,
                "training_seed": CANONICAL_TRAINING_SEED,
                "baseline_obs_0p08": baseline,
                "best_obs_0p08": best,
                "best_std": best_std,
                "std_0p08_obs_0p08": std08,
                "std_0p08_gain_over_baseline": std08 - baseline,
                "std_0p08_regret_to_best": best - std08,
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def _lockbox_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cols = [col.strip() for col in line.strip().strip("|").split("|")]
        if len(cols) != 7 or cols[0] not in TASKS or not cols[1].isdigit():
            continue
        try:
            best_std_value = float(cols[4])
        except ValueError:
            continue
        if not 0.0 <= best_std_value <= 0.08:
            continue
        baseline = _parse_float(cols[2])
        best = _parse_float(cols[3])
        std08 = _parse_float(cols[5])
        rows.append(
            {
                "task": cols[0],
                "training_seed": int(cols[1]),
                "baseline_obs_0p08": baseline,
                "best_obs_0p08": best,
                "best_std": cols[4],
                "std_0p08_obs_0p08": std08,
                "std_0p08_gain_over_baseline": _parse_float(cols[6]),
                "std_0p08_regret_to_best": best - std08,
                "source": str(path.relative_to(ROOT)),
            }
        )
    expected = {(task, seed) for task in TASKS for seed in (3073, 3074)}
    found = {(row["task"], row["training_seed"]) for row in rows}
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Missing lockbox rows: {missing}")
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    summaries: list[dict] = []
    for task in TASKS:
        task_rows = sorted(by_task[task], key=lambda row: row["training_seed"])
        seeds = [int(row["training_seed"]) for row in task_rows]
        if seeds != [3072, 3073, 3074]:
            raise ValueError(f"{task} has unexpected training seeds: {seeds}")
        summary = {"task": task, "training_seeds": seeds, "n_training_seeds": len(seeds)}
        for key in (
            "baseline_obs_0p08",
            "std_0p08_obs_0p08",
            "std_0p08_gain_over_baseline",
            "best_obs_0p08",
            "std_0p08_regret_to_best",
        ):
            values = [float(row[key]) for row in task_rows]
            summary[f"{key}_mean"] = _mean(values)
            summary[f"{key}_pstdev"] = _pstdev(values)
        summary["best_std_values"] = [row["best_std"] for row in task_rows]
        summary["best_std_range"] = f"{min(summary['best_std_values'])}--{max(summary['best_std_values'])}"
        summaries.append(summary)
    return summaries


def build_payload() -> dict:
    rows = _canonical_rows(CANONICAL_EVALS) + _lockbox_rows(LOCKBOX_NOTE)
    rows = sorted(rows, key=lambda row: (TASKS.index(row["task"]), row["training_seed"]))
    return {
        "metadata": {
            "schema_version": "paper1-training-seed-gaussian-lockbox-0.1",
            "scope": "LeWM Gaussian sweep, observation-only Gaussian endpoint pixels_std0.08, training seeds 3072/3073/3074; each point is the three-evaluation-seed mean from seeds 42/43/44 with 100 episodes per eval seed",
            "source_artifacts": [
                str(CANONICAL_EVALS.relative_to(ROOT)),
                str(LOCKBOX_NOTE.relative_to(ROOT)),
            ],
            "note": "Seed 3072 is computed from the canonical JSON; seeds 3073/3074 are parsed from the completed lockbox note.",
        },
        "per_seed_rows": rows,
        "task_summary_rows": _summarize(rows),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Training-Seed Gaussian Lockbox",
        "",
        "This artifact summarizes the completed LeWM Gaussian sweep over three training seeds: canonical seed 3072 plus lockbox seeds 3073 and 3074.",
        "Each point is the observation-only Gaussian endpoint `pixels_std0.08` mean over eval seeds 42/43/44 with 100 episodes per eval seed.",
        "",
        "## Three-seed task summary",
        "",
        "| Task | base obs0.08 | std0.08 obs0.08 | gain | best obs0.08 | std0.08 regret | best std range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["task_summary_rows"]:
        lines.append(
            "| {task} | {base} +/- {base_sd} | {std08} +/- {std08_sd} | {gain} +/- {gain_sd} | {best} +/- {best_sd} | {regret} +/- {regret_sd} | {best_range} |".format(
                task=row["task"],
                base=_fmt_float(row["baseline_obs_0p08_mean"]),
                base_sd=_fmt_float(row["baseline_obs_0p08_pstdev"]),
                std08=_fmt_float(row["std_0p08_obs_0p08_mean"]),
                std08_sd=_fmt_float(row["std_0p08_obs_0p08_pstdev"]),
                gain=_fmt_float(row["std_0p08_gain_over_baseline_mean"]),
                gain_sd=_fmt_float(row["std_0p08_gain_over_baseline_pstdev"]),
                best=_fmt_float(row["best_obs_0p08_mean"]),
                best_sd=_fmt_float(row["best_obs_0p08_pstdev"]),
                regret=_fmt_float(row["std_0p08_regret_to_best_mean"]),
                regret_sd=_fmt_float(row["std_0p08_regret_to_best_pstdev"]),
                best_range=row["best_std_range"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-seed rows",
            "",
            "| Task | seed | baseline obs0.08 | best obs0.08 | best std | std0.08 obs0.08 | std0.08 gain | regret |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["per_seed_rows"]:
        lines.append(
            "| {task} | {seed} | {base} | {best} | {best_std} | {std08} | {gain} | {regret} |".format(
                task=row["task"],
                seed=row["training_seed"],
                base=_fmt_float(row["baseline_obs_0p08"]),
                best=_fmt_float(row["best_obs_0p08"]),
                best_std=row["best_std"],
                std08=_fmt_float(row["std_0p08_obs_0p08"]),
                gain=_fmt_float(row["std_0p08_gain_over_baseline"]),
                regret=_fmt_float(row["std_0p08_regret_to_best"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)
    payload = build_payload()
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json.relative_to(ROOT)}")
    print(f"wrote {args.out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
