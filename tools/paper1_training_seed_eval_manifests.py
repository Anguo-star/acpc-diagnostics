"""Build LeWM three-training-seed eval manifests for Paper 1 diagnostics.

The Phase-0 ACPC runner consumes canonical-eval-shaped JSON. Seed 3072 uses the
released canonical JSON for audited behavior metrics; seeds 3073/3074 use the
completed lockbox checkpoints' ``eval_results/eval_summary.csv`` files. The
output manifests keep the canonical task/std layout while pointing at local
checkpoint directories and normalizing ``origin`` to ``clean``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
CANONICAL_3072 = DATA_DIR / "canonical_evals_20260517.json"


def _default_data_root() -> Path | None:
    for name in ("PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


DEFAULT_DATA_ROOT = _default_data_root()
DEFAULT_OUT_DIR = DATA_DIR / "training_seed_eval_manifests"
SEEDS = (3072, 3073, 3074)
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
TASKS = {
    "TwoRoom": {"folder": "lewm-tworooms", "slug": "tworoom"},
    "PushT": {"folder": "lewm-pusht", "slug": "pusht"},
    "Reacher": {"folder": "lewm-reacher", "slug": "reacher"},
    "Cube": {"folder": "lewm-cube", "slug": "cube"},
}
REQUIRED_METRICS = ("clean", "pixels_std0.08", "pixels_goal_std0.08")


def _std_suffix(std_key: str) -> str:
    return f"{int(round(float(std_key) * 100)):03d}"


def _subdir(task: str, std_key: str, seed: int) -> str:
    slug = TASKS[task]["slug"]
    if seed == 3072:
        return f"{slug}_lewm_20260430" if std_key == "0.0" else f"{slug}_lewm_noise_0to{_std_suffix(std_key)}_p1"
    return f"{slug}_lewm_baseline_seed{seed}" if std_key == "0.0" else f"{slug}_lewm_noise_0to{_std_suffix(std_key)}_p1_seed{seed}"


def _run_path(data_root: Path, task: str, subdir: str) -> Path:
    return data_root / TASKS[task]["folder"] / "ckpt" / subdir


def _parse_values(raw: str) -> list[float]:
    if not raw:
        return []
    return [float(part) for part in raw.split(";") if part]


def _read_eval_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    metrics: dict[str, Any] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") != "success_rate":
                continue
            group = str(row["group"])
            metrics[group] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "sem": float(row["sem"]),
                "n": int(row["n_seeds"]),
                "seeds": [int(s) for s in row["seeds"].split(",") if s and s != "-"],
                "values": _parse_values(row.get("values", "")),
            }
    if "origin" in metrics and "clean" not in metrics:
        metrics["clean"] = dict(metrics["origin"])
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"{path} missing metrics: {missing}")
    return metrics


def _seed3072_manifest(data_root: Path) -> dict[str, Any]:
    data = json.loads(CANONICAL_3072.read_text())
    for task in TASKS:
        for std_key, entry in data[task].items():
            subdir = _subdir(task, std_key, 3072)
            entry["subdir"] = subdir
            entry["path"] = str(_run_path(data_root, task, subdir))
            if "origin" not in entry["metrics"] and "clean" in entry["metrics"]:
                entry["metrics"]["origin"] = dict(entry["metrics"]["clean"])
    return data


def _lockbox_manifest(data_root: Path, seed: int) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for task in TASKS:
        data[task] = {}
        for std_key in STD_KEYS:
            subdir = _subdir(task, std_key, seed)
            run_path = _run_path(data_root, task, subdir)
            data[task][std_key] = {
                "subdir": subdir,
                "path": str(run_path),
                "metrics": _read_eval_summary(run_path / "eval_results" / "eval_summary.csv"),
            }
    return data


def _validate(data: dict[str, Any], seed: int) -> None:
    count = 0
    for task in TASKS:
        keys = set(data.get(task, {}))
        if keys != set(STD_KEYS):
            raise ValueError(f"seed {seed} task {task} std coverage mismatch: {sorted(keys)}")
        for std_key, entry in data[task].items():
            count += 1
            for metric in REQUIRED_METRICS:
                if metric not in entry.get("metrics", {}):
                    raise ValueError(f"seed {seed} {task}/{std_key} missing {metric}")
            ckpt = Path(entry["path"]) / f"{entry['subdir']}_epoch_10_object.ckpt"
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
    if count != 36:
        raise ValueError(f"seed {seed} expected 36 entries, found {count}")


def build(seed: int, data_root: Path) -> dict[str, Any]:
    if seed == 3072:
        data = _seed3072_manifest(data_root)
    elif seed in (3073, 3074):
        data = _lockbox_manifest(data_root, seed)
    else:
        raise ValueError(f"unsupported seed: {seed}")
    data["_metadata"] = {
        "schema_version": "paper1-training-seed-eval-manifest-0.1",
        "training_seed": seed,
        "source": "canonical_evals_20260517.json" if seed == 3072 else "checkpoint eval_results/eval_summary.csv",
        "data_root": str(data_root),
        "tasks": list(TASKS),
        "std_keys": list(STD_KEYS),
        "required_metrics": list(REQUIRED_METRICS),
    }
    _validate(data, seed)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.data_root is None:
        parser.error(
            "--data-root is required unless PAPER1_DATA_ROOT, DATA_ROOT, "
            "or STABLEWM_HOME is set"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        payload = build(seed, args.data_root)
        out = args.out_dir / f"lewm_seed{seed}_evals.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
