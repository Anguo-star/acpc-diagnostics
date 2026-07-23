#!/usr/bin/env python3
"""Join frozen PLDM predictions with behavior for three training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHOS = tuple(index / 100 for index in range(9))
TRAINING_SEEDS = (3072, 3073, 3074)
EVALUATION_SEEDS = (42, 43, 44)
FIELDS = (
    "model_family",
    "training_family_id",
    "training_seed",
    "task",
    "training_rho",
    "rho",
    "stressor_family",
    "stressor_severity",
    "atr_horizon_v2_q90",
    "ir_raw_q90",
    "ir_relative_q90",
    "smpr",
    "sr_delta010",
    "joint_score",
    "frozen_gate_pass",
    "clean_score",
    "clean_eval_score",
    "stress_score",
    "obs_sigma_008_score",
    "clean_score_by_evaluation_seed",
    "stress_score_by_evaluation_seed",
    "base_clean_score",
    "base_stress_score",
    "best_stress_score",
    "recovery_score_threshold",
    "normalized_recovery",
    "clean_constraint_pass",
    "behavior_label",
    "recovery_label",
    "split_name",
    "protocol_sha256",
    "diagnostics_sha256",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _metric(entry: Mapping[str, Any], name: str) -> tuple[float, list[float]]:
    metric = entry.get("metrics", {}).get(name, {})
    values = [_finite(value, name=f"{name}/value") for value in metric.get("values", [])]
    seeds = tuple(int(seed) for seed in metric.get("seeds", EVALUATION_SEEDS))
    if len(values) != 3 or seeds != EVALUATION_SEEDS or int(metric.get("n", -1)) != 3:
        raise ValueError(f"{name}: expected evaluation seeds 42/43/44")
    mean = _finite(metric.get("mean"), name=f"{name}/mean")
    if not math.isclose(mean, sum(values) / 3, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name}: mean/value mismatch")
    return mean, values


def build(
    *,
    prediction_paths: Mapping[int, Path],
    manifest_paths: Mapping[int, Path],
) -> list[dict[str, Any]]:
    predictions: dict[tuple[int, str, float], dict[str, str]] = {}
    manifests: dict[int, dict[str, Any]] = {}
    protocol_hashes: set[str] = set()
    for seed in TRAINING_SEEDS:
        manifest = _load_json(manifest_paths[seed])
        metadata = manifest.get("_metadata", {})
        if int(metadata.get("training_seed", -1)) != seed:
            raise ValueError(f"seed {seed}: manifest seed mismatch")
        if metadata.get("status") != "complete":
            raise ValueError(f"seed {seed}: incomplete manifest")
        manifests[seed] = manifest
        rows = _read_csv(prediction_paths[seed])
        if len(rows) != len(TASKS) * len(RHOS):
            raise ValueError(f"seed {seed}: expected 36 frozen predictions")
        for row in rows:
            row_seed = int(float(row["training_seed"]))
            task = str(row["task"])
            rho = float(row["training_rho"])
            key = (row_seed, task, rho)
            if row_seed != seed or key in predictions:
                raise ValueError(f"invalid or duplicate prediction row {key}")
            predictions[key] = row
            protocol_hashes.add(str(row["protocol_sha256"]))

    expected = {
        (seed, task, rho)
        for seed in TRAINING_SEEDS
        for task in TASKS
        for rho in RHOS
    }
    if set(predictions) != expected:
        raise ValueError("PLDM prediction grid is incomplete")
    if len(protocol_hashes) != 1:
        raise ValueError("PLDM predictions use different frozen protocols")

    output: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        manifest = manifests[seed]
        for task in TASKS:
            entries = {rho: manifest[task][str(rho)] for rho in RHOS}
            behavior = {
                rho: {
                    "clean": _metric(entry, "clean"),
                    "stress": _metric(entry, "pixels_std0.08"),
                }
                for rho, entry in entries.items()
            }
            base_clean = behavior[0.0]["clean"][0]
            base_stress = behavior[0.0]["stress"][0]
            best_stress = max(item["stress"][0] for item in behavior.values())
            recovery_threshold = base_stress + 0.8 * (best_stress - base_stress)
            recovery_denominator = max(best_stress - base_stress, 1e-12)
            base_ir = _finite(
                predictions[(seed, task, 0.0)]["atr_horizon_v2_q90"],
                name=f"{seed}/{task}/base IR",
            )
            if base_ir <= 0:
                raise ValueError(f"{seed}/{task}: non-positive base IR")
            for rho in RHOS:
                prediction = predictions[(seed, task, rho)]
                clean_mean, clean_values = behavior[rho]["clean"]
                stress_mean, stress_values = behavior[rho]["stress"]
                ir = _finite(
                    prediction["atr_horizon_v2_q90"],
                    name=f"{seed}/{task}/{rho}/IR",
                )
                clean_pass = clean_mean >= base_clean - 5.0
                recovery = stress_mean >= recovery_threshold and clean_pass
                output.append(
                    {
                        "model_family": "PLDM",
                        "training_family_id": f"pldm_canonical_seed{seed}",
                        "training_seed": seed,
                        "task": task,
                        "training_rho": rho,
                        "rho": rho,
                        "stressor_family": prediction["stressor_family"],
                        "stressor_severity": float(prediction["stressor_severity"]),
                        "atr_horizon_v2_q90": ir,
                        "ir_raw_q90": ir,
                        "ir_relative_q90": ir / base_ir,
                        "smpr": float(prediction["smpr"]),
                        "sr_delta010": float(prediction["smpr"]),
                        "joint_score": float(prediction["joint_score"]),
                        "frozen_gate_pass": prediction["frozen_gate_pass"],
                        "clean_score": clean_mean,
                        "clean_eval_score": clean_mean,
                        "stress_score": stress_mean,
                        "obs_sigma_008_score": stress_mean,
                        "clean_score_by_evaluation_seed": json.dumps(
                            clean_values, separators=(",", ":")
                        ),
                        "stress_score_by_evaluation_seed": json.dumps(
                            stress_values, separators=(",", ":")
                        ),
                        "base_clean_score": base_clean,
                        "base_stress_score": base_stress,
                        "best_stress_score": best_stress,
                        "recovery_score_threshold": recovery_threshold,
                        "normalized_recovery": (
                            stress_mean - base_stress
                        )
                        / recovery_denominator,
                        "clean_constraint_pass": str(clean_pass).lower(),
                        "behavior_label": str(recovery).lower(),
                        "recovery_label": str(recovery).lower(),
                        "split_name": prediction["split_name"],
                        "protocol_sha256": prediction["protocol_sha256"],
                        "diagnostics_sha256": prediction["diagnostics_sha256"],
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in TRAINING_SEEDS:
        parser.add_argument(f"--predictions-{seed}", type=Path, required=True)
        parser.add_argument(f"--manifest-{seed}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = {
        seed: getattr(args, f"predictions_{seed}") for seed in TRAINING_SEEDS
    }
    manifests = {
        seed: getattr(args, f"manifest_{seed}") for seed in TRAINING_SEEDS
    }
    rows = build(prediction_paths=predictions, manifest_paths=manifests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
