#!/usr/bin/env python3
"""Build a strict, provenance-complete Paper 1 horizon-v2 ACPC artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
SCHEMA_VERSION = "paper1-acpc-horizon-v2-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_strict(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metric(entry: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = entry.get("metrics", {}).get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest entry is missing metric {name}")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _validate_source_metadata(
    metadata: Mapping[str, Any],
    *,
    path: Path,
    expected_task: str,
    n_sequences: int,
    num_noise_draws: int,
    anchor_seed: int,
) -> None:
    _require(
        metadata.get("schema_version") == "paper1-acpc-phase0-0.2",
        f"{path}: expected source schema 0.2",
    )
    _require(metadata.get("methods") == ["LeWM"], f"{path}: expected LeWM source")
    _require(metadata.get("tasks") == [expected_task], f"{path}: task mismatch")
    _require(
        tuple(metadata.get("std_keys", [])) == STD_KEYS,
        f"{path}: expected the full nine-point rho grid",
    )
    protocol = metadata.get("protocol", {})
    _require(
        protocol.get("radius_metric") == "horizon_weighted_stacked_l2_v2",
        f"{path}: source is not horizon-v2",
    )
    _require(protocol.get("rollout_horizon") == 8, f"{path}: expected H=8")
    _require(
        protocol.get("horizon_weights") == "uniform_1_over_H",
        f"{path}: expected uniform horizon weights",
    )
    _require(
        protocol.get("num_noise_draws") == num_noise_draws,
        f"{path}: noise-draw count mismatch",
    )
    _require(
        protocol.get("anchor_seed") == anchor_seed,
        f"{path}: anchor seed mismatch",
    )
    _require(
        metadata.get("status_counts") == {"ok": len(STD_KEYS)},
        f"{path}: source has missing or failed rows",
    )
    _require(not metadata.get("errors"), f"{path}: source records errors")
    _require(not metadata.get("missing_rows"), f"{path}: source records missing rows")
    _require(n_sequences > 0, "n_sequences must be positive")


def build_artifact(
    *,
    inputs: Sequence[Path],
    eval_manifest_path: Path,
    model_family: str,
    training_seed: int,
    evaluation_seeds: Sequence[int],
    n_sequences: int,
    num_noise_draws: int,
    anchor_seed: int,
) -> dict[str, Any]:
    _require(model_family == "LeWM", "this calibration builder currently accepts LeWM only")
    _require(len(inputs) == len(TASKS), "exactly four task source artifacts are required")
    _require(len(set(evaluation_seeds)) == 3, "exactly three distinct evaluation seeds are required")
    _require(eval_manifest_path.is_file(), f"missing eval manifest: {eval_manifest_path}")
    eval_manifest = _load_strict(eval_manifest_path)
    manifest_meta = eval_manifest.get("_metadata", {})
    _require(
        manifest_meta.get("training_seed") == training_seed,
        "eval manifest training seed does not match the requested calibration seed",
    )
    _require(
        tuple(manifest_meta.get("std_keys", [])) == STD_KEYS,
        "eval manifest does not contain the full rho grid",
    )

    rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    source_metadata: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}
    for path in inputs:
        _require(path.is_file(), f"missing source artifact: {path}")
        payload = _load_strict(path)
        source_rows = payload.get("rows", [])
        _require(
            isinstance(source_rows, list) and len(source_rows) == len(STD_KEYS),
            f"{path}: expected nine rows",
        )
        tasks = {str(row.get("task")) for row in source_rows}
        _require(len(tasks) == 1, f"{path}: source must contain exactly one task")
        task = next(iter(tasks))
        _require(task in TASKS, f"{path}: unknown task {task}")
        _require(task not in seen_tasks, f"duplicate task source: {task}")
        seen_tasks.add(task)
        metadata = payload.get("metadata", {})
        _validate_source_metadata(
            metadata,
            path=path,
            expected_task=task,
            n_sequences=n_sequences,
            num_noise_draws=num_noise_draws,
            anchor_seed=anchor_seed,
        )
        source_paths[task] = path
        source_metadata[task] = metadata

        by_std = {str(row.get("std_key")): row for row in source_rows}
        _require(set(by_std) == set(STD_KEYS), f"{path}: rho coverage mismatch")
        for std_key in STD_KEYS:
            row = dict(by_std[std_key])
            _require(row.get("status") == "ok", f"{path}: non-ok row {std_key}")
            _require(row.get("method") == model_family, f"{path}: model family mismatch")
            _require(row.get("n_sequences") == n_sequences, f"{path}: anchor count mismatch")
            _require(
                row.get("num_noise_draws") == num_noise_draws,
                f"{path}: draw count mismatch",
            )
            _require(
                row.get("radius_metric") == "horizon_weighted_stacked_l2_v2",
                f"{path}: non-canonical row {std_key}",
            )
            _require(
                row.get("stepwise_rollout_q90_is_atr") is False,
                f"{path}: compatibility field is mislabeled",
            )
            _require(
                "acpc_h_l2_p90" not in row,
                f"{path}: ambiguous old ATR field is present",
            )
            for key in (
                "atr_horizon_v2_q90",
                "horizon_radius_v2_unnormalized_q90",
                "model_load_time",
                "data_io_time",
                "prediction_time",
                "fixed_pool_time",
                "wall_time_per_row",
            ):
                _finite_number(row.get(key), name=f"{task}/{std_key}/{key}")

            manifest_entry = eval_manifest[task][std_key]
            clean = _metric(manifest_entry, "clean")
            stress = _metric(manifest_entry, "pixels_std0.08")
            clean_values = list(clean.get("values", []))
            stress_values = list(stress.get("values", []))
            _require(
                len(clean_values) == len(evaluation_seeds)
                and len(stress_values) == len(evaluation_seeds),
                f"{task}/{std_key}: evaluation-seed value count mismatch",
            )
            _require(
                math.isclose(
                    _finite_number(row.get("clean_success"), name="clean_success"),
                    _finite_number(clean.get("mean"), name="clean mean"),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{task}/{std_key}: clean mean mismatch",
            )
            _require(
                math.isclose(
                    _finite_number(
                        row.get("pixels_std0.08_success"),
                        name="pixels_std0.08_success",
                    ),
                    _finite_number(stress.get("mean"), name="stress mean"),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{task}/{std_key}: stress mean mismatch",
            )
            row.update(
                {
                    "model_family": model_family,
                    "training_seed": int(training_seed),
                    "training_rho": float(std_key),
                    "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
                    "clean_success_by_evaluation_seed": clean_values,
                    "pixels_std0.08_success_by_evaluation_seed": stress_values,
                    "behavior_evaluation_count": len(evaluation_seeds),
                    "calibration_split": "CAL",
                }
            )
            rows.append(row)

    _require(seen_tasks == set(TASKS), f"task coverage mismatch: {seen_tasks}")
    task_order = {task: index for index, task in enumerate(TASKS)}
    rows.sort(key=lambda row: (task_order[row["task"]], float(row["std_key"])))

    script_path = Path(__file__).resolve()
    implementation_paths = {
        "canonical_metric": ROOT / "tools" / "paper1_acpc_metrics.py",
        "acpc_runner": ROOT / "tools" / "paper1_phase0_acpc.py",
    }
    all_sources = {"eval_manifest": eval_manifest_path, **source_paths}
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": {
                name: str(path)
                for name, path in all_sources.items()
            },
            "source_hashes": {
                name: _sha256(path)
                for name, path in all_sources.items()
            },
            "source_metadata": source_metadata,
            "implementation_paths": {
                name: str(path.relative_to(ROOT))
                for name, path in implementation_paths.items()
            },
            "implementation_hashes": {
                name: _sha256(path)
                for name, path in implementation_paths.items()
            },
            "protocol_hash": None,
            "protocol_hash_status": "calibration_input_pre_freeze",
            "model_family": model_family,
            "training_seed": int(training_seed),
            "training_seed_semantics": "one independently trained LeWM checkpoint family",
            "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
            "evaluation_seed_semantics": (
                "closed-loop values are conditional evaluation replicates and are "
                "not training-seed replications"
            ),
            "status": "complete",
            "status_counts": {"ok": len(rows)},
            "missing_rows": [],
            "errors": [],
            "protocol": {
                "radius_metric": "horizon_weighted_stacked_l2_v2",
                "rollout_horizon": 8,
                "horizon_weights": "uniform_1_over_H",
                "atr_quantile": 0.90,
                "normalization": (
                    "per_anchor_observed_clean_transition_l2_q50_"
                    "including_history_future_boundary"
                ),
                "noise_draw_aggregation": (
                    "per_anchor_mean_then_checkpoint_quantile"
                ),
                "n_sequences": int(n_sequences),
                "num_noise_draws": int(num_noise_draws),
                "anchor_seed": int(anchor_seed),
                "corruption": "observation_only_gaussian_std0.08_clean_goal",
            },
        },
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--model-family", default="LeWM")
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--num-noise-draws", type=int, default=5)
    parser.add_argument("--anchor-seed", type=int, default=9101)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "assets" / "paper1_data" / "acpc_horizon_v2_lewm.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = build_artifact(
        inputs=args.input,
        eval_manifest_path=args.eval_manifest,
        model_family=args.model_family,
        training_seed=args.training_seed,
        evaluation_seeds=args.evaluation_seeds,
        n_sequences=args.n_sequences,
        num_noise_draws=args.num_noise_draws,
        anchor_seed=args.anchor_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
