#!/usr/bin/env python3
"""Merge canonical PLDM horizon-v2 ATR shards under the frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.build_external_acpc_horizon_v2_artifact import (
    ROOT,
    _frozen_protocol,
)


TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_strict(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def build_external_artifact(
    *,
    inputs: Sequence[Path],
    manifest_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol_original = protocol_path.read_bytes()
    protocol_mtime = protocol_path.stat().st_mtime_ns
    protocol, protocol_sha = _frozen_protocol(protocol_path)
    manifest = _load_strict(manifest_path)
    manifest_meta = manifest.get("_metadata", {})
    _require(
        manifest_meta.get("schema_version")
        == "paper1-pldm-canonical-eval-manifest-0.2",
        "PLDM manifest schema mismatch",
    )
    _require(manifest_meta.get("status") == "complete", "PLDM manifest is incomplete")
    _require(manifest_meta.get("model_family") == "PLDM", "PLDM manifest family mismatch")
    training_seed = int(manifest_meta["training_seed"])
    family_id = str(manifest_meta["training_family_id"])
    evaluation_seeds = [
        int(seed)
        for seed in protocol["behavior_evaluation"]["evaluation_seeds"]
    ]
    _require(
        manifest_meta.get("evaluation_seeds") == evaluation_seeds,
        "PLDM evaluation seeds do not match the frozen protocol",
    )
    _require(len(inputs) == len(TASKS), "exactly four PLDM task shards are required")

    source_paths: dict[str, str] = {
        "manifest": str(manifest_path),
        "protocol": str(protocol_path),
    }
    source_hashes: dict[str, str] = {
        "manifest": _sha256(manifest_path),
        "protocol": protocol_sha,
    }
    source_metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    sampling = protocol["diagnostic_sampling"]
    for path in inputs:
        payload = _load_strict(path)
        metadata = payload.get("metadata", {})
        source_rows = payload.get("rows", [])
        _require(
            metadata.get("schema_version") == "paper1-acpc-phase0-0.2",
            f"{path}: source schema mismatch",
        )
        _require(metadata.get("methods") == ["PLDM"], f"{path}: source is not PLDM")
        _require(metadata.get("status_counts") == {"ok": 9}, f"{path}: row failures")
        _require(metadata.get("missing_rows") == [], f"{path}: missing rows")
        _require(metadata.get("errors") == [], f"{path}: recorded errors")
        _require(isinstance(source_rows, list) and len(source_rows) == 9, f"{path}: expected nine rows")
        tasks = {str(row.get("task")) for row in source_rows}
        _require(len(tasks) == 1, f"{path}: expected one task")
        task = next(iter(tasks))
        _require(task in TASKS and task not in seen_tasks, f"{path}: duplicate/unknown task")
        seen_tasks.add(task)
        _require(metadata.get("tasks") == [task], f"{path}: metadata task mismatch")
        source_key = f"atr_source_{task}"
        source_paths[source_key] = str(path)
        source_hashes[source_key] = _sha256(path)
        source_metadata[task] = metadata
        source_protocol = metadata.get("protocol", {})
        expected_source_protocol = {
            "radius_metric": protocol["radius_metric"],
            "rollout_horizon": protocol["rollout_horizon"],
            "horizon_weights": protocol["horizon_weights"],
            "num_noise_draws": int(sampling["num_noise_draws"]),
            "anchor_seed": int(sampling["anchor_seed"]),
        }
        for key, expected in expected_source_protocol.items():
            _require(source_protocol.get(key) == expected, f"{path}: protocol mismatch: {key}")
        by_std = {str(row.get("std_key")): row for row in source_rows}
        _require(set(by_std) == set(STD_KEYS), f"{path}: rho coverage mismatch")
        for std_key in STD_KEYS:
            row = by_std[std_key]
            key = (task, std_key)
            entry: Mapping[str, Any] = manifest[task][std_key]
            _require(row.get("status") == "ok", f"{key}: ATR row is not ok")
            _require(row.get("method") == "PLDM", f"{key}: method mismatch")
            _require(row.get("subdir") == entry.get("subdir"), f"{key}: subdir mismatch")
            _require(
                Path(str(row.get("run_path"))).name
                == Path(str(entry.get("path"))).name,
                f"{key}: legacy run identity mismatch",
            )
            _require(
                Path(str(row.get("model_file"))).resolve()
                == Path(str(entry.get("model_file"))).resolve(),
                f"{key}: model file mismatch",
            )
            model_file = Path(str(entry["model_file"])).resolve()
            checkpoint_sha = _sha256(model_file)
            _require(
                checkpoint_sha
                == manifest_meta["source_hashes"][f"checkpoint_{task}_{std_key}"],
                f"{key}: checkpoint hash mismatch",
            )
            _require(row.get("n_sequences") == sampling["n_anchors"], f"{key}: anchor count mismatch")
            _require(row.get("num_noise_draws") == sampling["num_noise_draws"], f"{key}: draw count mismatch")
            _require(row.get("noise_std") == sampling["evaluation_noise_std"], f"{key}: noise mismatch")
            _require(row.get("corrupt_goal") is sampling["corrupt_goal"], f"{key}: goal policy mismatch")
            _require(row.get("rollout_horizon_actual") == protocol["rollout_horizon"], f"{key}: horizon mismatch")
            _require(row.get("radius_metric") == protocol["radius_metric"], f"{key}: radius metric mismatch")
            _require(row.get("stepwise_rollout_q90_is_atr") is False, f"{key}: legacy field mislabeled")
            _require(row.get("embedding_space") == "normalized", f"{key}: embedding-space mismatch")
            _require(row.get("corruption_type") == "gaussian_noise", f"{key}: corruption mismatch")
            _require(
                row.get("noise_draw_seed_rule") == "seed+1009+7919*draw_index",
                f"{key}: draw seed rule mismatch",
            )
            _require(
                row.get("noise_draw_seeds")
                == [
                    int(sampling["anchor_seed"]) + 1009 + 7919 * index
                    for index in range(int(sampling["num_noise_draws"]))
                ],
                f"{key}: draw seeds mismatch",
            )
            _require(
                row.get("normalization") == "per_anchor_clean_transition_l2_q50",
                f"{key}: normalization mismatch",
            )
            clean_scales = [float(value) for value in row["clean_transition_scale"]]
            _require(len(clean_scales) == sampling["n_anchors"], f"{key}: scale coverage mismatch")
            for metric_name, row_name in (
                ("clean", "clean_success"),
                ("pixels_std0.08", "pixels_std0.08_success"),
            ):
                metric = entry.get("metrics", {}).get(metric_name, {})
                _require(metric.get("seeds") == evaluation_seeds, f"{key}: eval seeds mismatch")
                _require(
                    math.isclose(
                        _finite(row.get(row_name), name=f"{key}/{row_name}"),
                        _finite(metric.get("mean"), name=f"{key}/{metric_name} mean"),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                    f"{key}: {metric_name} mean mismatch",
                )
            rows.append(
                {
                    "status": "ok",
                    "model_family": "PLDM",
                    "training_family_id": family_id,
                    "training_seed": training_seed,
                    "task": task,
                    "std_key": std_key,
                    "training_rho": float(std_key),
                    "atr_horizon_v2_q90": _finite(
                        row.get("atr_horizon_v2_q90"),
                        name=f"{key}/ATR",
                    ),
                    "radius_metric": row["radius_metric"],
                    "rollout_horizon_actual": int(row["rollout_horizon_actual"]),
                    "n_sequences": int(row["n_sequences"]),
                    "num_noise_draws": int(row["num_noise_draws"]),
                    "noise_std": float(row["noise_std"]),
                    "corrupt_goal": bool(row["corrupt_goal"]),
                    "clean_transition_scale_min": min(clean_scales),
                    "clean_transition_scale_zero_count": sum(
                        value == 0.0 for value in clean_scales
                    ),
                    "model_file": str(model_file),
                    "checkpoint_sha256": checkpoint_sha,
                    "split_name": "E2",
                }
            )
    _require(seen_tasks == set(TASKS), "PLDM task coverage mismatch")
    rows.sort(key=lambda row: (TASKS.index(row["task"]), float(row["training_rho"])))
    canonical_metric = ROOT / "tools" / "paper1_acpc_metrics.py"
    runner = ROOT / "tools" / "paper1_phase0_acpc.py"
    _require(
        _sha256(canonical_metric) == protocol["source_hashes"]["canonical_metric"],
        "canonical metric no longer matches the frozen protocol",
    )
    calibration_atr = _load_strict(ROOT / protocol["source_paths"]["calibration_atr"])
    _require(
        _sha256(runner)
        == calibration_atr.get("metadata", {}).get("implementation_hashes", {}).get(
            "acpc_runner"
        ),
        "PLDM ATR runner hash differs from CAL",
    )
    _require(set(source_paths) == set(source_hashes), "source provenance keys differ")
    script_path = Path(__file__).resolve()
    artifact = {
        "metadata": {
            "schema_version": "paper1-acpc-horizon-v2-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "source_metadata": source_metadata,
            "implementation_paths": {
                "canonical_metric": str(canonical_metric.relative_to(ROOT)),
                "acpc_runner": str(runner.relative_to(ROOT)),
            },
            "implementation_hashes": {
                "canonical_metric": _sha256(canonical_metric),
                "acpc_runner": _sha256(runner),
            },
            "protocol_hash": protocol_sha,
            "protocol_sha256": protocol_sha,
            "protocol_hash_status": "E2_external_bound_to_immutable_protocol",
            "model_family": "PLDM",
            "training_family_id": family_id,
            "training_seed": training_seed,
            "training_seed_semantics": "one independently trained PLDM checkpoint family",
            "evaluation_seeds": evaluation_seeds,
            "evaluation_seed_semantics": manifest_meta["evaluation_seed_semantics"],
            "split_name": "E2",
            "artifact_role": "pldm_canonical_external_atr",
            "behavior_blind": True,
            "raw_source_manifest_note": (
                "raw ATR shards retain legacy /home/ag run_path strings; actual "
                "checkpoint identity is bound by the v2 manifest and model SHA"
            ),
            "status": "complete",
            "status_counts": {"ok": len(rows)},
            "missing_rows": [],
            "errors": [],
            "protocol": {
                "radius_metric": protocol["radius_metric"],
                "rollout_horizon": protocol["rollout_horizon"],
                "horizon_weights": protocol["horizon_weights"],
                "atr_quantile": protocol["atr_quantile"],
                "normalization": protocol["normalization"],
                "noise_draw_aggregation": protocol["noise_draw_aggregation"],
                "n_sequences": int(sampling["n_anchors"]),
                "num_noise_draws": int(sampling["num_noise_draws"]),
                "anchor_seed": int(sampling["anchor_seed"]),
                "corruption": "observation_only_gaussian_std0.08_clean_goal",
            },
        },
        "rows": rows,
    }
    _require(protocol_path.read_bytes() == protocol_original, "builder changed protocol bytes")
    _require(protocol_path.stat().st_mtime_ns == protocol_mtime, "builder changed protocol mtime")
    return artifact


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.out.resolve() == args.protocol.resolve():
        raise ValueError("ATR output cannot be the frozen protocol path")
    artifact = build_external_artifact(
        inputs=args.input,
        manifest_path=args.manifest,
        protocol_path=args.protocol,
    )
    _write_exclusive(args.out, artifact)
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
