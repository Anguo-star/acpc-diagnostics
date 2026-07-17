#!/usr/bin/env python3
"""Merge task-sharded SMPR v2 runs into one strict calibration artifact."""

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
SCHEMA_VERSION = "paper1-smpr-v2-merged-1.0"


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


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def build_artifact(
    *,
    inputs: Sequence[Path],
    reference_atr_path: Path,
    model_family: str,
    family_id: str,
    training_seed: int,
) -> dict[str, Any]:
    _require(len(inputs) == len(TASKS), "exactly four task inputs are required")
    reference = _load_strict(reference_atr_path)
    _require(
        reference.get("metadata", {}).get("schema_version")
        == "paper1-acpc-horizon-v2-1.0",
        "reference ATR artifact has the wrong schema",
    )
    reference_rows = {
        (str(row["task"]), str(row["std_key"])): _finite(
            row["atr_horizon_v2_q90"],
            name="reference ATR",
        )
        for row in reference.get("rows", [])
        if row.get("status") == "ok"
    }
    _require(len(reference_rows) == 36, "reference ATR coverage must be 36 rows")

    rows: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    source_metadata: dict[str, Any] = {}
    seen_tasks: set[str] = set()
    canonical_protocol: Mapping[str, Any] | None = None
    for path in inputs:
        _require(path.is_file(), f"missing SMPR source: {path}")
        payload = _load_strict(path)
        metadata = payload.get("metadata", {})
        _require(
            metadata.get("schema_version") == "paper1-smpr-v2-1.0",
            f"{path}: wrong SMPR source schema",
        )
        _require(metadata.get("status") == "complete", f"{path}: incomplete source")
        _require(
            metadata.get("status_counts") == {"ok": len(STD_KEYS)},
            f"{path}: source row failures",
        )
        _require(metadata.get("model_family") == model_family, f"{path}: family mismatch")
        _require(
            metadata.get("training_family_id") == family_id,
            f"{path}: family id mismatch",
        )
        _require(
            metadata.get("training_seed") == training_seed,
            f"{path}: training seed mismatch",
        )
        protocol = metadata.get("protocol", {})
        if canonical_protocol is None:
            canonical_protocol = protocol
        else:
            _require(protocol == canonical_protocol, f"{path}: protocol mismatch")
        source_rows = payload.get("rows", [])
        _require(
            isinstance(source_rows, list) and len(source_rows) == len(STD_KEYS),
            f"{path}: expected nine rows",
        )
        tasks = {str(row.get("task")) for row in source_rows}
        _require(len(tasks) == 1, f"{path}: expected one task")
        task = next(iter(tasks))
        _require(task in TASKS and task not in seen_tasks, f"{path}: duplicate/unknown task")
        seen_tasks.add(task)
        source_paths[task] = path
        source_metadata[task] = metadata
        by_std = {str(row.get("std_key")): row for row in source_rows}
        _require(set(by_std) == set(STD_KEYS), f"{path}: rho coverage mismatch")
        for std_key in STD_KEYS:
            row = dict(by_std[std_key])
            key = (task, std_key)
            _require(row.get("status") == "ok", f"{key}: row is not ok")
            _require(row.get("atr_reference_match") is True, f"{key}: ATR mismatch")
            _require(
                _finite(row.get("atr_reference_abs_error"), name=f"{key}/ATR error")
                <= 1e-6,
                f"{key}: ATR error exceeds tolerance",
            )
            _require(
                math.isclose(
                    _finite(row.get("same_state_tube_radius"), name=f"{key}/tube"),
                    reference_rows[key],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                f"{key}: tube radius differs from merged ATR",
            )
            _require(
                row.get("radius_metric") == "horizon_weighted_stacked_l2_v2",
                f"{key}: non-canonical radius",
            )
            _require(
                row.get("pair_rule") == "task_grounded_near_boundary_v2",
                f"{key}: wrong pair rule",
            )
            _require(
                row.get("margin_delta_norm") == 0.10,
                f"{key}: positive margin is not 0.10",
            )
            _require(
                int(row.get("semantic_pair_count", 0)) > 0,
                f"{key}: zero semantic pairs",
            )
            _finite(row.get("smpr"), name=f"{key}/SMPR")
            rows.append(row)
    _require(seen_tasks == set(TASKS), f"task coverage mismatch: {seen_tasks}")
    task_order = {task: index for index, task in enumerate(TASKS)}
    rows.sort(key=lambda row: (task_order[row["task"]], float(row["std_key"])))

    script_path = Path(__file__).resolve()
    implementation_paths = {
        "runner": ROOT / "paper1" / "scripts" / "smpr_sensitivity.py",
        "semantic_margin": ROOT / "tools" / "paper1_semantic_margin.py",
        "canonical_metric": ROOT / "tools" / "paper1_acpc_metrics.py",
    }
    all_sources = {"reference_atr": reference_atr_path, **source_paths}
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
            "training_family_id": family_id,
            "training_seed": int(training_seed),
            "training_seed_semantics": "one independently trained LeWM checkpoint family",
            "evaluation_seeds": [42, 43, 44],
            "evaluation_seed_semantics": (
                "conditional closed-loop evaluation replicates, not training seeds"
            ),
            "status": "complete",
            "status_counts": {"ok": len(rows)},
            "missing_rows": [],
            "errors": [],
            "protocol": canonical_protocol,
        },
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--reference-atr", type=Path, required=True)
    parser.add_argument("--model-family", default="LeWM")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "assets"
        / "paper1_data"
        / "smpr_calibration_lewm_seed3072_v2.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = build_artifact(
        inputs=args.input,
        reference_atr_path=args.reference_atr,
        model_family=args.model_family,
        family_id=args.family_id,
        training_seed=args.training_seed,
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
