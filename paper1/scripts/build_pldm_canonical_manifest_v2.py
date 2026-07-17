#!/usr/bin/env python3
"""Build a provenance-complete PLDM canonical manifest without guessing seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
EVALUATION_SEEDS = (42, 43, 44)


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


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def build_manifest(
    *,
    legacy_path: Path,
    loadability_path: Path,
) -> dict[str, Any]:
    legacy = _load_strict(legacy_path)
    loadability = _load_strict(loadability_path)
    _require(set(legacy) == set(TASKS), "legacy PLDM task coverage mismatch")
    _require(
        loadability.get("schema_version") == "paper1-pldm-checkpoint-loadability-1.0",
        "PLDM loadability schema mismatch",
    )
    _require(
        loadability.get("source", {}).get("sha256") == _sha256(legacy_path),
        "loadability audit does not bind the supplied legacy manifest",
    )
    audit_rows = loadability.get("rows", [])
    _require(isinstance(audit_rows, list) and len(audit_rows) == 36, "expected 36 loadability rows")
    audit_index = {
        (str(row["task"]), str(row["std_key"])): row
        for row in audit_rows
    }
    expected_keys = {(task, std_key) for task in TASKS for std_key in STD_KEYS}
    _require(set(audit_index) == expected_keys, "PLDM loadability key coverage mismatch")

    source_paths: dict[str, str] = {
        "legacy_manifest": str(legacy_path),
        "loadability_audit": str(loadability_path),
    }
    source_hashes: dict[str, str] = {
        "legacy_manifest": _sha256(legacy_path),
        "loadability_audit": _sha256(loadability_path),
    }
    discovered_training_seeds: set[int] = set()
    canonical: dict[str, Any] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for task in TASKS:
        _require(set(legacy[task]) == set(STD_KEYS), f"{task}: legacy rho coverage mismatch")
        canonical[task] = {}
        for std_key in STD_KEYS:
            key = (task, std_key)
            audit = audit_index[key]
            _require(audit.get("status") == "ok", f"{key}: checkpoint is not loadable")
            run_path = Path(str(audit["model_file"])).resolve().parent
            model_path = Path(str(audit["model_file"])).resolve()
            config_path = run_path / "config.yaml"
            _require(model_path.is_file(), f"{key}: model file is missing")
            _require(config_path.is_file(), f"{key}: training config is missing")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            _require(isinstance(config, Mapping), f"{key}: invalid training config")
            training_seed = int(config["seed"])
            discovered_training_seeds.add(training_seed)
            image_noise = config.get("image_noise", {})
            _require(
                image_noise.get("type") in {"gaussian", "gaussian_noise"},
                f"{key}: non-Gaussian training",
            )
            _require(
                math.isclose(
                    _finite(image_noise.get("std_max"), name=f"{key}/std_max"),
                    float(std_key),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{key}: config std_max mismatch",
            )
            entry = legacy[task][std_key]
            _require(entry.get("subdir") == audit.get("subdir"), f"{key}: subdir mismatch")
            _require(Path(str(entry.get("path"))).name == run_path.name, f"{key}: legacy path mismatch")
            metrics: dict[str, Any] = {}
            for metric_name, metric_value in entry.get("metrics", {}).items():
                metric = dict(metric_value)
                values = list(metric.get("values", []))
                _require(len(values) == len(EVALUATION_SEEDS), f"{key}/{metric_name}: eval coverage mismatch")
                _require(int(metric.get("n")) == len(values), f"{key}/{metric_name}: n mismatch")
                for index, value in enumerate(values):
                    _finite(value, name=f"{key}/{metric_name}/seed{EVALUATION_SEEDS[index]}")
                metric["seeds"] = list(EVALUATION_SEEDS)
                metrics[metric_name] = metric
            for required_metric in ("clean", "pixels_std0.08", "pixels_goal_std0.08"):
                _require(required_metric in metrics, f"{key}: missing {required_metric}")
            canonical[task][std_key] = {
                "path": str(run_path),
                "subdir": run_path.name,
                "model_file": str(model_path),
                "metrics": metrics,
            }
            source_key = f"{task}_{std_key}"
            source_paths[f"config_{source_key}"] = str(config_path)
            source_hashes[f"config_{source_key}"] = _sha256(config_path)
            source_paths[f"checkpoint_{source_key}"] = str(model_path)
            source_hashes[f"checkpoint_{source_key}"] = _sha256(model_path)
            checkpoint_rows.append(
                {
                    "task": task,
                    "std_key": std_key,
                    "training_seed": training_seed,
                    "subdir": run_path.name,
                    "config_path": str(config_path),
                    "config_sha256": source_hashes[f"config_{source_key}"],
                    "model_file": str(model_path),
                    "model_sha256": source_hashes[f"checkpoint_{source_key}"],
                }
            )
    _require(
        len(discovered_training_seeds) == 1,
        f"canonical PLDM family spans multiple training seeds: {sorted(discovered_training_seeds)}",
    )
    training_seed = next(iter(discovered_training_seeds))
    _require(set(source_paths) == set(source_hashes), "source provenance keys differ")
    script_path = Path(__file__).resolve()
    return {
        "_metadata": {
            "schema_version": "paper1-pldm-canonical-eval-manifest-0.2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "model_family": "PLDM",
            "training_family_id": f"pldm_canonical_seed{training_seed}",
            "training_seed": training_seed,
            "training_seed_source": "all 36 checkpoint-local config.yaml files",
            "training_seed_semantics": "one independently trained PLDM checkpoint family",
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "evaluation_seed_semantics": "conditional evaluation variability, not training-run replication",
            "tasks": list(TASKS),
            "std_keys": list(STD_KEYS),
            "status": "complete",
            "status_counts": {"ok": len(checkpoint_rows)},
            "missing_rows": [],
            "errors": [],
            "checkpoint_rows": checkpoint_rows,
        },
        **canonical,
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--loadability", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = build_manifest(
        legacy_path=args.legacy,
        loadability_path=args.loadability,
    )
    _write_exclusive(args.out, artifact)
    metadata = artifact["_metadata"]
    print(
        f"wrote {args.out} (36 checkpoints, training seed "
        f"{metadata['training_seed']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
