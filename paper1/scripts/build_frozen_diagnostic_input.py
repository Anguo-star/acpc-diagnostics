#!/usr/bin/env python3
"""Join protocol-bound held-out ATR and SMPR artifacts into blind input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.build_external_acpc_horizon_v2_artifact import (
    ROOT,
    _frozen_protocol,
)


TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHO_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
ROW_FIELDS = (
    "status",
    "model_family",
    "training_family_id",
    "training_seed",
    "task",
    "training_rho",
    "stressor_family",
    "stressor_severity",
    "atr_horizon_v2_q90",
    "smpr",
    "split_name",
)
FORBIDDEN_FIELD_PARTS = (
    "success",
    "score",
    "label",
    "return",
    "recovery",
    "clean",
    "pixels",
    "ground_truth",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    path: Path,
    schema: str,
    role: str,
    protocol_sha: str,
    expected_seeds: set[int],
) -> tuple[int, str]:
    _require(metadata.get("schema_version") == schema, f"{path}: schema mismatch")
    _require(metadata.get("status") == "complete", f"{path}: incomplete artifact")
    _require(metadata.get("status_counts") == {"ok": 36}, f"{path}: row failures")
    _require(metadata.get("missing_rows") == [], f"{path}: missing rows")
    _require(metadata.get("errors") == [], f"{path}: recorded errors")
    _require(metadata.get("artifact_role") == role, f"{path}: artifact role mismatch")
    _require(metadata.get("behavior_blind") is True, f"{path}: artifact is not behavior blind")
    _require(metadata.get("protocol_sha256") == protocol_sha, f"{path}: protocol hash mismatch")
    _require(metadata.get("split_name") == "TEST", f"{path}: split is not TEST")
    _require(metadata.get("model_family") == "LeWM", f"{path}: family is not LeWM")
    seed = int(metadata.get("training_seed"))
    family_id = str(metadata.get("training_family_id"))
    _require(seed in expected_seeds, f"{path}: unexpected held-out seed {seed}")
    _require(family_id == f"lewm_seed{seed}", f"{path}: family id/seed mismatch")
    _require(
        set(metadata.get("source_paths", {})) == set(metadata.get("source_hashes", {})),
        f"{path}: source provenance keys differ",
    )
    return seed, family_id


def build_diagnostic_input(
    *,
    protocol_path: Path,
    atr_paths: Sequence[Path],
    smpr_paths: Sequence[Path],
) -> dict[str, Any]:
    protocol_original = protocol_path.read_bytes()
    protocol_mtime = protocol_path.stat().st_mtime_ns
    protocol, protocol_sha = _frozen_protocol(protocol_path)
    expected_seeds = {
        int(seed)
        for seed in protocol["external_policy"]["heldout_lewm_training_seeds"]
    }
    _require(len(atr_paths) == len(expected_seeds), "one ATR artifact per held-out seed is required")
    _require(len(smpr_paths) == len(expected_seeds), "one SMPR artifact per held-out seed is required")
    expected_keys = {
        (seed, task, rho)
        for seed in expected_seeds
        for task in TASKS
        for rho in RHO_GRID
    }

    atr_rows: dict[tuple[int, str, float], tuple[dict[str, Any], Path]] = {}
    atr_by_seed: dict[int, Path] = {}
    for path in atr_paths:
        payload = _load_strict(path)
        metadata = payload.get("metadata", {})
        seed, family_id = _validate_metadata(
            metadata,
            path=path,
            schema="paper1-acpc-horizon-v2-1.0",
            role="heldout_external_atr",
            protocol_sha=protocol_sha,
            expected_seeds=expected_seeds,
        )
        _require(seed not in atr_by_seed, f"duplicate ATR seed {seed}")
        atr_by_seed[seed] = path
        _require(
            metadata.get("implementation_hashes", {}).get("canonical_metric")
            == protocol["source_hashes"]["canonical_metric"],
            f"{path}: canonical ATR implementation hash mismatch",
        )
        rows = payload.get("rows", [])
        _require(isinstance(rows, list) and len(rows) == 36, f"{path}: expected 36 rows")
        for row in rows:
            rho = _finite(row.get("training_rho"), name="ATR training rho")
            key = (seed, str(row.get("task")), rho)
            _require(key not in atr_rows, f"duplicate ATR row {key}")
            _require(row.get("status") == "ok", f"{key}: ATR row is not ok")
            _require(row.get("training_family_id") == family_id, f"{key}: ATR family mismatch")
            _require(row.get("split_name") == "TEST", f"{key}: ATR split mismatch")
            _require(
                row.get("radius_metric") == protocol["radius_metric"],
                f"{key}: ATR radius metric mismatch",
            )
            _finite(row.get("atr_horizon_v2_q90"), name=f"{key}/ATR")
            atr_rows[key] = (row, path)

    smpr_rows: dict[tuple[int, str, float], tuple[dict[str, Any], Path]] = {}
    smpr_by_seed: dict[int, Path] = {}
    for path in smpr_paths:
        payload = _load_strict(path)
        metadata = payload.get("metadata", {})
        seed, family_id = _validate_metadata(
            metadata,
            path=path,
            schema="paper1-smpr-v2-merged-1.0",
            role="heldout_external_smpr",
            protocol_sha=protocol_sha,
            expected_seeds=expected_seeds,
        )
        _require(seed not in smpr_by_seed, f"duplicate SMPR seed {seed}")
        smpr_by_seed[seed] = path
        implementation_hashes = metadata.get("implementation_hashes", {})
        for local_key, protocol_key in (
            ("canonical_metric", "canonical_metric"),
            ("semantic_margin", "semantic_margin"),
            ("runner", "smpr_runner"),
        ):
            _require(
                implementation_hashes.get(local_key)
                == protocol["source_hashes"][protocol_key],
                f"{path}: {local_key} implementation hash mismatch",
            )
        _require(
            metadata.get("source_hashes", {}).get("reference_atr")
            == _sha256(atr_by_seed[seed]),
            f"{path}: referenced ATR artifact hash mismatch",
        )
        rows = payload.get("rows", [])
        _require(isinstance(rows, list) and len(rows) == 36, f"{path}: expected 36 rows")
        for row in rows:
            rho = _finite(row.get("training_rho"), name="SMPR training rho")
            key = (seed, str(row.get("task")), rho)
            _require(key not in smpr_rows, f"duplicate SMPR row {key}")
            _require(row.get("status") == "ok", f"{key}: SMPR row is not ok")
            _require(row.get("training_family_id") == family_id, f"{key}: SMPR family mismatch")
            _require(row.get("split_name") == "TEST", f"{key}: SMPR split mismatch")
            _require(row.get("atr_reference_match") is True, f"{key}: ATR reference mismatch")
            _require(
                _finite(row.get("atr_reference_abs_error"), name=f"{key}/ATR error") <= 1e-6,
                f"{key}: ATR reference error exceeds tolerance",
            )
            _finite(row.get("smpr"), name=f"{key}/SMPR")
            smpr_rows[key] = (row, path)

    _require(set(atr_rows) == expected_keys, "held-out ATR key coverage mismatch")
    _require(set(smpr_rows) == expected_keys, "held-out SMPR key coverage mismatch")
    diagnostic_rows: list[dict[str, Any]] = []
    for key in sorted(
        expected_keys,
        key=lambda value: (value[0], TASKS.index(value[1]), value[2]),
    ):
        seed, task, rho = key
        atr = atr_rows[key][0]
        smpr = smpr_rows[key][0]
        atr_value = _finite(atr["atr_horizon_v2_q90"], name=f"{key}/ATR")
        reference_value = _finite(
            smpr["reference_atr_horizon_v2_q90"],
            name=f"{key}/SMPR reference ATR",
        )
        _require(
            math.isclose(atr_value, reference_value, rel_tol=1e-6, abs_tol=1e-6),
            f"{key}: ATR/SMPR joined value mismatch",
        )
        _require(
            Path(str(atr.get("model_file"))).resolve()
            == Path(str(smpr.get("model_file"))).resolve(),
            f"{key}: ATR/SMPR checkpoint path mismatch",
        )
        _require(
            atr.get("checkpoint_sha256") == smpr.get("checkpoint_sha256"),
            f"{key}: ATR/SMPR checkpoint hash mismatch",
        )
        _require(
            _sha256(Path(str(atr["model_file"]))) == atr["checkpoint_sha256"],
            f"{key}: live checkpoint hash mismatch",
        )
        diagnostic_rows.append(
            {
                "status": "ok",
                "model_family": "LeWM",
                "training_family_id": f"lewm_seed{seed}",
                "training_seed": seed,
                "task": task,
                "training_rho": rho,
                "stressor_family": "gaussian",
                "stressor_severity": float(
                    protocol["diagnostic_sampling"]["evaluation_noise_std"]
                ),
                "atr_horizon_v2_q90": atr_value,
                "smpr": _finite(smpr["smpr"], name=f"{key}/SMPR"),
                "split_name": "TEST",
            }
        )
    _require(
        all(tuple(row) == ROW_FIELDS for row in diagnostic_rows),
        "diagnostic row field allowlist/order mismatch",
    )
    _require(
        not any(
            part in field.lower()
            for field in ROW_FIELDS
            for part in FORBIDDEN_FIELD_PARTS
        ),
        "diagnostic field allowlist leaks behavior",
    )

    source_paths: dict[str, str] = {"protocol": str(protocol_path)}
    source_hashes: dict[str, str] = {"protocol": protocol_sha}
    for seed in sorted(expected_seeds):
        for kind, paths in (("atr", atr_by_seed), ("smpr", smpr_by_seed)):
            name = f"{kind}_lewm_seed{seed}"
            source_paths[name] = str(paths[seed])
            source_hashes[name] = _sha256(paths[seed])
    script_path = Path(__file__).resolve()
    artifact = {
        "metadata": {
            "schema_version": "paper1-frozen-diagnostic-input-1.0",
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "protocol_sha256": protocol_sha,
            "model_family": "LeWM",
            "training_seeds": sorted(expected_seeds),
            "training_seed_semantics": "two independent held-out LeWM TEST training runs",
            "evaluation_seed_semantics": protocol["evaluation_seed_semantics"],
            "split_name": "TEST",
            "status": "complete",
            "status_counts": {"ok": len(diagnostic_rows)},
            "missing_rows": [],
            "errors": [],
            "behavior_blind": True,
            "threshold_search_available": False,
            "strict_external_contract": "lewm_heldout_gaussian_v1",
            "operator_blinding": (
                "not claimed; raw diagnostic runners consumed eval manifests, "
                "while this frozen-apply input contains no behavior fields"
            ),
        },
        "rows": diagnostic_rows,
    }
    _require(set(source_paths) == set(source_hashes), "source provenance keys differ")
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--atr", type=Path, action="append", required=True)
    parser.add_argument("--smpr", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.out.resolve() == args.protocol.resolve():
        raise ValueError("diagnostic output cannot be the frozen protocol path")
    artifact = build_diagnostic_input(
        protocol_path=args.protocol,
        atr_paths=args.atr,
        smpr_paths=args.smpr,
    )
    _write_exclusive(args.out, artifact)
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
