#!/usr/bin/env python3
"""Join canonical PLDM ATR and SMPR into behavior-blind E2 gate input."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from paper1.scripts.build_external_acpc_horizon_v2_artifact import (
    ROOT,
    _frozen_protocol,
)
from paper1.scripts.build_frozen_diagnostic_input import (
    FORBIDDEN_FIELD_PARTS,
    ROW_FIELDS,
)


TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHO_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    role: str,
    schema: str,
    protocol_sha: str,
) -> tuple[str, int]:
    _require(metadata.get("schema_version") == schema, f"{role}: schema mismatch")
    _require(metadata.get("artifact_role") == role, f"{role}: role mismatch")
    _require(metadata.get("model_family") == "PLDM", f"{role}: family mismatch")
    _require(metadata.get("split_name") == "E2", f"{role}: split mismatch")
    _require(metadata.get("behavior_blind") is True, f"{role}: not behavior blind")
    _require(metadata.get("status") == "complete", f"{role}: incomplete")
    _require(metadata.get("status_counts") == {"ok": 36}, f"{role}: row failures")
    _require(metadata.get("missing_rows") == [], f"{role}: missing rows")
    _require(metadata.get("errors") == [], f"{role}: recorded errors")
    _require(metadata.get("protocol_sha256") == protocol_sha, f"{role}: protocol mismatch")
    _require(
        set(metadata.get("source_paths", {})) == set(metadata.get("source_hashes", {})),
        f"{role}: source provenance keys differ",
    )
    family_id = str(metadata["training_family_id"])
    training_seed = int(metadata["training_seed"])
    _require(
        family_id == f"pldm_canonical_seed{training_seed}",
        f"{role}: family id/seed mismatch",
    )
    return family_id, training_seed


def build_diagnostic_input(
    *,
    protocol_path: Path,
    atr_path: Path,
    smpr_path: Path,
) -> dict[str, Any]:
    protocol_original = protocol_path.read_bytes()
    protocol_mtime = protocol_path.stat().st_mtime_ns
    protocol, protocol_sha = _frozen_protocol(protocol_path)
    atr = _load_strict(atr_path)
    smpr = _load_strict(smpr_path)
    atr_meta = atr.get("metadata", {})
    smpr_meta = smpr.get("metadata", {})
    atr_family, atr_seed = _validate_metadata(
        atr_meta,
        role="pldm_canonical_external_atr",
        schema="paper1-acpc-horizon-v2-1.0",
        protocol_sha=protocol_sha,
    )
    smpr_family, smpr_seed = _validate_metadata(
        smpr_meta,
        role="pldm_canonical_external_smpr",
        schema="paper1-smpr-v2-merged-1.0",
        protocol_sha=protocol_sha,
    )
    _require((atr_family, atr_seed) == (smpr_family, smpr_seed), "ATR/SMPR family mismatch")
    _require(
        atr_meta.get("implementation_hashes", {}).get("canonical_metric")
        == protocol["source_hashes"]["canonical_metric"],
        "ATR canonical metric hash mismatch",
    )
    for local_key, protocol_key in (
        ("canonical_metric", "canonical_metric"),
        ("semantic_margin", "semantic_margin"),
        ("runner", "smpr_runner"),
    ):
        _require(
            smpr_meta.get("implementation_hashes", {}).get(local_key)
            == protocol["source_hashes"][protocol_key],
            f"SMPR implementation hash mismatch: {local_key}",
        )
    _require(
        smpr_meta.get("source_hashes", {}).get("reference_atr") == _sha256(atr_path),
        "SMPR source does not reference the supplied ATR artifact",
    )
    expected_keys = {(task, rho) for task in TASKS for rho in RHO_GRID}
    atr_index: dict[tuple[str, float], Mapping[str, Any]] = {}
    for row in atr.get("rows", []):
        key = (str(row.get("task")), _finite(row.get("training_rho"), name="ATR rho"))
        _require(key not in atr_index, f"duplicate ATR row {key}")
        _require(row.get("status") == "ok", f"{key}: ATR row not ok")
        _require(row.get("radius_metric") == protocol["radius_metric"], f"{key}: ATR metric mismatch")
        atr_index[key] = row
    smpr_index: dict[tuple[str, float], Mapping[str, Any]] = {}
    for row in smpr.get("rows", []):
        key = (str(row.get("task")), _finite(row.get("training_rho"), name="SMPR rho"))
        _require(key not in smpr_index, f"duplicate SMPR row {key}")
        _require(row.get("status") == "ok", f"{key}: SMPR row not ok")
        _require(row.get("atr_reference_match") is True, f"{key}: ATR reference mismatch")
        _require(
            _finite(row.get("atr_reference_abs_error"), name=f"{key}/ATR error") <= 1e-6,
            f"{key}: ATR reference error exceeds tolerance",
        )
        smpr_index[key] = row
    _require(set(atr_index) == expected_keys, "PLDM ATR coverage mismatch")
    _require(set(smpr_index) == expected_keys, "PLDM SMPR coverage mismatch")

    rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys, key=lambda value: (TASKS.index(value[0]), value[1])):
        task, rho = key
        atr_row = atr_index[key]
        smpr_row = smpr_index[key]
        atr_value = _finite(atr_row["atr_horizon_v2_q90"], name=f"{key}/ATR")
        reference_value = _finite(
            smpr_row["reference_atr_horizon_v2_q90"],
            name=f"{key}/SMPR reference ATR",
        )
        _require(
            math.isclose(atr_value, reference_value, rel_tol=1e-6, abs_tol=1e-6),
            f"{key}: joined ATR mismatch",
        )
        _require(
            Path(str(atr_row.get("model_file"))).resolve()
            == Path(str(smpr_row.get("model_file"))).resolve(),
            f"{key}: ATR/SMPR checkpoint path mismatch",
        )
        _require(
            atr_row.get("checkpoint_sha256")
            == smpr_row.get("checkpoint_sha256"),
            f"{key}: ATR/SMPR checkpoint hash mismatch",
        )
        _require(
            _sha256(Path(str(atr_row["model_file"])))
            == atr_row["checkpoint_sha256"],
            f"{key}: live checkpoint hash mismatch",
        )
        rows.append(
            {
                "status": "ok",
                "model_family": "PLDM",
                "training_family_id": atr_family,
                "training_seed": atr_seed,
                "task": task,
                "training_rho": rho,
                "stressor_family": "gaussian",
                "stressor_severity": float(
                    protocol["diagnostic_sampling"]["evaluation_noise_std"]
                ),
                "atr_horizon_v2_q90": atr_value,
                "smpr": _finite(smpr_row["smpr"], name=f"{key}/SMPR"),
                "split_name": "E2",
            }
        )
    _require(all(tuple(row) == ROW_FIELDS for row in rows), "diagnostic row allowlist mismatch")
    _require(
        not any(
            part in field.lower()
            for field in ROW_FIELDS
            for part in FORBIDDEN_FIELD_PARTS
        ),
        "diagnostic field allowlist leaks behavior",
    )
    source_paths = {
        "protocol": str(protocol_path),
        "atr_pldm_canonical": str(atr_path),
        "smpr_pldm_canonical": str(smpr_path),
    }
    source_hashes = {
        "protocol": protocol_sha,
        "atr_pldm_canonical": _sha256(atr_path),
        "smpr_pldm_canonical": _sha256(smpr_path),
    }
    script_path = Path(__file__).resolve()
    artifact = {
        "metadata": {
            "schema_version": "paper1-frozen-diagnostic-input-1.0",
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "protocol_sha256": protocol_sha,
            "model_family": "PLDM",
            "training_family_id": atr_family,
            "training_seed": atr_seed,
            "training_seed_semantics": "one independently trained PLDM checkpoint family",
            "evaluation_seed_semantics": "conditional evaluation variability, not training-run replication",
            "split_name": "E2",
            "status": "complete",
            "status_counts": {"ok": len(rows)},
            "missing_rows": [],
            "errors": [],
            "behavior_blind": True,
            "threshold_search_available": False,
            "strict_external_contract": "pldm_canonical_gaussian_e2_v1",
            "operator_blinding": (
                "not claimed; raw diagnostic runners consumed eval manifests, "
                "while this frozen-apply input contains no behavior fields"
            ),
        },
        "rows": rows,
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
    parser.add_argument("--atr", type=Path, required=True)
    parser.add_argument("--smpr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.out.resolve() == args.protocol.resolve():
        raise ValueError("diagnostic output cannot be the frozen protocol path")
    artifact = build_diagnostic_input(
        protocol_path=args.protocol,
        atr_path=args.atr,
        smpr_path=args.smpr,
    )
    _write_exclusive(args.out, artifact)
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
