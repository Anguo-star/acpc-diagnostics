#!/usr/bin/env python3
"""Freeze the numerical-correction v2 ACPC planner-stability panel."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V1_PROTOCOL = ROOT / "paper1/config/acpc_planner_stability_protocol_v1.json"
V1_EXECUTION = ROOT / "paper1/config/acpc_planner_stability_execution_v1.json"
V1_RESULT_ROOT = ROOT / "paper1/results/acpc_planner_stability_v1/formal"
V2_PROTOCOL = ROOT / "paper1/config/acpc_planner_stability_protocol_v2.json"
V2_EXECUTION = ROOT / "paper1/config/acpc_planner_stability_execution_v2.json"
V2_RESULT_ROOT = "paper1/results/acpc_planner_stability_v2/formal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def v1_attempt_records(execution: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for shard in execution["authorized_shards"]:
        path = ROOT / shard["output_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        metadata = payload.get("metadata", {})
        records.append(
            {
                "shard_id": shard["shard_id"],
                "analysis_role": shard["analysis_role"],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "status": metadata.get("status"),
                "expected_rows": metadata.get("expected_rows"),
                "actual_rows": metadata.get("actual_rows"),
                "error_count": metadata.get("error_count"),
                "errors": payload.get("errors", []),
            }
        )
    fixed = [r for r in records if r["analysis_role"] == "fixed_reduced"]
    adaptive = [r for r in records if r["analysis_role"] != "fixed_reduced"]
    if len(fixed) != 8 or len(adaptive) != 16:
        raise RuntimeError("unexpected v1 role counts")
    if not all(
        r["status"] == "partial"
        and r["actual_rows"] == 100
        and r["error_count"] == 3
        and all(
            e.get("error") == "RuntimeError('signed perturbed-gap identity mismatch')"
            for e in r["errors"]
        )
        for r in fixed
    ):
        raise RuntimeError("v1 fixed-pool failure pattern changed")
    if not all(
        r["status"] == "complete"
        and r["actual_rows"] == r["expected_rows"]
        and r["error_count"] == 0
        for r in adaptive
    ):
        raise RuntimeError("v1 adaptive result pattern changed")
    return records


def main() -> int:
    if V2_RESULT_ROOT and (ROOT / V2_RESULT_ROOT).exists():
        raise SystemExit("v2 result root already exists; refuse post-execution freeze")
    v1_protocol = read_json(V1_PROTOCOL)
    v1_execution = read_json(V1_EXECUTION)
    if v1_protocol.get("status") != "frozen_pre_execution":
        raise RuntimeError("v1 protocol is not frozen")
    if v1_execution.get("status") != "frozen_pre_execution":
        raise RuntimeError("v1 execution manifest is not frozen")
    if v1_execution["parent_protocol"]["sha256"] != sha256(V1_PROTOCOL):
        raise RuntimeError("v1 parent-protocol hash mismatch")

    now = datetime.now(timezone.utc).isoformat()
    attempt_records = v1_attempt_records(v1_execution)
    protocol = copy.deepcopy(v1_protocol)
    protocol.update(
        {
            "schema_version": "paper1-acpc-planner-stability-protocol-2.0",
            "protocol_id": "paper1-acpc-planner-stability-v2",
            "frozen_at_utc": now,
            "analysis_commit_parent": git_head(),
            "status": "frozen_pre_execution",
            "immutable": True,
        }
    )
    protocol["provenance"].update(
        {
            "v2_role": (
                "prospectively frozen numerical-correction replication of every v1 "
                "shard; no outcome-dependent parameter or threshold changes"
            ),
            "v1_results_retained": True,
        }
    )
    protocol["correction_history"] = {
        "supersedes_protocol": {
            "path": str(V1_PROTOCOL.relative_to(ROOT)),
            "sha256": sha256(V1_PROTOCOL),
        },
        "supersedes_execution": {
            "path": str(V1_EXECUTION.relative_to(ROOT)),
            "sha256": sha256(V1_EXECUTION),
        },
        "observed_before_v2_freeze": (
            "all eight v1 fixed_reduced shards produced their 100 identity rows "
            "and rejected all three nonzero severities on the same float32 signed-gap "
            "identity; all sixteen adaptive shards completed"
        ),
        "root_cause": (
            "float32 cancellation in an algebraic identity guard at real cost scale; "
            "the candidate costs, candidate ordering, and exact winner are unchanged"
        ),
        "only_change": (
            "evaluate signed-gap algebra and derived cost differences in float64 after "
            "the already-computed float32 costs; all frozen arguments, estimands, "
            "candidate pools, thresholds, checkpoints, and claim gates are unchanged"
        ),
        "v1_results_must_be_retained": True,
        "v1_attempts": attempt_records,
    }
    source_paths = set(protocol["source_hashes"])
    source_paths.update(
        {
            "paper1/scripts/freeze_acpc_planner_stability_protocol_v2.py",
            "paper1/scripts/run_acpc_planner_stability_shards.py",
            "tests/test_paper1_acpc_planner_protocol_v2.py",
            "tests/test_paper1_acpc_planner_stability_v2.py",
            "tools/paper1_acpc_planner_stability_audit_v2.py",
        }
    )
    protocol["source_hashes"] = {
        path: sha256(ROOT / path) for path in sorted(source_paths)
    }
    write_json(V2_PROTOCOL, protocol)

    execution = copy.deepcopy(v1_execution)
    execution.update(
        {
            "schema_version": "paper1-acpc-planner-stability-execution-2.0",
            "protocol_id": "paper1-acpc-planner-stability-v2",
            "created_utc": now,
            "parent_protocol": {
                "path": str(V2_PROTOCOL.relative_to(ROOT)),
                "sha256": sha256(V2_PROTOCOL),
            },
            "result_root": V2_RESULT_ROOT,
            "status": "frozen_pre_execution",
            "immutable": True,
            "post_result_protocol_edits_forbidden": True,
            "supersedes_execution": {
                "path": str(V1_EXECUTION.relative_to(ROOT)),
                "sha256": sha256(V1_EXECUTION),
            },
            "v1_results_must_be_retained": True,
        }
    )
    for shard in execution["authorized_shards"]:
        shard["output_path"] = f"{V2_RESULT_ROOT}/{shard['shard_id']}.json"
        if shard["analysis_role"] == "fixed_reduced":
            shard["runner"] = "tools/paper1_acpc_planner_stability_audit_v2.py"
    execution["source_hashes"] = {
        path: protocol["source_hashes"][path]
        for path in (
            "paper1/scripts/run_acpc_planner_stability_shards.py",
            "tools/paper1_acpc_planner_stability_audit_v2.py",
            "tools/paper1_acpc_adaptive_cem_audit.py",
        )
    }
    write_json(V2_EXECUTION, execution)
    print(f"wrote {V2_PROTOCOL.relative_to(ROOT)}")
    print(f"wrote {V2_EXECUTION.relative_to(ROOT)}")
    print("v1 attempts bound:", len(attempt_records))
    print("v2 authorized shards:", len(execution["authorized_shards"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
