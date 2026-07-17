#!/usr/bin/env python3
"""Freeze the source-complete v4 three-seed planner replication protocol."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper1.scripts.freeze_acpc_planner_three_seed_protocol import (
    ROOT,
    authorized_shards,
    git_head,
    read_json,
    sha256,
    validate_reference_panel,
    write_json,
)


V3_PROTOCOL = ROOT / "paper1/config/acpc_planner_stability_protocol_v3.json"
V3_EXECUTION = ROOT / "paper1/config/acpc_planner_stability_execution_v3.json"
V3_RESULT_ROOT = ROOT / "paper1/results/acpc_planner_stability_v3/formal"
V4_PROTOCOL = ROOT / "paper1/config/acpc_planner_stability_protocol_v4.json"
V4_EXECUTION = ROOT / "paper1/config/acpc_planner_stability_execution_v4.json"
V4_RESULT_ROOT = "paper1/results/acpc_planner_stability_v4/formal"

SOURCE_PATHS = (
    "paper1/scripts/freeze_acpc_planner_three_seed_protocol.py",
    "paper1/scripts/freeze_acpc_planner_three_seed_protocol_v4.py",
    "paper1/scripts/run_acpc_planner_stability_shards.py",
    "paper1/scripts/summarize_acpc_planner_stability.py",
    "paper1/scripts/summarize_acpc_planner_three_seed.py",
    "tests/test_paper1_acpc_planner_three_seed.py",
    "tests/test_paper1_acpc_planner_three_seed_v4.py",
    "tools/paper1_acpc_planner_stability_audit_v2.py",
    "tools/paper1_acpc_adaptive_cem_audit.py",
    "tools/paper1_operational_protocol.py",
)


def _source_drift(protocol: dict[str, Any]) -> list[dict[str, str]]:
    drift = []
    for path, expected in protocol["source_hashes"].items():
        actual = sha256(ROOT / path)
        if actual != expected:
            drift.append({"path": path, "expected": expected, "actual": actual})
    return drift


def main() -> int:
    if V4_PROTOCOL.exists() or V4_EXECUTION.exists():
        raise SystemExit("v4 protocol already exists; refuse post-freeze overwrite")
    if V4_RESULT_ROOT and (ROOT / V4_RESULT_ROOT).exists():
        raise SystemExit("v4 result root already exists; refuse post-execution freeze")
    if V3_RESULT_ROOT.exists():
        raise SystemExit("v3 was executed; it cannot be superseded as unexecuted")

    v3_protocol = read_json(V3_PROTOCOL)
    v3_execution = read_json(V3_EXECUTION)
    if v3_protocol.get("status") != "frozen_pre_execution":
        raise RuntimeError("v3 protocol is not frozen")
    if v3_execution.get("status") != "frozen_pre_execution":
        raise RuntimeError("v3 execution is not frozen")
    if v3_execution["parent_protocol"]["sha256"] != sha256(V3_PROTOCOL):
        raise RuntimeError("v3 parent-protocol hash mismatch")
    drift = _source_drift(v3_protocol)
    if [row["path"] for row in drift] != [
        "paper1/scripts/summarize_acpc_planner_three_seed.py"
    ]:
        raise RuntimeError(f"unexpected v3 pre-execution source drift: {drift}")

    reference = validate_reference_panel()
    checkpoints, shards = authorized_shards()
    if len(checkpoints) != 16 or len(shards) != 48:
        raise RuntimeError("unexpected replication checkpoint or shard count")
    for shard in shards:
        shard["output_path"] = f"{V4_RESULT_ROOT}/{shard['shard_id']}.json"

    now = datetime.now(timezone.utc).isoformat()
    source_hashes = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    protocol = copy.deepcopy(v3_protocol)
    protocol.update(
        {
            "schema_version": "paper1-acpc-planner-stability-protocol-4.0",
            "protocol_id": "paper1-acpc-planner-stability-v4",
            "frozen_at_utc": now,
            "analysis_commit_parent": git_head(),
            "status": "frozen_pre_execution",
            "immutable": True,
            "reference_seed3074_panel": reference,
            "checkpoints": checkpoints,
            "source_hashes": source_hashes,
        }
    )
    protocol["pre_execution_correction"] = {
        "supersedes_unexecuted_protocol": {
            "path": str(V3_PROTOCOL.relative_to(ROOT)),
            "sha256": sha256(V3_PROTOCOL),
        },
        "supersedes_unexecuted_execution": {
            "path": str(V3_EXECUTION.relative_to(ROOT)),
            "sha256": sha256(V3_EXECUTION),
        },
        "v3_result_root_absent_at_v4_freeze": True,
        "issue": (
            "v3 omitted the transitive legacy summarizer dependency that supplies "
            "the frozen Ridge and result-validation helpers"
        ),
        "correction": (
            "bind both summarizer source files and derive the output protocol_id "
            "from the active protocol"
        ),
        "v3_source_drift_after_correction": drift,
        "unchanged": [
            "checkpoints",
            "candidate pools and CEM budgets",
            "trajectory/probe/CEM seeds",
            "responses and feature sets",
            "leave-one-task-out split",
            "per-seed aggregation and claim gates",
        ],
        "results_observed_before_correction": False,
    }
    write_json(V4_PROTOCOL, protocol)

    execution = copy.deepcopy(v3_execution)
    execution.update(
        {
            "schema_version": "paper1-acpc-planner-stability-execution-4.0",
            "protocol_id": protocol["protocol_id"],
            "created_utc": now,
            "parent_protocol": {
                "path": str(V4_PROTOCOL.relative_to(ROOT)),
                "sha256": sha256(V4_PROTOCOL),
            },
            "result_root": V4_RESULT_ROOT,
            "authorized_shards": shards,
            "status": "frozen_pre_execution",
            "immutable": True,
            "supersedes_unexecuted_execution": protocol[
                "pre_execution_correction"
            ]["supersedes_unexecuted_execution"],
        }
    )
    execution["source_hashes"] = {
        path: source_hashes[path]
        for path in (
            "paper1/scripts/run_acpc_planner_stability_shards.py",
            "tools/paper1_acpc_planner_stability_audit_v2.py",
            "tools/paper1_acpc_adaptive_cem_audit.py",
        )
    }
    write_json(V4_EXECUTION, execution)
    print(f"wrote {V4_PROTOCOL.relative_to(ROOT)}")
    print(f"wrote {V4_EXECUTION.relative_to(ROOT)}")
    print("superseded unexecuted v3; observed results: 0")
    print(f"new authorized shards: {len(shards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
