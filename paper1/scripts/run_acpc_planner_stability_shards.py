#!/usr/bin/env python3
"""Run one task queue from the frozen ACPC planner-stability manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "paper1/config/acpc_planner_stability_protocol_v1.json"
DEFAULT_ADDENDUM = (
    ROOT / "paper1/config/acpc_planner_stability_execution_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_argument(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    command.append("--" + name.replace("_", "-"))
    if isinstance(value, list):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def _expected_rows(shard: Mapping[str, Any]) -> int:
    arguments = shard["arguments"]
    return (
        int(arguments["n_blocks"])
        * len(arguments["severities"])
        * int(arguments["draws"])
    )


def valid_shard(
    path: Path,
    *,
    shard: Mapping[str, Any],
    protocol_hash: str,
    addendum_hash: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = payload.get("metadata", {})
    expected_rows = _expected_rows(shard)
    return bool(
        metadata.get("status") == "complete"
        and not payload.get("errors")
        and int(metadata.get("expected_rows", -1)) == expected_rows
        and int(metadata.get("actual_rows", -1)) == expected_rows
        and metadata.get("protocol_sha256") == protocol_hash
        and metadata.get("execution_addendum_sha256") == addendum_hash
        and metadata.get("script_sha256") == _sha256(ROOT / shard["runner"])
        and metadata.get("checkpoint_sha256") == shard["checkpoint_sha256"]
    )


def _command(
    shard: Mapping[str, Any],
    *,
    protocol: Path,
    addendum: Path,
    device: str,
) -> list[str]:
    command = [sys.executable, str(ROOT / shard["runner"])]
    for name, value in shard["arguments"].items():
        _append_argument(command, name, value)
    command.extend(["--checkpoint", shard["checkpoint_path"]])
    command.extend(["--shard-id", shard["shard_id"]])
    command.extend(["--protocol", str(protocol)])
    command.extend(["--execution-addendum", str(addendum)])
    command.extend(["--out", str(ROOT / shard["output_path"])])
    command.extend(["--device", device])
    return [
        "timeout",
        "--signal=TERM",
        "--kill-after=60s",
        f"{int(shard['timeout_seconds'])}s",
        *command,
    ]


def _retain_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retained = path.with_name(f"{path.name}.attempt_{stamp}")
    counter = 1
    while retained.exists():
        retained = path.with_name(f"{path.name}.attempt_{stamp}_{counter}")
        counter += 1
    shutil.move(path, retained)
    return retained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "full"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    protocol = args.protocol.resolve()
    addendum = args.addendum.resolve()
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    addendum_payload = json.loads(addendum.read_text(encoding="utf-8"))
    protocol_hash = _sha256(protocol)
    addendum_hash = _sha256(addendum)
    if protocol_payload.get("status") != "frozen_pre_execution":
        raise SystemExit("protocol is not frozen_pre_execution")
    if addendum_payload.get("status") != "frozen_pre_execution":
        raise SystemExit("execution addendum is not frozen_pre_execution")
    if addendum_payload["parent_protocol"]["sha256"] != protocol_hash:
        raise SystemExit("frozen protocol hash mismatch")
    execution = protocol_payload["execution"]
    if int(execution.get("max_concurrent_jobs", -1)) != 4:
        raise SystemExit("protocol does not authorize four concurrent jobs")
    if not bool(execution.get("one_process_per_gpu")):
        raise SystemExit("protocol does not require one process per GPU")

    task_key = args.task.casefold()
    shards = [
        shard
        for shard in addendum_payload["authorized_shards"]
        if str(shard["arguments"]["task"]).casefold() == task_key
    ]
    if not shards:
        allowed = sorted(
            {str(s["arguments"]["task"]) for s in addendum_payload["authorized_shards"]}
        )
        raise SystemExit(f"unknown task {args.task!r}; choose one of {allowed}")

    print(f"mode={args.mode} task={args.task} selected_shards={len(shards)}", flush=True)
    if args.mode == "plan":
        for shard in shards:
            print(shard["shard_id"], shard["runner"], shard["output_path"])
        return 0

    threads = str(execution["native_threads_per_job"])
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_root = ROOT / addendum_payload["result_root"] / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, shard in enumerate(shards, start=1):
        output = ROOT / shard["output_path"]
        if valid_shard(
            output,
            shard=shard,
            protocol_hash=protocol_hash,
            addendum_hash=addendum_hash,
        ):
            print(f"[{index}/{len(shards)}] skip valid {shard['shard_id']}", flush=True)
            continue
        retained = _retain_existing(output)
        if retained is not None:
            print(f"retained prior artifact as {retained}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = log_root / f"{shard['shard_id']}.log"
        print(f"[{index}/{len(shards)}] run {shard['shard_id']}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                _command(
                    shard,
                    protocol=protocol,
                    addendum=addendum,
                    device=args.device,
                ),
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        valid = valid_shard(
            output,
            shard=shard,
            protocol_hash=protocol_hash,
            addendum_hash=addendum_hash,
        )
        print(
            f"[{index}/{len(shards)}] exit={completed.returncode} valid={valid} "
            f"log={log}",
            flush=True,
        )
        if completed.returncode != 0 or not valid:
            failures += 1
    print(f"task={args.task} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
