#!/usr/bin/env python3
"""Audit whether all canonical Paper1 PLDM checkpoints deserialize.

The audit is intentionally limited to checkpoint resolution and
``tools.paper1_phase0_acpc.load_model`` deserialization. It does not run a
forward pass, load datasets, or claim that the full remediation smoke benchmark
has passed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CANONICAL = ROOT / "assets/paper1_data/canonical_evals_pldm_20260522.json"
DEFAULT_OUT = ROOT / "paper1/results/pldm_checkpoint_loadability_v1.json"
EXPECTED_ROWS = 36


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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


def _manifest_rows(
    canonical: Mapping[str, Any],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    rows: list[tuple[str, str, Mapping[str, Any]]] = []
    for task in sorted(canonical):
        task_rows = canonical[task]
        if not isinstance(task_rows, Mapping):
            raise TypeError(f"canonical task block {task!r} must be a mapping")
        for std_key in sorted(task_rows, key=float):
            entry = task_rows[std_key]
            if not isinstance(entry, Mapping):
                raise TypeError(f"canonical row {task}/{std_key} must be a mapping")
            rows.append((str(task), str(std_key), entry))
    return rows


def _timing_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=DEFAULT_CANONICAL,
        help="Canonical PLDM eval manifest (default: %(default)s).",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        type=Path,
        default=[],
        help="Additional model root; repeat for multiple task roots.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to load_model (default: cpu).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path (default: %(default)s).",
    )
    return parser


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit_started = time.perf_counter()
    canonical_path = _repo_path(args.canonical).resolve()
    source_read_started = time.perf_counter()
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    source_read_seconds = time.perf_counter() - source_read_started
    if not isinstance(canonical, Mapping):
        raise TypeError("canonical PLDM manifest must be a mapping")

    manifest_rows = _manifest_rows(canonical)
    model_roots = [_repo_path(path).resolve() for path in args.model_root]

    module_import_started = time.perf_counter()
    phase0 = importlib.import_module("tools.paper1_phase0_acpc")
    module_import_seconds = time.perf_counter() - module_import_started

    runtime_setup_started = time.perf_counter()
    phase0._ensure_runtime_deps()
    runtime_dependency_setup_seconds = time.perf_counter() - runtime_setup_started

    rows: list[dict[str, Any]] = []
    successful_load_times: list[float] = []
    first_successful_row: dict[str, Any] | None = None

    for index, (task, std_key, entry) in enumerate(manifest_rows, start=1):
        model_file, searched_paths = phase0.resolve_model_file(
            str(entry.get("path", "")),
            str(entry.get("subdir", "")),
            model_roots,
        )
        row: dict[str, Any] = {
            "row_index": index,
            "task": task,
            "std_key": std_key,
            "subdir": entry.get("subdir"),
            "manifest_path": entry.get("path"),
            "model_file": str(model_file) if model_file else None,
            "model_file_size_bytes": model_file.stat().st_size if model_file else None,
            "searched_paths": searched_paths,
            "status": "missing_model" if model_file is None else "pending",
            "load_seconds": None,
            "cleanup_seconds": None,
            "model_class": None,
            "error": None,
        }

        model = None
        if model_file is not None:
            load_started = time.perf_counter()
            try:
                model = phase0.load_model(str(model_file), args.device)
                row["load_seconds"] = time.perf_counter() - load_started
                row["model_class"] = (
                    f"{type(model).__module__}.{type(model).__qualname__}"
                )
                row["status"] = "ok"
                successful_load_times.append(row["load_seconds"])
                if first_successful_row is None:
                    first_successful_row = {
                        "row_index": index,
                        "task": task,
                        "std_key": std_key,
                    }
            except Exception as exc:  # noqa: BLE001 - preserve every row failure.
                row["load_seconds"] = time.perf_counter() - load_started
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                cleanup_started = time.perf_counter()
                if model is not None:
                    del model
                gc.collect()
                if args.device.startswith("cuda") and phase0.torch.cuda.is_available():
                    phase0.torch.cuda.empty_cache()
                row["cleanup_seconds"] = time.perf_counter() - cleanup_started

        rows.append(row)
        load_display = (
            "n/a" if row["load_seconds"] is None else f"{row['load_seconds']:.6f}"
        )
        print(
            f"[{index:02d}/{len(manifest_rows):02d}] {task} std={std_key} "
            f"status={row['status']} load_seconds={load_display}",
            flush=True,
        )

    status_counts = Counter(str(row["status"]) for row in rows)
    class_counts = Counter(
        str(row["model_class"]) for row in rows if row["model_class"] is not None
    )
    first_load_seconds = successful_load_times[0] if successful_load_times else None
    warm_load_times = successful_load_times[1:]
    setup_seconds = module_import_seconds + runtime_dependency_setup_seconds
    cold_end_to_end_seconds = (
        setup_seconds + first_load_seconds if first_load_seconds is not None else None
    )
    all_rows_loadable = (
        len(rows) == EXPECTED_ROWS
        and status_counts.get("ok", 0) == EXPECTED_ROWS
        and len(status_counts) == 1
    )

    return {
        "schema_version": "paper1-pldm-checkpoint-loadability-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": _display_path(canonical_path),
            "sha256": _sha256(canonical_path),
        },
        "code_commit": _git_commit(),
        "script": {
            "path": _display_path(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "model_roots": [str(path) for path in model_roots],
        "device": args.device,
        "timing_semantics": {
            "source_read_seconds": "JSON read and parse only.",
            "module_import_seconds": "Import tools.paper1_phase0_acpc only.",
            "runtime_dependency_setup_seconds": (
                "Import torch, stable-worldmodel, and representation-analysis dependencies before "
                "the first checkpoint load."
            ),
            "load_seconds": (
                "Per-row tools.paper1_phase0_acpc.load_model(path, device) only; excludes "
                "dependency setup and cleanup."
            ),
            "cleanup_seconds": "Per-row model deletion, Python GC, and optional CUDA cache cleanup.",
            "cold_end_to_end_seconds": (
                "module_import_seconds + runtime_dependency_setup_seconds + first_load_seconds."
            ),
            "warm_load_seconds": "Successful load_seconds after the first successful row.",
        },
        "summary": {
            "expected_rows": EXPECTED_ROWS,
            "manifest_rows": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "model_class_counts": dict(sorted(class_counts.items())),
            "all_rows_loadable": all_rows_loadable,
            "source_read_seconds": source_read_seconds,
            "module_import_seconds": module_import_seconds,
            "runtime_dependency_setup_seconds": runtime_dependency_setup_seconds,
            "setup_seconds": setup_seconds,
            "first_successful_row": first_successful_row,
            "first_load_seconds": first_load_seconds,
            "cold_end_to_end_seconds": cold_end_to_end_seconds,
            "warm_load_seconds": _timing_summary(warm_load_times),
            "all_successful_load_seconds": _timing_summary(successful_load_times),
            "successful_load_seconds_sum": sum(successful_load_times),
            "cleanup_seconds_sum": sum(
                float(row["cleanup_seconds"] or 0.0) for row in rows
            ),
            "audit_compute_wall_seconds": time.perf_counter() - audit_started,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "rows": rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run_audit(args)
    out = _repo_path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(out)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if payload["summary"]["all_rows_loadable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
