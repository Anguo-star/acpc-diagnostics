#!/usr/bin/env python3
"""Snapshot Paper 1 remediation inputs without changing canonical artifacts.

This training-free audit records immutable hashes, validates released JSON,
resolves checkpoint paths, captures the legacy ATR semantics, and reports every
Phase-0 blocker. Supply ``--model-root`` after model/data mounts are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.paper1_phase0_acpc import TASK_DATASETS, resolve_model_file
from tools.paper1_unseen_eval_grid import FAMILY_META


ROOT = Path(__file__).resolve().parents[2]
TARGET_COMMIT = "c943fdf75cd71bc08e5466e1700676069728b7d2"
TARGET_PLAN_BLOB = "62677a1b2a03e57a3f21dab9bc13d6bf52c438fc"
DEFAULT_OUT = ROOT / "paper1/results/remediation_phase0_snapshot_v1.json"
DEFAULT_REPORT = ROOT / "paper1/results/remediation_phase0_snapshot_v1.md"

SNAPSHOT_PATHS = (
    "paper1/main.pdf",
    "paper1/main_blind.pdf",
    "paper1/main.tex",
    "DATA_MANIFEST.md",
    "paper1/docs/PAPER1_OFF_AXIS_THEORY_SMPR_NOVELTY_REMEDIATION_PLAN_20260710.md",
    "assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json",
    "assets/paper1_data/training_seed_gaussian_lockbox.json",
    "assets/paper1_data/training_seed_eval_manifests/lewm_seed3072_evals.json",
    "assets/paper1_data/training_seed_eval_manifests/lewm_seed3073_evals.json",
    "assets/paper1_data/training_seed_eval_manifests/lewm_seed3074_evals.json",
    "assets/paper1_data/acpc_phase0_lewm_three_seed.json",
    "assets/paper1_data/semantic_task_grounded_margin_lewm_three_seed.json",
    "assets/paper1_data/margin_flip_curve_lewm_three_seed.json",
    "assets/paper1_data/canonical_evals_pldm_20260522.json",
    "assets/paper1_data/canonical_diagnostics_pldm_20260522.json",
    "assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json",
    "assets/paper1_data/acpc_basin_diagnostics_pldm.json",
    "assets/paper1_data/acpc_phase0_clean_goal_seed9101.json",
    "assets/paper1_data/unseen_origin_vs_std008_strongest_s3072.json",
    "assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json",
    "assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json",
    "assets/paper1_data/unseen_phase0_acpc_fullstress.json",
    "assets/paper1_data/unseen_atr_smpr_summary_20260707.json",
    "assets/paper1_data/prospective_validation_summary.json",
    "assets/paper1_data/canonical_blur_baselines_20260523.json",
    "assets/paper1_data/target_view_closed_loop_summary.json",
    "assets/paper1_data/target_view_diagnostic_manifest_v1.json",
    "assets/paper1_data/cem_trace_audit_20260704.json",
    "assets/paper1_data/robust_cem_pilot_20260704.json",
    "assets/paper1_data/robust_cem_eval100x3_iteration_summary_20260705.json",
    "paper1/results/sample_level_certificate_full_sweep_audit.json",
    "paper1/results/jvp_hutchinson_sensitivity_audit.json",
    "paper1/results/gaussian_sensitivity_audit.json",
    "paper1/results/remediation_phase1_smoke_v2.json",
)

PLDM_EVALS = ROOT / "assets/paper1_data/canonical_evals_pldm_20260522.json"
PLDM_LOADABILITY = ROOT / "paper1/results/pldm_checkpoint_loadability_v1.json"
TARGET_VIEW_SUMMARY = ROOT / "assets/paper1_data/target_view_closed_loop_summary.json"
TARGET_VIEW_BUILDER = ROOT / "tools/paper1_target_view_diagnostic_manifest.py"
TARGET_VIEW_MANIFEST = (
    ROOT / "assets/paper1_data/target_view_diagnostic_manifest_v1.json"
)
PHASE1_SMOKE = ROOT / "paper1/results/remediation_phase1_smoke_v2.json"
LEGACY_PHASE0 = ROOT / "tools/paper1_phase0_acpc.py"
LEGACY_SHIFT = ROOT / "tools/repr_analysis/predictor_sensitivity.py"
LEGACY_FULL_SWEEP = ROOT / "paper1/scripts/build_full_sweep_diagnostics.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, repr(exc)
    output = proc.stdout.strip()
    if proc.returncode and proc.stderr.strip():
        output = proc.stderr.strip()
    return proc.returncode, output


def _json_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - preserve parse failures in the audit.
        return {"parse_status": "error", "error": repr(exc)}
    summary: dict[str, Any] = {
        "parse_status": "ok",
        "top_level_type": type(value).__name__,
    }
    if isinstance(value, dict):
        summary["top_level_keys"] = sorted(str(key) for key in value)
    elif isinstance(value, list):
        summary["top_level_length"] = len(value)
    return summary


def _file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    record: dict[str, Any] = {"path": relative, "exists": path.is_file()}
    if not path.is_file():
        record["status"] = "missing"
        return record
    record.update(
        {"status": "ok", "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    )
    if path.suffix == ".json":
        record["json"] = _json_summary(path)
        if record["json"]["parse_status"] != "ok":
            record["status"] = "invalid_json"
    return record


def _pldm_checkpoint_audit(model_roots: Sequence[Path]) -> dict[str, Any]:
    try:
        data = json.loads(PLDM_EVALS.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "blocked_invalid_manifest",
            "manifest": str(PLDM_EVALS.relative_to(ROOT)),
            "expected_rows": 36,
            "manifest_rows": 0,
            "resolved_model_files": 0,
            "missing_model_files": 36,
            "error": repr(exc),
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    for task in sorted(data):
        task_rows = data[task]
        if not isinstance(task_rows, Mapping):
            continue
        for std_key, entry in sorted(task_rows.items(), key=lambda item: float(item[0])):
            model_file, tried = resolve_model_file(
                str(entry.get("path", "")),
                str(entry.get("subdir", "")),
                model_roots,
            )
            rows.append(
                {
                    "task": task,
                    "std_key": std_key,
                    "subdir": entry.get("subdir"),
                    "manifest_path": entry.get("path"),
                    "resolved_model_file": str(model_file) if model_file else None,
                    "resolution_status": "resolved" if model_file else "missing",
                    "searched_paths": tried,
                    "deserialization_status": "not_run",
                }
            )
    resolved = sum(row["resolution_status"] == "resolved" for row in rows)
    expected = 36
    status = (
        "ready_for_deserialization_smoke"
        if len(rows) == expected and resolved == expected
        else "blocked_missing_models"
    )
    return {
        "status": status,
        "manifest": str(PLDM_EVALS.relative_to(ROOT)),
        "expected_rows": expected,
        "manifest_rows": len(rows),
        "resolved_model_files": resolved,
        "missing_model_files": len(rows) - resolved,
        "model_roots": [str(path) for path in model_roots],
        "rows": rows,
        "note": (
            "Path resolution is not a load test; deserialization and a forward "
            "smoke must still succeed."
        ),
    }


def _legacy_atr_audit() -> dict[str, Any]:
    phase0_text = LEGACY_PHASE0.read_text(encoding="utf-8")
    shift_text = LEGACY_SHIFT.read_text(encoding="utf-8")
    sweep_text = LEGACY_FULL_SWEEP.read_text(encoding="utf-8")
    checks = {
        "rollout_passed_to_shift_stats":
            "rollout_stats = _shift_stats(pred_clean, pred_noisy)" in phase0_text,
        "shift_stats_flattens_all_tokens":
            "clean = clean.reshape(-1, clean.size(-1))" in shift_text,
        "shift_stats_reports_l2_p90":
            '"l2_p90": _safe_quantile(l2, 0.9)' in shift_text,
        "downstream_normalizes_p90_by_transition_median":
            '"phase0_atr_q90": fnum(row.get("acpc_h_l2_p90")) / max(' in sweep_text,
    }
    return {
        "status": "confirmed" if all(checks.values()) else "implementation_drift_detected",
        "legacy_field": "acpc_h_l2_p90",
        "legacy_alias": "phase0_atr_q90 / atr_q90",
        "input_shape_before_aggregation": "B x H x D rollout predictions",
        "aggregation_unit": "flattened B*H step tokens",
        "quantile": 0.90,
        "normalization":
            "checkpoint-level p90 divided by clean transition L2 median downstream",
        "theorem_aligned_horizon_radius": False,
        "required_v2_semantics":
            "one weighted-stacked horizon L2 radius per anchor before checkpoint quantile",
        "source_files": {
            str(LEGACY_PHASE0.relative_to(ROOT)): _sha256(LEGACY_PHASE0),
            str(LEGACY_SHIFT.relative_to(ROOT)): _sha256(LEGACY_SHIFT),
            str(LEGACY_FULL_SWEEP.relative_to(ROOT)): _sha256(LEGACY_FULL_SWEEP),
        },
        "source_checks": checks,
    }


def _pldm_loadability_audit() -> dict[str, Any]:
    if not PLDM_LOADABILITY.is_file():
        return {
            "status": "pending",
            "artifact_path": str(PLDM_LOADABILITY.relative_to(ROOT)),
            "reason": "loadability artifact does not exist",
        }
    try:
        payload = json.loads(PLDM_LOADABILITY.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "invalid",
            "artifact_path": str(PLDM_LOADABILITY.relative_to(ROOT)),
            "error": repr(exc),
        }
    summary = payload.get("summary", {})
    source = payload.get("source", {})
    verified = (
        payload.get("schema_version") == "paper1-pldm-checkpoint-loadability-1.0"
        and payload.get("code_commit") == TARGET_COMMIT
        and source.get("sha256") == _sha256(PLDM_EVALS)
        and summary.get("expected_rows") == 36
        and summary.get("manifest_rows") == 36
        and summary.get("status_counts") == {"ok": 36}
        and summary.get("all_rows_loadable") is True
    )
    return {
        "status": "verified" if verified else "mismatch",
        "artifact_path": str(PLDM_LOADABILITY.relative_to(ROOT)),
        "artifact_sha256": _sha256(PLDM_LOADABILITY),
        "code_commit": payload.get("code_commit"),
        "device": payload.get("device"),
        "summary": summary,
    }


def _dataset_audit(model_roots: Sequence[Path]) -> dict[str, Any]:
    rows = []
    for task, dataset_name in TASK_DATASETS.items():
        candidates = []
        for root in model_roots:
            candidates.extend(
                [
                    root / f"{dataset_name}.h5",
                    root / "datasets" / f"{dataset_name}.h5",
                ]
            )
        resolved = next((path for path in candidates if path.is_file()), None)
        rows.append(
            {
                "task": task,
                "dataset_name": dataset_name,
                "resolved_path": str(resolved) if resolved else None,
                "status": "resolved" if resolved else "missing",
                "searched_paths": [str(path) for path in candidates],
            }
        )
    resolved_count = sum(row["status"] == "resolved" for row in rows)
    return {
        "status": "ready" if resolved_count == len(TASK_DATASETS) else "blocked_missing_datasets",
        "expected_datasets": len(TASK_DATASETS),
        "resolved_datasets": resolved_count,
        "missing_datasets": len(TASK_DATASETS) - resolved_count,
        "rows": rows,
    }


def _target_view_audit() -> dict[str, Any]:
    summary = (
        _json_summary(TARGET_VIEW_SUMMARY)
        if TARGET_VIEW_SUMMARY.is_file()
        else {"parse_status": "missing"}
    )
    builder_exists = TARGET_VIEW_BUILDER.is_file()
    manifest_summary = (
        _json_summary(TARGET_VIEW_MANIFEST)
        if TARGET_VIEW_MANIFEST.is_file()
        else {"parse_status": "missing"}
    )
    manifest_metadata: Mapping[str, Any] = {}
    if manifest_summary.get("parse_status") == "ok":
        manifest_payload = json.loads(
            TARGET_VIEW_MANIFEST.read_text(encoding="utf-8")
        )
        metadata = manifest_payload.get("_metadata")
        if isinstance(metadata, Mapping):
            manifest_metadata = metadata
    has_structured_provenance = (
        manifest_metadata.get("schema_version")
        == "paper1-target-view-diagnostic-manifest-1.0"
        and manifest_metadata.get("status") == "ok"
        and manifest_metadata.get("actual_rows") == 64
        and manifest_metadata.get("actual_matched_pairs") == 32
        and manifest_metadata.get("missing_rows") == []
        and manifest_metadata.get("errors") == []
        and manifest_metadata.get("builder", {}).get("sha256")
        == _sha256(TARGET_VIEW_BUILDER)
    )
    if not TARGET_VIEW_SUMMARY.is_file():
        status = "blocked_missing_summary"
    elif summary.get("parse_status") != "ok":
        status = "blocked_invalid_summary"
    elif not builder_exists:
        status = "blocked_missing_manifest_builder"
    elif manifest_summary.get("parse_status") != "ok":
        status = "blocked_missing_or_invalid_canonical_manifest"
    elif not has_structured_provenance:
        status = "blocked_manifest_contract_mismatch"
    else:
        status = "verified"
    return {
        "status": status,
        "summary_path": str(TARGET_VIEW_SUMMARY.relative_to(ROOT)),
        "summary": summary,
        "builder_path": str(TARGET_VIEW_BUILDER.relative_to(ROOT)),
        "builder_exists": builder_exists,
        "manifest_path": str(TARGET_VIEW_MANIFEST.relative_to(ROOT)),
        "manifest": manifest_summary,
        "manifest_sha256": (
            _sha256(TARGET_VIEW_MANIFEST)
            if TARGET_VIEW_MANIFEST.is_file()
            else None
        ),
        "manifest_rows": manifest_metadata.get("actual_rows"),
        "manifest_matched_pairs": manifest_metadata.get(
            "actual_matched_pairs"
        ),
        "legacy_live_conflicts": manifest_metadata.get(
            "legacy_conflict_count"
        ),
        "structured_metadata_present": has_structured_provenance,
        "note":
            "The canonical manifest binds live eval files and checkpoints; the "
            "legacy aggregate summary is advisory only.",
    }


def _smoke_benchmark_audit() -> dict[str, Any]:
    if not PHASE1_SMOKE.is_file():
        return {
            "status": "pending",
            "artifact_path": str(PHASE1_SMOKE.relative_to(ROOT)),
            "reason": "Phase-1 smoke artifact does not exist",
        }
    try:
        payload = json.loads(PHASE1_SMOKE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "invalid",
            "artifact_path": str(PHASE1_SMOKE.relative_to(ROOT)),
            "error": repr(exc),
        }
    metadata = payload.get("metadata", {})
    checks = payload.get("checks", {})
    rows = payload.get("benchmark_rows", [])
    verified = (
        metadata.get("schema_version")
        == "paper1-remediation-phase1-smoke-1.0"
        and metadata.get("code_commit") == TARGET_COMMIT
        and metadata.get("status") == "pass"
        and payload.get("gate") == "Gate 1"
        and payload.get("gate_status") == "pass"
        and len(rows) == 8
        and all(row.get("status") == "ok" for row in rows)
        and all(checks.values())
    )
    finite_wall = [
        float(row["wall_time_per_row"])
        for row in rows
        if row.get("wall_time_per_row") is not None
    ]
    finite_peak = [
        int(row["peak_gpu_memory"])
        for row in rows
        if row.get("peak_gpu_memory") is not None
    ]
    return {
        "status": "verified" if verified else "mismatch",
        "artifact_path": str(PHASE1_SMOKE.relative_to(ROOT)),
        "artifact_sha256": _sha256(PHASE1_SMOKE),
        "protocol": payload.get("selected_protocol"),
        "checks": checks,
        "benchmark_rows": len(rows),
        "wall_time_seconds_total": sum(finite_wall),
        "wall_time_seconds_max_per_row": max(finite_wall, default=None),
        "peak_gpu_memory_bytes": max(finite_peak, default=None),
        "source_hashes": metadata.get("source_hashes", {}),
    }


def _severity_audit() -> dict[str, Any]:
    families = {}
    for family in ("gaussian_blur", "resize"):
        metadata = FAMILY_META[family]
        values = list(metadata["default_magnitudes"])
        families[family] = {
            "launcher_env_key": metadata["env_key"],
            "enumerated_magnitudes": values,
            "identity": values[0],
            "non_identity": values[1:],
        }
    return {
        "status": "enumerated_from_implementation",
        "source": "tools/paper1_unseen_eval_grid.py:FAMILY_META",
        "source_sha256": _sha256(ROOT / "tools/paper1_unseen_eval_grid.py"),
        "families": families,
        "selection_status": "not_frozen_for_v2",
    }


def _environment_audit() -> dict[str, Any]:
    commands = {
        name: shutil.which(name)
        for name in ("python", "pytest", "latexmk", "pdflatex", "bibtex", "nvidia-smi")
    }
    gpu: dict[str, Any] = {"status": "not_available"}
    if commands["nvidia-smi"]:
        proc = subprocess.run(
            [
                commands["nvidia-smi"],
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            gpu = {"status": "ok", "device_count": len(rows), "devices": rows}
        else:
            gpu = {"status": "error", "error": proc.stderr.strip()}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "commands": commands,
        "gpu": gpu,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    git = payload["git_context"]
    files = payload["files"]
    missing = [row["path"] for row in files if row["status"] != "ok"]
    pldm = payload["pldm_checkpoint_audit"]
    loadability = payload["pldm_loadability_audit"]
    datasets = payload["dataset_audit"]
    target = payload["target_view_audit"]
    severity = payload["severity_audit"]["families"]
    lines = [
        "# Paper 1 remediation Phase 0 snapshot",
        "",
        f"- Created UTC: `{payload['created_utc']}`",
        f"- Local branch / HEAD: `{git['branch']}` / `{git['head']}`",
        f"- Required baseline: `{TARGET_COMMIT}`",
        f"- Baseline alignment: **{git['baseline_alignment']}**",
        f"- Target plan blob: **{git['target_plan_blob_status']}**",
        "",
        "## Immutable inputs",
        "",
        f"- Files audited: {len(files)}",
        f"- Missing or invalid: {len(missing)}",
    ]
    lines.extend(f"  - `{path}`" for path in missing)
    lines.extend(
        [
            "",
            "## Legacy ATR semantics",
            "",
            (
                "The current `acpc_h_l2_p90` is q90 over flattened `B×H` "
                "step tokens, followed downstream by checkpoint-level "
                "clean-transition-median normalization. It is not one stacked "
                "horizon radius per anchor."
            ),
            "",
            "## PLDM checkpoint audit",
            "",
            f"- Manifest rows: {pldm['manifest_rows']} / {pldm['expected_rows']}",
            f"- Resolved model files: {pldm['resolved_model_files']}",
            f"- Missing model files: {pldm['missing_model_files']}",
            f"- Status: **{pldm['status']}**",
            f"- Full CPU deserialization: **{loadability['status']}**",
            "",
            "## Dataset audit",
            "",
            f"- Resolved H5 datasets: {datasets['resolved_datasets']} / {datasets['expected_datasets']}",
            f"- Status: **{datasets['status']}**",
            "",
            "## Target-view audit",
            "",
            f"- Summary JSON: `{target['summary']['parse_status']}`",
            f"- Canonical manifest builder exists: `{target['builder_exists']}`",
            f"- Structured metadata present: `{target['structured_metadata_present']}`",
            f"- Canonical rows / pairs: {target['manifest_rows']} / {target['manifest_matched_pairs']}",
            f"- Status: **{target['status']}**",
            "",
            "## Enumerated stressor severities",
            "",
            f"- Gaussian blur: {', '.join(severity['gaussian_blur']['enumerated_magnitudes'])}",
            f"- Resize: {', '.join(severity['resize']['enumerated_magnitudes'])}",
            "- V2 severity selection remains unfrozen; no behavior results were used.",
            "",
            "## Gate 0",
            "",
            f"Status: **{payload['gate0']['status']}**",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["gate0"]["blockers"])
    lines.extend(
        [
            "",
            "The runtime smoke and target-view manifest are accepted only when "
            "their stored schema, row counts, checks, hashes, and current builder "
            "binding all validate.",
            "",
        ]
    )
    return "\n".join(lines)


def build_snapshot(model_roots: Sequence[Path]) -> dict[str, Any]:
    head_rc, head = _git("rev-parse", "HEAD")
    branch_rc, branch = _git("branch", "--show-current")
    plan_rc, plan_blob = _git(
        "hash-object",
        "paper1/docs/PAPER1_OFF_AXIS_THEORY_SMPR_NOVELTY_REMEDIATION_PLAN_20260710.md",
    )
    files = [_file_record(path) for path in SNAPSHOT_PATHS]
    pldm = _pldm_checkpoint_audit(model_roots)
    loadability = _pldm_loadability_audit()
    datasets = _dataset_audit(model_roots)
    target_view = _target_view_audit()
    smoke_benchmark = _smoke_benchmark_audit()
    missing_inputs = [row["path"] for row in files if row["status"] != "ok"]
    blockers = []
    if head_rc or head != TARGET_COMMIT:
        blockers.append(
            f"local HEAD {head or '<unavailable>'} is not requested baseline {TARGET_COMMIT}"
        )
    if missing_inputs:
        blockers.append(f"{len(missing_inputs)} required inputs are missing or invalid")
    if pldm["status"] != "ready_for_deserialization_smoke":
        blockers.append(
            f"PLDM model resolution incomplete "
            f"({pldm['resolved_model_files']}/{pldm['expected_rows']})"
        )
    if loadability["status"] != "verified":
        blockers.append(
            f"PLDM full deserialization audit is not verified: {loadability['status']}"
        )
    if datasets["status"] != "ready":
        blockers.append(
            f"dataset resolution incomplete "
            f"({datasets['resolved_datasets']}/{datasets['expected_datasets']})"
        )
    if target_view["status"] != "verified":
        blockers.append(f"target-view manifest not buildable: {target_view['status']}")
    if smoke_benchmark["status"] != "verified":
        blockers.append(
            "checkpoint/dataset-dependent runtime smoke benchmark is not "
            f"verified: {smoke_benchmark['status']}"
        )
    return {
        "schema_version": "paper1-remediation-phase0-snapshot-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": list(SNAPSHOT_PATHS),
        "source_hashes": {
            row["path"]: row.get("sha256") for row in files if row.get("sha256")
        },
        "git_context": {
            "repository": "qun-team/wm_exp",
            "required_branch": "ag/dev",
            "required_commit": TARGET_COMMIT,
            "head": head if head_rc == 0 else None,
            "branch": branch if branch_rc == 0 else None,
            "baseline_alignment": "exact" if head == TARGET_COMMIT else "mismatch",
            "target_plan_blob_expected": TARGET_PLAN_BLOB,
            "target_plan_blob_observed": plan_blob if plan_rc == 0 else None,
            "target_plan_blob_status":
                "verified" if plan_blob == TARGET_PLAN_BLOB else "mismatch",
        },
        "files": files,
        "legacy_atr_semantics": _legacy_atr_audit(),
        "pldm_checkpoint_audit": pldm,
        "pldm_loadability_audit": loadability,
        "dataset_audit": datasets,
        "target_view_audit": target_view,
        "severity_audit": _severity_audit(),
        "environment": _environment_audit(),
        "smoke_benchmark": smoke_benchmark,
        "gate0": {
            "status": "blocked" if blockers else "pass",
            "blockers": blockers,
            "missing_inputs": missing_inputs,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_roots = [Path(path).expanduser().resolve() for path in args.model_root]
    payload = build_snapshot(model_roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "report": str(args.report),
                "gate0": payload["gate0"]["status"],
                "blockers": len(payload["gate0"]["blockers"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
