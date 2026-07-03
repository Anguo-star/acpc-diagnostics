"""Build a Paper 1 unseen-perturbation pilot artifact.

The input is the manifest written by ``tools.paper1_unseen_eval_grid``. This
script reads each job's ``eval_summary.csv`` and optional diagnostics summary,
then writes one compact JSON artifact for review before any values are added to
the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = "assets/paper1_data/unseen_perturbation_pilot_seed3072_manifest.json"
DEFAULT_OUT = "assets/paper1_data/unseen_perturbation_pilot_seed3072.json"
DEFAULT_SCHEMA_OUT = "assets/paper1_data/unseen_perturbation_pilot_seed3072.schema.json"
DEFAULT_GAUSSIAN_REFERENCE = "assets/paper1_data/canonical_evals_20260517.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return _repo_root() / p


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _default_data_root() -> str | None:
    for key in ("PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _relative_to_or_self(path: Path, parent: Path) -> str:
    try:
        return str(path.relative_to(parent))
    except ValueError:
        return str(path)


def _resolve_with_root(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs).expanduser()
    if p.is_absolute():
        return p
    return root / p


def _parse_values(raw: str) -> list[float]:
    vals = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals


def _read_eval_summary(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row["group"]
            metric = row["metric"]
            values = _parse_values(row.get("values", ""))
            groups.setdefault(group, {})[metric] = {
                "n_seeds": int(row["n_seeds"]),
                "seeds": row["seeds"],
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "sem": float(row["sem"]),
                "values": values,
            }
    return groups


def _extract_success(eval_rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    success = {}
    for group, metrics in eval_rows.items():
        if "success_rate" in metrics:
            success[group] = metrics["success_rate"]
    return success


def _best_success(success: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not success:
        return None
    group, row = max(success.items(), key=lambda kv: kv[1]["mean"])
    return {"group": group, **row}


def _worst_success(success: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not success:
        return None
    group, row = min(success.items(), key=lambda kv: kv[1]["mean"])
    return {"group": group, **row}


def _strongest_magnitude(family: str, magnitudes: list[str]) -> str | None:
    if not magnitudes:
        return None
    numeric = [(float(m), m) for m in magnitudes]
    if family in ("gaussian_noise", "gaussian_blur"):
        return max(numeric, key=lambda x: x[0])[1]
    if family == "resize":
        return min(numeric, key=lambda x: x[0])[1]
    return None


def _primary_scope(apply_to: Any) -> str:
    raw = str(apply_to)
    if raw in ("1", "pixels", "pixel", "obs", "observation"):
        return "pixels"
    if raw in ("2", "goal"):
        return "goal"
    if raw in ("3", "pixels+goal", "pixels_goal"):
        return "pixels_goal"
    return "pixels"


def _primary_stress_group(job: dict[str, Any]) -> str | None:
    family = job["family"]
    magnitude = _strongest_magnitude(family, [str(m) for m in job.get("magnitudes", [])])
    if magnitude is None:
        return None
    scope = _primary_scope(job.get("apply_to"))
    if family == "gaussian_noise":
        if float(magnitude) == 0.0:
            return "origin"
        return f"{scope}_std{magnitude}"
    if family == "gaussian_blur":
        if float(magnitude) == 1.0:
            return "origin"
        return f"{scope}_blur_ks{magnitude}"
    if family == "resize":
        if float(magnitude) == 1.0:
            return "origin"
        return f"{scope}_rs_factor{magnitude}"
    return None


def _reference_gaussian(
    reference: dict[str, Any] | None,
    task: str,
    std_key: str,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    try:
        metrics = reference[task][std_key]["metrics"]
    except KeyError:
        return None
    return {
        "clean": metrics.get("clean"),
        "pixels_std0.08": metrics.get("pixels_std0.08"),
        "pixels_goal_std0.08": metrics.get("pixels_goal_std0.08"),
    }


def _job_key(job: dict[str, Any]) -> tuple[str, str, str]:
    return job["task"], job["std_key"], job["family"]


def _aggregate_family_plateaus(results: dict[str, Any]) -> dict[str, Any]:
    """Lightweight summaries for review, not paper-facing statistical tests."""
    by_task_family: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for task, by_std in results.items():
        for std_key, by_family in by_std.items():
            for family, entry in by_family.items():
                stress = entry.get("primary_stress_success")
                if stress is None:
                    continue
                by_task_family.setdefault((task, family), []).append((float(stress["mean"]), std_key))

    summaries: dict[str, Any] = {}
    for (task, family), rows in sorted(by_task_family.items()):
        means = [v for v, _ in rows]
        best_mean = max(means)
        tolerance = 0.05 * max(1.0, abs(best_mean))
        plateau = [std for value, std in rows if best_mean - value <= tolerance]
        summaries.setdefault(task, {})[family] = {
            "n_checkpoints": len(rows),
            "best_primary_stress_mean": best_mean,
            "mean_primary_stress_success": statistics.fmean(means),
            "population_std_primary_stress_success": statistics.pstdev(means) if len(means) > 1 else 0.0,
            "plateau_std_keys_within_5pct_of_best": sorted(plateau, key=lambda x: float(x)),
            "note": "Uses the strongest configured severity for each family; review only.",
        }
    return summaries


def build(
    *,
    manifest_path: Path,
    out_path: Path,
    schema_path: Path | None,
    root_override: str | None,
    gaussian_reference_path: Path | None,
    allow_missing: bool,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    root_raw = root_override or _default_data_root() or manifest["metadata"].get("root")
    if not root_raw:
        raise ValueError("Pass --root or set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME.")
    root = Path(root_raw).expanduser()
    reference = _load_json(gaussian_reference_path) if gaussian_reference_path else None

    results: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[dict[str, str]] = []
    for job in manifest["jobs"]:
        task, std_key, family = _job_key(job)
        eval_summary = _resolve_with_root(root, job["eval_summary_rel"])
        diagnostics_dir = _resolve_with_root(root, job["diagnostics_dir_rel"])
        diagnostics_summary = diagnostics_dir / "diagnostics_summary.json"

        if not eval_summary.is_file():
            missing.append(
                {
                    "task": task,
                    "std_key": std_key,
                    "family": family,
                    "missing": job["eval_summary_rel"],
                }
            )
            if not allow_missing:
                raise FileNotFoundError(f"missing eval summary: {eval_summary}")
            continue

        eval_rows = _read_eval_summary(eval_summary)
        success = _extract_success(eval_rows)
        primary_group = _primary_stress_group(job)
        primary_stress_success = success.get(primary_group) if primary_group else None
        entry: dict[str, Any] = {
            "subdir": job["subdir"],
            "checkpoint_rel": job["checkpoint_rel"],
            "result_dir_rel": job["result_dir_rel"],
            "eval_summary_rel": job["eval_summary_rel"],
            "diagnostics_dir_rel": job["diagnostics_dir_rel"],
            "diagnostics_enabled": bool(job.get("diagnostics_enabled", False)),
            "magnitudes": job.get("magnitudes", []),
            "apply_to": job.get("apply_to"),
            "eval": eval_rows,
            "success_rate": success,
            "best_success": _best_success(success),
            "worst_success": _worst_success(success),
            "primary_stress_group": primary_group,
            "primary_stress_success": primary_stress_success,
            "gaussian_reference": _reference_gaussian(reference, task, std_key),
        }
        if diagnostics_summary.is_file():
            entry["diagnostics_summary"] = _load_json(diagnostics_summary)
        else:
            entry["diagnostics_summary"] = None
            if job.get("diagnostics_enabled", False):
                missing.append(
                    {
                        "task": task,
                        "std_key": std_key,
                        "family": family,
                        "missing": job["diagnostics_dir_rel"] + "/diagnostics_summary.json",
                    }
                )
                if not allow_missing:
                    raise FileNotFoundError(f"missing diagnostics summary: {diagnostics_summary}")

        results.setdefault(task, {}).setdefault(std_key, {})[family] = entry

    payload = {
        "metadata": {
            "schema_version": "paper1-unseen-perturbation-pilot-1.0",
            "source_manifest": _relative_to_or_self(manifest_path, _repo_root()),
            "root_note": "Root is a runtime prefix; use --root to rebuild on another machine.",
            "train_seed": manifest["metadata"].get("train_seed"),
            "epoch": manifest["metadata"].get("epoch"),
            "tasks": manifest["metadata"].get("tasks"),
            "families": manifest["metadata"].get("families"),
            "eval_seeds": manifest["metadata"].get("eval_seeds"),
            "eval_base_seed": manifest["metadata"].get("eval_base_seed"),
            "num_eval": manifest["metadata"].get("num_eval"),
            "status": "pilot artifact for review; do not cite in main text until manually audited",
        },
        "results": results,
        "family_plateau_summaries": _aggregate_family_plateaus(results),
        "missing": missing,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if schema_path is not None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Paper 1 unseen perturbation pilot artifact",
            "type": "object",
            "required": ["metadata", "results", "missing"],
            "properties": {
                "metadata": {"type": "object"},
                "results": {"type": "object"},
                "family_plateau_summaries": {"type": "object"},
                "missing": {"type": "array"},
            },
        }
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--schema-out", default=DEFAULT_SCHEMA_OUT)
    p.add_argument("--root", default=None, help="Override runtime root from the manifest.")
    p.add_argument(
        "--gaussian-reference",
        default=DEFAULT_GAUSSIAN_REFERENCE,
        help="Canonical Gaussian eval artifact used only as a reference column. Pass '' to disable.",
    )
    p.add_argument("--allow-missing", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = _resolve_repo_path(args.manifest)
    out_path = _resolve_repo_path(args.out)
    schema_path = _resolve_repo_path(args.schema_out) if args.schema_out else None
    gaussian_reference_path = (
        _resolve_repo_path(args.gaussian_reference) if args.gaussian_reference else None
    )
    payload = build(
        manifest_path=manifest_path,
        out_path=out_path,
        schema_path=schema_path,
        root_override=args.root,
        gaussian_reference_path=gaussian_reference_path,
        allow_missing=args.allow_missing,
    )
    print(f"wrote {out_path}")
    if schema_path is not None:
        print(f"wrote {schema_path}")
    print(f"missing entries: {len(payload['missing'])}")


if __name__ == "__main__":
    main()
