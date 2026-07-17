"""Summarize the Paper 1 unseen-perturbation Phase-0 ACPC subset.

This is a review artifact builder.  It joins small Phase-0 paired diagnostics
for selected blur/resize cases with the completed std=0.0 vs std=0.08 unseen
closed-loop eval artifacts.  It intentionally does not modify the Gaussian ACPC
basin release artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CASES = (
    "TwoRoom:gaussian_blur:15",
    "Reacher:gaussian_blur:15",
    "PushT:resize:0.25",
    "Cube:resize:0.25",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return _repo_root() / p


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _relative_to_repo(path: Path) -> str:
    try:
        return path.relative_to(_repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _task_slug(task: str) -> str:
    return task.lower()


def _case_parts(case: str) -> tuple[str, str, float]:
    parts = case.split(":")
    if len(parts) != 3:
        raise ValueError(f"case must be Task:family:magnitude, got {case!r}")
    task, family, magnitude = parts
    return task, family, float(magnitude)


def _row_by_std(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in raw.get("rows", []):
        rows[str(row.get("std_key"))] = row
    return rows


def _finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _metric(row: Mapping[str, Any], key: str) -> float:
    return _finite(row.get(key))


def _delta(after: float, before: float) -> float:
    if not math.isfinite(after) or not math.isfinite(before):
        return float("nan")
    return after - before


def _eval_entry(
    unseen_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
    task: str,
    std_key: str,
    family: str,
) -> Mapping[str, Any]:
    return unseen_by_seed[seed]["results"][task][std_key][family]


def _mean(entry: Mapping[str, Any] | None) -> float:
    if entry is None:
        return float("nan")
    return _finite(entry.get("mean"))


def _eval_summary(
    unseen_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
    task: str,
    family: str,
) -> dict[str, float | str | None]:
    base = _eval_entry(unseen_by_seed, seed=seed, task=task, std_key="0.0", family=family)
    robust = _eval_entry(unseen_by_seed, seed=seed, task=task, std_key="0.08", family=family)
    base_origin = _mean(base.get("success_rate", {}).get("origin"))
    robust_origin = _mean(robust.get("success_rate", {}).get("origin"))
    base_stress = _mean(base.get("primary_stress_success"))
    robust_stress = _mean(robust.get("primary_stress_success"))
    base_drop = base_origin - base_stress
    robust_drop = robust_origin - robust_stress
    return {
        "primary_stress_group": base.get("primary_stress_group"),
        "baseline_origin_success": base_origin,
        "baseline_stress_success": base_stress,
        "std008_origin_success": robust_origin,
        "std008_stress_success": robust_stress,
        "stress_success_delta": _delta(robust_stress, base_stress),
        "baseline_drop": base_drop,
        "std008_drop": robust_drop,
        "drop_improvement": _delta(base_drop, robust_drop),
    }


def _diagnostic_pair(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = rows.get("0.0")
    robust = rows.get("0.08")
    if base is None or robust is None:
        return {
            "status": "missing_pair",
            "baseline_status": base.get("status") if base else None,
            "std008_status": robust.get("status") if robust else None,
        }

    metrics = {
        "acpc_h_norm_by_transition": "acpc_h_norm_by_transition",
        "acpc_h_l2_median": "acpc_h_l2_median",
        "pcc_abs_median": "pcc_abs_median",
        "cra_spearman_mean": "cra_spearman_mean",
        "elite_overlap_mean": "elite_overlap_mean",
        "maf_flip_rate": "maf_flip_rate",
        "encoder_shift_to_nn_l2": "encoder_shift_to_nn_l2",
        "sprr": "sprr",
    }
    out: dict[str, Any] = {
        "status": "ok" if base.get("status") == robust.get("status") == "ok" else "row_status_issue",
        "baseline_status": base.get("status"),
        "std008_status": robust.get("status"),
    }
    for label, key in metrics.items():
        before = _metric(base, key)
        after = _metric(robust, key)
        out[f"baseline_{label}"] = before
        out[f"std008_{label}"] = after
        out[f"delta_{label}"] = _delta(after, before)

    out["directional_checks"] = {
        "acpc_lower": out["delta_acpc_h_norm_by_transition"] < 0,
        "pcc_lower": out["delta_pcc_abs_median"] < 0,
        "cra_higher": out["delta_cra_spearman_mean"] > 0,
        "elite_overlap_higher": out["delta_elite_overlap_mean"] > 0,
        "maf_lower": out["delta_maf_flip_rate"] < 0,
    }
    checks = out["directional_checks"]
    out["diagnostic_improvement_count"] = int(sum(bool(v) for v in checks.values()))
    return out


def _mean_of(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_finite(r.get(key)) for r in rows]
    values = [v for v in values if math.isfinite(v)]
    return statistics.fmean(values) if values else float("nan")


def build(
    *,
    raw_dir: Path,
    out_path: Path,
    schema_path: Path | None,
    seeds: Sequence[int],
    cases: Sequence[str],
    unseen_template: str,
) -> dict[str, Any]:
    unseen_by_seed = {
        seed: _load_json(_resolve_repo_path(unseen_template.format(seed=seed))) for seed in seeds
    }

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for seed in seeds:
        for case in cases:
            task, family, magnitude = _case_parts(case)
            raw_path = raw_dir / f"seed{seed}_{_task_slug(task)}_{family}.json"
            if not raw_path.is_file():
                missing.append({"seed": seed, "task": task, "family": family, "missing": str(raw_path)})
                continue
            raw = _load_json(raw_path)
            diag = _diagnostic_pair(_row_by_std(raw))
            ev = _eval_summary(unseen_by_seed, seed=seed, task=task, family=family)
            stress_delta = _finite(ev["stress_success_delta"])
            diag_count = int(diag.get("diagnostic_improvement_count", 0))
            if stress_delta >= 5.0 and diag_count >= 4:
                reading = "positive_transfer_with_diagnostic_alignment"
            elif stress_delta < 5.0 and diag_count >= 4:
                reading = "diagnostic_improves_more_than_unseen_score"
            elif stress_delta >= 5.0:
                reading = "score_improves_without_full_diagnostic_alignment"
            else:
                reading = "boundary_or_neutral_transfer"
            rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "family": family,
                    "magnitude": magnitude,
                    "case": case,
                    "eval": ev,
                    "diagnostics": diag,
                    "reading": reading,
                    "raw_phase0_artifact": _relative_to_repo(raw_path),
                }
            )

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        by_task[task] = {
            "n": len(task_rows),
            "mean_stress_success_delta": _mean_of(
                [row["eval"] for row in task_rows], "stress_success_delta"
            ),
            "mean_drop_improvement": _mean_of([row["eval"] for row in task_rows], "drop_improvement"),
            "mean_delta_acpc_h_norm_by_transition": _mean_of(
                [row["diagnostics"] for row in task_rows],
                "delta_acpc_h_norm_by_transition",
            ),
            "mean_delta_pcc_abs_median": _mean_of(
                [row["diagnostics"] for row in task_rows], "delta_pcc_abs_median"
            ),
            "mean_delta_cra_spearman_mean": _mean_of(
                [row["diagnostics"] for row in task_rows], "delta_cra_spearman_mean"
            ),
            "mean_delta_maf_flip_rate": _mean_of(
                [row["diagnostics"] for row in task_rows], "delta_maf_flip_rate"
            ),
            "readings": sorted({str(row["reading"]) for row in task_rows}),
        }

    payload = {
        "metadata": {
            "schema_version": "paper1-unseen-phase0-acpc-subset-1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "review artifact; supports appendix boundary analysis, not the main Gaussian claim",
            "seeds": list(seeds),
            "cases": list(cases),
            "unseen_artifact_template": unseen_template,
            "phase0_protocol": {
                "std_keys": ["0.0", "0.08"],
                "goal": "clean",
                "paired_corruption": "observation history only",
                "lower_is_better": ["acpc_h_norm_by_transition", "pcc_abs_median", "maf_flip_rate"],
                "higher_is_better": ["cra_spearman_mean", "elite_overlap_mean", "sprr"],
            },
        },
        "rows": rows,
        "summary_by_task": by_task,
        "missing": missing,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if schema_path is not None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Paper 1 unseen perturbation Phase-0 ACPC subset",
            "type": "object",
            "required": ["metadata", "rows", "summary_by_task", "missing"],
            "properties": {
                "metadata": {"type": "object"},
                "rows": {"type": "array"},
                "summary_by_task": {"type": "object"},
                "missing": {"type": "array"},
            },
        }
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="assets/paper1_data/unseen_phase0_acpc_subset_raw")
    p.add_argument("--out", default="assets/paper1_data/unseen_phase0_acpc_subset.json")
    p.add_argument("--schema-out", default="assets/paper1_data/unseen_phase0_acpc_subset.schema.json")
    p.add_argument("--seeds", nargs="+", type=int, default=[3072, 3073, 3074])
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    p.add_argument(
        "--unseen-template",
        default="assets/paper1_data/unseen_origin_vs_std008_strongest_s{seed}.json",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    payload = build(
        raw_dir=_resolve_repo_path(args.raw_dir),
        out_path=_resolve_repo_path(args.out),
        schema_path=_resolve_repo_path(args.schema_out) if args.schema_out else None,
        seeds=args.seeds,
        cases=args.cases,
        unseen_template=args.unseen_template,
    )
    print(f"wrote {_resolve_repo_path(args.out)}")
    if args.schema_out:
        print(f"wrote {_resolve_repo_path(args.schema_out)}")
    print(f"rows: {len(payload['rows'])}; missing: {len(payload['missing'])}")
    for task, summary in payload["summary_by_task"].items():
        print(
            f"{task}: stress_delta={summary['mean_stress_success_delta']:.2f}, "
            f"delta_acpc={summary['mean_delta_acpc_h_norm_by_transition']:.3f}, "
            f"readings={','.join(summary['readings'])}"
        )


if __name__ == "__main__":
    main()
