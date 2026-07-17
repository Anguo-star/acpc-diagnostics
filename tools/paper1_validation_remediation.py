"""Validation/protocol remediation summaries for Paper 1.

This script builds a compact artifact for the top-conference remediation pass.
It uses only released JSON/Markdown summaries and does not load models,
checkpoints, or datasets.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "paper1_data"
TRAINING_SEED_LOCKBOX = DATA_DIR / "training_seed_gaussian_lockbox.json"
UNSEEN_PHASE0 = DATA_DIR / "unseen_phase0_acpc_subset.json"
UNSEEN_PHASE0_FULLSTRESS = DATA_DIR / "unseen_phase0_acpc_fullstress.json"
NO_RETRAIN_AUDIT = DATA_DIR / "no_retrain_diagnostic_audit.json"
THREE_SEED_DIAGNOSTIC_VALIDATION = DATA_DIR / "three_seed_diagnostic_validation.json"
SEMANTIC_MARGIN_PASSRATE = DATA_DIR / "semantic_margin_passrate_lewm_three_seed.json"
UNSEEN_SCORE_ARTIFACTS = [
    DATA_DIR / "unseen_origin_vs_std008_strongest_s3072.json",
    DATA_DIR / "unseen_origin_vs_std008_strongest_s3073.json",
    DATA_DIR / "unseen_origin_vs_std008_strongest_s3074.json",
]
UNSEEN_STRESS_KEYS = {
    "gaussian_blur": "pixels_blur_ks15",
    "resize": "pixels_rs_factor0.25",
}
SELECTED_UNSEEN_STRESS = {
    "TwoRoom": "gaussian_blur",
    "Reacher": "gaussian_blur",
    "PushT": "resize",
    "Cube": "resize",
}
TASKS = ["TwoRoom", "PushT", "Reacher", "Cube"]
DEFAULT_OUT_JSON = DATA_DIR / "prospective_validation_summary.json"
DEFAULT_OUT_MD = DATA_DIR / "prospective_validation_summary.md"

SIGNED_METRICS = [
    ("ACPC-H/trans. delta", "delta_acpc_h_norm_by_transition", -1.0, "lower is better"),
    ("PCC delta", "delta_pcc_abs_median", -1.0, "lower is better"),
    ("CRA delta", "delta_cra_spearman_mean", 1.0, "higher is better"),
    ("MAF delta", "delta_maf_flip_rate", -1.0, "lower is better"),
]
SEMANTIC_GUARD_PROTOCOL = [
    {
        "task": "PushT",
        "semantic_factor": "T-block pose/contact relative to pusher",
        "available_source": "dataset state column is configured for PushT analysis",
        "probe_rule": "different-state pair must cross a pose/contact threshold while same-state clean/noisy views share state",
        "release_status": "reported as a task-state proxy margin pass-rate in semantic_margin_passrate_lewm_three_seed.json; finer oracle labels remain an extension",
    },
    {
        "task": "TwoRoom",
        "semantic_factor": "room/doorway/topology and target-region relation",
        "available_source": "derive from trajectory position/proprio and map topology",
        "probe_rule": "different-state pair must differ in room or doorway side under comparable visual nuisance",
        "release_status": "reported as a task-state proxy margin pass-rate from pos_agent geometry in semantic_margin_passrate_lewm_three_seed.json; topology-specific labels remain an extension",
    },
    {
        "task": "Reacher",
        "semantic_factor": "joint/target geometry and end-effector-to-target relation",
        "available_source": "qpos/goal_qpos are used by eval set-state callables",
        "probe_rule": "different-state pair must differ in target quadrant or end-effector-target distance bin",
        "release_status": "reported as a task-state proxy margin pass-rate in semantic_margin_passrate_lewm_three_seed.json; finer oracle labels remain an extension",
    },
    {
        "task": "Cube",
        "semantic_factor": "cube pose and gripper-object/goal relation",
        "available_source": "qpos plus goal block position/quaternion are used by eval callables",
        "probe_rule": "different-state pair must differ in object pose/goal relation beyond tolerance",
        "release_status": "reported as a task-state proxy margin pass-rate in semantic_margin_passrate_lewm_three_seed.json; finer oracle labels remain an extension",
    },
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rankdata(values: Sequence[float]) -> list[float]:
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = 0.5 * (j + k - 1)
        for _, idx in pairs[j:k]:
            ranks[idx] = rank
        j = k
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_rankdata(xs), _rankdata(ys))


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def _pstdev(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def _success(entry: dict, group: str) -> float:
    return float(entry["eval"][group]["success_rate"]["mean"])


def _three_seed_unseen_score_summary() -> dict:
    rows = []
    for artifact in UNSEEN_SCORE_ARTIFACTS:
        data = _load(artifact)
        seed = int(data["metadata"]["train_seed"])
        for task, task_block in data["results"].items():
            for family, stress_key in UNSEEN_STRESS_KEYS.items():
                if family not in task_block.get("0.0", {}) or family not in task_block.get("0.08", {}):
                    continue
                baseline = task_block["0.0"][family]
                robust = task_block["0.08"][family]
                base_origin = _success(baseline, "origin")
                base_stress = _success(baseline, stress_key)
                robust_origin = _success(robust, "origin")
                robust_stress = _success(robust, stress_key)
                rows.append(
                    {
                        "task": task,
                        "family": family,
                        "training_seed": seed,
                        "baseline_origin_success": base_origin,
                        "baseline_stress_success": base_stress,
                        "std008_origin_success": robust_origin,
                        "std008_stress_success": robust_stress,
                        "stress_success_delta": robust_stress - base_stress,
                        "drop_improvement": (base_origin - base_stress) - (robust_origin - robust_stress),
                        "source": str(artifact.relative_to(ROOT)),
                    }
                )

    coverage = {
        f"{task}:{family}": sorted(
            int(row["training_seed"])
            for row in rows
            if row["task"] == task and row["family"] == family
        )
        for task in TASKS
        for family in UNSEEN_STRESS_KEYS
    }
    missing = {key: seeds for key, seeds in coverage.items() if seeds != [3072, 3073, 3074]}
    if missing:
        raise ValueError(f"Unexpected three-seed unseen coverage: {missing}")

    def summarize(task_rows: list[dict], task: str, family: str) -> dict:
        out = {
            "task": task,
            "family": family,
            "training_seeds": sorted(int(row["training_seed"]) for row in task_rows),
            "n_training_seeds": len(task_rows),
        }
        for key in (
            "baseline_stress_success",
            "std008_stress_success",
            "stress_success_delta",
            "drop_improvement",
        ):
            values = [float(row[key]) for row in task_rows]
            out[f"{key}_mean"] = _mean(values)
            out[f"{key}_pstdev"] = _pstdev(values)
        return out

    selected_rows = []
    task_family_rows = []
    for task in TASKS:
        for family in UNSEEN_STRESS_KEYS:
            task_rows = [row for row in rows if row["task"] == task and row["family"] == family]
            summary = summarize(task_rows, task, family)
            task_family_rows.append(summary)
            if SELECTED_UNSEEN_STRESS[task] == family:
                selected_rows.append(summary)

    return {
        "scope": "unseen score aggregate over training seeds 3072/3073/3074; each seed point is the mean over eval seeds 42/43/44 from source artifacts with num_eval=300",
        "selected_stress_policy": SELECTED_UNSEEN_STRESS,
        "per_seed_rows": sorted(rows, key=lambda row: (row["task"], row["family"], row["training_seed"])),
        "selected_stress_rows": selected_rows,
        "task_family_rows": task_family_rows,
        "coverage": coverage,
    }


def _heldout_metric_rows(unseen: dict) -> tuple[list[dict], dict]:
    rows = unseen["rows"]
    stress_delta = [float(row["eval"]["stress_success_delta"]) for row in rows]
    drop_improvement = [float(row["eval"]["drop_improvement"]) for row in rows]
    out = []
    signed_values = []
    for label, key, sign, direction in SIGNED_METRICS:
        values = [sign * float(row["diagnostics"][key]) for row in rows]
        signed_values.append(values)
        out.append(
            {
                "metric": label,
                "direction": direction,
                "n": len(rows),
                "spearman_vs_stress_success_delta": _spearman(values, stress_delta),
                "pearson_vs_stress_success_delta": _pearson(values, stress_delta),
                "spearman_vs_drop_improvement": _spearman(values, drop_improvement),
                "pearson_vs_drop_improvement": _pearson(values, drop_improvement),
            }
        )

    composite_scores = [0.0] * len(rows)
    for values in signed_values:
        ranks = _rankdata(values)
        # Higher signed values are better; negative rank makes lower composite score better.
        for i, rank in enumerate(ranks):
            composite_scores[i] -= rank
    out.append(
        {
            "metric": "Composite signed-rank rule",
            "direction": "aggregate rank over ACPC/PCC/CRA/MAF directions",
            "n": len(rows),
            "spearman_vs_stress_success_delta": _spearman([-s for s in composite_scores], stress_delta),
            "pearson_vs_stress_success_delta": _pearson([-s for s in composite_scores], stress_delta),
            "spearman_vs_drop_improvement": _spearman([-s for s in composite_scores], drop_improvement),
            "pearson_vs_drop_improvement": _pearson([-s for s in composite_scores], drop_improvement),
        }
    )
    top_diag = set(sorted(range(len(rows)), key=lambda i: composite_scores[i])[:4])
    top_stress = set(sorted(range(len(rows)), key=lambda i: stress_delta[i], reverse=True)[:4])
    top_drop = set(sorted(range(len(rows)), key=lambda i: drop_improvement[i], reverse=True)[:4])
    topk = {
        "k": 4,
        "topk_by_composite": [rows[i]["case"] + f":seed{rows[i]['seed']}" for i in sorted(top_diag)],
        "stress_success_delta_topk_hit_count": len(top_diag & top_stress),
        "drop_improvement_topk_hit_count": len(top_diag & top_drop),
        "stress_success_delta_topk_total": len(top_stress),
        "drop_improvement_topk_total": len(top_drop),
    }
    return out, topk


def build_payload() -> dict:
    training = _load(TRAINING_SEED_LOCKBOX)
    unseen = _load(UNSEEN_PHASE0)
    unseen_fullstress = _load(UNSEEN_PHASE0_FULLSTRESS)
    no_retrain = _load(NO_RETRAIN_AUDIT)
    three_seed_diag = _load(THREE_SEED_DIAGNOSTIC_VALIDATION)
    semantic_margin = _load(SEMANTIC_MARGIN_PASSRATE)
    heldout_rows, topk = _heldout_metric_rows(unseen)
    fullstress_rows, fullstress_topk = _heldout_metric_rows(unseen_fullstress)
    return {
        "metadata": {
            "schema_version": "paper1-validation-remediation-0.3",
            "source_artifacts": [
                str(TRAINING_SEED_LOCKBOX.relative_to(ROOT)),
                str(THREE_SEED_DIAGNOSTIC_VALIDATION.relative_to(ROOT)),
                str(SEMANTIC_MARGIN_PASSRATE.relative_to(ROOT)),
                str(UNSEEN_PHASE0.relative_to(ROOT)),
                str(UNSEEN_PHASE0_FULLSTRESS.relative_to(ROOT)),
                str(NO_RETRAIN_AUDIT.relative_to(ROOT)),
                *[str(path.relative_to(ROOT)) for path in UNSEEN_SCORE_ARTIFACTS],
            ],
            "scope": "No retraining. Summarizes completed three-training-seed Gaussian behavior, full-grid three-seed fixed-rule diagnostic validation, task-state proxy margin pass-rate, and bounded unseen-stressor scope checks.",
        },
        "three_training_seed_gaussian_summary": training["task_summary_rows"],
        "three_seed_full_grid_diagnostic_validation": three_seed_diag["summary"],
        "three_seed_diagnostic_split_summaries": three_seed_diag.get("split_summaries", []),
        "three_seed_diagnostic_selection_rows": three_seed_diag["selection_rows"],
        "semantic_margin_passrate": semantic_margin["summary_rows"],
        "semantic_margin_coverage": semantic_margin["coverage"],
        "three_seed_unseen_score_summary": _three_seed_unseen_score_summary(),
        "heldout_unseen_validation": {
            "split": "appendix scope check: training seeds 3072/3073/3074; selected unseen perturbation cases; fixed std_max comparison 0.0 vs 0.08",
            "n_rows": len(unseen["rows"]),
            "metric_rows": heldout_rows,
            "topk_summary": topk,
            "summary_by_task": unseen["summary_by_task"],
        },
        "fullstress_unseen_validation": {
            "split": "appendix scope check: training seeds 3072/3073/3074; all task by blur/resize strongest endpoints; fixed std_max comparison 0.0 vs 0.08",
            "n_rows": len(unseen_fullstress["rows"]),
            "metric_rows": fullstress_rows,
            "topk_summary": fullstress_topk,
            "summary_by_task": unseen_fullstress["summary_by_task"],
        },
        "existing_full_grid_frozen_rule_audit": no_retrain["summary"],
        "semantic_discriminability_protocol": SEMANTIC_GUARD_PROTOCOL,
        "remaining_validation_work": [
            "Extend the fixed diagnostic rule to additional perturbation families and method families after this three-seed Gaussian validation.",
            "Extend state-proxy pair construction to finer oracle contact/topology/goal-relation labels if the claim is expanded beyond matched Gaussian diagnostics.",
            "Keep training-seed uncertainty as the primary behavior statistic and evaluation-seed variance as the secondary decomposition.",
        ],
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Prospective Validation Remediation Summary",
        "",
        "This artifact separates main completed validation evidence from appendix scope checks and reproducibility details.",
        "",
        "## Three-seed unseen score aggregate",
        "",
        "Scores include training seeds 3072/3073/3074 and are treated as a bounded unseen-stressor scope check.",
        "",
        "| Task | selected stress | baseline stress | std0.08 stress | stress delta | drop improvement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["three_seed_unseen_score_summary"]["selected_stress_rows"]:
        lines.append(
            "| {task} | {family} | {base} +/- {base_sd} | {rob} +/- {rob_sd} | {delta} +/- {delta_sd} | {drop} +/- {drop_sd} |".format(
                task=row["task"],
                family=row["family"],
                base=_fmt(row["baseline_stress_success_mean"]),
                base_sd=_fmt(row["baseline_stress_success_pstdev"]),
                rob=_fmt(row["std008_stress_success_mean"]),
                rob_sd=_fmt(row["std008_stress_success_pstdev"]),
                delta=_fmt(row["stress_success_delta_mean"]),
                delta_sd=_fmt(row["stress_success_delta_pstdev"]),
                drop=_fmt(row["drop_improvement_mean"]),
                drop_sd=_fmt(row["drop_improvement_pstdev"]),
            )
        )

    lines.extend(
        [
            "",
            "## Three-seed fixed-rule Gaussian diagnostic validation",
            "",
            "Seed 3072 is the development grid used to freeze the aggregate-rank rule; seeds 3073/3074 are held-out training seeds.",
            "",
            "| Split | seeds | blocks | candidates | within 5pp | regret mean +/- std | bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in payload["three_seed_diagnostic_split_summaries"]:
        ci = split["bootstrap_ci95_mean_selected_regret_to_best_pp"]
        lines.append(
            "| {split_name} | {seeds} | {blocks} | {cands} | {within}/{blocks} | {regret} +/- {regret_sd} | [{lo}, {hi}] |".format(
                split_name=split["split"],
                seeds=",".join(str(s) for s in split["training_seeds"]),
                blocks=split["n_task_seed_blocks"],
                cands=split["n_checkpoint_candidates"],
                within=split["within_5pp_hits"],
                regret=_fmt(split["mean_selected_regret_to_best_pp"]),
                regret_sd=_fmt(split["pstdev_selected_regret_to_best_pp"]),
                lo=_fmt(ci[0]),
                hi=_fmt(ci[1]),
            )
        )
    lines.extend(
        [
            "",
            "## Task-state proxy margin pass-rate",
            "",
            "| Task | std | pass-rate | same-state radius | state-proxy diff | margin |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["semantic_margin_passrate"]:
        lines.append(
            "| {task} | {std} | {pass_rate} +/- {pass_sd} | {same} +/- {same_sd} | {diff} +/- {diff_sd} | {margin} +/- {margin_sd} |".format(
                task=row["task"],
                std=row["std_key"],
                pass_rate=_fmt(row["semantic_margin_pass_rate_mean"]),
                pass_sd=_fmt(row["semantic_margin_pass_rate_pstdev"]),
                same=_fmt(row["same_state_noisy_radius_median_mean"]),
                same_sd=_fmt(row["same_state_noisy_radius_median_pstdev"]),
                diff=_fmt(row["semantic_diff_l2_median_mean"]),
                diff_sd=_fmt(row["semantic_diff_l2_median_pstdev"]),
                margin=_fmt(row["semantic_margin_median_mean"]),
                margin_sd=_fmt(row["semantic_margin_median_pstdev"]),
            )
        )

    lines.extend(
        [
            "",
            "## Matched three-seed unseen diagnostic validation slice",
            "",
            "Split: training seeds 3072/3073/3074; selected unseen perturbation cases; fixed comparison std_max 0.0 -> 0.08.",
            "",
            "| Metric | rho vs stress delta | r vs stress delta | rho vs drop improvement | r vs drop improvement | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["heldout_unseen_validation"]["metric_rows"]:
        lines.append(
            "| {metric} | {rho_s} | {r_s} | {rho_d} | {r_d} | {n} |".format(
                metric=row["metric"],
                rho_s=_fmt(row["spearman_vs_stress_success_delta"]),
                r_s=_fmt(row["pearson_vs_stress_success_delta"]),
                rho_d=_fmt(row["spearman_vs_drop_improvement"]),
                r_d=_fmt(row["pearson_vs_drop_improvement"]),
                n=row["n"],
            )
        )
    topk = payload["heldout_unseen_validation"]["topk_summary"]
    lines.extend(
        [
            "",
            f"Top-{topk['k']} agreement: composite signed-rank top-k hits {topk['stress_success_delta_topk_hit_count']}/{topk['stress_success_delta_topk_total']} for stress-success delta and {topk['drop_improvement_topk_hit_count']}/{topk['drop_improvement_topk_total']} for drop improvement.",
            "",
            "## Full blur/resize unseen diagnostic validation slice",
            "",
            "Split: training seeds 3072/3073/3074; all task by blur/resize strongest endpoints; fixed comparison std_max 0.0 -> 0.08.",
            "",
            "| Metric | rho vs stress delta | r vs stress delta | rho vs drop improvement | r vs drop improvement | n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["fullstress_unseen_validation"]["metric_rows"]:
        lines.append(
            "| {metric} | {rho_s} | {r_s} | {rho_d} | {r_d} | {n} |".format(
                metric=row["metric"],
                rho_s=_fmt(row["spearman_vs_stress_success_delta"]),
                r_s=_fmt(row["pearson_vs_stress_success_delta"]),
                rho_d=_fmt(row["spearman_vs_drop_improvement"]),
                r_d=_fmt(row["pearson_vs_drop_improvement"]),
                n=row["n"],
            )
        )
    full_topk = payload["fullstress_unseen_validation"]["topk_summary"]
    lines.extend(
        [
            "",
            "Top-{k} agreement: composite signed-rank top-k hits {stress_hit}/{stress_total} for stress-success delta and {drop_hit}/{drop_total} for drop improvement.".format(
                k=full_topk["k"],
                stress_hit=full_topk["stress_success_delta_topk_hit_count"],
                stress_total=full_topk["stress_success_delta_topk_total"],
                drop_hit=full_topk["drop_improvement_topk_hit_count"],
                drop_total=full_topk["drop_improvement_topk_total"],
            ),
            "",
            "## Semantic state proxies",
            "",
            "| Task | semantic factor | available source | status |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["semantic_discriminability_protocol"]:
        lines.append(
            "| {task} | {factor} | {source} | {status} |".format(
                task=row["task"],
                factor=row["semantic_factor"],
                source=row["available_source"],
                status=row["release_status"],
            )
        )
    lines.extend(
        [
            "",
            "## Remaining validation work",
            "",
        ]
    )
    for item in payload["remaining_validation_work"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)
    payload = build_payload()
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json.relative_to(ROOT)}")
    print(f"wrote {args.out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
