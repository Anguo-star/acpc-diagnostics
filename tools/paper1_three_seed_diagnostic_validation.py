"""Summarize three-seed full-grid Phase-0 diagnostic validation for Paper 1."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_IN = DATA_DIR / "acpc_phase0_lewm_three_seed.json"
DEFAULT_OUT_JSON = DATA_DIR / "three_seed_diagnostic_validation.json"
DEFAULT_OUT_MD = DATA_DIR / "three_seed_diagnostic_validation.md"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
DIAGNOSTIC_RULE = (
    ("ACPC-H/trans", "acpc_h_norm_by_transition", -1.0, "lower"),
    ("PCC", "pcc_abs_median", -1.0, "lower"),
    ("CRA", "cra_spearman_mean", 1.0, "higher"),
    ("MAF", "maf_flip_rate", -1.0, "lower"),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rankdata(values: Sequence[float]) -> list[float]:
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for _, idx in pairs[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_rankdata(xs), _rankdata(ys))


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def _pstdev(values: Sequence[float]) -> float:
    mu = _mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int = 9101, n_resamples: int = 10000) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_resamples):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (n_resamples - 1))]
    hi = means[int(0.975 * (n_resamples - 1))]
    return lo, hi


def _selection_summary(rows: Sequence[dict], *, name: str, description: str) -> dict:
    regrets = [float(r["selected_regret_to_best_pp"]) for r in rows]
    top2_hits = [float(r["top2_overlap"]) for r in rows]
    lo, hi = _bootstrap_mean_ci(regrets)
    return {
        "split": name,
        "description": description,
        "n_task_seed_blocks": len(rows),
        "n_checkpoint_candidates": 8 * len(rows),
        "training_seeds": sorted({int(r["training_seed"]) for r in rows}),
        "exact_best_hits": sum(1 for r in rows if bool(r["selected_exact_best"])),
        "within_5pp_hits": sum(1 for r in rows if bool(r["selected_within_5pp_of_best"])),
        "mean_selected_regret_to_best_pp": _mean(regrets),
        "pstdev_selected_regret_to_best_pp": _pstdev(regrets),
        "bootstrap_ci95_mean_selected_regret_to_best_pp": [lo, hi],
        "mean_top2_overlap": _mean(top2_hits),
        "top2_total_per_block": 2,
    }


def _fmt(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _validate(rows: Sequence[dict]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if len(ok_rows) != 108:
        raise ValueError(f"expected 108 ok rows, found {len(ok_rows)}")
    coverage = {
        (r["task"], int(r["training_seed"]), str(r["std_key"]))
        for r in ok_rows
    }
    expected = {(task, seed, std) for task in TASKS for seed in SEEDS for std in STD_KEYS}
    if coverage != expected:
        missing = sorted(expected - coverage)
        extra = sorted(coverage - expected)
        raise ValueError(f"coverage mismatch missing={missing[:5]} extra={extra[:5]}")


def _block_score(candidates: Sequence[dict]) -> dict[str, float]:
    scores = {str(r["std_key"]): 0.0 for r in candidates}
    for _, key, sign, _ in DIAGNOSTIC_RULE:
        values = [sign * float(r[key]) for r in candidates]
        ranks = _rankdata([-v for v in values])  # signed higher is better, rank 0 is best.
        for row, rank in zip(candidates, ranks):
            scores[str(row["std_key"])] += float(rank)
    return scores


def _selection_rows(rows: Sequence[dict]) -> list[dict]:
    out = []
    for task in TASKS:
        for seed in SEEDS:
            block = [r for r in rows if r["task"] == task and int(r["training_seed"]) == seed]
            base = next(r for r in block if str(r["std_key"]) == "0.0")
            candidates = [r for r in block if str(r["std_key"]) != "0.0"]
            scores = _block_score(candidates)
            selected = min(candidates, key=lambda r: (scores[str(r["std_key"])], str(r["std_key"])))
            best = max(candidates, key=lambda r: float(r["pixels_std0.08_success"]))
            top2_diag = {str(r["std_key"]) for r in sorted(candidates, key=lambda r: (scores[str(r["std_key"])], str(r["std_key"])))[:2]}
            top2_score = {str(r["std_key"]) for r in sorted(candidates, key=lambda r: float(r["pixels_std0.08_success"]), reverse=True)[:2]}
            selected_success = float(selected["pixels_std0.08_success"])
            best_success = float(best["pixels_std0.08_success"])
            out.append(
                {
                    "task": task,
                    "training_seed": seed,
                    "baseline_px08_success": float(base["pixels_std0.08_success"]),
                    "selected_std": str(selected["std_key"]),
                    "selected_px08_success": selected_success,
                    "closed_loop_best_std": str(best["std_key"]),
                    "closed_loop_best_px08_success": best_success,
                    "selected_regret_to_best_pp": best_success - selected_success,
                    "selected_within_5pp_of_best": (best_success - selected_success) <= 5.0,
                    "selected_exact_best": str(selected["std_key"]) == str(best["std_key"]),
                    "top2_overlap": len(top2_diag & top2_score),
                    "top2_total": 2,
                    "selected_rank_score": float(scores[str(selected["std_key"])]),
                    "selected_acpc_h_norm_by_transition": float(selected["acpc_h_norm_by_transition"]),
                    "selected_pcc_abs_median": float(selected["pcc_abs_median"]),
                    "selected_cra_spearman_mean": float(selected["cra_spearman_mean"]),
                    "selected_maf_flip_rate": float(selected["maf_flip_rate"]),
                }
            )
    return out


def _correlation_rows(rows: Sequence[dict]) -> list[dict]:
    candidates = [r for r in rows if str(r["std_key"]) != "0.0"]
    scopes: list[tuple[str, list[dict]]] = [("all_tasks", candidates)]
    scopes.extend((task, [r for r in candidates if r["task"] == task]) for task in TASKS)
    out = []
    for scope, scope_rows in scopes:
        px08 = [float(r["pixels_std0.08_success"]) for r in scope_rows]
        negative_drop = [-float(r["corruption_drop"]) for r in scope_rows]
        for label, key, sign, direction in DIAGNOSTIC_RULE:
            values = [sign * float(r[key]) for r in scope_rows]
            out.append(
                {
                    "scope": scope,
                    "metric": label,
                    "direction": direction,
                    "n": len(scope_rows),
                    "spearman_vs_px08_success": _spearman(values, px08),
                    "pearson_vs_px08_success": _pearson(values, px08),
                    "spearman_vs_negative_drop": _spearman(values, negative_drop),
                    "pearson_vs_negative_drop": _pearson(values, negative_drop),
                }
            )
    return out


def build_payload(input_path: Path) -> dict:
    data = _load(input_path)
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    _validate(rows)
    selections = _selection_rows(rows)
    split_summaries = [
        _selection_summary(
            [r for r in selections if int(r["training_seed"]) == 3072],
            name="development_seed_3072",
            description="Seed 3072 development grid used to freeze metric computation and aggregate-rank rule before reading independent training seeds.",
        ),
        _selection_summary(
            [r for r in selections if int(r["training_seed"]) in (3073, 3074)],
            name="heldout_training_seeds_3073_3074",
            description="Independent held-out training seeds evaluated after the rule is fixed; no closed-loop score is used by the selector.",
        ),
        _selection_summary(
            selections,
            name="all_training_seeds_3072_3073_3074",
            description="Complete three-seed LeWM Gaussian full grid.",
        ),
    ]
    summary = dict(split_summaries[-1])
    summary["checkpoint_candidates_per_block"] = 8
    return {
        "metadata": {
            "schema_version": "paper1-three-seed-diagnostic-validation-0.2",
            "source_artifact": str(input_path.relative_to(ROOT)),
            "scope": "Fixed no-retraining ACPC/PCC/CRA/MAF rule over LeWM 4-task x 9-std full grid and training seeds 3072/3073/3074.",
            "robustness_endpoint": "pixels_std0.08_success",
            "selection_rule": "Among nonzero std checkpoints, select lowest aggregate rank over ACPC-H/trans low, PCC low, CRA high, MAF low. Metric computation and rule are fixed on seed 3072 before evaluating independent training seeds 3073/3074.",
            "development_training_seed": 3072,
            "heldout_training_seeds": [3073, 3074],
        },
        "summary": summary,
        "split_summaries": split_summaries,
        "selection_rows": selections,
        "correlation_rows": _correlation_rows(rows),
    }


def _write_md(path: Path, payload: dict) -> None:
    s = payload["summary"]
    lines = [
        "# Three-Seed Diagnostic Validation",
        "",
        "Fixed rule: rank nonzero checkpoints by ACPC-H/trans low, PCC low, CRA high, and MAF low; select the lowest aggregate rank. Robustness endpoint is observation-only `pixels_std0.08_success`.",
        "",
        "Protocol: seed 3072 is the development grid used to freeze metric computation and the aggregate-rank rule; seeds 3073/3074 are independent held-out training seeds evaluated after the rule is fixed.",
        "",
        "## Split Summary",
        "",
        "| Split | seeds | blocks | candidates | exact best | within 5pp | regret mean +/- std | bootstrap 95% CI | top-2 overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in payload["split_summaries"]:
        ci = split["bootstrap_ci95_mean_selected_regret_to_best_pp"]
        lines.append(
            "| {split} | {seeds} | {blocks} | {cands} | {exact}/{blocks} | {within}/{blocks} | {mean} +/- {std} | [{lo}, {hi}] | {top2}/2 |".format(
                split=split["split"],
                seeds=",".join(str(s) for s in split["training_seeds"]),
                blocks=split["n_task_seed_blocks"],
                cands=split["n_checkpoint_candidates"],
                exact=split["exact_best_hits"],
                within=split["within_5pp_hits"],
                mean=_fmt(split["mean_selected_regret_to_best_pp"]),
                std=_fmt(split["pstdev_selected_regret_to_best_pp"]),
                lo=_fmt(ci[0]),
                hi=_fmt(ci[1]),
                top2=_fmt(split["mean_top2_overlap"]),
            )
        )
    lines.extend([
        "",
        "## Summary",
        "",
        f"Blocks: {s['n_task_seed_blocks']} task-seed blocks; training seeds {s['training_seeds']}; 8 nonzero checkpoints per block.",
        f"Exact best hits: {s['exact_best_hits']}/{s['n_task_seed_blocks']}; within-5pp hits: {s['within_5pp_hits']}/{s['n_task_seed_blocks']}; mean regret to best: {_fmt(s['mean_selected_regret_to_best_pp'])} +/- {_fmt(s['pstdev_selected_regret_to_best_pp'])} pp; mean top-2 overlap: {_fmt(s['mean_top2_overlap'])}/2.",
        "",
        "## Selection Rows",
        "",
        "| Task | seed | selected std | selected px08 | best std | best px08 | regret | within 5pp | top-2 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ])
    for row in payload["selection_rows"]:
        lines.append(
            "| {task} | {seed} | {sel_std} | {sel_px} | {best_std} | {best_px} | {regret} | {within} | {top2}/2 |".format(
                task=row["task"],
                seed=row["training_seed"],
                sel_std=row["selected_std"],
                sel_px=_fmt(row["selected_px08_success"]),
                best_std=row["closed_loop_best_std"],
                best_px=_fmt(row["closed_loop_best_px08_success"]),
                regret=_fmt(row["selected_regret_to_best_pp"]),
                within="yes" if row["selected_within_5pp_of_best"] else "no",
                top2=row["top2_overlap"],
            )
        )
    lines.extend([
        "",
        "## Correlations",
        "",
        "| Scope | metric | rho vs px08 | r vs px08 | rho vs -drop | r vs -drop | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["correlation_rows"]:
        lines.append(
            "| {scope} | {metric} | {rho_px} | {r_px} | {rho_d} | {r_d} | {n} |".format(
                scope=row["scope"],
                metric=row["metric"],
                rho_px=_fmt(row["spearman_vs_px08_success"]),
                r_px=_fmt(row["pearson_vs_px08_success"]),
                rho_d=_fmt(row["spearman_vs_negative_drop"]),
                r_d=_fmt(row["pearson_vs_negative_drop"]),
                n=row["n"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)
    payload = build_payload(args.input)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_md(args.out_md, payload)
    print(f"wrote {args.out_json.relative_to(ROOT)}")
    print(f"wrote {args.out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
