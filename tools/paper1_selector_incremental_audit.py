"""Incremental selector audit for Paper 1.

This audit asks whether ACPC-family readouts explain robustness variation after
controlling for the simple sources that reviewers can reasonably suspect:
training-noise level, task, and independent training seed.  It is intentionally
no-retraining and uses the released three-seed Phase-0 LeWM grid.
"""
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
DEFAULT_OUT_JSON = DATA_DIR / "selector_incremental_audit_20260704.json"
DEFAULT_OUT_MD = DATA_DIR / "selector_incremental_audit_20260704.md"
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
NONZERO_STD_KEYS = ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
DIAGNOSTIC_RULE = (
    ("ACPC-H/trans.", "acpc_h_norm_by_transition", -1.0),
    ("PCC", "pcc_abs_median", -1.0),
    ("CRA", "cra_spearman_mean", 1.0),
    ("MAF", "maf_flip_rate", -1.0),
)
METRICS = (
    ("Aggregate ACPC/PCC/CRA/MAF", "aggregate_quality", 1.0),
    ("ACPC-H/trans.", "acpc_h_norm_by_transition", -1.0),
    ("PCC", "pcc_abs_median", -1.0),
    ("CRA", "cra_spearman_mean", 1.0),
    ("MAF", "maf_flip_rate", -1.0),
    ("Elite overlap", "elite_overlap_mean", 1.0),
)
OUTCOMES = (
    ("obs0.08 success", "pixels_std0.08_success"),
    ("reduced drop", "reduced_drop"),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / len(values)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = _mean(xs)
    my = _mean(ys)
    dx = [float(x) - mx for x in xs]
    dy = [float(y) - my for y in ys]
    vx = sum(x * x for x in dx)
    vy = sum(y * y for y in dy)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / math.sqrt(vx * vy)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [float(rhs)] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= 1e-12:
            aug[col][col] += 1e-8
            pivot = col
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= denom
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) <= 1e-15:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _ols_fit(x: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    p = len(x[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row, yi in zip(x, y):
        for i in range(p):
            xty[i] += row[i] * yi
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    return _solve(xtx, xty)


def _predict(x: Sequence[Sequence[float]], beta: Sequence[float]) -> list[float]:
    return [sum(row[j] * beta[j] for j in range(len(beta))) for row in x]


def _residuals(x: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    beta = _ols_fit(x, y)
    pred = _predict(x, beta)
    return [float(yi) - pi for yi, pi in zip(y, pred)]


def _sse(values: Sequence[float]) -> float:
    return sum(float(v) * float(v) for v in values)


def _r2(x: Sequence[Sequence[float]], y: Sequence[float]) -> float:
    residual = _residuals(x, y)
    mean_y = _mean(y)
    total = sum((float(yi) - mean_y) ** 2 for yi in y)
    if total <= 1e-12:
        return 0.0
    return 1.0 - _sse(residual) / total


def _control_row(row: dict) -> list[float]:
    std = float(row["std_key"])
    task = row["task"]
    seed = int(row["training_seed"])
    # Intercept, a conservative two-term training-noise-level control, task FE,
    # and training-seed FE.  TwoRoom and seed 3072 are the reference categories.
    return [
        1.0,
        std,
        std * std,
        1.0 if task == "PushT" else 0.0,
        1.0 if task == "Reacher" else 0.0,
        1.0 if task == "Cube" else 0.0,
        1.0 if seed == 3073 else 0.0,
        1.0 if seed == 3074 else 0.0,
    ]


def _validate(rows: Sequence[dict]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    expected = {(task, seed, std) for task in TASKS for seed in SEEDS for std in ("0.0",) + NONZERO_STD_KEYS}
    got = {(r["task"], int(r["training_seed"]), str(r["std_key"])) for r in ok_rows}
    if got != expected:
        raise ValueError(f"coverage mismatch missing={sorted(expected - got)[:5]} extra={sorted(got - expected)[:5]}")


def _block_scores(candidates: Sequence[dict]) -> dict[tuple[str, int, str], float]:
    scores: dict[tuple[str, int, str], float] = {}
    for task in TASKS:
        for seed in SEEDS:
            block = [r for r in candidates if r["task"] == task and int(r["training_seed"]) == seed]
            if len(block) != len(NONZERO_STD_KEYS):
                raise ValueError(f"missing block rows for {task}/{seed}: {len(block)}")
            block_scores = {str(r["std_key"]): 0.0 for r in block}
            for _, key, sign in DIAGNOSTIC_RULE:
                signed = [sign * float(r[key]) for r in block]
                # Higher signed value is better, so rank the negated value with rank 1 best.
                ranks = _rankdata([-v for v in signed])
                for row, rank in zip(block, ranks):
                    block_scores[str(row["std_key"])] += rank
            for row in block:
                scores[(task, seed, str(row["std_key"]))] = -block_scores[str(row["std_key"])]
    return scores


def _prepare_rows(data: dict) -> list[dict]:
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    _validate(rows)
    candidates = [dict(r) for r in rows if str(r["std_key"]) != "0.0"]
    aggregate = _block_scores(candidates)
    for row in candidates:
        row["reduced_drop"] = -float(row["corruption_drop"])
        row["aggregate_quality"] = aggregate[(row["task"], int(row["training_seed"]), str(row["std_key"]))]
    return sorted(candidates, key=lambda r: (r["task"], int(r["training_seed"]), float(r["std_key"])))


def _partial_stats(rows: Sequence[dict], metric_key: str, metric_sign: float, outcome_key: str) -> dict:
    y = _rankdata([float(r[outcome_key]) for r in rows])
    m = _rankdata([metric_sign * float(r[metric_key]) for r in rows])
    controls = [_control_row(r) for r in rows]
    y_res = _residuals(controls, y)
    m_res = _residuals(controls, m)
    partial_r = _corr(m_res, y_res)
    base_r2 = _r2(controls, y)
    full_r2 = _r2([row + [mi] for row, mi in zip(controls, m)], y)
    return {
        "n": len(rows),
        "partial_r": partial_r,
        "partial_r2": partial_r * partial_r if math.isfinite(partial_r) else float("nan"),
        "base_r2": base_r2,
        "full_r2": full_r2,
        "incremental_r2": full_r2 - base_r2,
        "metric_residuals": m_res,
        "outcome_residuals": y_res,
    }


def _block_permutation_p(
    rows: Sequence[dict],
    metric_residuals: Sequence[float],
    outcome_residuals: Sequence[float],
    observed: float,
    *,
    n_permutations: int,
    seed: int,
) -> float:
    block_keys = [(task, seed_id) for task in TASKS for seed_id in SEEDS]
    by_block: dict[tuple[str, int], list[int]] = {key: [] for key in block_keys}
    for idx, row in enumerate(rows):
        by_block[(row["task"], int(row["training_seed"]))].append(idx)
    for key in block_keys:
        by_block[key].sort(key=lambda i: float(rows[i]["std_key"]))
        if len(by_block[key]) != len(NONZERO_STD_KEYS):
            raise ValueError(f"bad block length {key}: {len(by_block[key])}")

    rng = random.Random(seed)
    exceed = 0
    abs_obs = abs(observed)
    keys = list(block_keys)
    for _ in range(n_permutations):
        permuted = keys[:]
        rng.shuffle(permuted)
        shuffled = [0.0] * len(rows)
        for src_key, dst_key in zip(keys, permuted):
            src_idx = by_block[src_key]
            dst_idx = by_block[dst_key]
            for si, di in zip(src_idx, dst_idx):
                shuffled[di] = metric_residuals[si]
        val = _corr(shuffled, outcome_residuals)
        if abs(val) >= abs_obs:
            exceed += 1
    return (exceed + 1.0) / (n_permutations + 1.0)


def _leave_one(rows: Sequence[dict], metric_key: str, metric_sign: float, outcome_key: str, field: str) -> list[dict]:
    values = sorted({r[field] for r in rows})
    out = []
    for value in values:
        subset = [r for r in rows if r[field] != value]
        stats = _partial_stats(subset, metric_key, metric_sign, outcome_key)
        out.append({"held_out": value, "n": len(subset), "partial_r": stats["partial_r"]})
    return out


def _round(value: float, digits: int = 4) -> float | None:
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def build_payload(input_path: Path, *, n_permutations: int, seed: int) -> dict:
    rows = _prepare_rows(_load(input_path))
    metric_rows = []
    for metric_label, metric_key, metric_sign in METRICS:
        for outcome_label, outcome_key in OUTCOMES:
            stats = _partial_stats(rows, metric_key, metric_sign, outcome_key)
            p_value = _block_permutation_p(
                rows,
                stats["metric_residuals"],
                stats["outcome_residuals"],
                stats["partial_r"],
                n_permutations=n_permutations,
                seed=seed + len(metric_rows) * 17,
            )
            task_loso = _leave_one(rows, metric_key, metric_sign, outcome_key, "task")
            seed_loso = _leave_one(rows, metric_key, metric_sign, outcome_key, "training_seed")
            metric_rows.append({
                "metric": metric_label,
                "metric_key": metric_key,
                "outcome": outcome_label,
                "outcome_key": outcome_key,
                "direction": "higher signed metric is better",
                "n": stats["n"],
                "controls": ["std_max", "std_max^2", "task fixed effects", "training-seed fixed effects"],
                "partial_r": stats["partial_r"],
                "partial_r2": stats["partial_r2"],
                "base_r2": stats["base_r2"],
                "full_r2": stats["full_r2"],
                "incremental_r2": stats["incremental_r2"],
                "block_permutation_p_two_sided": p_value,
                "leave_one_task_out": task_loso,
                "leave_one_training_seed_out": seed_loso,
            })

    compact_rows = []
    by_metric = {label: {} for label, _, _ in METRICS}
    for row in metric_rows:
        by_metric[row["metric"]][row["outcome"]] = row
    for metric_label, _, _ in METRICS:
        reduced = by_metric[metric_label]["reduced drop"]
        success = by_metric[metric_label]["obs0.08 success"]
        compact_rows.append({
            "metric": metric_label,
            "reduced_drop_partial_r": reduced["partial_r"],
            "reduced_drop_partial_r2": reduced["partial_r2"],
            "reduced_drop_incremental_r2": reduced["incremental_r2"],
            "reduced_drop_block_permutation_p": reduced["block_permutation_p_two_sided"],
            "obs008_success_partial_r": success["partial_r"],
            "obs008_success_partial_r2": success["partial_r2"],
            "obs008_success_incremental_r2": success["incremental_r2"],
            "obs008_success_block_permutation_p": success["block_permutation_p_two_sided"],
        })

    return {
        "metadata": {
            "schema_version": "paper1-selector-incremental-audit-0.1",
            "source_artifact": str(input_path.relative_to(ROOT)),
            "scope": "No-retraining incremental explanatory audit over the 96 nonzero LeWM Gaussian rows: 4 tasks x 3 training seeds x 8 nonzero std_max checkpoints.",
            "row_unit": "task-training-seed-checkpoint",
            "controls": ["std_max", "std_max^2", "task fixed effects", "training-seed fixed effects"],
            "rank_transform": True,
            "block_permutation": "permutes residual metric blocks across the 12 task-training-seed blocks, preserving each 8-row std grid within a block",
            "n_block_permutations": n_permutations,
            "random_seed": seed,
            "interpretation": "Tests residual diagnostic signal beyond training-noise-level, task, and training-seed controls; not a selector dominance or closed-loop guarantee claim.",
        },
        "compact_rows": compact_rows,
        "metric_rows": metric_rows,
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _write_md(path: Path, payload: dict) -> None:
    lines = [
        "# Selector Incremental Audit",
        "",
        "This no-retraining audit asks whether the paired ACPC-family readouts explain robustness variation after controlling for `std_max`, `std_max^2`, task fixed effects, and training-seed fixed effects. Rows are the 96 nonzero LeWM Gaussian checkpoints.",
        "",
        "| Metric | partial r vs reduced drop | partial R2 | incr. R2 | block p | partial r vs obs0.08 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["compact_rows"]:
        lines.append(
            "| {metric} | {rd_r} | {rd_r2} | {rd_inc} | {p} | {obs_r} |".format(
                metric=row["metric"],
                rd_r=_fmt(row["reduced_drop_partial_r"]),
                rd_r2=_fmt(row["reduced_drop_partial_r2"]),
                rd_inc=_fmt(row["reduced_drop_incremental_r2"]),
                p=_fmt(row["reduced_drop_block_permutation_p"]),
                obs_r=_fmt(row["obs008_success_partial_r"]),
            )
        )
    lines.extend([
        "",
        "Reading: reduced drop is the cleaner residual target than absolute noisy success. The audit tests incremental explanatory signal; it does not establish a superior checkpoint selector.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args(argv)
    payload = build_payload(args.input, n_permutations=args.n_permutations, seed=args.seed)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_md(args.out_md, payload)
    print(f"wrote {args.out_json.relative_to(ROOT)}")
    print(f"wrote {args.out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
