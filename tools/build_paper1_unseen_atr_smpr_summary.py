"""Build Paper 1 unseen-stressor ATR/SMPR summary.

This joins strongest-severity unseen closed-loop score rows with paper-facing
unseen-stressor diagnostics:
- ATR_q90 = acpc_h_l2_p90 / clean_transition_l2_median from the unseen Phase-0
  paired diagnostic rows under the same stressor.
- SMPR_m0 = task-grounded near-boundary semantic margin pass-rate recomputed by
  tools.paper1_semantic_margin under the same stressor.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_SUBSET = DATA_DIR / "unseen_phase0_acpc_subset.json"
DEFAULT_BLUR_SMPR = DATA_DIR / "semantic_task_grounded_margin_unseen_blur_lewm_three_seed.json"
DEFAULT_RESIZE_SMPR = DATA_DIR / "semantic_task_grounded_margin_unseen_resize_lewm_three_seed.json"
DEFAULT_OUT = DATA_DIR / "unseen_atr_smpr_summary_20260707.json"
DEFAULT_MD_OUT = DATA_DIR / "unseen_atr_smpr_summary_20260707.md"
STD_KEYS = ("0.0", "0.08")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return fmean(vals) if vals else float("nan")


def _pstdev(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    mu = fmean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _metric_block(values: Sequence[float]) -> dict[str, float]:
    return {"mean": _mean(values), "pstdev": _pstdev(values)}


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xvals, yvals = zip(*pairs)
    mx = fmean(xvals)
    my = fmean(yvals)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    deny = math.sqrt(sum((y - my) ** 2 for y in yvals))
    return num / (denx * deny) if denx > 0 and deny > 0 else float("nan")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xvals, yvals = zip(*pairs)
    return _pearson(_rankdata(xvals), _rankdata(yvals))


def _row_by_std(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("std_key")): row for row in raw.get("rows", [])}


def _atr(row: Mapping[str, Any]) -> float:
    p90 = _finite(row.get("acpc_h_l2_p90"))
    scale = _finite(row.get("clean_transition_l2_median"))
    return p90 / scale if scale > 0 else float("nan")


def _smpr_index(*payloads: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    out: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for payload in payloads:
        for row in payload.get("rows", []):
            if row.get("status") == "ok":
                out[(str(row["task"]), str(row["std_key"]), int(row["training_seed"]))] = row
    return out


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(x: float, ndigits: int = 2) -> str:
    return "nan" if not math.isfinite(float(x)) else f"{float(x):.{ndigits}f}"


def build(*, subset_path: Path, blur_smpr_path: Path, resize_smpr_path: Path, out_path: Path, md_out_path: Path | None) -> dict[str, Any]:
    subset = _load(subset_path)
    blur_smpr = _load(blur_smpr_path)
    resize_smpr = _load(resize_smpr_path)
    smpr = _smpr_index(blur_smpr, resize_smpr)

    rows: list[dict[str, Any]] = []
    for item in subset.get("rows", []):
        raw_path = ROOT / str(item["raw_phase0_artifact"])
        raw_rows = _row_by_std(_load(raw_path))
        base = raw_rows["0.0"]
        robust = raw_rows["0.08"]
        task = str(item["task"])
        seed = int(item["seed"])
        base_smpr = smpr[(task, "0.0", seed)]
        robust_smpr = smpr[(task, "0.08", seed)]
        atr_base = _atr(base)
        atr_robust = _atr(robust)
        smpr_base = _finite(base_smpr.get("semantic_margin_pass_rate"))
        smpr_robust = _finite(robust_smpr.get("semantic_margin_pass_rate"))
        ev = item["eval"]
        rows.append(
            {
                "seed": seed,
                "task": task,
                "family": item["family"],
                "magnitude": item["magnitude"],
                "case": item["case"],
                "baseline_stress_success": _finite(ev.get("baseline_stress_success")),
                "std008_stress_success": _finite(ev.get("std008_stress_success")),
                "stress_success_delta": _finite(ev.get("stress_success_delta")),
                "ATR_q90_0.0": atr_base,
                "ATR_q90_0.08": atr_robust,
                "ATR_drop": atr_base - atr_robust,
                "SMPR_0.0": smpr_base,
                "SMPR_0.08": smpr_robust,
                "SMPR_gain": smpr_robust - smpr_base,
                "raw_phase0_artifact": item["raw_phase0_artifact"],
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for task in ("TwoRoom", "Reacher", "PushT", "Cube"):
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        summary_rows.append(
            {
                "task": task,
                "family": task_rows[0]["family"],
                "magnitude": task_rows[0]["magnitude"],
                "training_seeds": sorted(int(row["seed"]) for row in task_rows),
                "baseline_stress_success": _metric_block([row["baseline_stress_success"] for row in task_rows]),
                "std008_stress_success": _metric_block([row["std008_stress_success"] for row in task_rows]),
                "stress_success_delta": _metric_block([row["stress_success_delta"] for row in task_rows]),
                "ATR_q90_0.0": _metric_block([row["ATR_q90_0.0"] for row in task_rows]),
                "ATR_q90_0.08": _metric_block([row["ATR_q90_0.08"] for row in task_rows]),
                "ATR_drop": _metric_block([row["ATR_drop"] for row in task_rows]),
                "SMPR_0.0": _metric_block([row["SMPR_0.0"] for row in task_rows]),
                "SMPR_0.08": _metric_block([row["SMPR_0.08"] for row in task_rows]),
                "SMPR_gain": _metric_block([row["SMPR_gain"] for row in task_rows]),
            }
        )

    score_delta = [row["stress_success_delta"] for row in rows]
    atr_drop = [row["ATR_drop"] for row in rows]
    smpr_gain = [row["SMPR_gain"] for row in rows]
    payload = {
        "metadata": {
            "schema_version": "paper1-unseen-atr-smpr-summary-20260707-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "strongest-severity unseen-stressor diagnostics under the same blur/resize stressor as the closed-loop score rows; no retraining",
            "atr_definition": "ATR_q90 = acpc_h_l2_p90 / clean_transition_l2_median from unseen Phase-0 paired diagnostic rows; lower is better.",
            "smpr_definition": "SMPR_m0 = task-grounded near-boundary semantic margin pass-rate recomputed under the same unseen stressor; higher is better.",
            "source_artifacts": [
                _relative(subset_path),
                _relative(blur_smpr_path),
                _relative(resize_smpr_path),
            ],
            "std_keys": list(STD_KEYS),
            "training_seeds": [3072, 3073, 3074],
        },
        "rows": rows,
        "summary_rows": summary_rows,
        "correlations": {
            "seed_rows_n": len(rows),
            "pearson_stress_delta_vs_ATR_drop": _pearson(score_delta, atr_drop),
            "spearman_stress_delta_vs_ATR_drop": _spearman(score_delta, atr_drop),
            "pearson_stress_delta_vs_SMPR_gain": _pearson(score_delta, smpr_gain),
            "spearman_stress_delta_vs_SMPR_gain": _spearman(score_delta, smpr_gain),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if md_out_path is not None:
        lines = [
            "# Paper1 Unseen ATR/SMPR Summary",
            "",
            "Values are population mean over training seeds 3072/3073/3074. ATR and SMPR are reported as raw no-noise -> noise-trained values under the same unseen stressor; lower ATR and higher SMPR are better.",
            "",
            "| Task | stressor | no-noise score | noise-trained score | ATR raw | SMPR raw |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in summary_rows:
            stressor = f"{row['family']} {row['magnitude']:g}"
            lines.append(
                "| {task} | {stressor} | {base} +/- {base_sd} | {rob} +/- {rob_sd} | {atr_base} -> {atr_rob} | {smpr_base} -> {smpr_rob} |".format(
                    task=row["task"],
                    stressor=stressor,
                    base=_fmt(row["baseline_stress_success"]["mean"]),
                    base_sd=_fmt(row["baseline_stress_success"]["pstdev"]),
                    rob=_fmt(row["std008_stress_success"]["mean"]),
                    rob_sd=_fmt(row["std008_stress_success"]["pstdev"]),
                    atr_base=_fmt(row["ATR_q90_0.0"]["mean"]),
                    atr_rob=_fmt(row["ATR_q90_0.08"]["mean"]),
                    smpr_base=_fmt(row["SMPR_0.0"]["mean"]),
                    smpr_rob=_fmt(row["SMPR_0.08"]["mean"]),
                )
            )
        lines.extend([
            "",
            "Seed-row correlations are descriptive for this bounded scope check, not a formal transfer theorem.",
            "",
            "```json",
            json.dumps(payload["correlations"], indent=2, sort_keys=True),
            "```",
        ])
        md_out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--blur-smpr", type=Path, default=DEFAULT_BLUR_SMPR)
    parser.add_argument("--resize-smpr", type=Path, default=DEFAULT_RESIZE_SMPR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()
    payload = build(
        subset_path=args.subset,
        blur_smpr_path=args.blur_smpr,
        resize_smpr_path=args.resize_smpr,
        out_path=args.out,
        md_out_path=args.md_out,
    )
    print(f"wrote {args.out}")
    if args.md_out:
        print(f"wrote {args.md_out}")
    for row in payload["summary_rows"]:
        print(
            f"{row['task']}: score={row['baseline_stress_success']['mean']:.2f}->{row['std008_stress_success']['mean']:.2f}, "
            f"ATR={row['ATR_q90_0.0']['mean']:.2f}->{row['ATR_q90_0.08']['mean']:.2f}, "
            f"SMPR={row['SMPR_0.0']['mean']:.2f}->{row['SMPR_0.08']['mean']:.2f}"
        )


if __name__ == "__main__":
    main()
