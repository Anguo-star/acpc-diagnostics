"""No-retraining remediation summaries for Paper 1.

This script consumes existing Phase-0 paired ACPC artifacts and writes a compact
summary used by the top-conference remediation pass. It does not load models,
checkpoints, or datasets; it only recomputes tables from released JSON files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_OUT_JSON = DATA_DIR / "no_retrain_diagnostic_audit.json"
DEFAULT_OUT_MD = DATA_DIR / "no_retrain_diagnostic_audit.md"
METHOD_ARTIFACTS = {
    "LeWM": DATA_DIR / "heldout_selection_phase0_seed9101.json",
    "PLDM": DATA_DIR / "heldout_selection_phase0_pldm_seed9101.json",
}
TASKS = ["TwoRoom", "PushT", "Reacher", "Cube"]
DIAGNOSTIC_RULE_METRICS = [
    ("acpc_h_norm_by_transition", "low"),
    ("pcc_abs_median", "low"),
    ("cra_spearman_mean", "high"),
    ("maf_flip_rate", "low"),
]
CORRELATION_METRICS = [
    ("Encoder radius", "encoder_shift_to_nn_l2", -1.0),
    ("ACPC-H/trans.", "acpc_h_norm_by_transition", -1.0),
    ("PCC", "pcc_abs_median", -1.0),
    ("CRA", "cra_spearman_mean", 1.0),
    ("MAF", "maf_flip_rate", -1.0),
]


def _load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [r for r in data["rows"] if r.get("status") == "ok"]


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


def _rank_score(rows: Sequence[dict]) -> dict[str, float]:
    scores = {str(row["std_key"]): 0.0 for row in rows}
    for key, direction in DIAGNOSTIC_RULE_METRICS:
        ordered = sorted(rows, key=lambda row: float(row[key]), reverse=(direction == "high"))
        for rank, row in enumerate(ordered):
            scores[str(row["std_key"])] += float(rank)
    return scores


def _selection_summary(method: str, rows: Sequence[dict]) -> list[dict]:
    out = []
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        base = next(r for r in task_rows if str(r["std_key"]) == "0.0")
        candidates = [r for r in task_rows if str(r["std_key"]) != "0.0"]
        scores = _rank_score(candidates)
        selected = min(candidates, key=lambda r: (scores[str(r["std_key"])], str(r["std_key"])))
        best = max(candidates, key=lambda r: float(r["pixels_goal_std0.08_success"]))
        out.append(
            {
                "method": method,
                "task": task,
                "base_px08_success": float(base["pixels_goal_std0.08_success"]),
                "selected_std": str(selected["std_key"]),
                "selected_px08_success": float(selected["pixels_goal_std0.08_success"]),
                "closed_loop_best_std": str(best["std_key"]),
                "closed_loop_best_px08_success": float(best["pixels_goal_std0.08_success"]),
                "gap_to_best_pp": float(best["pixels_goal_std0.08_success"] - selected["pixels_goal_std0.08_success"]),
                "selected_rank_score": float(scores[str(selected["std_key"])]),
                "selected_acpc_h_norm_by_transition": float(selected["acpc_h_norm_by_transition"]),
                "selected_pcc_abs_median": float(selected["pcc_abs_median"]),
                "selected_cra_spearman_mean": float(selected["cra_spearman_mean"]),
                "selected_maf_flip_rate": float(selected["maf_flip_rate"]),
            }
        )
    return out


def _correlation_summary(method: str, rows: Sequence[dict]) -> list[dict]:
    px08 = [float(r["pixels_goal_std0.08_success"]) for r in rows]
    negative_drop = [-float(r["corruption_drop"]) for r in rows]
    out = []
    for label, key, sign in CORRELATION_METRICS:
        values = [sign * float(r[key]) for r in rows]
        out.append(
            {
                "method": method,
                "metric": label,
                "n": len(rows),
                "spearman_vs_px08_success": _spearman(values, px08),
                "spearman_vs_negative_drop": _spearman(values, negative_drop),
                "signed_direction": "higher_is_better_after_sign",
            }
        )
    return out


def _fmt(x: float, digits: int = 2) -> str:
    return f"{float(x):.{digits}f}"


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# No-Retraining Diagnostic Audit",
        "",
        "This artifact is computed only from existing Phase-0 paired ACPC JSON files. ",
        "It is a frozen-rule sanity audit, not a true held-out prospective validation.",
        "",
        "## Composite diagnostic selection",
        "",
        "Rule: among nonzero training-noise checkpoints, select the row with the lowest ",
        "aggregate rank over ACPC-H/transition (low), PCC (low), CRA (high), and MAF (low).",
        "",
        "| Method | Task | selected std | selected px08 | closed-loop best std | best px08 | gap pp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["selection_rows"]:
        lines.append(
            "| {method} | {task} | {selected_std} | {selected_px08} | {best_std} | {best_px08} | {gap} |".format(
                method=row["method"],
                task=row["task"],
                selected_std=row["selected_std"],
                selected_px08=_fmt(row["selected_px08_success"]),
                best_std=row["closed_loop_best_std"],
                best_px08=_fmt(row["closed_loop_best_px08_success"]),
                gap=_fmt(row["gap_to_best_pp"]),
            )
        )
    lines.extend(
        [
            "",
            "## Spearman correlations",
            "",
            "Metric signs are oriented so larger values mean a better diagnostic reading.",
            "",
            "| Method | Metric | rho vs px08 success | rho vs -drop | n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["correlation_rows"]:
        lines.append(
            "| {method} | {metric} | {rho_px} | {rho_drop} | {n} |".format(
                method=row["method"],
                metric=row["metric"],
                rho_px=_fmt(row["spearman_vs_px08_success"]),
                rho_drop=_fmt(row["spearman_vs_negative_drop"]),
                n=row["n"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def build_payload() -> dict:
    all_selection = []
    all_corr = []
    for method, artifact in METHOD_ARTIFACTS.items():
        rows = _load_rows(artifact)
        all_selection.extend(_selection_summary(method, rows))
        all_corr.extend(_correlation_summary(method, rows))
    gaps = [row["gap_to_best_pp"] for row in all_selection]
    return {
        "metadata": {
            "schema_version": "paper1-no-retrain-diagnostic-audit-0.1",
            "source_artifacts": {m: str(p.relative_to(ROOT)) for m, p in METHOD_ARTIFACTS.items()},
            "selection_rule": "lowest aggregate rank over ACPC-H/trans low, PCC low, CRA high, MAF low among nonzero training-noise checkpoints",
            "scope": "existing Phase-0 full-grid artifacts only; no model loading, no retraining, no true held-out prospective split",
        },
        "summary": {
            "selection_rows": len(all_selection),
            "mean_gap_to_closed_loop_best_pp": sum(gaps) / len(gaps),
            "max_gap_to_closed_loop_best_pp": max(gaps),
            "exact_best_hits": sum(1 for g in gaps if abs(g) < 1e-9),
            "within_2pp_hits": sum(1 for g in gaps if g <= 2.0),
            "within_6pp_hits": sum(1 for g in gaps if g <= 6.0),
        },
        "selection_rows": all_selection,
        "correlation_rows": all_corr,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)
    payload = build_payload()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
