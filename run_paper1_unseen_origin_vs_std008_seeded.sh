#!/usr/bin/env bash
# Seed-aware wrapper for the Paper1 std=0.0 vs std=0.08 unseen eval.
#
# The base launcher reads the paper-facing canonical JSON, whose subdir fields
# point at the original seed-3072-style checkpoint names.  Independent lockbox
# training seeds use suffixes such as *_seed3073 and *_seed3074, so this wrapper
# builds a temporary seed-specific canonical file before delegating to the base
# launcher.
#
# Usage:
#   DATA_ROOT=/path/to/world_model/quentinll TRAIN_SEED=3073 \
#     EVAL_GPUS="0 1 2 3" bash run_paper1_unseen_origin_vs_std008_seeded.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TRAIN_SEED="${TRAIN_SEED:-3072}"
BASE_CANONICAL="${BASE_CANONICAL:-${CANONICAL:-assets/paper1_data/canonical_evals_20260517.json}}"

if [[ "${TRAIN_SEED}" == "3072" && -z "${FORCE_SEED_CANONICAL:-}" ]]; then
    export CANONICAL="${BASE_CANONICAL}"
else
    SEED_CANONICAL="${SEED_CANONICAL:-/tmp/paper1_seed${TRAIN_SEED}_canonical.json}"
    python - "${BASE_CANONICAL}" "${SEED_CANONICAL}" "${TRAIN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = int(sys.argv[3])

slugs = {
    "TwoRoom": "tworoom",
    "PushT": "pusht",
    "Reacher": "reacher",
    "Cube": "cube",
}

with base_path.open() as f:
    data = json.load(f)

for task, slug in slugs.items():
    if task not in data:
        raise SystemExit(f"missing task in canonical JSON: {task}")
    for std_key, entry in data[task].items():
        if std_key == "0.0":
            entry["subdir"] = f"{slug}_lewm_baseline_seed{seed}"
        else:
            std_int = int(round(float(std_key) * 100))
            entry["subdir"] = f"{slug}_lewm_noise_0to{std_int:03d}_p1_seed{seed}"

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
print(f"[paper1-unseen-seeded] wrote seed canonical: {out_path}")
PY
    export CANONICAL="${SEED_CANONICAL}"
fi

export TRAIN_SEED

if [[ -z "${MANIFEST_OUT:-}" ]]; then
    export MANIFEST_OUT="assets/paper1_data/unseen_origin_vs_std008_strongest_s${TRAIN_SEED}_manifest.json"
fi
if [[ -z "${ARTIFACT_OUT:-}" ]]; then
    export ARTIFACT_OUT="assets/paper1_data/unseen_origin_vs_std008_strongest_s${TRAIN_SEED}.json"
fi
if [[ -z "${SCHEMA_OUT:-}" ]]; then
    export SCHEMA_OUT="assets/paper1_data/unseen_origin_vs_std008_strongest_s${TRAIN_SEED}.schema.json"
fi

exec bash run_paper1_unseen_origin_vs_std008_eval.sh
