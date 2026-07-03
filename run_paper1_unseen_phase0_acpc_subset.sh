#!/usr/bin/env bash
# Seed-aware Phase-0 ACPC subset for the Paper 1 unseen-perturbation follow-up.
#
# This runner keeps the Gaussian ACPC basin artifact separate.  It uses the
# Phase-0 paired ACPC/PCC/CRA/MAF runner on a small blur/resize subset selected
# from the completed std=0.0 vs std=0.08 unseen eval:
#
#   positive transfer: TwoRoom blur, Reacher blur
#   boundary cases:   PushT resize, Cube resize
#
# Usage:
#   DATA_ROOT=/path/to/world_model/quentinll \
#     bash run_paper1_unseen_phase0_acpc_subset.sh
#
# Common overrides:
#   TRAIN_SEEDS="3073" DRY_RUN=1 bash run_paper1_unseen_phase0_acpc_subset.sh
#   DEVICE=cuda:0 N_SEQUENCES=64 RANDOM_ACTION_TRIALS=32 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
if [ -z "${DATA_ROOT}" ]; then
    echo "Set DATA_ROOT, PAPER1_DATA_ROOT, or STABLEWM_HOME to the world_model/quentinll root." >&2
    exit 1
fi
export STABLEWM_HOME="${DATA_ROOT}"

TRAIN_SEEDS="${TRAIN_SEEDS:-3073 3074}"
CASES="${CASES:-TwoRoom:gaussian_blur:15 Reacher:gaussian_blur:15 PushT:resize:0.25 Cube:resize:0.25}"
STD_KEYS="${STD_KEYS:-0.0 0.08}"
BASE_CANONICAL="${BASE_CANONICAL:-assets/paper1_data/canonical_evals_20260517.json}"
RAW_DIR="${RAW_DIR:-assets/paper1_data/unseen_phase0_acpc_subset_raw}"
OUT="${OUT:-assets/paper1_data/unseen_phase0_acpc_subset.json}"
SCHEMA_OUT="${SCHEMA_OUT:-assets/paper1_data/unseen_phase0_acpc_subset.schema.json}"
N_SEQUENCES="${N_SEQUENCES:-100}"
RANDOM_ACTION_TRIALS="${RANDOM_ACTION_TRIALS:-64}"
DIAG_SEED="${DIAG_SEED:-9101}"
FUTURE_STEPS="${FUTURE_STEPS:-9}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-8}"
FRAMESKIP="${FRAMESKIP:-5}"
IMG_SIZE="${IMG_SIZE:-224}"
ONLY_MISSING="${ONLY_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RAW_DIR}"

model_roots=(
    "${DATA_ROOT}/lewm-tworooms"
    "${DATA_ROOT}/lewm-pusht"
    "${DATA_ROOT}/lewm-reacher"
    "${DATA_ROOT}/lewm-cube"
)

write_seed_canonical() {
    local seed="$1"
    local out="/tmp/paper1_seed${seed}_canonical.json"
    python - "${BASE_CANONICAL}" "${out}" "${seed}" "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = int(sys.argv[3])
root = Path(sys.argv[4]).expanduser()

slugs = {
    "TwoRoom": "tworoom",
    "PushT": "pusht",
    "Reacher": "reacher",
    "Cube": "cube",
}
task_dirs = {
    "TwoRoom": "lewm-tworooms",
    "PushT": "lewm-pusht",
    "Reacher": "lewm-reacher",
    "Cube": "lewm-cube",
}

data = json.loads(base_path.read_text(encoding="utf-8"))
for task, slug in slugs.items():
    if task not in data:
        raise SystemExit(f"missing task in canonical JSON: {task}")
    for std_key, entry in data[task].items():
        if std_key == "0.0":
            subdir = f"{slug}_lewm_baseline_seed{seed}"
        else:
            std_int = int(round(float(std_key) * 100))
            subdir = f"{slug}_lewm_noise_0to{std_int:03d}_p1_seed{seed}"
        entry["subdir"] = subdir
        entry["path"] = str(root / task_dirs[task] / "ckpt" / subdir)

out_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
PY
}

model_root_args=()
for root in "${model_roots[@]}"; do
    model_root_args+=(--model-root "${root}")
done

printf '[unseen-phase0-subset] DATA_ROOT=%s\n' "${DATA_ROOT}"
printf '[unseen-phase0-subset] train_seeds=%s\n' "${TRAIN_SEEDS}"
printf '[unseen-phase0-subset] cases=%s\n' "${CASES}"
printf '[unseen-phase0-subset] std_keys=%s n_sequences=%s random_action_trials=%s\n' \
    "${STD_KEYS}" "${N_SEQUENCES}" "${RANDOM_ACTION_TRIALS}"
printf '[unseen-phase0-subset] dry_run=%s only_missing=%s raw_dir=%s\n' \
    "${DRY_RUN}" "${ONLY_MISSING}" "${RAW_DIR}"

for seed in ${TRAIN_SEEDS}; do
    seed_canonical="$(write_seed_canonical "${seed}")"
    for case_spec in ${CASES}; do
        IFS=':' read -r task family magnitude <<< "${case_spec}"
        if [ -z "${task}" ] || [ -z "${family}" ] || [ -z "${magnitude}" ]; then
            echo "Invalid CASES entry '${case_spec}'; expected Task:family:magnitude." >&2
            exit 1
        fi
        task_slug="$(printf '%s' "${task}" | tr '[:upper:]' '[:lower:]')"
        raw_out="${RAW_DIR}/seed${seed}_${task_slug}_${family}.json"
        if [ "${ONLY_MISSING}" = "1" ] && [ "${DRY_RUN}" != "1" ] && [ -s "${raw_out}" ]; then
            printf '[unseen-phase0-subset] skip existing %s\n' "${raw_out}"
            continue
        fi

        args=(
            --methods LeWM
            --tasks "${task}"
            --std-keys ${STD_KEYS}
            --evals-lewm "${seed_canonical}"
            "${model_root_args[@]}"
            --out "${raw_out}"
            --noise-std "${magnitude}"
            --corruption-type "${family}"
            --clean-goal
            --n-sequences "${N_SEQUENCES}"
            --random-action-trials "${RANDOM_ACTION_TRIALS}"
            --future-steps "${FUTURE_STEPS}"
            --rollout-horizon "${ROLLOUT_HORIZON}"
            --frameskip "${FRAMESKIP}"
            --img-size "${IMG_SIZE}"
            --seed "${DIAG_SEED}"
        )
        if [ -n "${DEVICE:-}" ]; then
            args+=(--device "${DEVICE}")
        fi
        if [ "${DRY_RUN}" = "1" ]; then
            args+=(--dry-run)
        fi

        printf '[unseen-phase0-subset] seed=%s task=%s family=%s magnitude=%s\n' \
            "${seed}" "${task}" "${family}" "${magnitude}"
        python -m tools.paper1_phase0_acpc "${args[@]}"
    done
done

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

python -m tools.build_paper1_unseen_phase0_acpc_subset \
    --raw-dir "${RAW_DIR}" \
    --out "${OUT}" \
    --schema-out "${SCHEMA_OUT}" \
    --seeds ${TRAIN_SEEDS} \
    --cases ${CASES}
