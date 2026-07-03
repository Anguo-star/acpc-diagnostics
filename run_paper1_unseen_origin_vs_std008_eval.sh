#!/usr/bin/env bash
# Formal Paper1 unseen-perturbation eval for origin-trained vs noise-trained checkpoints.
#
# Default protocol:
#   - training seed/checkpoint grid: seed 3072, std_max in {0.0, 0.08}
#   - tasks: PushT, TwoRoom, Reacher, Cube
#   - perturbation families: gaussian_blur, resize
#   - severities: no-op plus strongest stress only
#       gaussian_blur = 1,15
#       resize = 1.0,0.25
#   - eval budget: 100 episodes x 3 eval seeds (42, 43, 44)
#   - diagnostics disabled; rerun representative jobs with DIAGNOSTICS=1 only after review.
#
# Usage:
#   DATA_ROOT=/path/to/world_model/quentinll bash run_paper1_unseen_origin_vs_std008_eval.sh
#
# Common overrides:
#   TASKS="PushT TwoRoom" bash run_paper1_unseen_origin_vs_std008_eval.sh
#   FAMILIES="gaussian_blur" bash run_paper1_unseen_origin_vs_std008_eval.sh
#   EVAL_GPUS="0 1 2 3" bash run_paper1_unseen_origin_vs_std008_eval.sh
#   DRY_RUN=1 bash run_paper1_unseen_origin_vs_std008_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
if [ -z "${DATA_ROOT}" ]; then
    echo "Set DATA_ROOT, PAPER1_DATA_ROOT, or STABLEWM_HOME to the world_model/quentinll root." >&2
    exit 1
fi

TRAIN_SEED="${TRAIN_SEED:-3072}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-paper1_unseen_origin_vs_std008_s${TRAIN_SEED}_strongest}"
TASKS="${TASKS:-PushT TwoRoom Reacher Cube}"
STD_KEYS="${STD_KEYS:-0.0 0.08}"
FAMILIES="${FAMILIES:-gaussian_blur resize}"
EPOCH="${EPOCH:-10}"
NUM_EVAL="${NUM_EVAL:-300}"
EVAL_SEEDS="${EVAL_SEEDS:-3}"
EVAL_BASE_SEED="${EVAL_BASE_SEED:-42}"
APPLY_TO="${APPLY_TO:-1}"
BLUR_KERNEL_SIZES="${BLUR_KERNEL_SIZES:-1,15}"
RESIZE_FACTORS="${RESIZE_FACTORS:-1.0,0.25}"
MANIFEST_OUT="${MANIFEST_OUT:-assets/paper1_data/unseen_origin_vs_std008_strongest_manifest.json}"
ARTIFACT_OUT="${ARTIFACT_OUT:-assets/paper1_data/unseen_origin_vs_std008_strongest.json}"
SCHEMA_OUT="${SCHEMA_OUT:-assets/paper1_data/unseen_origin_vs_std008_strongest.schema.json}"
CANONICAL="${CANONICAL:-assets/paper1_data/canonical_evals_20260517.json}"
DIAGNOSTICS="${DIAGNOSTICS:-0}"
KEEP_GOING="${KEEP_GOING:-1}"
ONLY_MISSING="${ONLY_MISSING:-1}"
DRY_RUN="${DRY_RUN:-0}"

args=(
    --root "${DATA_ROOT}"
    --canonical "${CANONICAL}"
    --manifest-out "${MANIFEST_OUT}"
    --tasks ${TASKS}
    --std-keys ${STD_KEYS}
    --families ${FAMILIES}
    --family-magnitudes "gaussian_blur=${BLUR_KERNEL_SIZES}"
    --family-magnitudes "resize=${RESIZE_FACTORS}"
    --train-seed "${TRAIN_SEED}"
    --output-prefix "${OUTPUT_PREFIX}"
    --epoch "${EPOCH}"
    --num-eval "${NUM_EVAL}"
    --eval-seeds "${EVAL_SEEDS}"
    --eval-base-seed "${EVAL_BASE_SEED}"
    --apply-to "${APPLY_TO}"
)

if [ -n "${EVAL_GPUS:-}" ]; then
    args+=(--eval-gpus "${EVAL_GPUS}")
fi
if [ "${DIAGNOSTICS}" = "1" ]; then
    args+=(--diagnostics)
fi
if [ "${KEEP_GOING}" = "1" ]; then
    args+=(--keep-going)
fi
if [ "${ONLY_MISSING}" = "1" ]; then
    args+=(--only-missing)
fi
if [ "${DRY_RUN}" = "1" ]; then
    args+=(--dry-run)
fi

printf '[paper1-unseen-std008] DATA_ROOT=%s\n' "${DATA_ROOT}"
printf '[paper1-unseen-std008] output_prefix=%s\n' "${OUTPUT_PREFIX}"
printf '[paper1-unseen-std008] tasks=%s\n' "${TASKS}"
printf '[paper1-unseen-std008] std_keys=%s\n' "${STD_KEYS}"
printf '[paper1-unseen-std008] families=%s\n' "${FAMILIES}"
printf '[paper1-unseen-std008] gaussian_blur kernels=%s resize factors=%s\n' \
    "${BLUR_KERNEL_SIZES}" "${RESIZE_FACTORS}"
printf '[paper1-unseen-std008] eval=%s episodes x %s seeds from %s\n' \
    "$(( NUM_EVAL / EVAL_SEEDS ))" "${EVAL_SEEDS}" "${EVAL_BASE_SEED}"
printf '[paper1-unseen-std008] diagnostics=%s dry_run=%s only_missing=%s\n' \
    "${DIAGNOSTICS}" "${DRY_RUN}" "${ONLY_MISSING}"

python -m tools.paper1_unseen_eval_grid "${args[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

python -m tools.build_paper1_unseen_eval_artifact \
    --manifest "${MANIFEST_OUT}" \
    --out "${ARTIFACT_OUT}" \
    --schema-out "${SCHEMA_OUT}" \
    --root "${DATA_ROOT}" \
    --allow-missing
