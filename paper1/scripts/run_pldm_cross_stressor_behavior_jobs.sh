#!/usr/bin/env bash
set -euo pipefail

root="/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"
canonical="assets/paper1_data/canonical_evals_pldm_v2.json"
source_root="paper1/results/remediation_phase2_external_sources/cross_stressor/pldm_canonical"
max_parallel_jobs="${PAPER1_MAX_PARALLEL_EVAL_JOBS:-1}"
# Preserve the original 100-env evaluation protocol. Resource control comes
# from one outer job at a time, disabled video encoding, bounded native
# threads, and the optimized episode scan. Changing this value can change the
# stochastic CEM candidate stream and therefore is not a pure performance knob.
eval_batch_size="${PAPER1_EVAL_BATCH_SIZE:-100}"
eval_timeout_seconds="${PAPER1_EVAL_TIMEOUT_SECONDS:-7200}"
eval_threads="${PAPER1_EVAL_THREADS:-2}"

for value_name in max_parallel_jobs eval_batch_size eval_timeout_seconds eval_threads; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
done

export OMP_NUM_THREADS="$eval_threads"
export MKL_NUM_THREADS="$eval_threads"
export OPENBLAS_NUM_THREADS="$eval_threads"
export NUMEXPR_NUM_THREADS="$eval_threads"
export PYTHONUNBUFFERED=1

echo "[pldm-cross-stressor] max_parallel_jobs=${max_parallel_jobs} batch_size=${eval_batch_size} timeout_seconds=${eval_timeout_seconds} native_threads=${eval_threads}"

run_job() {
  local gpu="$1"
  local task="$2"
  local std_key="$3"
  local family="$4"
  local magnitude="$5"
  local output_prefix="$6"
  local slug
  slug="$(printf '%s_%s_%s' "$task" "$std_key" "$family" | tr '[:upper:].' '[:lower:]p')"
  local manifest="${source_root}/${family}/behavior_job_${slug}.json"

  PYTHONPATH=. python -m tools.paper1_unseen_eval_grid \
    --root "$root" \
    --canonical "$canonical" \
    --manifest-out "$manifest" \
    --tasks "$task" \
    --std-keys "$std_key" \
    --families "$family" \
    --family-magnitudes "${family}=${magnitude}" \
    --train-seed 3072 \
    --output-prefix "$output_prefix" \
    --epoch 10 \
    --num-eval 300 \
    --eval-seeds 3 \
    --eval-base-seed 42 \
    --eval-gpus "$gpu" \
    --apply-to 1 \
    --trainer-file train_pldm.py \
    --config pldm \
    --post-train-eval-mode full \
    --extra-env "eval_batch_size=${eval_batch_size}" \
    --extra-env "eval_resume=1" \
    --extra-env "eval_save_video=0" \
    --extra-env "eval_timeout_seconds=${eval_timeout_seconds}" \
    --only-missing
}

run_wave() {
  local failed=0
  while (( $# )); do
    local pids=()
    local slot=0
    while (( slot < max_parallel_jobs && $# )); do
      echo "[pldm-cross-stressor] launch task=$2 std=$3 family=$4 magnitude=$5 gpu=$1"
      run_job "$1" "$2" "$3" "$4" "$5" "$6" &
      pids+=("$!")
      shift 6
      slot=$((slot + 1))
    done
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
  done
  return "$failed"
}

failed=0
if ! run_wave \
  0 PushT 0.08 gaussian_blur 15 paper1_cross_stressor_pldm_blur_endpoint \
  1 TwoRoom 0.08 gaussian_blur 15 paper1_cross_stressor_pldm_blur_endpoint \
  2 Reacher 0.08 gaussian_blur 15 paper1_cross_stressor_pldm_blur_endpoint \
  3 Cube 0.08 gaussian_blur 15 paper1_cross_stressor_pldm_blur_endpoint \
  4 PushT 0.0 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint \
  5 TwoRoom 0.0 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint \
  6 Reacher 0.0 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint \
  7 Cube 0.0 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint; then
  failed=1
fi

if ! run_wave \
  0 PushT 0.08 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint \
  1 TwoRoom 0.08 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint \
  2 Reacher 0.08 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint \
  3 Cube 0.08 resize 0.25 paper1_cross_stressor_pldm_resize_base_endpoint; then
  failed=1
fi

exit "$failed"
