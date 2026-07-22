#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gaussian_blur|resize> <magnitude> <output-name>" >&2
  exit 2
fi

stressor="$1"
magnitude="$2"
output_name="$3"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
model_base="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
: "${model_base:?Set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root}"
source_base="paper1/results/remediation_phase2_external_sources/cross_stressor"
training_seeds_raw="${PAPER1_TRAINING_SEEDS:-3072 3073 3074}"
max_parallel="${PAPER1_MAX_PARALLEL_DIAGNOSTIC_JOBS:-1}"
timeout_seconds="${PAPER1_DIAGNOSTIC_TIMEOUT_SECONDS:-7200}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"
gpus_raw="${PAPER1_DIAGNOSTIC_GPUS:-0}"

tasks=(TwoRoom PushT Reacher Cube)
slugs=(tworoom pusht reacher cube)
roots=(lewm-tworooms lewm-pusht lewm-reacher lewm-cube)
read -ra training_seeds <<< "$training_seeds_raw"
read -ra gpus <<< "$gpus_raw"

for value_name in max_parallel timeout_seconds native_threads; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
done
if (( ${#training_seeds[@]} == 0 || ${#gpus[@]} == 0 )); then
  echo "PAPER1_TRAINING_SEEDS and PAPER1_DIAGNOSTIC_GPUS must be non-empty" >&2
  exit 2
fi

valid_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '.metadata.status_counts == {"ok": 2} and
      .metadata.missing_rows == [] and .metadata.errors == [] and
      (.rows | length) == 2' "$path" >/dev/null
}

run_shard() {
  local training_seed="$1"
  local index="$2"
  local gpu="$3"
  local out_dir="${source_base}/lewm_seed${training_seed}/${output_name}"
  local out_path="${out_dir}/acpc_${slugs[$index]}_v2.json"
  mkdir -p "$out_dir"

  if valid_shard "$out_path"; then
    echo "[lewm-atr] skip seed=${training_seed} task=${tasks[$index]} stressor=${stressor}"
    return 0
  fi
  if [[ -e "$out_path" ]]; then
    mv -- "$out_path" "${out_path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  echo "[lewm-atr] start seed=${training_seed} task=${tasks[$index]} stressor=${stressor} gpu=${gpu}"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HOME="$diagnostic_home" \
    OMP_NUM_THREADS="$native_threads" \
    MKL_NUM_THREADS="$native_threads" \
    OPENBLAS_NUM_THREADS="$native_threads" \
    NUMEXPR_NUM_THREADS="$native_threads" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    timeout --signal=TERM --kill-after=60s "${timeout_seconds}s" \
    python tools/paper1_phase0_acpc.py \
      --methods LeWM \
      --tasks "${tasks[$index]}" \
      --std-keys 0.0 0.08 \
      --evals-lewm "assets/paper1_data/training_seed_eval_manifests/lewm_seed${training_seed}_evals.json" \
      --model-root "${model_base}/${roots[$index]}" \
      --out "$out_path" \
      --n-sequences 100 \
      --num-noise-draws 5 \
      --rollout-horizon 8 \
      --noise-std "$magnitude" \
      --corruption-type "$stressor" \
      --clean-goal \
      --embedding-space normalized \
      --seed 9101 \
      --device cuda
  valid_shard "$out_path"
}

failed=0
launch_count=0
specs=()
for training_seed in "${training_seeds[@]}"; do
  for index in "${!tasks[@]}"; do
    specs+=("${training_seed}|${index}")
  done
done

cursor=0
while (( cursor < ${#specs[@]} )); do
  pids=()
  while (( ${#pids[@]} < max_parallel && cursor < ${#specs[@]} )); do
    IFS='|' read -r training_seed index <<< "${specs[$cursor]}"
    gpu="${gpus[$((launch_count % ${#gpus[@]}))]}"
    run_shard "$training_seed" "$index" "$gpu" &
    pids+=("$!")
    cursor=$((cursor + 1))
    launch_count=$((launch_count + 1))
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

for training_seed in "${training_seeds[@]}"; do
  for index in "${!tasks[@]}"; do
    out_path="${source_base}/lewm_seed${training_seed}/${output_name}/acpc_${slugs[$index]}_v2.json"
    if ! valid_shard "$out_path"; then
      echo "invalid ATR shard: $out_path" >&2
      failed=1
    fi
  done
done

exit "$failed"
