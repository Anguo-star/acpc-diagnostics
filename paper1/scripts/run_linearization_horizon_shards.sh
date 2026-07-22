#!/usr/bin/env bash
set -euo pipefail

source_dir="paper1/results/remediation_phase3_linearization_sources"
model_base="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
: "${model_base:?Set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root}"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
timeout_seconds="${PAPER1_LINEARIZATION_TIMEOUT_SECONDS:-10800}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"
gpu="${PAPER1_DIAGNOSTIC_GPU:-0}"

tasks=(TwoRoom PushT Reacher Cube)
slugs=(tworoom pusht reacher cube)
roots=(lewm-tworooms lewm-pusht lewm-reacher lewm-cube)
seeds=(3072 3073 3074)
mkdir -p "$source_dir"

for value_name in timeout_seconds native_threads; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
done

valid_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '
      .metadata.schema_version == "paper1-linearization-horizon-audit-0.1" and
      .metadata.status == "complete" and
      .metadata.status_counts == {"ok": 3} and
      .metadata.frozen_protocol_sha256 == "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21" and
      (.checkpoint_rows | length) == 3 and
      (.calibration_rows | length) == 12 and
      (.horizon_rows | length) == 36 and
      (.probe_rows | length) == 24 and
      all(.checkpoint_rows[]; .status == "ok") and
      all(.calibration_rows[]; .status == "ok") and
      all(.horizon_rows[]; .status == "ok")
    ' "$path" >/dev/null
}

inputs=()
for task_index in "${!tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    stem="${slugs[$task_index]}_seed${seed}"
    out_json="${source_dir}/linearization_${stem}_v1.json"
    inputs+=(--input "$out_json")
    if valid_shard "$out_json"; then
      echo "[linearization] skip task=${tasks[$task_index]} seed=${seed}"
      continue
    fi
    if [[ -e "$out_json" ]]; then
      mv -- "$out_json" "${out_json}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    echo "[linearization] start task=${tasks[$task_index]} seed=${seed} gpu=${gpu} threads=${native_threads}"
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
      python tools/paper1_linearization_horizon_audit.py \
        --seeds "$seed" \
        --tasks "${tasks[$task_index]}" \
        --model-root "${model_base}/${roots[$task_index]}" \
        --n-sequences 16 \
        --num-noise-draws 8 \
        --hutchinson-probes 8 \
        --horizon-noise-draws 5 \
        --device cuda \
        --out-json "$out_json" \
        --checkpoint-csv "${source_dir}/linearization_${stem}_checkpoints.csv" \
        --calibration-csv "${source_dir}/linearization_${stem}_calibration.csv" \
        --horizon-csv "${source_dir}/linearization_${stem}_horizons.csv"
    if ! valid_shard "$out_json"; then
      echo "invalid linearization shard: $out_json" >&2
      exit 1
    fi
  done
done

env PYTHONPATH=. python paper1/scripts/build_linearization_horizon_artifact.py \
  "${inputs[@]}"
