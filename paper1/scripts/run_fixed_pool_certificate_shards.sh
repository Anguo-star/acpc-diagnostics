#!/usr/bin/env bash
set -euo pipefail

source_dir="paper1/results/remediation_phase3_certificate_sources"
model_base="/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
timeout_seconds="${PAPER1_CERTIFICATE_TIMEOUT_SECONDS:-7200}"
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
    jq -e \
      '.metadata.schema_version == "paper1-sample-level-certificate-0.2" and
       .metadata.status == "complete" and
       .metadata.status_counts == {"ok": 9} and
       .metadata.missing_rows == [] and .metadata.errors == [] and
       (.rows | length) == 9 and (.sample_rows | length) == 900 and
       all(.rows[];
         .sharp_cert_invariant_flip_count == 0 and
         (.coverage_by_K | length) == 4 and
         (.risk_coverage | length) >= 2)' \
      "$path" >/dev/null
}

inputs=()
for task_index in "${!tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    stem="${slugs[$task_index]}_seed${seed}"
    out_json="${source_dir}/certificate_${stem}_v2.json"
    inputs+=(--input "$out_json")
    if valid_shard "$out_json"; then
      echo "[certificate] skip task=${tasks[$task_index]} seed=${seed}"
      continue
    fi
    if [[ -e "$out_json" ]]; then
      mv -- "$out_json" "${out_json}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    echo "[certificate] start task=${tasks[$task_index]} seed=${seed} gpu=${gpu}"
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
      python tools/paper1_sample_level_certificate.py \
        --seeds "$seed" \
        --tasks "${tasks[$task_index]}" \
        --model-root "${model_base}/${roots[$task_index]}" \
        --include-samples \
        --n-sequences 100 \
        --random-action-trials 64 \
        --k-values 8 16 32 65 \
        --device cuda \
        --out-json "$out_json" \
        --out-csv "${source_dir}/certificate_${stem}_v2.csv" \
        --sample-csv "${source_dir}/certificate_${stem}_samples_v2.csv"
    if ! valid_shard "$out_json"; then
      echo "invalid certificate shard: $out_json" >&2
      exit 1
    fi
  done
done

env PYTHONPATH=. python paper1/scripts/fixed_pool_certificate_calibration.py \
  "${inputs[@]}"
