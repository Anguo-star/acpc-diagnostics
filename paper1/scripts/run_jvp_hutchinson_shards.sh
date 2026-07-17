#!/usr/bin/env bash
set -euo pipefail

source_dir="paper1/results/remediation_phase1_jvp_sources"
model_base="/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
timeout_seconds="${PAPER1_JVP_TIMEOUT_SECONDS:-7200}"
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
      '.metadata.schema_version == "paper1-jvp-hutchinson-sensitivity-0.2" and
       .metadata.status == "complete" and
       .metadata.status_counts == {"ok": 3} and
       .metadata.missing_rows == [] and .metadata.errors == [] and
       (.rows | length) == 3 and (.probe_rows | length) == 24 and
       all(.rows[];
         (.encoder_trace_mean_ci95_unclipped | length) == 2 and
         (.rollout_trace_mean_ci95_unclipped | length) == 2 and
         (.composed_trace_mean_ci95_unclipped | length) == 2 and
         (.kappa_submultiplicative_probe_ci95_unclipped | length) == 2 and
         (.kappa_relative_isotropic_probe_ci95_unclipped | length) == 2)' \
      "$path" >/dev/null
}

inputs=()
for task_index in "${!tasks[@]}"; do
  for seed in "${seeds[@]}"; do
    stem="${slugs[$task_index]}_seed${seed}"
    out_json="${source_dir}/jvp_${stem}_v2.json"
    inputs+=(--input "$out_json")
    if valid_shard "$out_json"; then
      echo "[jvp] skip task=${tasks[$task_index]} seed=${seed}"
      continue
    fi
    if [[ -e "$out_json" ]]; then
      mv -- "$out_json" "${out_json}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    echo "[jvp] start task=${tasks[$task_index]} seed=${seed} gpu=${gpu}"
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
      python tools/paper1_jvp_hutchinson_sensitivity_audit.py \
        --seeds "$seed" \
        --tasks "${tasks[$task_index]}" \
        --model-root "${model_base}/${roots[$task_index]}" \
        --n-sequences 16 \
        --hutchinson-probes 8 \
        --rollout-horizon 8 \
        --device cuda \
        --out-json "$out_json" \
        --out-csv "${source_dir}/jvp_${stem}_v2.csv" \
        --summary-csv "${source_dir}/jvp_${stem}_summary_v2.csv" \
        --table "${source_dir}/jvp_${stem}_table_v2.tex"
    if ! valid_shard "$out_json"; then
      echo "invalid JVP shard: $out_json" >&2
      exit 1
    fi
  done
done

env PYTHONPATH=. python paper1/scripts/build_jvp_hutchinson_full_artifact.py \
  "${inputs[@]}"
