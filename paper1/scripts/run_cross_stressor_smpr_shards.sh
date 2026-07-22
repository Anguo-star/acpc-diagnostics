#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <LeWM|PLDM> <gaussian_blur|resize> <magnitude> <output-name>" >&2
  exit 2
fi

method="$1"
stressor="$2"
magnitude="$3"
output_name="$4"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
model_base="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
: "${model_base:?Set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root}"
source_base="paper1/results/remediation_phase2_external_sources/cross_stressor"
gpu="${PAPER1_DIAGNOSTIC_GPU:-0}"
timeout_seconds="${PAPER1_DIAGNOSTIC_TIMEOUT_SECONDS:-7200}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"

tasks=(TwoRoom PushT Reacher Cube)
slugs=(tworoom pusht reacher cube)
roots=(lewm-tworooms lewm-pusht lewm-reacher lewm-cube)

for value_name in timeout_seconds native_threads; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
done

case "$method" in
  LeWM)
    training_seeds_raw="${PAPER1_TRAINING_SEEDS:-3072 3073 3074}"
    read -ra training_seeds <<< "$training_seeds_raw"
    ;;
  PLDM)
    training_seeds=(3072)
    ;;
  *)
    echo "method must be LeWM or PLDM; got '${method}'" >&2
    exit 2
    ;;
esac

valid_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '.metadata.schema_version == "paper1-smpr-v2-1.0" and
      .metadata.status_counts == {"ok": 2} and
      .metadata.missing_rows == [] and .metadata.errors == [] and
      (.rows | length) == 2' "$path" >/dev/null
}

failed=0
for training_seed in "${training_seeds[@]}"; do
  if [[ "$method" == "LeWM" ]]; then
    family_id="lewm_seed${training_seed}"
    manifest="assets/paper1_data/training_seed_eval_manifests/lewm_seed${training_seed}_evals.json"
    out_dir="${source_base}/lewm_seed${training_seed}/${output_name}"
  else
    family_id="pldm_canonical_seed3072"
    manifest="assets/paper1_data/canonical_evals_pldm_v2.json"
    out_dir="${source_base}/pldm_canonical/${output_name}"
  fi
  reference_atr="${out_dir}/acpc_horizon_v2_checkpoint_bound.json"
  if [[ ! -s "$reference_atr" ]]; then
    echo "missing reference ATR: $reference_atr" >&2
    exit 1
  fi

  for index in "${!tasks[@]}"; do
    out_path="${out_dir}/smpr_${slugs[$index]}_v2.json"
    mkdir -p "$out_dir"
    if valid_shard "$out_path"; then
      echo "[smpr] skip method=${method} seed=${training_seed} task=${tasks[$index]} stressor=${stressor}"
      continue
    fi
    if [[ -e "$out_path" ]]; then
      mv -- "$out_path" "${out_path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    fi

    echo "[smpr] start method=${method} seed=${training_seed} task=${tasks[$index]} stressor=${stressor} gpu=${gpu}"
    if ! env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      HOME="$diagnostic_home" \
      OMP_NUM_THREADS="$native_threads" \
      MKL_NUM_THREADS="$native_threads" \
      OPENBLAS_NUM_THREADS="$native_threads" \
      NUMEXPR_NUM_THREADS="$native_threads" \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH=. \
      timeout --signal=TERM --kill-after=60s "${timeout_seconds}s" \
      python paper1/scripts/smpr_sensitivity.py \
        --method "$method" \
        --family-id "$family_id" \
        --training-seed "$training_seed" \
        --evals "$manifest" \
        --reference-atr "$reference_atr" \
        --model-root "${model_base}/${roots[$index]}" \
        --tasks "${tasks[$index]}" \
        --std-keys 0.0 0.08 \
        --n-sequences 100 \
        --num-noise-draws 5 \
        --rollout-horizon 8 \
        --radius-quantile 0.90 \
        --local-quantile 0.35 \
        --margin-delta-norm 0.10 \
        --noise-std "$magnitude" \
        --corruption-type "$stressor" \
        --anchor-seed 9101 \
        --embedding-space normalized \
        --device cuda \
        --out "$out_path"; then
      failed=1
      continue
    fi
    if ! valid_shard "$out_path"; then
      echo "invalid SMPR shard: $out_path" >&2
      failed=1
    fi
  done
done

exit "$failed"
