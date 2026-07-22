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
canonical="assets/paper1_data/canonical_evals_pldm_v2.json"
source_base="paper1/results/remediation_phase2_external_sources/cross_stressor/pldm_canonical"
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

valid_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '.metadata.status_counts == {"ok": 2} and
      .metadata.missing_rows == [] and .metadata.errors == [] and
      (.rows | length) == 2' "$path" >/dev/null
}

failed=0
for index in "${!tasks[@]}"; do
  out_dir="${source_base}/${output_name}"
  out_path="${out_dir}/acpc_${slugs[$index]}_v2.json"
  mkdir -p "$out_dir"
  if valid_shard "$out_path"; then
    echo "[pldm-atr] skip task=${tasks[$index]} stressor=${stressor}"
    continue
  fi
  if [[ -e "$out_path" ]]; then
    mv -- "$out_path" "${out_path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  echo "[pldm-atr] start task=${tasks[$index]} stressor=${stressor} gpu=${gpu}"
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
    python tools/paper1_phase0_acpc.py \
      --methods PLDM \
      --tasks "${tasks[$index]}" \
      --std-keys 0.0 0.08 \
      --evals-pldm "$canonical" \
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
      --device cuda; then
    failed=1
    continue
  fi
  if ! valid_shard "$out_path"; then
    echo "invalid ATR shard: $out_path" >&2
    failed=1
  fi
done

exit "$failed"
