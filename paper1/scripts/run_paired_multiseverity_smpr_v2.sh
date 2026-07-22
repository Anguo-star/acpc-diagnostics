#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
case "$mode" in
  smoke|full) ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

protocol="paper1/config/paired_multiseverity_protocol_v1.json"
protocol_hash_file="paper1/config/paired_multiseverity_protocol_v1.sha256"
addendum_v1="paper1/config/paired_multiseverity_execution_addendum_v1.json"
addendum_v1_hash_file="paper1/config/paired_multiseverity_execution_addendum_v1.sha256"
addendum_v2="paper1/config/paired_multiseverity_execution_addendum_v2.json"
addendum_v2_hash_file="paper1/config/paired_multiseverity_execution_addendum_v2.sha256"
raw_root="paper1/results/multiseverity_v1/raw"
reference_root="paper1/results/multiseverity_v1/reference"
model_base="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
: "${model_base:?Set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root}"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
gpu="${PAPER1_DIAGNOSTIC_GPU:-0}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"
if [[ "$mode" == "smoke" ]]; then
  timeout_seconds="${PAPER1_DIAGNOSTIC_TIMEOUT_SECONDS:-1800}"
else
  timeout_seconds="${PAPER1_DIAGNOSTIC_TIMEOUT_SECONDS:-7200}"
fi

verify_frozen_file() {
  local path="$1"
  local hash_file="$2"
  local label="$3"
  [[ -f "$path" && -f "$hash_file" ]] || {
    echo "missing frozen $label contract" >&2
    exit 1
  }
  local expected actual
  read -r expected _ < "$hash_file"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if ! [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || [[ "$actual" != "$expected" ]]; then
    echo "frozen $label hash mismatch: expected=$expected actual=$actual" >&2
    exit 1
  fi
}

verify_frozen_file "$protocol" "$protocol_hash_file" "protocol"
verify_frozen_file "$addendum_v1" "$addendum_v1_hash_file" "execution addendum v1"
verify_frozen_file "$addendum_v2" "$addendum_v2_hash_file" "execution addendum v2"
for value_name in native_threads timeout_seconds; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$value_name must be a positive integer; got '$value'" >&2
    exit 2
  fi
done

task_meta() {
  case "$1" in
    TwoRoom) printf '%s|%s\n' "tworoom" "lewm-tworooms" ;;
    PushT) printf '%s|%s\n' "pusht" "lewm-pusht" ;;
    Reacher) printf '%s|%s\n' "reacher" "lewm-reacher" ;;
    Cube) printf '%s|%s\n' "cube" "lewm-cube" ;;
    *)
      echo "unsupported task: $1" >&2
      return 2
      ;;
  esac
}

severity_slug() {
  case "$1" in
    gaussian_blur) printf 'gaussian_blur_ks%s\n' "$2" ;;
    resize) printf 'resize_factor%s\n' "${2//./p}" ;;
    *)
      echo "unsupported family: $1" >&2
      return 2
      ;;
  esac
}

valid_reference() {
  local path="$1"
  local raw_atr="$2"
  local seed="$3"
  local task="$4"
  local family="$5"
  local magnitude="$6"
  [[ -s "$path" && -s "$raw_atr" ]] || return 1
  local raw_sha
  raw_sha="$(sha256sum "$raw_atr" | awk '{print $1}')"
  jq -e --arg task "$task" --arg family "$family" --arg raw_sha "$raw_sha"     --argjson seed "$seed" --argjson magnitude "$magnitude" '
    .metadata.schema_version == "paper1-acpc-horizon-v2-1.0" and
    .metadata.artifact_role == "paired_multiseverity_atr_reference" and
    .metadata.status == "complete" and
    .metadata.training_seed == $seed and .metadata.task == $task and
    .metadata.corruption_type == $family and
    .metadata.stressor_severity == $magnitude and
    .metadata.source_sha256 == $raw_sha and
    (.rows | length) == 2 and
    (.rows | all(
      .status == "ok" and .training_seed == $seed and .task == $task and
      .corruption_type == $family and .stressor_severity == $magnitude
    ))' "$path" >/dev/null
}

valid_smpr() {
  local path="$1"
  local reference_atr="$2"
  local seed="$3"
  local task="$4"
  local family="$5"
  local magnitude="$6"
  [[ -s "$path" && -s "$reference_atr" ]] || return 1
  local reference_sha
  reference_sha="$(sha256sum "$reference_atr" | awk '{print $1}')"
  jq -e --arg task "$task" --arg family "$family"     --arg reference_path "$reference_atr" --arg reference_sha "$reference_sha"     --argjson seed "$seed" --argjson magnitude "$magnitude" '
    .metadata.schema_version == "paper1-smpr-v2-1.0" and
    .metadata.status == "complete" and .metadata.status_counts == {"ok": 2} and
    .metadata.missing_rows == [] and .metadata.errors == [] and
    .metadata.source_paths.reference_atr == $reference_path and
    .metadata.source_hashes.reference_atr == $reference_sha and
    (.rows | length) == 2 and
    (.rows | all(
      .status == "ok" and .training_seed == $seed and .task == $task and
      .corruption_type == $family and .noise_std == $magnitude and
      .atr_reference_match == true and .atr_reference_abs_error <= 0.000001
    ))' "$path" >/dev/null
}

run_shard() {
  local training_seed="$1"
  local task="$2"
  local family="$3"
  local magnitude="$4"
  local meta task_slug model_root severity_dir
  meta="$(task_meta "$task")"
  IFS='|' read -r task_slug model_root <<< "$meta"
  severity_dir="$(severity_slug "$family" "$magnitude")"
  local raw_dir="${raw_root}/lewm_seed${training_seed}/${severity_dir}"
  local raw_atr="${raw_dir}/acpc_${task_slug}_v2.json"
  local reference_dir="${reference_root}/lewm_seed${training_seed}/${severity_dir}"
  local reference_atr="${reference_dir}/acpc_${task_slug}_horizon_v2_checkpoint_bound.json"
  local out_path="${raw_dir}/smpr_${task_slug}_v2.json"
  [[ -s "$raw_atr" ]] || {
    echo "missing ATR raw shard: $raw_atr" >&2
    return 1
  }
  mkdir -p "$raw_dir" "$reference_dir"

  if ! valid_reference "$reference_atr" "$raw_atr" "$training_seed" "$task" "$family" "$magnitude"; then
    if [[ -e "$reference_atr" ]]; then
      mv -- "$reference_atr" "${reference_atr}.pre_v2_or_invalid_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    python -m paper1.scripts.build_paired_multiseverity_atr_reference       --raw "$raw_atr"       --training-seed "$training_seed"       --protocol "$protocol"       --out "$reference_atr"
  fi
  if valid_smpr "$out_path" "$reference_atr" "$training_seed" "$task" "$family" "$magnitude"; then
    echo "[multiseverity-smpr-v2] skip seed=$training_seed task=$task family=$family magnitude=$magnitude"
    return 0
  fi
  if [[ -e "$out_path" ]]; then
    mv -- "$out_path" "${out_path}.pre_v2_or_invalid_$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  echo "[multiseverity-smpr-v2] start seed=$training_seed task=$task family=$family magnitude=$magnitude gpu=$gpu"
  env     CUDA_VISIBLE_DEVICES="$gpu"     HOME="$diagnostic_home"     OMP_NUM_THREADS="$native_threads"     MKL_NUM_THREADS="$native_threads"     OPENBLAS_NUM_THREADS="$native_threads"     NUMEXPR_NUM_THREADS="$native_threads"     PYTHONUNBUFFERED=1     PYTHONPATH=.     timeout --signal=TERM --kill-after=60s "${timeout_seconds}s"     python paper1/scripts/smpr_sensitivity.py       --method LeWM       --family-id "lewm_seed${training_seed}"       --training-seed "$training_seed"       --evals "assets/paper1_data/training_seed_eval_manifests/lewm_seed${training_seed}_evals.json"       --reference-atr "$reference_atr"       --model-root "${model_base}/${model_root}"       --tasks "$task"       --std-keys 0.0 0.08       --n-sequences 100       --num-noise-draws 5       --rollout-horizon 8       --radius-quantile 0.90       --local-quantile 0.35       --margin-delta-norm 0.10       --noise-std "$magnitude"       --corruption-type "$family"       --anchor-seed 9101       --embedding-space normalized       --evaluation-seeds 42 43 44       --device cuda       --out "$out_path"
  valid_smpr "$out_path" "$reference_atr" "$training_seed" "$task" "$family" "$magnitude"
  echo "[multiseverity-smpr-v2] done  seed=$training_seed task=$task family=$family magnitude=$magnitude"
}

mapfile -t blur_severities < <(
  jq -r '.stressors.gaussian_blur.prospective_nonidentity[] | tostring' "$protocol"
)
mapfile -t resize_severities < <(
  jq -r '.stressors.resize.prospective_nonidentity[] | tostring' "$protocol"
)

if [[ "$mode" == "smoke" ]]; then
  run_shard 3072 TwoRoom gaussian_blur "${blur_severities[0]}"
else
  for training_seed in 3072 3073 3074; do
    for task in TwoRoom PushT Reacher Cube; do
      for magnitude in "${blur_severities[@]}"; do
        run_shard "$training_seed" "$task" gaussian_blur "$magnitude"
      done
      for magnitude in "${resize_severities[@]}"; do
        run_shard "$training_seed" "$task" resize "$magnitude"
      done
    done
  done
fi
