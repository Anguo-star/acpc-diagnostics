#!/usr/bin/env bash
set -euo pipefail

source_dir="paper1/results/remediation_phase3_baseline_sources"
model_base="${PAPER1_DATA_ROOT:-${DATA_ROOT:-${STABLEWM_HOME:-}}}"
: "${model_base:?Set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME to the dataset root}"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
timeout_seconds="${PAPER1_BASELINE_TIMEOUT_SECONDS:-7200}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"
gpu="${PAPER1_DIAGNOSTIC_GPU:-0}"

tasks=(TwoRoom PushT Reacher Cube)
slugs=(tworoom pusht reacher cube)
roots=(lewm-tworooms lewm-pusht lewm-reacher lewm-cube)
gaussian_rhos=(0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
endpoint_rhos=(0.0 0.08)
e4_rhos=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
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
  local expected="$2"
  [[ -s "$path" ]] &&
    jq -e --argjson expected "$expected" \
      '.metadata.schema_version == "paper1-diagnostic-baseline-raw-1.0" and
       .metadata.status == "complete" and
       .metadata.status_counts == {"ok": $expected} and
       .metadata.behavior_blind == true and
       .metadata.threshold_search_available == false and
       .metadata.missing_rows == [] and .metadata.errors == [] and
       (.rows | length) == $expected and
       all(.rows[];
         .status == "ok" and .reference_atr_match == true)' \
      "$path" >/dev/null
}

run_shard() {
  local method="$1"
  local family_id="$2"
  local training_seed="$3"
  local seed_semantics="$4"
  local split_name="$5"
  local stressor_family="$6"
  local severity="$7"
  local corruption_type="$8"
  local evals="$9"
  local reference_atr="${10}"
  local task_index="${11}"
  local context="${12}"
  local branch="${13}"
  shift 13
  local rhos=("$@")
  local expected="${#rhos[@]}"
  local out="${source_dir}/baseline_${context}_${slugs[$task_index]}.json"
  if valid_shard "$out" "$expected"; then
    echo "[baseline] skip context=${context} task=${tasks[$task_index]}"
    return
  fi
  if [[ -e "$out" ]]; then
    mv -- "$out" "${out}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  echo "[baseline] start context=${context} task=${tasks[$task_index]} gpu=${gpu}"
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
    python paper1/scripts/diagnostic_baseline_benchmark.py \
      --method "$method" \
      --family-id "$family_id" \
      --training-seed "$training_seed" \
      --training-seed-semantics "$seed_semantics" \
      --split-name "$split_name" \
      --stressor-family "$stressor_family" \
      --stressor-severity "$severity" \
      --branch "$branch" \
      --evals "$evals" \
      --reference-atr "$reference_atr" \
      --model-root "${model_base}/${roots[$task_index]}" \
      --tasks "${tasks[$task_index]}" \
      --std-keys "${rhos[@]}" \
      --n-sequences 100 \
      --num-noise-draws 5 \
      --rollout-horizon 8 \
      --noise-std "$severity" \
      --corruption-type "$corruption_type" \
      --anchor-seed 9101 \
      --embedding-space normalized \
      --device cuda \
      --out "$out"
  if ! valid_shard "$out" "$expected"; then
    echo "invalid baseline shard: $out" >&2
    exit 1
  fi
}

# Gaussian CAL/E1 LeWM.
for seed in 3072 3073 3074; do
  manifest="assets/paper1_data/training_seed_eval_manifests/lewm_seed${seed}_evals.json"
  if [[ "$seed" == "3072" ]]; then
    split="CAL"
    reference="assets/paper1_data/acpc_horizon_v2_lewm.json"
  else
    split="E1"
    reference="paper1/results/remediation_phase2_external_sources/lewm_seed${seed}/acpc_horizon_v2_checkpoint_bound.json"
  fi
  for task_index in "${!tasks[@]}"; do
    run_shard \
      LeWM "lewm_seed${seed}" "$seed" \
      "independently trained LeWM checkpoint seed" \
      "$split" gaussian_noise 0.08 gaussian_noise \
      "$manifest" "$reference" "$task_index" \
      "gaussian_lewm_seed${seed}" "" "${gaussian_rhos[@]}"
  done
done

# Gaussian E2 canonical PLDM.
for task_index in "${!tasks[@]}"; do
  run_shard \
    PLDM pldm_canonical_seed3072 3072 \
    "one independently trained PLDM checkpoint family" \
    E2 gaussian_noise 0.08 gaussian_noise \
    assets/paper1_data/canonical_evals_pldm_v2.json \
    paper1/results/remediation_phase2_external_sources/pldm_canonical/acpc_horizon_v2_checkpoint_bound.json \
    "$task_index" gaussian_pldm_canonical "" "${gaussian_rhos[@]}"
done

# E3 strongest blur/resize, all base/endpoint pairs.
for seed in 3072 3073 3074; do
  manifest="assets/paper1_data/training_seed_eval_manifests/lewm_seed${seed}_evals.json"
  for stressor in blur resize; do
    if [[ "$stressor" == "blur" ]]; then
      artifact_dir="gaussian_blur"
      severity=15
      corruption=gaussian_blur
    else
      artifact_dir=resize
      severity=0.25
      corruption=resize
    fi
    reference="paper1/results/remediation_phase2_external_sources/cross_stressor/lewm_seed${seed}/${artifact_dir}/acpc_horizon_v2_checkpoint_bound.json"
    for task_index in "${!tasks[@]}"; do
      run_shard \
        LeWM "lewm_seed${seed}" "$seed" \
        "independently trained LeWM checkpoint seed" \
        E3-L "$stressor" "$severity" "$corruption" \
        "$manifest" "$reference" "$task_index" \
        "cross_${stressor}_lewm_seed${seed}" "" "${endpoint_rhos[@]}"
    done
  done
done

for stressor in blur resize; do
  if [[ "$stressor" == "blur" ]]; then
    artifact_dir="gaussian_blur"
    severity=15
    corruption=gaussian_blur
  else
    artifact_dir=resize
    severity=0.25
    corruption=resize
  fi
  reference="paper1/results/remediation_phase2_external_sources/cross_stressor/pldm_canonical/${artifact_dir}/acpc_horizon_v2_checkpoint_bound.json"
  for task_index in "${!tasks[@]}"; do
    run_shard \
      PLDM pldm_canonical_seed3072 3072 \
      "one independently trained PLDM checkpoint family" \
      E3-P "$stressor" "$severity" "$corruption" \
      assets/paper1_data/canonical_evals_pldm_v2.json \
      "$reference" "$task_index" \
      "cross_${stressor}_pldm_canonical" "" "${endpoint_rhos[@]}"
  done
done

# E4 full-sequence/target-view matched checkpoints.
for branch in full_sequence target_view; do
  evals="assets/paper1_data/target_view_runner_manifests/target_view_${branch}_evals_v1.json"
  reference="paper1/results/remediation_phase2_external_sources/target_view/${branch}/acpc_horizon_v2_checkpoint_bound.json"
  for task_index in "${!tasks[@]}"; do
    run_shard \
      LeWM "lewm_seed3072_${branch}" 3072 \
      "one existing LeWM training run per task/std/branch" \
      E4 gaussian_noise 0.08 gaussian_noise \
      "$evals" "$reference" "$task_index" \
      "target_view_${branch}" "$branch" "${e4_rhos[@]}"
  done
done
