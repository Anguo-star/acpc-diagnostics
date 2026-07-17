#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "atr" && "$1" != "smpr" ]]; then
  echo "usage: $0 <atr|smpr>" >&2
  exit 2
fi

stage="$1"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
model_base="/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll"
source_base="paper1/results/remediation_phase2_external_sources/target_view"
manifest="assets/paper1_data/target_view_diagnostic_manifest_v1.json"
runner_manifest_dir="assets/paper1_data/target_view_runner_manifests"
branches_raw="${PAPER1_TARGET_VIEW_BRANCHES:-full_sequence target_view}"
timeout_seconds="${PAPER1_DIAGNOSTIC_TIMEOUT_SECONDS:-7200}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"
gpu="${PAPER1_DIAGNOSTIC_GPU:-0}"

tasks=(TwoRoom PushT Reacher Cube)
slugs=(tworoom pusht reacher cube)
roots=(lewm-tworooms lewm-pusht lewm-reacher lewm-cube)
rhos=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
read -ra branches <<< "$branches_raw"

for value_name in timeout_seconds native_threads; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
done
if (( ${#branches[@]} == 0 )); then
  echo "PAPER1_TARGET_VIEW_BRANCHES must be non-empty" >&2
  exit 2
fi

valid_shard() {
  local path="$1"
  local schema="$2"
  [[ -s "$path" ]] &&
    jq -e --arg schema "$schema" \
      '.metadata.schema_version == $schema and
       .metadata.status_counts == {"ok": 8} and
       .metadata.missing_rows == [] and .metadata.errors == [] and
       (.rows | length) == 8' "$path" >/dev/null
}

run_with_limits() {
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
    "$@"
}

for branch in "${branches[@]}"; do
  if [[ "$branch" != "full_sequence" && "$branch" != "target_view" ]]; then
    echo "invalid target-view branch: $branch" >&2
    exit 2
  fi
  evals="${runner_manifest_dir}/target_view_${branch}_evals_v1.json"
  out_dir="${source_base}/${branch}"
  mkdir -p "$out_dir"
  inputs=()
  schema="paper1-acpc-phase0-0.2"
  if [[ "$stage" == "smpr" ]]; then
    schema="paper1-smpr-v2-1.0"
  fi

  for index in "${!tasks[@]}"; do
    out_path="${out_dir}/${stage}_${slugs[$index]}_v2.json"
    inputs+=(--input "$out_path")
    if valid_shard "$out_path" "$schema"; then
      echo "[target-view-${stage}] skip branch=${branch} task=${tasks[$index]}"
      continue
    fi
    if [[ -e "$out_path" ]]; then
      mv -- "$out_path" "${out_path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    echo "[target-view-${stage}] start branch=${branch} task=${tasks[$index]} gpu=${gpu}"
    if [[ "$stage" == "atr" ]]; then
      run_with_limits \
        python tools/paper1_phase0_acpc.py \
          --methods LeWM \
          --tasks "${tasks[$index]}" \
          --std-keys "${rhos[@]}" \
          --evals-lewm "$evals" \
          --model-root "${model_base}/${roots[$index]}" \
          --out "$out_path" \
          --n-sequences 100 \
          --num-noise-draws 5 \
          --rollout-horizon 8 \
          --noise-std 0.08 \
          --corruption-type gaussian_noise \
          --clean-goal \
          --embedding-space normalized \
          --seed 9101 \
          --device cuda
    else
      reference="${out_dir}/acpc_horizon_v2_checkpoint_bound.json"
      if [[ ! -s "$reference" ]]; then
        echo "missing merged ATR reference: $reference" >&2
        exit 2
      fi
      run_with_limits \
        python paper1/scripts/smpr_sensitivity.py \
          --method LeWM \
          --family-id "lewm_seed3072_${branch}" \
          --training-seed 3072 \
          --evals "$evals" \
          --reference-atr "$reference" \
          --model-root "${model_base}/${roots[$index]}" \
          --tasks "${tasks[$index]}" \
          --std-keys "${rhos[@]}" \
          --out "$out_path" \
          --n-sequences 100 \
          --num-noise-draws 5 \
          --rollout-horizon 8 \
          --noise-std 0.08 \
          --corruption-type gaussian_noise \
          --anchor-seed 9101 \
          --embedding-space normalized \
          --device cuda
    fi
    if ! valid_shard "$out_path" "$schema"; then
      echo "invalid target-view ${stage} shard: $out_path" >&2
      exit 1
    fi
  done

  if [[ "$stage" == "atr" ]]; then
    run_with_limits \
      python paper1/scripts/build_target_view_diagnostic_artifact.py \
        --kind atr \
        --branch "$branch" \
        --manifest "$manifest" \
        "${inputs[@]}" \
        --out "${out_dir}/acpc_horizon_v2_checkpoint_bound.json"
  else
    run_with_limits \
      python paper1/scripts/build_target_view_diagnostic_artifact.py \
        --kind smpr \
        --branch "$branch" \
        --manifest "$manifest" \
        --reference-atr "${out_dir}/acpc_horizon_v2_checkpoint_bound.json" \
        "${inputs[@]}" \
        --out "${out_dir}/smpr_v2_checkpoint_bound.json"
  fi
done
