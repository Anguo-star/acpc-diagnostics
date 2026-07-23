#!/usr/bin/env bash
set -euo pipefail

model_base="${PAPER1_MODEL_BASE:-/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll}"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"
output_root="${PAPER1_PLDM_OUTPUT_ROOT:-paper1/results/pldm_multiseed_v2}"
protocol="paper1/config/frozen_diagnostic_protocol_v1.json"
gpus_raw="${PAPER1_DIAGNOSTIC_GPUS:-0 1 2 3 4 5 6 7}"
training_seeds_raw="${PAPER1_TRAINING_SEEDS:-3073 3074}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"

tasks=(TwoRoom PushT Reacher Cube)
slugs=(tworoom pusht reacher cube)
roots=(lewm-tworooms lewm-pusht lewm-reacher lewm-cube)
read -ra gpus <<< "$gpus_raw"
read -ra training_seeds <<< "$training_seeds_raw"

if (( ${#gpus[@]} == 0 || ${#training_seeds[@]} == 0 )); then
  echo "PAPER1_DIAGNOSTIC_GPUS and PAPER1_TRAINING_SEEDS must be non-empty" >&2
  exit 2
fi

valid_atr_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '.metadata.status_counts == {"ok": 9} and
      .metadata.missing_rows == [] and .metadata.errors == [] and
      (.rows | length) == 9' "$path" >/dev/null
}

valid_smpr_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '.metadata.status_counts == {"ok": 9} and
      .metadata.missing_rows == [] and .metadata.errors == [] and
      (.rows | length) == 9' "$path" >/dev/null
}

run_atr_shard() {
  local training_seed="$1"
  local task_index="$2"
  local gpu="$3"
  local seed_dir="${output_root}/seed${training_seed}"
  local out="${seed_dir}/raw/acpc_${slugs[$task_index]}_v2.json"
  mkdir -p "${seed_dir}/raw"
  if valid_atr_shard "$out"; then
    echo "[pldm-atr] skip seed=${training_seed} task=${tasks[$task_index]}"
    return
  fi
  if [[ -e "$out" ]]; then
    mv -- "$out" "${out}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  echo "[pldm-atr] start seed=${training_seed} task=${tasks[$task_index]} gpu=${gpu}"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HOME="$diagnostic_home" \
    MPLCONFIGDIR="${diagnostic_home}/.cache/matplotlib" \
    OMP_NUM_THREADS="$native_threads" \
    MKL_NUM_THREADS="$native_threads" \
    OPENBLAS_NUM_THREADS="$native_threads" \
    NUMEXPR_NUM_THREADS="$native_threads" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    python tools/paper1_phase0_acpc.py \
      --methods PLDM \
      --tasks "${tasks[$task_index]}" \
      --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
      --evals-pldm "assets/paper1_data/training_seed_eval_manifests/pldm_seed${training_seed}_evals.json" \
      --model-root "${model_base}/${roots[$task_index]}" \
      --out "$out" \
      --n-sequences 100 \
      --num-noise-draws 5 \
      --rollout-horizon 8 \
      --noise-std 0.08 \
      --corruption-type gaussian_noise \
      --clean-goal \
      --embedding-space normalized \
      --seed 9101 \
      --device cuda
  valid_atr_shard "$out"
}

run_smpr_shard() {
  local training_seed="$1"
  local task_index="$2"
  local gpu="$3"
  local seed_dir="${output_root}/seed${training_seed}"
  local out="${seed_dir}/raw/smpr_${slugs[$task_index]}_v2.json"
  local reference="${seed_dir}/acpc_horizon_v2_checkpoint_bound.json"
  if valid_smpr_shard "$out"; then
    echo "[pldm-smpr] skip seed=${training_seed} task=${tasks[$task_index]}"
    return
  fi
  if [[ -e "$out" ]]; then
    mv -- "$out" "${out}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  echo "[pldm-smpr] start seed=${training_seed} task=${tasks[$task_index]} gpu=${gpu}"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HOME="$diagnostic_home" \
    MPLCONFIGDIR="${diagnostic_home}/.cache/matplotlib" \
    OMP_NUM_THREADS="$native_threads" \
    MKL_NUM_THREADS="$native_threads" \
    OPENBLAS_NUM_THREADS="$native_threads" \
    NUMEXPR_NUM_THREADS="$native_threads" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    python paper1/scripts/smpr_sensitivity.py \
      --method PLDM \
      --family-id "pldm_canonical_seed${training_seed}" \
      --training-seed "$training_seed" \
      --evals "assets/paper1_data/training_seed_eval_manifests/pldm_seed${training_seed}_evals.json" \
      --reference-atr "$reference" \
      --model-root "${model_base}/${roots[$task_index]}" \
      --tasks "${tasks[$task_index]}" \
      --std-keys 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 \
      --n-sequences 100 \
      --num-noise-draws 5 \
      --rollout-horizon 8 \
      --radius-quantile 0.90 \
      --local-quantile 0.35 \
      --margin-delta-norm 0.10 \
      --noise-std 0.08 \
      --corruption-type gaussian_noise \
      --anchor-seed 9101 \
      --embedding-space normalized \
      --device cuda \
      --out "$out"
  valid_smpr_shard "$out"
}

run_parallel_stage() {
  local stage="$1"
  local launch_index=0
  local pids=()
  for training_seed in "${training_seeds[@]}"; do
    for task_index in "${!tasks[@]}"; do
      local gpu="${gpus[$((launch_index % ${#gpus[@]}))]}"
      if [[ "$stage" == "atr" ]]; then
        run_atr_shard "$training_seed" "$task_index" "$gpu" &
      else
        run_smpr_shard "$training_seed" "$task_index" "$gpu" &
      fi
      pids+=("$!")
      launch_index=$((launch_index + 1))
    done
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

run_parallel_stage atr

for training_seed in "${training_seeds[@]}"; do
  seed_dir="${output_root}/seed${training_seed}"
  merged_atr="${seed_dir}/acpc_horizon_v2_checkpoint_bound.json"
  if [[ ! -s "$merged_atr" ]]; then
    python -m paper1.scripts.build_external_pldm_acpc_horizon_v2_artifact \
      --input "${seed_dir}/raw/acpc_tworoom_v2.json" \
      --input "${seed_dir}/raw/acpc_pusht_v2.json" \
      --input "${seed_dir}/raw/acpc_reacher_v2.json" \
      --input "${seed_dir}/raw/acpc_cube_v2.json" \
      --manifest "assets/paper1_data/training_seed_eval_manifests/pldm_seed${training_seed}_evals.json" \
      --protocol "$protocol" \
      --out "$merged_atr"
  fi
done

run_parallel_stage smpr

for training_seed in "${training_seeds[@]}"; do
  seed_dir="${output_root}/seed${training_seed}"
  merged_atr="${seed_dir}/acpc_horizon_v2_checkpoint_bound.json"
  merged_smpr="${seed_dir}/smpr_v2_checkpoint_bound.json"
  diagnostic_input="${seed_dir}/diagnostic_input.json"
  blind_predictions="${seed_dir}/frozen_predictions_blind.csv"
  if [[ ! -s "$merged_smpr" ]]; then
    python -m paper1.scripts.build_external_pldm_smpr_v2_artifact \
      --input "${seed_dir}/raw/smpr_tworoom_v2.json" \
      --input "${seed_dir}/raw/smpr_pusht_v2.json" \
      --input "${seed_dir}/raw/smpr_reacher_v2.json" \
      --input "${seed_dir}/raw/smpr_cube_v2.json" \
      --reference-atr "$merged_atr" \
      --protocol "$protocol" \
      --out "$merged_smpr"
  fi
  if [[ ! -s "$diagnostic_input" ]]; then
    python -m paper1.scripts.build_pldm_frozen_diagnostic_input \
      --protocol "$protocol" \
      --atr "$merged_atr" \
      --smpr "$merged_smpr" \
      --out "$diagnostic_input"
  fi
  if [[ ! -s "$blind_predictions" ]]; then
    python -m paper1.scripts.frozen_external_validation apply \
      --protocol "$protocol" \
      --diagnostics "$diagnostic_input" \
      --out "$blind_predictions"
  fi
done

echo "PLDM multi-seed Gaussian diagnostics complete: ${training_seeds[*]}"
