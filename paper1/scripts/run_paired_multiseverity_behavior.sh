#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
case "$mode" in
  plan|smoke|full) ;;
  *)
    echo "usage: $0 [plan|smoke|full]" >&2
    exit 2
    ;;
esac

root="${PAPER1_DATA_ROOT:-/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll}"
protocol="paper1/config/paired_multiseverity_protocol_v1.json"
protocol_hash_file="paper1/config/paired_multiseverity_protocol_v1.sha256"
manifest_root="paper1/results/multiseverity_v1/manifests"
gpu="${PAPER1_EVAL_GPU:-0}"
eval_timeout="${PAPER1_EVAL_TIMEOUT_SECONDS:-7200}"
outer_timeout="${PAPER1_OUTER_JOB_TIMEOUT_SECONDS:-7800}"
native_threads="${PAPER1_EVAL_THREADS:-2}"

for value_name in eval_timeout outer_timeout native_threads; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$value_name must be a positive integer; got '$value'" >&2
    exit 2
  fi
done
[[ -f "$protocol" ]] || {
  echo "missing frozen prospective protocol: $protocol" >&2
  exit 1
}
[[ -f "$protocol_hash_file" ]] || {
  echo "missing frozen protocol hash: $protocol_hash_file" >&2
  exit 1
}
read -r expected_protocol_sha _ < "$protocol_hash_file"
actual_protocol_sha="$(sha256sum "$protocol" | awk '{print $1}')"
if ! [[ "$expected_protocol_sha" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid frozen protocol hash: $expected_protocol_sha" >&2
  exit 1
fi
if [[ "$actual_protocol_sha" != "$expected_protocol_sha" ]]; then
  echo "frozen protocol hash mismatch: expected=$expected_protocol_sha actual=$actual_protocol_sha" >&2
  exit 1
fi
mkdir -p "$manifest_root"

export OMP_NUM_THREADS="$native_threads"
export MKL_NUM_THREADS="$native_threads"
export OPENBLAS_NUM_THREADS="$native_threads"
export NUMEXPR_NUM_THREADS="$native_threads"
export PYTHONUNBUFFERED=1

magnitudes_for() {
  local family="$1"
  case "$family" in
    gaussian_blur)
      jq -r '.stressors.gaussian_blur.eval_magnitudes_with_identity | map(tostring) | join(",")' "$protocol"
      ;;
    resize)
      jq -r '.stressors.resize.eval_magnitudes_with_identity | map(tostring) | join(",")' "$protocol"
      ;;
    *)
      echo "unsupported family: $family" >&2
      return 2
      ;;
  esac
}

run_job() {
  local training_seed="$1"
  local task="$2"
  local std_key="$3"
  local family="$4"
  local magnitudes
  magnitudes="$(magnitudes_for "$family")"
  local std_slug="${std_key//./p}"
  local task_slug
  task_slug="$(printf '%s' "$task" | tr '[:upper:]' '[:lower:]')"
  local manifest="${manifest_root}/behavior_s${training_seed}_${task_slug}_std${std_slug}_${family}.json"
  local output_prefix="paper1_unseen_origin_vs_std008_s${training_seed}_strongest"
  local canonical="assets/paper1_data/training_seed_eval_manifests/lewm_seed${training_seed}_evals.json"

  args=(
    python -m tools.paper1_unseen_eval_grid
    --root "$root"
    --canonical "$canonical"
    --manifest-out "$manifest"
    --tasks "$task"
    --std-keys "$std_key"
    --families "$family"
    --family-magnitudes "${family}=${magnitudes}"
    --train-seed "$training_seed"
    --output-prefix "$output_prefix"
    --epoch 10
    --num-eval 300
    --eval-seeds 3
    --eval-base-seed 42
    --eval-gpus "$gpu"
    --apply-to 1
    --trainer-file train.py
    --config lewm
    --post-train-eval-mode full
    --extra-env "eval_max_concurrency=1"
    --extra-env "eval_resume=1"
    --extra-env "eval_save_video=0"
    --extra-env "eval_timeout_seconds=${eval_timeout}"
    --only-missing
  )

  if [[ "$mode" == "plan" ]]; then
    echo "[multiseverity-behavior] plan seed=${training_seed} task=${task} std=${std_key} family=${family} gpu=${gpu}"
    "${args[@]}" --dry-run
    return 0
  fi
  echo "[multiseverity-behavior] start seed=${training_seed} task=${task} std=${std_key} family=${family} gpu=${gpu}"
  timeout --signal=TERM --kill-after=60s "${outer_timeout}s" "${args[@]}"
  "${args[@]}" --dry-run
  jq -e '(.jobs | length) == 1 and (.jobs | all(.complete == true))' "$manifest" >/dev/null
  echo "[multiseverity-behavior] done  seed=${training_seed} task=${task} std=${std_key} family=${family}"
}

if [[ "$mode" == "smoke" || "$mode" == "plan" ]]; then
  run_job 3072 TwoRoom 0.0 gaussian_blur
  run_job 3072 TwoRoom 0.08 gaussian_blur
else
  for training_seed in 3072 3073 3074; do
    for task in TwoRoom PushT Reacher Cube; do
      for std_key in 0.0 0.08; do
        for family in gaussian_blur resize; do
          run_job "$training_seed" "$task" "$std_key" "$family"
        done
      done
    done
  done
fi

