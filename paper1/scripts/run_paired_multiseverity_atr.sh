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
raw_root="paper1/results/multiseverity_v1/raw"
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

for value_name in native_threads timeout_seconds; do
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
  local family="$1"
  local magnitude="$2"
  case "$family" in
    gaussian_blur) printf 'gaussian_blur_ks%s\n' "$magnitude" ;;
    resize) printf 'resize_factor%s\n' "${magnitude//./p}" ;;
    *)
      echo "unsupported family: $family" >&2
      return 2
      ;;
  esac
}

valid_shard() {
  local path="$1"
  [[ -s "$path" ]] &&
    jq -e '.metadata.status_counts == {"ok": 2} and
      .metadata.missing_rows == [] and .metadata.errors == [] and
      (.rows | length) == 2' "$path" >/dev/null
}

run_shard() {
  local training_seed="$1"
  local task="$2"
  local family="$3"
  local magnitude="$4"
  local meta task_slug model_root
  meta="$(task_meta "$task")"
  IFS='|' read -r task_slug model_root <<< "$meta"
  local severity_dir
  severity_dir="$(severity_slug "$family" "$magnitude")"
  local out_dir="${raw_root}/lewm_seed${training_seed}/${severity_dir}"
  local out_path="${out_dir}/acpc_${task_slug}_v2.json"
  mkdir -p "$out_dir"

  if valid_shard "$out_path"; then
    echo "[multiseverity-atr] skip seed=${training_seed} task=${task} family=${family} magnitude=${magnitude}"
    return 0
  fi
  if [[ -e "$out_path" ]]; then
    mv -- "$out_path" "${out_path}.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  echo "[multiseverity-atr] start seed=${training_seed} task=${task} family=${family} magnitude=${magnitude} gpu=${gpu}"
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
    python tools/paper1_phase0_acpc.py \
      --methods LeWM \
      --tasks "$task" \
      --std-keys 0.0 0.08 \
      --evals-lewm "assets/paper1_data/training_seed_eval_manifests/lewm_seed${training_seed}_evals.json" \
      --model-root "${model_base}/${model_root}" \
      --out "$out_path" \
      --n-sequences 100 \
      --num-noise-draws 5 \
      --rollout-horizon 8 \
      --noise-std "$magnitude" \
      --corruption-type "$family" \
      --clean-goal \
      --embedding-space normalized \
      --seed 9101 \
      --device cuda
  valid_shard "$out_path"
  echo "[multiseverity-atr] done  seed=${training_seed} task=${task} family=${family} magnitude=${magnitude}"
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
