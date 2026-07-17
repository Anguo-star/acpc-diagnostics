#!/usr/bin/env bash
set -euo pipefail

gpu="${PAPER1_DIAGNOSTIC_GPU:-0}"
native_threads="${PAPER1_DIAGNOSTIC_THREADS:-2}"
timeout_seconds="${PAPER1_SMPR_CONTROL_TIMEOUT_SECONDS:-7200}"
diagnostic_home="${PAPER1_DIAGNOSTIC_HOME:-/tmp/paper1_swm_home}"

for value_name in native_threads timeout_seconds; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${value_name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
done

valid_outputs() {
  [[ -s assets/paper1_data/smpr_sensitivity_v2.json ]] &&
  [[ -s assets/paper1_data/smpr_controls_v2.json ]] &&
  [[ -s assets/paper1_data/smpr_oracle_guard_v2.json ]] &&
  jq -e '.metadata.schema_version == "paper1-smpr-sensitivity-v2-1.0" and .metadata.status == "complete" and .count_contract.full_calibration_grid_rows == 864 and .count_contract.pairing_mve_rows == 48' assets/paper1_data/smpr_sensitivity_v2.json >/dev/null &&
  jq -e '.metadata.schema_version == "paper1-smpr-controls-v2-1.0" and .metadata.status == "complete" and .correctness_gates.constant_collapse_rejected_all_mve_rows == true and .claim_decisions.task_grounded_increment_established == false' assets/paper1_data/smpr_controls_v2.json >/dev/null &&
  jq -e '.metadata.schema_version == "paper1-smpr-oracle-guard-v2-1.0" and .metadata.status == "complete" and .count_contract.observed_rows == 4' assets/paper1_data/smpr_oracle_guard_v2.json >/dev/null
}

if valid_outputs; then
  echo "[smpr-controls] skip complete MVE"
  exit 0
fi

echo "[smpr-controls] start TwoRoom+PushT base+endpoint gpu=${gpu}"
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
  python paper1/scripts/smpr_controls.py

valid_outputs
