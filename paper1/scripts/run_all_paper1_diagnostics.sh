#!/usr/bin/env bash
# Rebuild/verify Paper1 diagnostic artifacts without concurrent checkpoint jobs.
set -euo pipefail

export PAPER1_DIAGNOSTIC_THREADS="${PAPER1_DIAGNOSTIC_THREADS:-2}"
export PAPER1_DIAGNOSTIC_GPU="${PAPER1_DIAGNOSTIC_GPU:-0}"

if ! [[ "$PAPER1_DIAGNOSTIC_THREADS" =~ ^[0-9]+$ ]] || (( PAPER1_DIAGNOSTIC_THREADS < 1 )); then
  echo "PAPER1_DIAGNOSTIC_THREADS must be a positive integer" >&2
  exit 2
fi

# The default path is CPU-only and safe for release checks.  Historical
# aggregate regeneration is opt-in because it can overwrite archived figures.
python -m paper1.scripts.build_diagnostic_manifest

if [[ "${REBUILD_LEGACY_AGGREGATES:-0}" == "1" ]]; then
  python -m paper1.scripts.build_full_sweep_diagnostics
  python -m paper1.scripts.plot_full_sweep_diagnostics
  python -m paper1.scripts.plot_endpoint_atr_smpr
  python -m paper1.scripts.fixed_pool_tail_audit
  python -m paper1.scripts.heldout_diagnostic_validation
  python -m paper1.scripts.threshold_quantile_sensitivity
  python -m paper1.scripts.plot_fixed_pool_event_rates
  python -m paper1.scripts.plot_gaussian_sensitivity_mechanism
fi

# Compatibility: the old flag now selects the bounded serial remediation path;
# it no longer launches the retired monolithic checkpoint commands.
run_checkpoint_audits="${RUN_REMEDIATION_AUDITS:-${RUN_CHECKPOINT_AUDITS:-0}}"
if [[ "$run_checkpoint_audits" == "1" ]]; then
  echo "[paper1] checkpoint audits run serially on GPU ${PAPER1_DIAGNOSTIC_GPU} with ${PAPER1_DIAGNOSTIC_THREADS} native threads"
  bash paper1/scripts/run_jvp_hutchinson_shards.sh
  bash paper1/scripts/run_diagnostic_baseline_shards.sh
  bash paper1/scripts/run_fixed_pool_certificate_shards.sh
  bash paper1/scripts/run_smpr_controls_mve.sh
  bash paper1/scripts/run_linearization_horizon_shards.sh
fi

if [[ "${RUN_EXTERNAL_AUDITS:-0}" == "1" ]]; then
  echo "[paper1] external checkpoint audits run serially on GPU ${PAPER1_DIAGNOSTIC_GPU} with ${PAPER1_DIAGNOSTIC_THREADS} native threads"
  bash paper1/scripts/run_cross_stressor_lewm_atr_shards.sh
  bash paper1/scripts/run_cross_stressor_smpr_shards.sh
  bash paper1/scripts/run_pldm_cross_stressor_atr_shards.sh
  bash paper1/scripts/run_target_view_diagnostic_shards.sh
fi

# The internal manuscript-consistency checker is not part of the public
# code release; artifact provenance is documented in DATA_MANIFEST.md.

