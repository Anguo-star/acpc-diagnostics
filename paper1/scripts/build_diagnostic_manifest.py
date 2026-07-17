#!/usr/bin/env python3
"""Create Paper1 diagnostic remediation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .utils_paper1_io import ROOT, RHO_GRID, SEEDS, TASKS

DEFAULT_OUT = ROOT / "paper1" / "results" / "diagnostic_manifest.json"

FROZEN_PROTOCOL = "paper1/config/frozen_diagnostic_protocol_v1.json"
PROSPECTIVE_MULTISEVERITY_PROTOCOL = "paper1/config/paired_multiseverity_protocol_v1.json"
PROSPECTIVE_MULTISEVERITY_ADDENDUM_V1 = "paper1/config/paired_multiseverity_execution_addendum_v1.json"
PROSPECTIVE_MULTISEVERITY_ADDENDUM_V2 = "paper1/config/paired_multiseverity_execution_addendum_v2.json"
P1_PROSPECTIVE_PROTOCOL = "paper1/config/p1_prospective_seed3075_protocol_v1.json"
P1_PROSPECTIVE_README = (
    "paper1/results/target_aligned_acpc_prospective_v1/README.md"
)
P1_PROSPECTIVE_ADJUDICATION = (
    "paper1/results/target_aligned_acpc_prospective_v1/"
    "adjudication_four_task_seed3075_goal25_base_v1.json"
)
P1_PROSPECTIVE_META = (
    "paper1/results/target_aligned_acpc_prospective_v1/"
    "meta_four_task_seeds3073_3074_3075_goal25_base_endpoint_v1.json"
)
P1_PROSPECTIVE_RAW = tuple(
    "paper1/results/target_aligned_acpc_prospective_v1/"
    f"mve_{task}_seed3075_base_goal25_v1_16block.json"
    for task in ("tworoom", "pusht", "reacher", "cube")
)
P1_RETROSPECTIVE_ADJUDICATION = (
    "paper1/results/target_aligned_acpc_dev/"
    "adjudication_four_task_seed3072_goal25_base_endpoint_retrospective_v1.json"
)
PLANNER_PROTOCOL = "paper1/config/acpc_planner_stability_protocol_v4.json"
PLANNER_EXECUTION = "paper1/config/acpc_planner_stability_execution_v4.json"
PLANNER_SUMMARY = "paper1/results/acpc_planner_stability_v4/summary.json"
PUBLIC_V1_ARTIFACTS = (
    FROZEN_PROTOCOL,
    "paper1/results/frozen_external_validation_summary_v3.json",
    "paper1/tables/table_pldm_architecture_portability.tex",
    "paper1/results/external_validation/cross_stressor_fixed_rho_summary.json",
    "paper1/results/external_validation/target_view_frozen_summary.json",
    "paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json",
    "paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json",
    "paper1/results/jvp_hutchinson_sensitivity_audit_v2.json",
    "paper1/results/linearization_horizon_sensitivity_v1.json",
    "paper1/results/fixed_pool_candidatewise_certificate_summary.json",
    "assets/paper1_data/smpr_sensitivity_v2.json",
    "assets/paper1_data/smpr_controls_v2.json",
    "assets/paper1_data/smpr_oracle_guard_v2.json",
)
CLAIM_ALIGNED_EXTENSION_ARTIFACTS = (
    "paper1/config/target_aligned_acpc_four_task_contract_v2.json",
    P1_PROSPECTIVE_PROTOCOL,
    P1_PROSPECTIVE_README,
    "paper1/results/target_aligned_acpc_dev/adjudication_four_task_seed3073_goal25_base_endpoint_v1.json",
    "paper1/results/target_aligned_acpc_dev/adjudication_four_task_seed3074_goal25_base_endpoint_v1.json",
    "paper1/results/target_aligned_acpc_dev/meta_four_task_seeds3073_3074_goal25_base_endpoint_v1.json",
    P1_RETROSPECTIVE_ADJUDICATION,
    *P1_PROSPECTIVE_RAW,
    P1_PROSPECTIVE_ADJUDICATION,
    P1_PROSPECTIVE_META,
    PLANNER_PROTOCOL,
    PLANNER_EXECUTION,
    PLANNER_SUMMARY,
    "paper1/results/three_pillar_evidence_summary.json",
    "paper1/tables/table_target_aligned_acpc.tex",
    "paper1/tables/table_target_aligned_acpc_absolute.tex",
    "paper1/tables/table_acpc_planner_increment.tex",
    "paper1/tables/table_acpc_planner_absolute.tex",
    "paper1/tables/table_full_sweep_compact.tex",
    "paper1/tables/table_seed_transfer_audit.tex",
    "paper1/tables/table_cross_stressor_transfer.tex",
    "assets/paper1_figs/fig_acpc_planner_evidence.pdf",
    "assets/paper1_figs/fig_full_sweep_diagnostics.pdf",
    "assets/paper1_figs/fig_acpc_basin_tsne.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    artifact_hashes = {
        rel: sha256_file(ROOT / rel)
        for rel in PUBLIC_V1_ARTIFACTS
    }
    claim_aligned_hashes = {
        rel: sha256_file(ROOT / rel)
        for rel in CLAIM_ALIGNED_EXTENSION_ARTIFACTS
    }
    protocol_hash = artifact_hashes[FROZEN_PROTOCOL]
    prospective_protocol_hash = sha256_file(ROOT / PROSPECTIVE_MULTISEVERITY_PROTOCOL)
    prospective_addendum_v1_hash = sha256_file(ROOT / PROSPECTIVE_MULTISEVERITY_ADDENDUM_V1)
    prospective_addendum_v2_hash = sha256_file(ROOT / PROSPECTIVE_MULTISEVERITY_ADDENDUM_V2)
    planner_summary = json.loads((ROOT / PLANNER_SUMMARY).read_text(encoding="utf-8"))
    planner_increment = {
        response: {
            "equal_task_relative_mae_reduction": analysis["lobo_ridge"][
                "equal_task_relative_mae_reduction"
            ],
            "sample_sd_across_training_seeds": analysis["lobo_ridge"][
                "relative_mae_reduction_sample_sd"
            ],
            "tasks_improved": analysis["lobo_ridge"]["tasks_improved"],
            "positive_training_seeds": analysis["lobo_ridge"][
                "positive_seed_count"
            ],
            "training_seed_gates_passed": analysis["lobo_ridge"][
                "gate_pass_count"
            ],
        }
        for response, analysis in planner_summary[
            "predeclared_incremental_analyses"
        ].items()
    }
    data = {
        "schema_version": "paper1-diagnostic-manifest-2.1",
        "tasks": TASKS,
        "training_seeds": SEEDS,
        "eval_seeds": [42, 43, 44],
        "rho_grid": [float(x) for x in RHO_GRID],
        "eval_noise_sigmas": [0.00, 0.03, 0.05, 0.08],
        "checkpoint_epoch": 10,
        "frozen_protocol_source": FROZEN_PROTOCOL,
        "frozen_protocol_sha256": protocol_hash,
        "protocol_development_split": "CAL: LeWM training seed 3072 Gaussian sweep only",
        "external_threshold_search": False,
        "closed_loop_eval_source": "assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json",
        "canonical_horizon_v2_source": "assets/paper1_data/acpc_horizon_v2_lewm.json",
        "diagnostic_source": "paper1/results/prospective_diagnostic/diagnostics_all_ckpts.csv",
        "smpr_sensitivity_source": "assets/paper1_data/smpr_sensitivity_v2.json",
        "smpr_control_source": "assets/paper1_data/smpr_controls_v2.json",
        "smpr_oracle_mve_source": "assets/paper1_data/smpr_oracle_guard_v2.json",
        "fixed_pool_summary_source": "assets/paper1_data/acpc_phase0_lewm_three_seed.json",
        "raw_fixed_pool_source": "paper1/results/fixed_pool_candidatewise_certificate_summary.json",
        "jacobian_audit_source": "exact-autograd matched-map audit: paper1/results/jvp_hutchinson_sensitivity_audit_v2.json; covariance-aware finite-difference calibration: paper1/results/linearization_horizon_sensitivity_v1.json",
        "component_baseline_source": "paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json",
        "public_v1_artifact_sha256": artifact_hashes,
        "claim_aligned_three_pillar_extension": {
            "artifact_sha256": claim_aligned_hashes,
            "evidence_bundle": "paper1/results/three_pillar_evidence_summary.json",
            "p1_provenance": (
                "seed3075 is fully prospective; seed3074 is a protocol-frozen "
                "four-task replication; seed3073 has development-era provenance; "
                "seed3072 remains a separate retrospective completeness check"
            ),
            "p1_prospective_protocol": P1_PROSPECTIVE_PROTOCOL,
            "p1_prospective_provenance_note": P1_PROSPECTIVE_README,
            "p1_prospective_raw_results": list(P1_PROSPECTIVE_RAW),
            "p1_prospective_adjudication": P1_PROSPECTIVE_ADJUDICATION,
            "p1_three_run_meta": P1_PROSPECTIVE_META,
            "p1_reported_training_seeds": [3073, 3074, 3075],
            "p1_retrospective_training_seed": 3072,
            "p1_retrospective_adjudication": P1_RETROSPECTIVE_ADJUDICATION,
            "p1_raw_runner_metadata_note": (
                "the raw MVE runner retains its legacy DEV-purpose string; "
                "prospective provenance is established by the pre-result frozen "
                "seed3075 protocol, checkpoint hashes, exact commands, and the "
                "separate prospective result directory"
            ),
            "p1_inference_scope": (
                "cluster intervals are conditional on the three listed trained "
                "checkpoints and are not population intervals over training runs"
            ),
            "p1_primary_target": "action-matched held-out future-error drift",
            "p1_regression_role": (
                "analysis-only conditional-information test; not a deployed "
                "future-label-dependent surrogate"
            ),
            "p2_calibration": "LeWM seed3072 only; seeds3073/3074 read-only TEST",
            "p3_calibration": "Gaussian only; blur/resize thresholds unchanged",
            "threshold_search_allowed": False,
        },
        "planner_aligned_extension": {
            "protocol_source": PLANNER_PROTOCOL,
            "execution_addendum_source": PLANNER_EXECUTION,
            "summary_source": PLANNER_SUMMARY,
            "authorized_shards": planner_summary["authorized_shard_count"],
            "validated_shards": planner_summary["validated_shard_count"],
            "joined_reduced_rows": planner_summary["joined_reduced_row_count"],
            "training_seeds": planner_summary["training_seeds"],
            "invariants": planner_summary["invariants"],
            "h5_increment": planner_increment,
            "fixed_pool_clean_regret_bound": (
                "for shared-pool nominal/probe winners w and w_tilde, "
                "0 <= C(w_tilde)-C(w) <= b(w_tilde)+b(w)"
            ),
            "claim_boundary": (
                "behavior-blind finite candidate pools; adaptive-CEM implications "
                "are conditional on aligned proposals and certified elite sets, "
                "not a closed-loop robustness guarantee"
            ),
        },
        "prospective_multiseverity_extension": {
            "completed_primary_pairs": 0,
            "evidence_status": "protocol-only; not public-v1 completed evidence",
            "execution_addendum_sha256": prospective_addendum_v2_hash,
            "execution_addendum_source": PROSPECTIVE_MULTISEVERITY_ADDENDUM_V2,
            "parent_execution_addendum_sha256": prospective_addendum_v1_hash,
            "parent_execution_addendum_source": PROSPECTIVE_MULTISEVERITY_ADDENDUM_V1,
            "expected_primary_pairs": 72,
            "protocol_sha256": prospective_protocol_hash,
            "protocol_source": PROSPECTIVE_MULTISEVERITY_PROTOCOL,
            "smoke_validation": {
                "completed_behavior_atr_smpr_triplets": 1,
                "primary_claim_eligible": False,
                "scope": "LeWM seed3072 TwoRoom gaussian_blur kernel_size=7",
                "zero_rule_direction_agreement": True,
            },
            "status": "frozen_protocol_with_v2_task_bound_smoke",
            "v1_reference_path_failure_disclosed": True,
        },
        "external_validation": {
            "E1_heldout_lewm": "paper1/results/frozen_external_validation_summary_v3.json",
            "E2_pldm_architecture_portability": "paper1/tables/table_pldm_architecture_portability.tex",
            "E3_fixed_rho_blur_resize": "paper1/results/external_validation/cross_stressor_fixed_rho_summary.json",
            "E4_failed_target_view": "paper1/results/external_validation/target_view_frozen_summary.json",
            "pldm_eval_seed_semantics": "conditional evaluation replicates, not independent training seeds",
        },
        "generated_outputs": [
            "paper1/results/full_sweep_diagnostics.csv",
            "paper1/results/full_sweep_diagnostics_summary.csv",
            "paper1/results/heldout_diagnostic_validation.csv",
            "paper1/results/heldout_gate_params.json",
            "paper1/results/fixed_pool_tail_audit.csv",
            "paper1/results/fixed_pool_tail_audit_summary.csv",
            "paper1/results/threshold_quantile_sensitivity.csv",
            "assets/paper1_figs/fig_full_sweep_diagnostics.png",
            "assets/paper1_figs/fig_full_sweep_diagnostic_region.png",
            "assets/paper1_figs/fig_full_sweep_planner_guard.png",
            "assets/paper1_figs/fig_endpoint_atr_smpr.png",
            "assets/paper1_figs/fig_heldout_diagnostic_validation.png",
            "assets/paper1_figs/fig_fixed_pool_tail_audit.png",
            "assets/paper1_figs/fig_top1_agreement_full_sweep.png",
            "assets/paper1_figs/fig_threshold_sensitivity.png",
            "paper1/results/sample_level_certificate_full_sweep_audit.csv",
            "paper1/results/sample_level_certificate_full_sweep_audit.json",
            "paper1/results/sample_level_certificate_full_sweep_samples.csv",
            "paper1/results/sample_level_certificate_full_sweep_summary.csv",
            "paper1/results/sample_level_certificate_recovery_alignment.csv",
            "paper1/results/sample_level_event_rate_wilson_ci.csv",
            "paper1/tables/table_sample_level_certificate_full_sweep.tex",
            "paper1/tables/table_sample_level_event_rate_ci.tex",
            "assets/paper1_figs/fig_fixed_pool_event_rates.png",
            "paper1/results/sample_level_certificate_endpoint_audit.csv",
            "paper1/results/sample_level_certificate_endpoint_audit.json",
            "paper1/results/sample_level_certificate_endpoint_samples.csv",
            "paper1/results/sample_level_certificate_endpoint_summary.csv",
            "paper1/tables/table_sample_level_certificate_endpoint.tex",
            "paper1/results/gaussian_sensitivity_audit.csv",
            "paper1/results/gaussian_sensitivity_audit.json",
            "paper1/results/gaussian_sensitivity_summary.csv",
            "paper1/tables/table_gaussian_sensitivity_audit.tex",
            "paper1/results/jvp_hutchinson_sensitivity_audit.csv",
            "paper1/results/jvp_hutchinson_sensitivity_audit.json",
            "paper1/results/jvp_hutchinson_sensitivity_summary.csv",
            "paper1/tables/table_jvp_hutchinson_sensitivity_audit.tex",
            "assets/paper1_figs/fig_gaussian_sensitivity_main.png",
            "assets/paper1_figs/fig_jvp_trace_decomposition_heatmap.png",
            "paper1/tables/table_endpoint_atr_smpr.tex",
            "paper1/tables/table_theory_evidence_map.tex",
            "paper1/results/joint_guard_side_validation.csv",
            "paper1/tables/table_joint_guard_side_validation.tex",
            "paper1/results/frozen_external_validation_summary_v3.json",
            "paper1/tables/table_pldm_architecture_portability.tex",
            "paper1/results/external_validation/cross_stressor_fixed_rho_summary.json",
            "paper1/results/external_validation/cross_stressor_fixed_rho_rows.csv",
            "paper1/results/external_validation/cross_stressor_all_pairs.csv",
            "paper1/tables/table_cross_stressor_paired_change.tex",
            "paper1/tables/table_cross_stressor_robustness_audit.tex",
            "paper1/results/three_pillar_evidence_summary.json",
            P1_PROSPECTIVE_ADJUDICATION,
            P1_PROSPECTIVE_META,
            PLANNER_SUMMARY,
            "paper1/tables/table_target_aligned_acpc.tex",
            "paper1/tables/table_target_aligned_acpc_absolute.tex",
            "paper1/tables/table_acpc_planner_increment.tex",
            "paper1/tables/table_acpc_planner_absolute.tex",
            "paper1/tables/table_full_sweep_compact.tex",
            "paper1/tables/table_seed_transfer_audit.tex",
            "paper1/tables/table_cross_stressor_transfer.tex",
            "assets/paper1_figs/fig_acpc_planner_evidence.pdf",
            "assets/paper1_figs/fig_cross_stressor_fixed_rho.png",
            "paper1/results/external_validation/target_view_frozen_summary.json",
            "paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json",
            "paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json",
            "paper1/results/jvp_hutchinson_sensitivity_audit_v2.json",
            "paper1/results/jvp_hutchinson_sensitivity_summary_v2.csv",
            "paper1/results/linearization_horizon_sensitivity_v1.json",
            "paper1/results/linearization_calibration_summary.csv",
            "paper1/results/horizon_quantile_sensitivity_summary.csv",
            "paper1/results/fixed_pool_candidatewise_certificate_summary.json",
            "paper1/results/fixed_pool_certificate_coverage_by_block.csv",
            "paper1/results/fixed_pool_risk_coverage.csv",
            "paper1/tables/table_fixed_pool_certificate_coverage.tex",
            "paper1/tables/table_linearization_calibration.tex",
            "paper1/tables/table_horizon_quantile_sensitivity.tex",
            "paper1/tables/table_smpr_sensitivity.tex",
            "paper1/tables/table_smpr_controls.tex",
            "assets/paper1_figs/fig_diagnostic_baseline_external.png",
            "assets/paper1_figs/fig_fixed_pool_certificate_calibration.png",
            "assets/paper1_figs/fig_linearization_calibration.png",
            "assets/paper1_figs/fig_smpr_radius_margin_decomposition.png",
        ],
        "sample_level_certificate_audit": {
            "scope": "full Gaussian sweep for four tasks and training seeds 3072/3073/3074",
            "n_sequences_per_seed_checkpoint": 100,
            "candidate_count": 65,
            "result_csv": "paper1/results/sample_level_certificate_full_sweep_summary.csv",
            "interpretation": "cert-pass and top1-flip separation strengthens fixed-pool mechanism audit; strict q10/q95 gaps remain negative, so this is not a calibrated certificate",
        },
        "gaussian_sensitivity_audit": {
            "scope": "base, recovery-onset, and endpoint checkpoints for four tasks and training seeds 3072/3073/3074",
            "small_sigmas": [0.005, 0.01, 0.02],
            "n_sequences_per_checkpoint": 100,
            "num_noise_draws_per_small_sigma": 5,
            "result_csv": "paper1/results/gaussian_sensitivity_summary.csv",
            "interpretation": "finite-difference local sensitivity proxy; not a global robustness guarantee",
        },
        "jvp_hutchinson_sensitivity_audit": {
            "scope": "base, recovery-onset, and endpoint checkpoints for four tasks and training seeds 3072/3073/3074",
            "n_sequences_per_checkpoint": 100,
            "hutchinson_probes_per_checkpoint": 8,
            "result_csv": "paper1/results/jvp_hutchinson_sensitivity_summary.csv",
            "interpretation": "exact-autograd JVP/Hutchinson local Frobenius-trace decomposition; not a full Jacobian matrix or closed-loop guarantee",
        },
        "joint_guard_side_validation": {
            "result_csv": "paper1/results/joint_guard_side_validation.csv",
            "interpretation": "legacy proxy view retained for provenance; public-v1 claim decisions use the v2 SMPR sensitivity/control/MVE artifacts",
        },
        "public_v1_claim_boundaries": {
            "coarse_checkpoint_label_long_horizon_increment_established": False,
            "coarse_checkpoint_label_correct_action_increment_established": False,
            "target_aligned_fragile_logged_h8_increment_established": True,
            "target_aligned_candidate_h5_universal_increment_established": False,
            "smpr_task_label_increment_established": False,
            "smpr_constant_collapse_rejected": True,
            "pldm_training_runs": 1,
            "fixed_pool_only": False,
            "adaptive_cem_panel_completed": True,
            "adaptive_cem_guarantee": False,
            "adaptive_cem_conditional_result": (
                "proposal updates remain aligned while every preceding elite-set "
                "certificate holds"
            ),
            "analysis_interface_portability": ["LeWM", "PLDM"],
            "absolute_calibration_scope": "model-family-specific",
            "cross_stressor_transfer_claim": "within-LeWM paired comparison across fixed blur/resize; absolute screening is conservative",
            "paired_change_reference_required": True,
            "paired_change_shared_zero_threshold_within_lewm": True,
            "shared_cross_model_absolute_threshold": False,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "The claim-aligned extension adds a prospectively frozen LeWM training run and a behavior-blind planner panel; it does not add closed-loop re-evaluation.",
            "Full-sweep sample-level fixed-pool event rates are recomputed from checkpoints; strict q10/q95 gaps remain negative and are not calibrated probability bounds.",
            "E3 separates family-calibrated absolute screening from a reference-based zero-threshold paired comparison shared across fixed LeWM blur/resize stressors.",
            "Wilson intervals quantify sample event-rate estimation uncertainty, not theorem-calibrated probabilities.",
            "Held-out gates are calibrated on calibration rows only; held-out labels are used only for evaluation.",
            "SMPR and fixed-pool top1 flip are guard-side checks interpreted jointly with ATR, not standalone robustness metrics.",
            "Exact-autograd JVP/Hutchinson traces decompose local encoder, rollout, and composed sensitivity but do not materialize a full Jacobian or prove closed-loop robustness.",
            "Coarse checkpoint labels do not establish H8/correct-action increment; the separate target-aligned fragile logged-future audit does.",
            "The sharp certificate is deterministic for the sampled ordered candidate pool; zero certified flips are an invariant check.",
            "The three-seed v4 planner panel validates the exact squared-goal-cost bound, fixed-pool certificates, and the conditional adaptive-CEM induction; decision-regret increment passes all three seed-level gates, cost drift is positive on all three seeds but passes two gates, and first-action RMS passes none.",
            "SMPR rejects complete collapse but does not establish incremental task-label, action, progressive-collapse, or four-task oracle relevance.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
