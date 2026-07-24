# Paper1 Diagnostic Remediation Scripts

Submission-facing scripts rebuild diagnostics and displays from the LeWM
training runs and the complete PLDM four-task Gaussian sweep. They do not
train a new checkpoint. Separately named paired_multiseverity runners reuse
fixed checkpoints, rerun bounded closed-loop evaluation, and never train.

## Current submission-facing ACPC bundle

The submission mainline is rebuilt from the future-drift adjudications, the
complete LeWM and PLDM Gaussian sweeps, the planner summary, and the 24-pair
LeWM stressor file:

```bash
python -m paper1.scripts.plot_acpc_ir_sr_overview
python -m paper1.scripts.build_future_drift_reader_display
python -m paper1.scripts.cross_task_selective_rule
python -m paper1.scripts.build_cross_stressor_ir_sr_comparison
python -m paper1.scripts.build_acpc_submission_assets
bash paper1/build.sh
```

The first command rebuilds Figure 1 from the committed, hash-recorded PushT
input frames. The next three rebuild the future-drift display, all 14 LeWM
cross-task threshold partitions, and the IR--SR blur/resize comparison
analysis. `build_acpc_submission_assets` writes the vector planner-evidence figure,
and compact planner/full-sweep tables. The PLDM diagnostic-behavior figure
reads the complete 108-row, three-training-run sweep at
`paper1/results/external_validation/pldm_frozen_rows_v2.csv`. These steps
perform no model evaluation or training.

To reproduce the two additional PLDM training families from local checkpoints,
first build their behavior and checkpoint-bound manifests, then run the frozen
GPU diagnostic pipeline:

```bash
python tools/build_canonical_evals_pldm.py \
  --root /path/to/pldm/task/roots \
  --training-seed 3073 \
  --out assets/paper1_data/training_seed_eval_manifests/pldm_seed3073_evals_legacy.json
python tools/build_canonical_evals_pldm.py \
  --root /path/to/pldm/task/roots \
  --training-seed 3074 \
  --out assets/paper1_data/training_seed_eval_manifests/pldm_seed3074_evals_legacy.json
python -m paper1.scripts.audit_pldm_checkpoint_loadability \
  --canonical assets/paper1_data/training_seed_eval_manifests/pldm_seed3073_evals_legacy.json \
  --model-root /path/to/pldm/task/roots/lewm-tworooms \
  --model-root /path/to/pldm/task/roots/lewm-pusht \
  --model-root /path/to/pldm/task/roots/lewm-reacher \
  --model-root /path/to/pldm/task/roots/lewm-cube \
  --out paper1/results/pldm_checkpoint_loadability_seed3073_v2.json
python -m paper1.scripts.build_pldm_canonical_manifest_v2 \
  --legacy assets/paper1_data/training_seed_eval_manifests/pldm_seed3073_evals_legacy.json \
  --loadability paper1/results/pldm_checkpoint_loadability_seed3073_v2.json \
  --out assets/paper1_data/training_seed_eval_manifests/pldm_seed3073_evals.json
# Repeat the preceding audit/build pair with seed3074 paths.
bash paper1/scripts/run_pldm_multiseed_gaussian_diagnostics.sh
python -m paper1.scripts.build_pldm_multiseed_validation \
  --predictions-3072 paper1/results/external_validation/pldm_frozen_predictions_blind_v2.csv \
  --manifest-3072 assets/paper1_data/canonical_evals_pldm_v2.json \
  --predictions-3073 paper1/results/pldm_multiseed_v2/seed3073/frozen_predictions_blind.csv \
  --manifest-3073 assets/paper1_data/training_seed_eval_manifests/pldm_seed3073_evals_legacy.json \
  --predictions-3074 paper1/results/pldm_multiseed_v2/seed3074/frozen_predictions_blind.csv \
  --manifest-3074 assets/paper1_data/training_seed_eval_manifests/pldm_seed3074_evals_legacy.json \
  --out paper1/results/external_validation/pldm_frozen_rows_v2.csv
```

`run_pldm_multiseed_gaussian_diagnostics.sh` defaults to seeds 3073/3074 and
eight GPUs, but accepts `PAPER1_MODEL_BASE`, `PAPER1_DIAGNOSTIC_GPUS`, and
`PAPER1_TRAINING_SEEDS`. It reuses the immutable metric protocol and refuses
incomplete 36-checkpoint families. `build_pldm_multiseed_validation.py` then
joins the frozen predictions with all three training runs to produce the
108-row paper-facing CSV. Checkpoint paths in released manifests are
machine-local provenance; portable reruns should regenerate them.

Reader-facing builders canonicalize diagnostic fields through
`ir_sr_compat.py`. It maps both immutable ATR/SMPR fields and released IR/DR-v1
fields into the current IR/SR schema without changing stored values. Equal
duplicate aliases are accepted, but aliases with inconsistent values fail
explicitly. The older `ir_dr_compat.py` entry point remains only for legacy-v1
reproduction.

The released linearization aggregate can be migrated to the same canonical
schema without reopening its machine-specific checkpoint paths:

```bash
python -m paper1.scripts.build_linearization_horizon_artifact \
  --legacy-artifact paper1/results/linearization_horizon_sensitivity_v1.json
```

This path records a schema-migration provenance block and leaves all measured
values unchanged.

If the source PushT H5 is available, regenerate Figure 1's paired inputs and
retained source metadata first with
`python -m paper1.scripts.build_acpc_overview_inputs --h5 /path/to/pusht_expert_train.h5`.
The planner input is the
three-seed v4 summary: 24 validated seed-3074 reference shards under
`paper1/results/acpc_planner_stability_v2/` plus 48 exact-protocol replication
shards for seeds 3072/3073 under
`paper1/results/acpc_planner_stability_v4/`. The numerical fixed-pool wrapper
changes only signed cost-gap algebra to float64 after the unchanged float32
model-cost path. The unexecuted v3 freeze was superseded before result
generation because its source manifest omitted a transitive summarizer
dependency; v4 binds that dependency without changing the experiment.

To inspect or reproduce one frozen planner task queue:

```bash
python paper1/scripts/run_acpc_planner_stability_shards.py plan \
  --task TwoRoom \
  --protocol paper1/config/acpc_planner_stability_protocol_v4.json \
  --addendum paper1/config/acpc_planner_stability_execution_v4.json \
  --device cuda:0
```

The legacy three-pillar evidence generator is archived and is not part of the
current paper dependency graph. The retained theory chain gives the sharp
candidate-cost bound, a shared-pool clean-regret bound, top-1/elite
certificates, and conditional adaptive-CEM alignment. Its empirical support is
the three-seed future-drift increment, planner panel, exhaustive cross-task
calibration, complete Gaussian sweep, and blur/resize selective-score transfer.

Run the CPU-only manifest/checker path from repository root:

```bash
bash paper1/scripts/run_all_paper1_diagnostics.sh
```

Rebuild checkpoint-level public-v1 audits only when the checkpoints and datasets are available:

```bash
PAPER1_DIAGNOSTIC_GPU=0 PAPER1_DIAGNOSTIC_THREADS=2 \
RUN_REMEDIATION_AUDITS=1 bash paper1/scripts/run_all_paper1_diagnostics.sh
```

This path is deliberately serial: one task/seed shard runs at a time on one GPU, every shard has a timeout, complete shards are validated and skipped, and native math libraries default to two threads. `RUN_CHECKPOINT_AUDITS=1` remains a compatibility alias for the same bounded path; it no longer launches the retired monolithic jobs.

External E3/E4 checkpoint recomputation is separately opt-in:

```bash
PAPER1_DIAGNOSTIC_GPU=0 PAPER1_DIAGNOSTIC_THREADS=2 \
RUN_EXTERNAL_AUDITS=1 bash paper1/scripts/run_all_paper1_diagnostics.sh
```

Rebuild the E3 aggregate, paired-change uncertainty, exact/deletion/selection
robustness audit, generated tables, and scatter from completed
diagnostic/evaluation artifacts (no training or evaluation):

```bash
python -m paper1.scripts.build_cross_stressor_external_validation \
  --baseline-diagnostics paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json \
  --force
```

Archived summary/plot regeneration can overwrite legacy figures and is therefore also explicit: `REBUILD_LEGACY_AGGREGATES=1`. No flag reruns closed-loop evaluation or training.

Archived public-v1 checkpoint steps retained for provenance:

```bash
bash paper1/scripts/run_jvp_hutchinson_shards.sh
bash paper1/scripts/run_diagnostic_baseline_shards.sh
bash paper1/scripts/run_fixed_pool_certificate_shards.sh
bash paper1/scripts/run_smpr_controls_mve.sh
bash paper1/scripts/run_linearization_horizon_shards.sh
```

Legacy summary-level steps retained for provenance:

```bash
python -m paper1.scripts.build_diagnostic_manifest
python -m paper1.scripts.build_full_sweep_diagnostics
python -m paper1.scripts.plot_full_sweep_diagnostics
python -m paper1.scripts.plot_pldm_sweep_diagnostics
python -m paper1.scripts.plot_endpoint_atr_smpr
python -m paper1.scripts.fixed_pool_tail_audit
python -m paper1.scripts.heldout_diagnostic_validation
python -m paper1.scripts.threshold_quantile_sensitivity
python -m tools.paper1_sample_level_certificate --out-json paper1/results/sample_level_certificate_full_sweep_audit.json --out-csv paper1/results/sample_level_certificate_full_sweep_audit.csv --sample-csv paper1/results/sample_level_certificate_full_sweep_samples.csv
python -m paper1.scripts.full_sweep_sample_level_certificate_summary
python -m paper1.scripts.sample_level_event_rate_ci
python -m paper1.scripts.plot_fixed_pool_event_rates
python -m tools.paper1_gaussian_sensitivity_audit --num-noise-draws 5
python -m tools.paper1_jvp_hutchinson_sensitivity_audit --n-sequences 100 --hutchinson-probes 8
python -m paper1.scripts.joint_guard_side_validation
python -m paper1.scripts.plot_gaussian_sensitivity_mechanism
```

Frozen prospective multi-severity extension (all commands are serial,
resumable, hash-gated, and watchdog-bounded):

~~~bash
# No eval: inspect checkpoint binding and pending behavior artifacts.
bash paper1/scripts/run_paired_multiseverity_behavior.sh plan

# One frozen seed/task/stressor/severity smoke.
bash paper1/scripts/run_paired_multiseverity_atr.sh smoke
bash paper1/scripts/run_paired_multiseverity_behavior.sh smoke
bash paper1/scripts/run_paired_multiseverity_smpr_v2.sh smoke

# Complete LeWM diagnostics first, then closed-loop behavior.
bash paper1/scripts/run_paired_multiseverity_atr.sh full
bash paper1/scripts/run_paired_multiseverity_smpr_v2.sh full
bash paper1/scripts/run_paired_multiseverity_behavior.sh full

# This refuses partial coverage; success requires all 72 paired rows.
python -m paper1.scripts.build_paired_multiseverity_summary
~~~

The parent protocol is paper1/config/paired_multiseverity_protocol_v1.json
(SHA-256 6712b4f595444d751d9c327262c288e37dbd80be7ddde9bb4fd336ed41119622).
The SMPR execution addendum transparently records that its wrapper/reference
adapter was connected after the behavior/ATR smoke had been inspected; it does
not alter the zero threshold, severity grid, sampling, or analysis. A smoke is
never primary-claim eligible, and PLDM must not start before the complete LeWM
72-row aggregate is locked.

The original `run_paired_multiseverity_smpr.sh` is retained as v1 provenance.
Its reference key omitted task; the v2 addendum records the stopped run and the
v2 runner binds every reference by seed, task, stressor, and severity.

Plot output notes:

- `plot_pldm_sweep_diagnostics` renders PLDM planning success, relative IR, and SR from `pldm_frozen_rows_v2.csv`, with run means/ranges over three independent training runs and no checkpoint-screening threshold.
- `plot_full_sweep_diagnostics` writes vector PDF figures by default: a main figure with separate behavior and ATR/SMPR axes per task, the diagnostic-region scatter, and a compact four-across appendix planner-guard figure; recovery shading is rendered as continuous majority-recovered ranges. The main sweep display divides ATR by each task$\times$seed no-noise value (base 1) while leaving SMPR on its original rate scale; frozen calibration uses the unrescaled ATR statistic.
- `plot_cross_stressor_submission` reads the locked all-pairs CSV and writes the 24-pair LeWM submission scatter as a vector PDF; it does not rerun diagnostics or evaluation.
- `plot_endpoint_atr_smpr` writes the two-panel endpoint dumbbell figure with base-to-noise-trained movement arrows.
- `plot_gaussian_sensitivity_mechanism` writes a two-panel endpoint/base lollipop figure for the main text and the trace-decomposition heatmap plus separate alignment panel for the appendix.
- `plot_fixed_pool_event_rates` writes the two-panel paired event-rate figure; conditional flip-given-cert-pass rates remain in the appendix table.

Important scope constraints:

- The immutable public-v1 protocol SHA-256 is `edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21`; checkpoint builders must reproduce this value in external artifacts.
- Matched component/horizon baselines do not establish empirical necessity for H=8 or correct-action conditioning.
- E3 distinguishes model-family-calibrated absolute screening from a reference-based paired comparison; one zero threshold is shared across fixed LeWM blur/resize stressors, not across absolute model-family scales.
- The sharp certificate applies only to the sampled ordered candidate pool; zero flips on cert-pass are a deterministic invariant check.
- SMPR v2 rejects constant collapse but does not establish incremental action, label, progressive-collapse, or four-task oracle relevance.
- The frozen SMPR-v2 numeric artifacts retain two legacy proxy-description strings. `paper1/results/smpr_v2_proxy_metadata_correction_v1.json` records the exact executed state-coordinate slices; numeric rows and selected pair indices are unchanged. Paper-facing text treats SMPR as a state-coordinate separation guard, not a semantic certificate.
- Full-sweep diagnostics join existing Gaussian evaluation, ATR, SMPR, and retained fixed-pool summaries.
- Full-sweep sample-level fixed-pool event rates are recomputed from checkpoints; strict q10/q95 gaps remain negative and are not treated as calibrated probability bounds.
- Wilson intervals quantify sample event-rate estimation uncertainty; they are not calibrated theorem probability bounds. The event-rate figure is regenerated from `paper1/results/sample_level_event_rate_wilson_ci.csv`.
- Held-out validation freezes diagnostic gates on calibration rows before evaluating held-out rows.
- Gaussian sensitivity is audited with finite differences using 100 sampled sequences and 5 noise draws per small sigma, plus an exact-autograd JVP/Hutchinson trace decomposition using 100 sampled sequences and 8 Rademacher probes per checkpoint; both are local checkpoint audits, not closed-loop guarantees.
- SMPR and fixed-pool top-1 flip are guard-side checks interpreted only jointly with ATR, not standalone robustness metrics.
- ATR q80/q95 and positive-margin SMPR variants are not inferred from retained ATR/SMPR summaries.
