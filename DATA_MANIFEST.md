# Paper 1 Data Manifest

This manifest documents the released evaluation aggregate for Paper 1:

- Canonical aggregate: `assets/paper1_data/canonical_evals_20260517.json`
- Schema: `assets/paper1_data/canonical_evals_20260517.schema.json`
- Canonical diagnostics: `assets/paper1_data/canonical_diagnostics_20260517.json` (2026-06-10 revision: the TwoRoom/PushT representative diagnostics used by `tab:diag-base-vs-best` were re-extracted from the per-checkpoint diagnostics after an audit found they duplicated the `*_lewm_hetero_default` values; see the JSON `metadata.table3_revision_20260610` note)
- Diagnostics schema: `assets/paper1_data/canonical_diagnostics_20260517.schema.json`
- External baseline sanity check: `assets/paper1_data/canonical_external_baselines_20260520.json`
- External baseline schema: `assets/paper1_data/canonical_external_baselines_20260520.schema.json`
- **PLDM cross-method replication aggregate**: `assets/paper1_data/canonical_evals_pldm_20260522.json` (36 ckpts: 4 tasks × 9 configs)
- **PLDM cross-method replication diagnostics**: `assets/paper1_data/canonical_diagnostics_pldm_20260522.json`
- **PLDM full diagnostics**: `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json` (full diagnostics-summary rows for the same 36 PLDM ckpts)
- **PLDM full diagnostics schema**: `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.schema.json`
- **Cross-method correlations**: `assets/paper1_data/cross_method_corr_pldm_20260522.json` (within-LeWM / within-PLDM / joint partial Spearman; consumed by the PLDM appendix)
- **PLDM full ACPC basin replication**: `assets/paper1_data/acpc_basin_diagnostics_pldm.json` (36 rows: 4 tasks × 9 PLDM configs)
- **Partial-correlation bootstrap CIs**: `assets/paper1_data/partial_corr_bootstrap_20260523.json` (95% percentile bootstrap intervals for the LeWM, PLDM, and joint partial-correlation tables)
- **Phase-0 paired ACPC diagnostics**: `assets/paper1_data/acpc_phase0_clean_goal_seed9101.json` (72 rows: LeWM + PLDM, 4 tasks × 9 configs; clean-goal observation-noise run consumed by the Phase-0 appendix)
- **Gaussian-noise ACPC basin diagnostics**: `assets/paper1_data/acpc_basin_diagnostics.json` (LeWM 36 ckpts, epoch-10 model objects, paired clean/noised views at Gaussian std 0.01..0.08; source for the ACPC basin table)
- **No-noise-baseline blur sanity check**: `assets/paper1_data/canonical_blur_baselines_20260523.json` (LeWM + PLDM baselines trained without input-noise augmentation, 4 tasks, blur eval only)
- **No-noise-baseline blur schema**: `assets/paper1_data/canonical_blur_baselines_20260523.schema.json`
- **Three-training-seed strongest-only unseen score artifacts**: `assets/paper1_data/unseen_origin_vs_std008_strongest_s3072.json`, `assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json`, and `assets/paper1_data/unseen_origin_vs_std008_strongest_s3074.json` (audited score artifacts for the unseen score aggregate over training seeds 3072/3073/3074; the `s3072` wrapper is derived from the earlier seed-3072 task-split artifacts `unseen_origin_vs_std008_strongest_tworoom.json` and `unseen_origin_vs_std008_strongest_reacher.json`; together cover 4 tasks × 3 training seeds × 2 std keys × 2 stress families with no-op plus blur k=15 / resize factor 0.25)
- **Unseen Phase-0 ACPC subset artifacts**: `assets/paper1_data/unseen_phase0_acpc_subset.json` (12 selected seed/task/stress case rows for training seeds 3072/3073/3074 joining 24 clean-goal Phase-0 diagnostic rows with strongest-only unseen eval scores) and `assets/paper1_data/unseen_phase0_acpc_fullstress.json` (24 task-family-seed rows over all four tasks, blur/resize strongest endpoints, and training seeds 3072/3073/3074; reduces selected-slice risk while remaining a bounded two-family scope audit)
- **Three-training-seed Gaussian replication summary**: `assets/paper1_data/training_seed_gaussian_lockbox.json` and `.md` (legacy filename; canonical seed 3072 plus independent seeds 3073/3074; reports training-seed mean/std for the main observation-noise 0.08 endpoint)
- **Three-training-seed full Gaussian sweep summary**: `assets/paper1_data/three_seed_gaussian_sweep_summary_20260706.json` and `.md` (36 task/std rows plus 108 per-training-seed rows; source for the main sweep figure and compact appendix sweep table)
- **Three-seed LeWM eval manifests**: `assets/paper1_data/training_seed_eval_manifests/lewm_seed{3072,3073,3074}_evals.json` (canonical-shaped manifests with local checkpoint paths and seed-specific eval metrics for the three-training-seed full-grid diagnostic run)
- **Three-seed PLDM eval manifests**: the historical seed-3072 aggregate plus `assets/paper1_data/training_seed_eval_manifests/pldm_seed{3073,3074}_evals_legacy.json` provide the behavior results for three independent PLDM training runs; the matching `pldm_seed{3073,3074}_evals.json` files bind the new checkpoints and configs used for diagnostics.
- **Three-seed PLDM frozen diagnostics**: `paper1/results/pldm_multiseed_v2/seed{3073,3074}/` contains the checkpoint-bound ATR, SMPR, strict diagnostic-input, and behavior-blind prediction artifacts for the two new runs. Together with the archived seed-3072 artifacts, `paper1/results/external_validation/pldm_frozen_rows_v2.csv` contains 108 rows (3 training seeds × 4 tasks × 9 checkpoints).
- **Three-seed LeWM Phase-0 full-grid diagnostics**: `assets/paper1_data/acpc_phase0_lewm_three_seed.json` plus per-seed component files (108 ok rows: 3 training seeds × 4 tasks × 9 Gaussian checkpoint rows; ACPC/PCC/CRA/MAF computed with clean goal and observation-history Gaussian noise)
- **Three-seed fixed-rule diagnostic validation**: `assets/paper1_data/three_seed_diagnostic_validation.json` and `.md` (ACPC/PCC/CRA/MAF ranking rule fixed on development seed 3072 and evaluated on held-out training seeds 3073/3074; reports 7/8 held-out within-5pp selection hits, 2.21 ± 1.83 pp held-out regret, and 10/12 within-5pp over all seeds)
- **Legacy selector-baseline audit**: `assets/paper1_data/selector_baseline_audit_20260704.json` and `.md` (retained for provenance only; not used as paper-facing selector evidence)
- **Plateau-membership audit**: `assets/paper1_data/selector_plateau_audit_20260704.json` and `.md` (same 96 nonzero LeWM rows; treats `std_max` as a candidate label inside each task-training-seed block; reports whether diagnostic top-half screens are enriched for best-5pp plateau members rather than ranking checkpoints inside a plateau)
- **Legacy residual diagnostic audits**: `assets/paper1_data/residual_diagnostic_audit_20260704.json`/`.md` and `assets/paper1_data/selector_incremental_audit_20260704.json`/`.md` (same 96 nonzero LeWM rows; retained for provenance only, not used as paper-facing selector evidence)
- **Margin-conditioned action-flip audit**: `assets/paper1_data/margin_flip_curve_lewm_three_seed.json` (96 ok threshold rows plus 2400 sample rows: training seeds 3072/3073/3074 × 4 tasks × {0.0, 0.08}; records clean top-1/top-2 candidate margins and clean/noisy top-1 flips for a fixed 65-action candidate pool).
- **Task-grounded near-boundary proxy margin pass-rate**: `assets/paper1_data/semantic_task_grounded_margin_lewm_three_seed.json` and `.md` (24 ok rows: training seeds 3072/3073/3074 × 4 tasks × {0.0, 0.08}; selects closest task-label-crossing neighbors inside the closest 35% state-distance neighborhood; reports proxy pass rates plus same-state noisy radius and proxy-different rollout distance). The earlier local/global proxy sanity artifacts remain at `assets/paper1_data/semantic_local_margin_lewm_three_seed.json` and `assets/paper1_data/semantic_margin_passrate_lewm_three_seed.json`.
- **Reduced-budget offline CEM trace audit**: `assets/paper1_data/cem_trace_audit_20260704.json` and `.md` (24 ok rows: training seeds 3072/3073/3074 × 4 tasks × {0.0, 0.08}; runs the actual CEMSolver clean/noisy traces on logged states with a reduced budget to audit planner-side sensitivity without replacing closed-loop evaluation)
- **Robust CEM low-concurrency pilot**: `assets/paper1_data/robust_cem_pilot_20260704.json` and `.md` (reduced-budget closed-loop smoke/pilot for `paper1/docs/ROBUST_CEM_CODEX.md`; uses `world.num_envs=1` and timeout-guarded evals; validates implementation integration but is negative for a main-method claim at this budget)
- **Robust CEM rank-based iteration**: `assets/paper1_data/robust_cem_iteration_20260705.json` and `paper1/docs/ROBUST_CEM_ITERATION_LOG_20260705.md` (ACPC/ranking-aligned final robust rerank over frozen origin LeWM checkpoints; moderate positive on TwoRoom Gaussian std=0.08 against standard and compute-matched CEM, with clean-drop caveat)
- **Robust CEM 100x3 iteration summary**: `assets/paper1_data/robust_cem_eval100x3_iteration_summary_20260705.json` and `paper1/docs/ROBUST_CEM_EVAL100X3_ITERATION_LOG_20260705.md` (frozen origin LeWM planner-side Robust CEM reranking at `num_eval=100 x 3`; finds only a borderline TwoRoom gain for `rank_vote + elite_mean`, no compute-matched separation, and a negative Reacher seed42 check, so the route is weak/no-go for a main Paper1 method claim)
- **Validation remediation summary**: `assets/paper1_data/prospective_validation_summary.json` and `.md` (summarizes three-training-seed Gaussian behavior, development/held-out fixed-rule diagnostic validation, the archived task-state proxy margin pass-rate, and bounded unseen-stressor scope checks)
- Scope: the current Gaussian-sweep analysis contains 108 LeWM and 108 PLDM checkpoints = 3 training seeds × 4 tasks × 9 configs (`base` + `std_max` 0.01..0.08) per model family. The original 36-checkpoint aggregates remain as seed-3072 provenance artifacts. Strongest-only unseen score artifacts cover LeWM training seeds 3072/3073/3074.
- Evaluation protocol: canonical Gaussian grids use **3 evaluation seeds** (`42`, `43`, `44`) × **100 trajectories per seed** for each checkpoint/seed point; strongest-only unseen score artifacts store `eval_base_seed=42`, `eval_seeds=3`, and `num_eval=300`.
- Seed clarification: both current LeWM and PLDM Gaussian-sweep figures report independent training seeds 3072/3073/3074. Evaluation seeds 42/43/44 quantify within-checkpoint evaluation variability and are not substitutes for training runs.

## Current three-seed PLDM extension (2026-07-23)

The current manuscript extends the archived one-family PLDM audit to three independently trained checkpoint families while keeping the frozen metric implementation and protocol unchanged. Threshold selection is performed within PLDM: for each evaluation task, the other three PLDM tasks select the IR--SR thresholds, and the selected pair is applied unchanged to the held-out task. The pooled 108-row result has balanced accuracy `0.681`, precision `0.667`, and recall `0.786`; the equal-task mean balanced accuracy is `0.699`.

| Artifact | Role | SHA-256 |
|---|---|---|
| `training_seed_eval_manifests/pldm_seed3073_evals.json` | Seed-3073 checkpoint-bound diagnostic manifest | `c65e963c8f82ad82700916ccbcdbcd25b02739296e934411a1bcded1466ce231` |
| `training_seed_eval_manifests/pldm_seed3073_evals_legacy.json` | Seed-3073 behavior aggregate | `bbfd35e28cc77bbbfad59340648fdc393c6d47d5882e19a657db3e5c26dfe5b3` |
| `training_seed_eval_manifests/pldm_seed3074_evals.json` | Seed-3074 checkpoint-bound diagnostic manifest | `3a3c9796dd39706506f83ac7a300b0fbd39c39b0aa8f632a5f5e00fc02612604` |
| `training_seed_eval_manifests/pldm_seed3074_evals_legacy.json` | Seed-3074 behavior aggregate | `3e0a8d8a12be1872eb94c9c74479dc07c9a572894b900ad8bff06414c30994d1` |
| `paper1/results/external_validation/pldm_frozen_rows_v2.csv` | Paper-facing 108-row PLDM diagnostic/behavior join | `a9dc22e467ec756ebf527e37a60cb6959ec676c9df9519906a1e7f112823fbe9` |
| `paper1/tables/table_pldm_architecture_portability_ir_sr_v2.tex` | Three-run architecture-portability summary | `cf1e68172beef19f110cfb653058f0f1b5f1aeaada2f9eb8b6b6769e22ccbf04` |
| `assets/paper1_figs/fig_pldm_sweep_diagnostics.pdf` | Three-run PLDM sweep figure | `634a2a17a1df9779f4b19a8de795a05b288e417a76555e0f96bba2bb0844b7ac` |

## Public-v1 remediation artifacts (2026-07-11)

The public-v1 diagnostic protocol is immutable at SHA-256 `edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21`. It was calibrated only on the LeWM seed-3072 Gaussian sweep. E1--E4, component baselines, certificate calibration, JVP/linearization audits, and SMPR controls read this protocol without threshold or severity search.

- **E1 strict held-out LeWM validation**: `paper1/results/frozen_external_validation_summary_v3.json` covers seeds 3073/3074 (72 checkpoint rows). The frozen gate obtains balanced accuracy/AUPRC `0.945/0.965`; these are independent training seeds.
- **Archived E2 PLDM architecture portability**: `paper1/tables/table_pldm_architecture_portability.tex` preserves the immutable public-v1 result for one independently trained four-task PLDM checkpoint family (36 rows). PLDM-local and task-relative LeWM-source calibration both gave BA/precision/recall `0.836/0.789/0.882` on that grid. The current three-seed extension above supersedes this result in the manuscript without rewriting the archived artifact.
- **E3 fixed-rho blur/resize absolute and paired tests**: `paper1/results/external_validation/cross_stressor_fixed_rho_summary.json` retains all 16 pre-mapped absolute endpoints and all 32 base-to-endpoint pairs from LeWM and canonical PLDM. The absolute frozen endpoint slice remains conservative (`0.563/0.737`, recall `0.375`). Within LeWM, one untuned `delta_joint_score > 0` rule shared across all 24 blur/resize pairs reaches balanced accuracy/AUPRC `0.889/0.974`, precision/recall `0.882/1.000`, Spearman `0.864`, and signed agreement `0.917`; a 5,000-repeat bootstrap over 12 task-training-seed blocks gives Spearman 95% interval `[0.641,0.932]`. The post-freeze audit adds an exact 12-block sign-flip `p=0.000244`, leave-one-task/seed remaining-set Spearman ranges `[0.748,0.897]`/`[0.753,0.940]`, and `22/24` correct stressed-success checkpoint choices (`16/16` for changes of at least 5pp) with mean regret `0.139pp`. The two classification false positives are neutral `+4pp` rows and have zero selection regret. Component bootstrap contrasts retain the boundary that joint/H8 is not uniquely superior. PLDM has one training family and supports analysis-interface portability, not shared absolute calibration.
- **E4 failed-repair falsification**: `paper1/results/external_validation/target_view_frozen_summary.json` retains both full-sequence and target-view branches. The target branch has no positive behavior rows and is rejected throughout, while the full-sequence branch has a high false-pass rate; target-view is therefore a failure control, not a successful repair.
- **Matched component baselines**: `paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json` contains 272 complete, behavior-blind rows on the same anchors/noise draws. Encoder-only, H1, action-zeroed, action-shuffled, and time-shuffled radii are competitive with correct-action H8, so the release does not claim empirical necessity of a long horizon or correct-action conditioning. `paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json` is a privileged Gaussian-only confound audit and is not an external baseline.
- **JVP and linearization**: `paper1/results/jvp_hutchinson_sensitivity_audit_v2.json` contains 36 complete checkpoint rows and 288 probes with separate bounded submultiplicative and relative-isotropic alignment quantities. `paper1/results/linearization_horizon_sensitivity_v1.json` uses the same 16 anchors, weighted H8 map, pixel-noise covariance, eight JVP probes, and small-noise draws for all 36 base/onset/endpoint rows; it also reports H in `{1,2,4,8}` and q in `{0.80,0.90,0.95}`. The local approximation is visibly nonlinear for part of the range and is not a global guarantee.
- **Sharp fixed-pool certificate**: `paper1/results/fixed_pool_candidatewise_certificate_summary.json` aggregates 108 checkpoints and 10,800 histories. Coarse/sharp coverage are `0.478/0.700`, observed top-1 flip is `0.208`, and flip conditional on sharp-cert failure is `0.694`. Zero flips on cert-pass are a deterministic invariant check, not independent evidence. The result applies only to the sampled ordered candidate pool.
- **SMPR sensitivity and controls**: `assets/paper1_data/smpr_sensitivity_v2.json`, `smpr_controls_v2.json`, and `smpr_oracle_guard_v2.json` report positive-margin sensitivity, constant-collapse/identical-view/action/label/pair controls, and a TwoRoom+PushT MVE. Constant collapse is rejected, but progressive-collapse, action-relevance, and task-label increments are not established. TwoRoom uses an environment-geometry proxy; PushT uses a sequence-goal state-derived proxy because the released HDF5 lacks simulator goal state. These are proxy-level guards, not four-task oracle semantics.
- **SMPR-v2 proxy metadata correction**: `paper1/results/smpr_v2_proxy_metadata_correction_v1.json` records the exact state-coordinate slices executed by the frozen SMPR-v2 artifacts. It corrects legacy descriptive strings for Reacher and Cube without changing numeric rows or pair indices. The paper therefore treats SMPR as a programmatic state-coordinate guard, not a semantic or action-relevance certificate.

| Artifact | Role | SHA-256 |
|---|---|---|
| `paper1/config/frozen_diagnostic_protocol_v1.json` | Immutable public-v1 metric/gate protocol | `edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21` |
| `paper1/results/frozen_external_validation_summary_v3.json` | E1 held-out LeWM frozen validation | `ec485a7026c1d2ff80295f4dc85dd3753ca12f2ede7d7c0137a13796070dfeba` |
| `paper1/tables/table_pldm_architecture_portability.tex` | E2 one-family PLDM architecture-portability summary | `b7f84b19896d20f23775d9bb75faddd91680b227f689b67a699ddc49c304ba83` |
| `paper1/results/external_validation/cross_stressor_fixed_rho_summary.json` | E3 absolute/paired blur-resize validation plus post-freeze robustness audit | `94077f772e8dd7641b47e161a17d4ec67cea695dc044cb9a0229857efc157453` |
| `paper1/tables/table_cross_stressor_robustness_audit.tex` | Generated E3 exact/deletion/selection audit table | `f97774bed394eaaf979ed2c8f835d73a999249489bb03a12c49c52dc4c9614bf` |
| `paper1/results/external_validation/target_view_frozen_summary.json` | E4 failed-repair falsification | `dba255daf282d1dbea7a102839e054cdd39b159a08a9ea9b1d3def7767477870` |
| `paper1/results/diagnostic_baselines/diagnostic_baseline_all_v1.json` | Matched component-baseline benchmark | `df43cfd80b0387bde31426a37445149646a247724c1b2dd61f801a97d6c4f3c8` |
| `paper1/results/diagnostic_baselines/gaussian_rho_confound_summary.json` | Privileged Gaussian-only rho audit | `3079d357d7dcca2643dc0a4ef9bb3297bf1de49cfcfae51ae632bdc272ed3591` |
| `paper1/results/jvp_hutchinson_sensitivity_audit_v2.json` | Matched-map exact-JVP audit | `50cd24027772129a1d594299ec50c2df908894f9a32f3790f43e2b5273936cc5` |
| `paper1/results/linearization_horizon_sensitivity_v1.json` | Covariance-aware linearization and H/q audit | `4697d6ac006348803b3597af15f276a4c0233b4c65081fd13164e7f5629e5ee1` |
| `paper1/results/fixed_pool_candidatewise_certificate_summary.json` | Sharp certificate, K sensitivity, and block bootstrap | `ccd8c0e35fa18cd4b1af4b6f43ed70af64ac21b97abfd4cf73e3cb69789e67fe` |
| `assets/paper1_data/smpr_sensitivity_v2.json` | Positive-margin sensitivity grid | `c61df36c5a3eeeed0749f7f6f4d84dd50a91ec3a60a643ad2097fc4cbe54b5c8` |
| `assets/paper1_data/smpr_controls_v2.json` | Collapse/action/label/pair controls | `55a53c5e8036bcca2e8be82186bbad665531e1cefc7a5a5a8045c4313d68cdf2` |
| `assets/paper1_data/smpr_oracle_guard_v2.json` | TwoRoom+PushT state-derived proxy MVE | `1224514237a958121e9e5a7d26515d98cf5c2bf59017c65060a17ba1ccd31203` |
| `paper1/results/smpr_v2_proxy_metadata_correction_v1.json` | Description-only correction for frozen SMPR-v2 proxy labels | `17d5080b3f7a713ad5d45288d0c3406fc5c759561d5fd83674c2697d9b368a5a` |

## Release Provenance Notes

- Paper-facing main evidence: `canonical_evals_20260517.json`, `three_seed_gaussian_sweep_summary_20260706.json`, `training_seed_gaussian_lockbox.json`, `acpc_phase0_lewm_three_seed.json`, `three_seed_diagnostic_validation.json`, `selector_plateau_audit_20260704.json`, `margin_flip_curve_lewm_three_seed.json`, `semantic_task_grounded_margin_lewm_three_seed.json`, `cem_trace_audit_20260704.json`, `prospective_validation_summary.json`, `unseen_phase0_acpc_fullstress.json`, `canonical_diagnostics_20260517.json`, `acpc_basin_diagnostics.json`, `canonical_evals_pldm_20260522.json`, `canonical_diagnostics_pldm_20260522.json`, `canonical_full_diagnostics_pldm_20260523.json`, `acpc_basin_diagnostics_pldm.json`, `partial_corr_bootstrap_20260523.json`, and `acpc_phase0_clean_goal_seed9101.json`.
- Scope-boundary / sanity artifacts: `canonical_blur_baselines_20260523.json` is eval-only blur stress; `acpc_phase0_diagnostics.json` is the archived observation+goal Phase-0 sanity run; `target_view_closed_loop_summary.json` is a negative target-view ablation; `canonical_external_baselines_20260520.json` is retained for backward-compatible sanity checks. The strongest-only unseen score artifacts support the three-training-seed unseen score aggregate in `prospective_validation_summary.json`; `unseen_phase0_acpc_subset.json` supports the matched selected three-training-seed appendix diagnostic slice, and `unseen_phase0_acpc_fullstress.json` supports the full blur/resize 24-row appendix scope audit. Margin-conditioned action-flip rates are reported in `margin_flip_curve_lewm_three_seed.json`; the older `semantic_task_grounded_margin_lewm_three_seed.json` records a legacy programmatic proxy and is not used as oracle evidence. The v2 controls add a TwoRoom environment-geometry proxy and a PushT sequence-goal state-derived proxy MVE. Simulator-derived contact/topology/action-value/cost-to-go labels across all four tasks remain future work.
- Contamination fix: the 2026-06-10 audit found that the TwoRoom and PushT representative diagnostic rows in `canonical_diagnostics_20260517.json` duplicated heteroscedastic-loss diagnostics. The affected representative fields were re-extracted from the intended per-checkpoint `diagnostics_summary.json` files. The release checker now guards these values so the PushT noise-sweep row cannot regress to the heteroscedastic `rank 76.4 -> 42.9` narrative.
- Manual revision status: no released numerical JSON result row is hand-edited for paper prose. The representative-diagnostics re-extraction above is recorded in JSON metadata and checked by `tools/check_paper1_consistency.py`; the strongest-only unseen score artifacts received a status-only metadata update after the three-seed score aggregate audit, with numeric rows unchanged.

| Artifact | Role | SHA-256 |
|---|---|---|
| `canonical_evals_20260517.json` | LeWM evaluation aggregate | `394c21142311e628232e510d0087b17828ab78d973727cff6b049cb50ed98e1a` |
| `canonical_diagnostics_20260517.json` | LeWM diagnostics and representative rows | `8012fd3bc5fb445bd5d00ea78d3c5df30f57c10f69b5175f8479b48370517443` |
| `acpc_basin_diagnostics.json` | LeWM Gaussian ACPC basin | `e0e468d2d7a94666e6bbcf8dafd24f32fe2dade03d66deb5619a86145a8dc521` |
| `canonical_evals_pldm_20260522.json` | PLDM evaluation aggregate | `e9bf3a49b91f3d17151db2cd94c696cd56f53d8c2be9670351f059ebb671df7a` |
| `canonical_diagnostics_pldm_20260522.json` | PLDM predictor diagnostics | `efb15cc8baafd1f80e5d5b67ffce59666ef909f7a4106dd50cc6a7c9fcf4c536` |
| `canonical_full_diagnostics_pldm_20260523.json` | PLDM five-layer diagnostics | `6a5b2ae47b09b4bd6fd6fba87846e7d9484e6beea2cfe24f75380c238c73fc7d` |
| `acpc_basin_diagnostics_pldm.json` | PLDM Gaussian ACPC basin | `dd6aeaa3e793ce09294b049b31ff2f7791c7b83fe0a6a375e6adba806f23e6e2` |
| `partial_corr_bootstrap_20260523.json` | Bootstrap CI aggregate | `e6cbba0893defd152b540150dcf86ee091fbe5cf4061131871228b6a59d51465` |
| `acpc_phase0_clean_goal_seed9101.json` | Clean-goal Phase-0 ACPC/PCC/CRA/MAF diagnostics | `0207daddc972bfd66829d5a521101284daed3cc60933046fa13766da73cde021` |
| `heldout_selection_phase0_seed9101.json` | LeWM component for clean-goal Phase-0 diagnostics | `4d8daf17a0d43b57c996bd3fc0ab4b1cba830844c9bbb8dae0b002cee2fb409c` |
| `heldout_selection_phase0_pldm_seed9101.json` | PLDM component for clean-goal Phase-0 diagnostics | `e46db5f49011ca199f213096a813db1ca69c2c6023ffde123d11f60dce751f91` |
| `acpc_phase0_diagnostics.json` | Archived observation+goal Phase-0 sanity run | `9654759b576216b7249a3bf5e2ee7b778318cf4de22babf3395a0757b3e644fd` |
| `target_view_closed_loop_summary.json` | Negative target-view ablation | `04f75ad72543fb98d51304a6dec12ceb1b2dc099e915e24a49862ed3451744d0` |
| `canonical_blur_baselines_20260523.json` | Eval-only blur sanity check | `8e4c18d9f354a585770e6eb389e4ceb1b449eea7a5e7758af40a324878e0700b` |
| `unseen_origin_vs_std008_strongest_tworoom.json` | Seed-3072 strongest-only unseen score, TwoRoom | `af0a4329155b2bd4acd59964318ebab2935f8a48f145e64279d5130fc709aa68` |
| `unseen_origin_vs_std008_strongest_reacher.json` | Seed-3072 strongest-only unseen score, Reacher/Cube/PushT | `90debe607cd302c31707890375c500ca2e62a4feeb05bc61f580f2dc5097566c` |
| `unseen_origin_vs_std008_strongest_s3072.json` | Seed-3072 strongest-only unseen score wrapper, all tasks | `5b12fd4d19484199178b13a3eae3c12867d6dd4a218b431d03ea69666eaaf0dd` |
| `unseen_origin_vs_std008_strongest_s3072_manifest.json` | Seed-3072 strongest-only unseen score wrapper manifest | `0f1cb6ed765e96f479dc5b60fe4c81d2ab05a9220fb1f09392cc389bb8aa1017` |
| `unseen_origin_vs_std008_strongest_s3072.schema.json` | Seed-3072 strongest-only unseen score wrapper schema | `525d59956dc1e8180e795c23f5667c552e05bb4155808f7fb317d7571f0d13da` |
| `unseen_origin_vs_std008_strongest_s3073.json` | Seed-3073 strongest-only unseen score lockbox | `c933785bc39b4ac556fbe69a3b00e3451402accfd3b64f907b8a29f306b8636a` |
| `unseen_origin_vs_std008_strongest_s3074.json` | Seed-3074 strongest-only unseen score lockbox | `243ef17d77ed2e72adfd6fbdf16459ed09dd33558869c31421c562ae520db637` |
| `unseen_phase0_acpc_subset.json` | Unseen Phase-0 ACPC subset review artifact | `7f8ff8a8f85c170b7eed0abdf16109e0f2d4f0f94f66dd498c5ab76a1c314ead` |
| `unseen_phase0_acpc_subset.schema.json` | Unseen Phase-0 ACPC subset schema | `8f688e77a27fbf69bb750bf26f90e8745b0b2e369ad6e62128b4a3040095b85f` |
| `unseen_phase0_acpc_fullstress.json` | Full blur/resize unseen Phase-0 ACPC scope audit | `77636682649c1f7a7099eca6fe7d2ba86244550a4c54571551d88298cd1ca51c` |
| `unseen_phase0_acpc_fullstress.schema.json` | Full blur/resize unseen Phase-0 ACPC scope-audit schema | `8f688e77a27fbf69bb750bf26f90e8745b0b2e369ad6e62128b4a3040095b85f` |
| `training_seed_gaussian_lockbox.json` | Three-training-seed Gaussian replication summary (legacy filename) | `526ff2fad2ba86c3c865341ae4bc9db9fad52f4825bf12818c3ab54a2af4aabb` |
| `three_seed_gaussian_sweep_summary_20260706.json` | Three-training-seed full Gaussian sweep summary | `1ce15c4e5090e0caa6c9eee3a250c3406a8892dab0462534ac7dd3f95ffae30f` |
| `three_seed_gaussian_sweep_summary_20260706.md` | Human-readable three-training-seed full Gaussian sweep summary | `c781fe567ac5b24583d2b81c19653085f4af2911d2ed0a74f9ab53ad34b40bb3` |
| `training_seed_eval_manifests/lewm_seed3072_evals.json` | Seed-3072 LeWM eval manifest for three-seed Phase-0 | `0f34532e78bf228eaa3f470b5728fb28fc88d34f47b40e728c7aadab6b4397c4` |
| `training_seed_eval_manifests/lewm_seed3073_evals.json` | Seed-3073 LeWM eval manifest for three-seed Phase-0 | `03cfcf889825cf2022a3b0e3d54a67e1f70e412b8d704226609cbe89360b0b5b` |
| `training_seed_eval_manifests/lewm_seed3074_evals.json` | Seed-3074 LeWM eval manifest for three-seed Phase-0 | `7a7dbd8c37992cf12f5b77bf1f7f841e9314c7e9eb31ddf67ce8afa4b5699316` |
| `acpc_phase0_lewm_seed3072.json` | Seed-3072 LeWM Phase-0 full-grid diagnostics | `1163c0f033fb8ce5b6afc9307d52db4bec4ad9f6f16a62f048defc4865e578ba` |
| `acpc_phase0_lewm_seed3073.json` | Seed-3073 LeWM Phase-0 full-grid diagnostics | `77d78b4a3193779b9b7d8d6a2e4db0f4d280ad3b408abeb0edd1f686e78a1394` |
| `acpc_phase0_lewm_seed3074.json` | Seed-3074 LeWM Phase-0 full-grid diagnostics | `ef68d50e294df3b8cce795f47f0adba33803fb964f043ff1ed95fb4e4de50cb6` |
| `acpc_phase0_lewm_three_seed.json` | Three-seed LeWM Phase-0 full-grid diagnostics | `8742c9dfbbb45f884fc4aa93d1adecfe03ca9d7b9f037e959b5ddda988d4e989` |
| `three_seed_diagnostic_validation.json` | Development/held-out three-seed fixed-rule diagnostic validation | `6275b0ea68e367aa302dbe1e19f1ab96f2a0485aee28d656fce9d8e028025ef0` |
| `selector_baseline_audit_20260704.json` | Legacy selector-baseline audit retained for provenance only | `95d7ffc04949b6c93f567ab6f1097280e963865fbe341159fdd38630b01e6219` |
| `selector_baseline_audit_20260704.md` | Human-readable legacy selector-baseline audit summary | `1584f65093c1328a970c25d79efb57cea7e5f7f4752d9a50cbf12431e0393ea5` |
| `selector_plateau_audit_20260704.json` | Plateau-membership diagnostic screen audit | `30a1dd59e8e941adbd567d1e8ffcd323b8aa836eff98e67d0b51c2566ff722ce` |
| `selector_plateau_audit_20260704.md` | Human-readable plateau-membership audit summary | `0292f581098381a978beea0fe0c22e297c87cd9694ac4888316daedd42921401` |
| `residual_diagnostic_audit_20260704.json` | Residual diagnostic audit controlling for std, task, and training seed | `304db7d19113f68ca2ca4299768203653184af0d02b8b50b2232aa75c26da75a` |
| `residual_diagnostic_audit_20260704.md` | Human-readable residual diagnostic audit summary | `1a5bf140616b3cee27a6459456fbda2229752126cf33cf28d3b4123603a269c3` |
| `selector_incremental_audit_20260704.json` | Legacy incremental selector audit with std/task/seed controls | `b9c3c5140746b84c45323329019caee02f2fa1a962d301fe2908dc5b2d003645` |
| `selector_incremental_audit_20260704.md` | Human-readable legacy incremental selector audit summary | `e8d18f895e95d8e4f8824bdb50c0f552f862a1123a783b53d2f938553ff44f71` |
| `semantic_margin_passrate_lewm_three_seed.json` | Three-seed median-distance task-state proxy margin sanity pass | `b4cea993839043f7a5759fbed9645dc290fc4373b3203aa5ca96442b81a1c3fa` |
| `margin_flip_curve_lewm_three_seed.json` | Three-seed margin-conditioned action-flip audit | `d5bcdc95acfdd2d2830ce7838f4d19168769b379876282008f6f0c570b988f88` |
| `semantic_local_margin_lewm_three_seed.json` | Three-seed local task-feature proxy margin pass-rate | `44f8df413d6ccfab47c4542eece1b30940ef76ed026c52b2d60b78866b7bc580` |
| `semantic_task_grounded_margin_lewm_three_seed.json` | Three-seed task-grounded near-boundary proxy margin pass-rate | `121a2a0c8cf2e4e4b9dde2a66fdb242d8fa70a2188f31c3166120642cc278f96` |
| `semantic_task_grounded_margin_lewm_three_seed.md` | Human-readable task-grounded near-boundary proxy margin summary | `5a957d4654999ff2b2b44bbeec6a97e096b12718efaa694cb8484cbc50ed2d0b` |
| `cem_trace_audit_20260704.json` | Reduced-budget offline CEMSolver trace audit | `03d26cdf985034b8cd7cb4491087456290b60ee89f384ffe7a60d9b68ca62f5e` |
| `cem_trace_audit_20260704.md` | Human-readable reduced-budget CEM trace summary | `7c97ed9dd6686c435d3392b2b9aec2fe531df22c9c09e11f6f8c163f08cd74e9` |
| `prospective_validation_summary.json` | Validation remediation summary, fixed-rule diagnostic split, archived task-state proxy margin pass-rate, and unseen scope check | `6534a97cf6130350f47a36404c75f1e9cca2de12dd2f5da5682cb854bad8a6d5` |

## Data Semantics

- Unit: every success-rate field is stored in **percentage points** on `[0, 100]`
- Aggregation: each metric stores `values = [seed42, seed43, seed44]`, `mean`, and `std`
- `std` convention: the JSON stores the **population standard deviation** across the 3 evaluation seeds (`ddof = 0`)
- Conditions released per checkpoint:
  - `clean`
  - `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
  - `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
  - `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`
- Paper 1 primary corrupted endpoint: `pixels_std0.08` (observation pixels corrupted, goal image kept clean). `pixels_goal_std0.08` is retained as a stronger full-visual-stream stress condition.
- Canonical lookup key for portability: use `subdir`
  - The JSON also stores an absolute local `path`, but downstream tools should not rely on that exact absolute prefix

## Per-Seed File Pattern

For every checkpoint subdirectory below, the raw per-seed metrics are expected at:

`<ckpt>/eval_results/<cond>_seed42_metrics.txt`

`<ckpt>/eval_results/<cond>_seed43_metrics.txt`

`<ckpt>/eval_results/<cond>_seed44_metrics.txt`

where `<cond>` is one of:

- `clean`
- `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
- `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
- `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`

## Task Roots

Set `DATA_ROOT` to the machine-local prefix that contains the released
`lewm-*` task directories. Keep that prefix outside the manifest and commands;
the released task roots below stay relative to `DATA_ROOT` so the same
artifacts can be moved between machines.

| Task | Local ckpt root |
|---|---|
| TwoRoom | `$DATA_ROOT/lewm-tworooms/ckpt` |
| PushT | `$DATA_ROOT/lewm-pusht/ckpt` |
| Reacher | `$DATA_ROOT/lewm-reacher/ckpt` |
| Cube | `$DATA_ROOT/lewm-cube/ckpt` |

## Released Checkpoints

| Task | `std_max` | Canonical `subdir` | Raw per-seed file pattern |
|---|---:|---|---|
| TwoRoom | 0.0 | `tworoom_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.01 | `tworoom_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.02 | `tworoom_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.03 | `tworoom_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.04 | `tworoom_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.05 | `tworoom_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.06 | `tworoom_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.07 | `tworoom_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.08 | `tworoom_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.0 | `pusht_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.01 | `pusht_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.02 | `pusht_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.03 | `pusht_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.04 | `pusht_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.05 | `pusht_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.06 | `pusht_lewm_noise_0to006_p1_20260507` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.07 | `pusht_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.08 | `pusht_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.0 | `reacher_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.01 | `reacher_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.02 | `reacher_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.03 | `reacher_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.04 | `reacher_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.05 | `reacher_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.06 | `reacher_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.07 | `reacher_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.08 | `reacher_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.0 | `cube_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.01 | `cube_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.02 | `cube_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.03 | `cube_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.04 | `cube_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.05 | `cube_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.06 | `cube_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.07 | `cube_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.08 | `cube_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |

## Consumer Notes

- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_evals_20260517.json` as the source of truth for the script-generated main evaluation figures and tables.
- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_diagnostics_20260517.json` as the source of truth for rollout-diagnostic implementation keys, the representative diagnostic table (`tab:diag-base-vs-best`), and the PushT fragility scatter (`fig:scatter`).
- The canonical diagnostics release stores:
  - 4 tasks × 9 ckpts of `predictor_target_to_nn_cos_ratio_at_max_std`
  - 4 tasks × 9 ckpts of `predictor_rollout_T8_l2_at_max_std`
  - finalized representative diagnostic values used by `tab:diag-base-vs-best`
  - published LeWM cross-checkpoint correlation and partial-correlation values
- The historical external baseline release stores one PushT PLDM run trained without input-noise augmentation (`pusht_pldm_baseline`) for backward compatibility. The original 36-checkpoint PLDM aggregate in `canonical_evals_pldm_20260522.json` is retained as the seed-3072 source.
- The current PLDM Gaussian sweep combines that source with the seed-3073/3074 manifests and frozen diagnostic artifacts listed above. It uses the same condition names, success-rate units, and evaluation seeds as the LeWM sweep while keeping thresholds model-family-specific.
- `canonical_full_diagnostics_pldm_20260523.json` stores the full `diagnostics_summary.json` row for every PLDM checkpoint and a compact base-vs-representative table used by the PLDM appendix. It is interpreted as a mechanism-boundary check: PLDM replicates the task-level fragility/recovery signature but does not reuse LeWM's exact compression-chain profile.
- `acpc_basin_diagnostics_pldm.json` stores the full PLDM 4 tasks × 9 configs Gaussian ACPC basin replication. It uses the same same-state clean/noised view protocol as `acpc_basin_diagnostics.json`; the PLDM appendix reports a compact baseline-vs-pixels-0.08-point-best summary from this full artifact.
- `partial_corr_bootstrap_20260523.json` stores the 95% percentile bootstrap CIs for the LeWM, PLDM, and joint partial-correlation claims quoted in the main text and PLDM appendix. It is generated by `tools/build_partial_corr_bootstrap.py` from the canonical eval/diagnostic JSONs.
- `acpc_phase0_clean_goal_seed9101.json` stores the clean-goal Phase-0 paired ACPC diagnostics (ACPC-1/H, PCC, CRA, MAF, ADM action-distance proxy, SPRR) for the LeWM and PLDM full sweep. It uses Gaussian observation-history noise at std 0.08 while keeping the goal image clean, matching the paper's primary `pixels_std0.08` endpoint. It is an appendix reference file for the paired-diagnostic definitions and is not used as the paper-facing held-out selector benchmark. The two component files `heldout_selection_phase0_seed9101.json` and `heldout_selection_phase0_pldm_seed9101.json` record the separate LeWM and PLDM runs merged into this release artifact. The older `acpc_phase0_diagnostics.json` is retained as an archived observation+goal sanity run.
- `acpc_basin_diagnostics.json` stores the paired Gaussian-noise ACPC basin diagnostic for all 36 LeWM canonical checkpoints. For each checkpoint it uses clean plus noised views at std 0.01..0.08, all rolled out under the same recorded action sequence. The main summary fields are `encoder_view_pair_l2_norm_by_nn`, `pred_view_pair_l2_norm_by_transition`, and `basin_contraction_pair_norm`. This artifact is intentionally Gaussian-noise-only to match the training sweep family; blur/resize are not mixed into the ACPC basin evidence.
- `unseen_phase0_acpc_subset.json` stores the matched unseen diagnostic subset: TwoRoom/Reacher blur as positive-transfer cases and PushT/Cube resize as boundary cases, each comparing `std_key` 0.0 vs 0.08 for training seeds 3072/3073/3074 with clean-goal Phase-0 paired diagnostics. It joins diagnostics with strongest-only unseen eval scores and is a review artifact for matched diagnostic analysis.
- Some released JSON rows retain the absolute checkpoint paths from the machine that produced the artifact. Treat those fields as historical provenance only. Portable reruns should resolve checkpoints from `DATA_ROOT` plus the relative task roots above.
- `canonical_blur_baselines_20260523.json` stores blur evals of LeWM/PLDM baselines trained without input-noise augmentation for kernel sizes 3/7/11/15 on `pixels`, `goal`, and `pixels_goal`. This is an eval-only cross-corruption sanity check for the blur appendix; it is not a blur-training sweep and is not mixed into the Gaussian-noise canonical tables.
- `selector_baseline_audit_20260704.json` stores a legacy selector-baseline audit for the three-training-seed Gaussian grid. It is retained for provenance only and is not used as paper-facing selector evidence.
- `selector_plateau_audit_20260704.json` stores the paper-facing plateau-membership audit for the same 96 nonzero LeWM rows. It treats `std_max` as a candidate label inside each task-training-seed block, defines a best-5pp practical plateau, and evaluates whether diagnostic top-half screens are enriched for plateau members rather than ranking checkpoints inside a plateau.
- `residual_diagnostic_audit_20260704.json` and `selector_incremental_audit_20260704.json` store legacy residual/control audits for the same 96 nonzero LeWM rows. They are retained for provenance and are not used as paper-facing selector evidence because the manuscript treats `std_max` as a candidate label inside each task-training-seed block rather than as a continuous point-prediction covariate.
- `semantic_task_grounded_margin_lewm_three_seed.json` stores the legacy task-grounded near-boundary proxy margin audit. It selects closest task-label-crossing neighbors inside the closest 35% state-distance neighborhood; it and the older local/global proxy files are retained for provenance and sanity checks. Public-v1 claim decisions use `smpr_sensitivity_v2.json`, `smpr_controls_v2.json`, and `smpr_oracle_guard_v2.json`, which establish constant-collapse rejection but not incremental action, label, or four-task oracle relevance.
- `semantic_task_grounded_margin_lewm_full_sweep_20260708.json` and `paper1/results/prospective_diagnostic/diagnostics_all_ckpts.csv` provide the full three-training-seed ATR/SMPR sweep inputs used by `paper1/results/diagnostic_region/`. The diagnostic-region tables summarize shared low-ATR/high-SMPR regions, within-block direction consistency, and recovered-vs-fragile separation; they are not used as a threshold-classifier F1/IoU ranking.
- `paper1/results/radius_margin_boundary_alignment.csv` and `paper1/results/fixed_pool_top1_agreement.csv` are generated by `tools/paper1_boundary_mechanism_audit.py` from the existing radius-margin summary CSVs. They provide the appendix boundary-aware recovery/proxy alignment and fixed-pool Top1Agree audit; they do not introduce new closed-loop evaluations or adaptive-CEM guarantees.
- `cem_trace_audit_20260704.json` stores the reduced-budget offline CEMSolver trace audit over base/std0.08 endpoints and training seeds 3072/3073/3074. It is a planner-side sensitivity diagnostic over logged states, not a closed-loop evaluation or full adaptive CEM guarantee.
- `tools/check_paper1_consistency.py` verifies:
  - the JSON exists
  - the released structure is 4 tasks × 9 configs
  - each config contains `clean`, `pixels_std0.05`, `pixels_std0.08`, `pixels_goal_std0.05`, and `pixels_goal_std0.08` with `mean`/`std`
  - the stored `mean`/`std` agree with the 3 released seed values
  - the historical external PLDM sanity-check aggregate is trained without input-noise augmentation and recomputes its reported means/stds/drop
  - the full PLDM aggregate is 4 tasks × 9 configs, contains required eval metrics, and recomputes means/stds from the three released seed values
  - the PLDM full-diagnostics aggregate is 4 tasks × 9 configs and contains the required five-layer diagnostic fields
  - the bootstrap CI aggregate exists, has the expected n=9 / n=18 scopes, and reproduces the headline PushT CI values quoted in the paper
  - the clean-goal Phase-0 ACPC aggregate is 2 methods × 4 tasks × 9 configs, all rows are `ok`, `corrupt_goal=false`, and the paired diagnostic fields used by the Phase-0 appendix are finite
  - the ACPC basin artifact covers LeWM 4 tasks × 9 configs, contains only Gaussian-noise std 0.01..0.08 variants, and stores finite encoder/prediction basin radii
  - the PLDM ACPC basin replication covers the full 4 tasks × 9 configs grid, all `ok`
  - the blur sanity-check aggregate covers 2 methods × 4 tasks × 12 blur conditions and recomputes means/stds/worst-blur drops from the three seed values
