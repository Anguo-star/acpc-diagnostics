# Paper 1 data manifest

This manifest defines the reader-facing evidence boundary for
**Diagnosing JEPA World Models with Action-Conditioned Predictive
Consistency**. The public repository contains curated aggregate inputs,
frozen protocols, generated tables/figures, and selected raw audit shards. It
does not contain model checkpoints, HDF5 datasets, manuscript source, or every
intermediate file produced during development.

## Statistical units

- The LeWM Gaussian sweep contains 4 tasks × 9 training settings × 3
  independently trained checkpoints (training seeds 3072, 3073, and 3074),
  for 108 checkpoint rows.
- Each reported planning success rate is evaluated with seeds 42, 43, and 44.
  These evaluation seeds are repeated trials of a checkpoint, not additional
  training seeds.
- The PLDM sweep contains 4 tasks × 9 training settings with one independently
  trained PLDM run per setting. Its three evaluation seeds must not be
  interpreted as three PLDM training runs.
- Success rates are stored in percentage points on `[0,100]`. Where a row
  stores a standard deviation over the three evaluation seeds, it uses the
  population convention (`ddof=0`).
- Gaussian training settings are the unaugmented baseline plus
  `std_max={0.01,0.02,...,0.08}`.

## Primary released inputs

| Path | Role |
|---|---|
| `paper1/results/full_sweep_diagnostics.csv` | 108-row LeWM behavior and checkpoint-diagnostic sweep |
| `paper1/results/full_sweep_diagnostics_summary.csv` | Task/seed-level sweep summary |
| `paper1/results/future_drift_three_seed_v1.csv` | Pair-level recorded-action future-error-drift experiment |
| `paper1/results/future_drift_three_seed_summary_v1.json` | Cross-validated future-drift summary |
| `paper1/results/acpc_planner_stability_v4/summary.json` | Planner-horizon ACPC summary over the frozen v4 shards |
| `paper1/results/cross_task_ir_sr_all_subsets_v2.csv` | All nonempty source-task calibration partitions in the current IR/SR schema |
| `paper1/results/cross_task_ir_sr_all_subsets_params_v2.json` | Current IR/SR calibration parameters |
| `paper1/results/cross_task_ir_sr_all_subsets_summary_v2.json` | Cross-task IR/SR summary consumed by later builders |
| `paper1/results/external_validation/pldm_frozen_rows_v2.csv` | Complete 36-row PLDM sweep |
| `paper1/results/external_validation/cross_stressor_all_pairs.csv` | Frozen LeWM/PLDM blur and resize pair rows |
| `paper1/results/external_validation/cross_stressor_three_source_ir_sr_v2.csv` | Reader-facing 24-row LeWM cross-stressor IR/SR comparison |
| `paper1/results/external_validation/cross_stressor_three_source_ir_sr_v2.json` | Machine-readable cross-stressor summary in the current schema |
| `paper1/results/linearization_horizon_sensitivity_ir_sr_v2.json` | Current-schema linearization/horizon aggregate migrated value-for-value from the released v1 aggregate |

The corresponding generated outputs are:

- `assets/paper1_figs/fig_acpc_ir_sr_overview.pdf`
- `assets/paper1_figs/fig_full_sweep_diagnostics.pdf`
- `assets/paper1_figs/fig_full_sweep_diagnostic_region.pdf`
- `assets/paper1_figs/fig_full_sweep_planner_guard.pdf`
- `assets/paper1_figs/fig_pldm_sweep_diagnostics.pdf`
- `assets/paper1_figs/fig_cross_task_ir_sr_source_coverage_v2.pdf`
- `assets/paper1_figs/fig_cross_stressor_ir_sr_comparison_v2.pdf`
- the current `paper1/tables/*ir_sr*.tex` tables.

Rebuild commands and the active dependency graph are documented in
`paper1/scripts/README.md`.

## Figure 1 input provenance

The input thumbnails are real PushT frames; the latent-space panels are
schematic illustrations. `build_acpc_overview_inputs.py` validates the source
dataset shape, episode/step identifiers, and raw frame hashes. The committed
metadata intentionally records a logical dataset name and filename rather
than a machine-specific absolute path.

| File | SHA-256 |
|---|---|
| `assets/paper1_figs/acpc_overview_inputs/metadata.json` | `82929eaec50e25a11d01f41fa0b9612d3b4fd98ef727461f2f76ffb9e749a709` |
| `pusht_ep8200_step48_clean.png` | `78f0dfabca82ee290a668e3434af1dd38cc4dda8c22be919bb7ac9cd05666d28` |
| `pusht_ep8200_step48_gaussian_std008.png` | `bf7437088e870a8585ad20bd477afffac5027020be50abc5f62bded8b5a8e5c2` |
| `pusht_ep8200_step124_different_state.png` | `d795b91f64ce77116ea39e80970609214fcf36a479334ba61958b01e51fa8538` |

## Primary artifact hashes

| File | SHA-256 |
|---|---|
| `full_sweep_diagnostics.csv` | `f7c24b6881f1b0811d4d4ab039834bf3ccd4390adfd40d861c9d10ca19494cd5` |
| `full_sweep_diagnostics_summary.csv` | `48f69fd21d3f272a13dde5151506818dd87465912dfbb8d29edc8d776532b1e0` |
| `future_drift_three_seed_v1.csv` | `c7fd82d0cd66dff804309164ace4bbd26f733e4a1c78aea53d6f3c105d2e25dd` |
| `future_drift_three_seed_summary_v1.json` | `452489d4dab3239e155a9755c991c3225b18338a124ed64e92aadd24aa2d6a04` |
| `acpc_planner_stability_v4/summary.json` | `a8ca3029cadcfd4d75e2e6906a9ff5a9072ae6b8bb13e8351752d10d60729cbb` |
| `cross_task_ir_dr_all_subsets_v1.csv` | `7d1d2120ad210b1a7e507426aa47498260ea52f213e0125cb69152f8279e2330` |
| `cross_task_ir_dr_all_subsets_params_v1.json` | `280823ed9a92078889353ddbd31a0563e526c22f326c5aa63b66760d25c72c9e` |
| `cross_task_ir_dr_all_subsets_summary_v1.json` | `ef7259c28286b61894fb5da71b0c0115bfd942f14e2213b375469eea187bf1f8` |
| `pldm_frozen_rows_v2.csv` | `d3026fbfb9804d5252367a49858509f7f2dc70993de985da57595d9524477a34` |
| `cross_stressor_all_pairs.csv` | `dbf26c84bb3670dd8dd00a3a535fd2beaf6157a7abdeeb23d0baea0579f07b02` |
| `cross_stressor_three_source_ir_dr_v1.csv` | `7c5448c3d8af54de9b867c5fd2bade6b4bf7ca86c53e2bc7f16f3ba5ea5d0495` |
| `cross_stressor_three_source_ir_dr_v1.json` | `e58cc707201aa4a3769b51910256329b91ba4c71171c5ea298302b9b86bdfd55` |
| `cross_task_ir_sr_all_subsets_v2.csv` | `03aca8b785bb1f3df4ee227fb6f62cc52e2982a3ab546388c6b39f3c7134ad74` |
| `cross_task_ir_sr_all_subsets_params_v2.json` | `012096b1639d4858da6e520d92377db73946bee92314366e9ea7b9ccbcce642a` |
| `cross_task_ir_sr_all_subsets_summary_v2.json` | `ea8789a958057bca4455d128e36b0e2127741d4f9e3d846255782e65a48e11f6` |
| `cross_stressor_three_source_ir_sr_v2.csv` | `59f09888e6820dc32609fe40c5c001c7e5270ca7b6e6093fa1388529976e295f` |
| `cross_stressor_three_source_ir_sr_v2.json` | `3652b77c5de5ab70d53fb92b66af5ce72102c485b79f2073718b3d30be3ea6d8` |
| `linearization_horizon_sensitivity_ir_sr_v2.json` | `39fb2cb9bbb1a9cfd9ef31575ee80970523fe86787603b6db5a625b49e5f267a` |
| `linearization_checkpoint_rows_ir_sr_v2.csv` | `60453e5a6d97b75c3f9e6ddf9366211c12067d5965c5f4df61f153d98fbed862` |
| `linearization_calibration_rows_ir_sr_v2.csv` | `07779e1a01e77106b7606909333e3a5d02f08d27f6d740ceaa9d5ca3bd70ea49` |
| `horizon_quantile_sensitivity_rows_ir_sr_v2.csv` | `26e0d1ba3304c7c72944408e442bbbdc2986d232bd3e2da1e4103020c6bd6b70` |
| `linearization_calibration_summary_ir_sr_v2.csv` | `1eb0a4ed14ce5574e001c05d497f7b94f543590d8b65dc974859d893815c5fdc` |
| `horizon_quantile_sensitivity_summary_ir_sr_v2.csv` | `29a585a97be204926ba2f603b04d725ddcc5cc90a1dce79d55bbdc202fc6e86b` |

## Frozen legacy IR/DR-v1 artifacts and column names

The five `*_ir_dr_*_v1` files listed and hash-recorded above remain byte-for-byte
unchanged for reproducibility. Their SR-v2 counterparts contain the same
values with only canonical field/schema names changed. Current consumers use
`paper1/scripts/ir_sr_compat.py`, which applies aliases including:

```text
atr_normalized_q90 -> ir_relative_q90
smpr_delta010      -> sr_delta010
dr_delta010        -> sr_delta010
dr_threshold       -> sr_threshold
```

`paper1/scripts/ir_dr_compat.py` remains only as the deprecated v1 API boundary.
The aliases do not change stored values. Current prose, scripts, tables, and
figures use Invariance Radius (IR) and Separation Rate (SR).

## Portability and provenance

- Historical JSON files may retain absolute checkpoint paths from the machine
  that produced them. Those paths document provenance; reruns should resolve
  checkpoints from the user's own `$STABLEWM_HOME` or explicit model root.
- HDF5 lookup supports both `$STABLEWM_HOME/<name>.h5` and
  `$STABLEWM_HOME/datasets/<name>.h5`.
- Frozen protocols and their `.sha256` sidecars bind experiment settings.
  Do not edit a protocol and continue using its old digest.
- PDF binaries may differ across rebuilds because of backend metadata even
  when the plotted data and extracted text are unchanged. CSV/JSON equality
  and source hashes are the primary deterministic checks.
- IR/SR screen outcomes do not certify closed-loop robustness, policy return,
  or adaptive-CEM behavior.
