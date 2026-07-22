# Paper 1 ACPC/IR/SR rebuild scripts

These scripts rebuild the released diagnostic displays from committed
machine-readable inputs. They do not train checkpoints, rerun closed-loop
evaluation, or build the manuscript.

## Current dependency graph

```text
full_sweep_diagnostics*.csv ──> plot_full_sweep_diagnostics
future_drift_three_seed* ─────> build_future_drift_reader_display
LeWM pair rows ───────────────> cross_task_selective_rule ──> IR/SR summary
PLDM frozen rows ─────────────> plot_pldm_sweep_diagnostics
IR/SR summary + stressor rows ─> build_cross_stressor_ir_sr_comparison
planner summary + sweep rows ─> build_acpc_submission_assets
Figure 1 input PNGs ──────────> plot_acpc_ir_sr_overview
released linearization v1 ───> build_linearization_horizon_artifact ─> IR/SR v2
```

The canonical CPU-only rebuild from repository root is:

```bash
python -m paper1.scripts.plot_acpc_ir_sr_overview
python -m paper1.scripts.plot_full_sweep_diagnostics
python -m paper1.scripts.build_future_drift_reader_display
python -m paper1.scripts.cross_task_selective_rule
python -m paper1.scripts.plot_pldm_sweep_diagnostics
python -m paper1.scripts.build_cross_stressor_ir_sr_comparison
python -m paper1.scripts.build_acpc_submission_assets
python -m paper1.scripts.plot_gaussian_sensitivity_mechanism
python -m paper1.scripts.build_linearization_horizon_artifact \
  --legacy-artifact paper1/results/linearization_horizon_sensitivity_v1.json
```

These commands write the committed figures/tables in place. To smoke-test the
core IR/SR path without changing the checkout, use temporary outputs:

```bash
python -m paper1.scripts.plot_acpc_ir_sr_overview \
  --out /tmp/acpc-overview.pdf --preview /tmp/acpc-overview.png

python -m paper1.scripts.cross_task_selective_rule \
  --rows-out /tmp/cross-task.csv \
  --params-out /tmp/cross-task-params.json \
  --summary-out /tmp/cross-task-summary.json \
  --table-out /tmp/cross-task.tex \
  --figure-out /tmp/cross-task.pdf

python -m paper1.scripts.build_cross_stressor_ir_sr_comparison \
  --p2-summary /tmp/cross-task-summary.json \
  --rows-out /tmp/cross-stressor.csv \
  --summary-out /tmp/cross-stressor.json \
  --table-out /tmp/cross-stressor-summary.tex \
  --all-pairs-table-out /tmp/cross-stressor-pairs.tex \
  --figure-out /tmp/cross-stressor.pdf

python -m paper1.scripts.plot_full_sweep_diagnostics \
  --out /tmp/full-sweep.pdf \
  --region-out /tmp/diagnostic-region.pdf \
  --planner-out /tmp/planner-guard.pdf

python -m paper1.scripts.plot_pldm_sweep_diagnostics \
  --out-fig /tmp/pldm-sweep.pdf

python -m paper1.scripts.build_acpc_submission_assets \
  --cross-task-summary /tmp/cross-task-summary.json \
  --planner-figure /tmp/planner.pdf \
  --increment-table /tmp/increment.tex \
  --absolute-table /tmp/absolute.tex \
  --sweep-table /tmp/sweep.tex \
  --pldm-table /tmp/pldm.tex
```

## Figure 1 inputs

The three frozen PushT input PNGs and their hashes are committed under
`assets/paper1_figs/acpc_overview_inputs/`. Re-extraction is optional and
requires the original 46 GB PushT HDF5 file:

```bash
python -m paper1.scripts.build_acpc_overview_inputs \
  --h5 /path/to/pusht_expert_train.h5
python -m paper1.scripts.plot_acpc_ir_sr_overview
```

The extractor validates the dataset shape, episode/step identifiers, and raw
pixel hashes before writing images. Its metadata records only the logical
dataset name and filename, not a machine-specific absolute path.

## Planner-horizon ACPC

The committed planner summary is
`paper1/results/acpc_planner_stability_v4/summary.json`. To inspect a frozen
task queue without running it:

```bash
python paper1/scripts/run_acpc_planner_stability_shards.py plan \
  --task TwoRoom \
  --protocol paper1/config/acpc_planner_stability_protocol_v4.json \
  --addendum paper1/config/acpc_planner_stability_execution_v4.json \
  --device cuda:0
```

Executing shards requires the bound checkpoints, datasets, and a CUDA device.
Historical absolute checkpoint paths in protocol/result JSON files are
provenance only; portable reruns should bind local paths explicitly.

## Terminology compatibility

Some frozen raw files and the released IR/DR-v1 derived artifacts predate the
current reader-facing name. `ir_sr_compat.py` maps them without changing stored
values:

```text
atr_normalized_q90 -> ir_relative_q90
smpr_delta010      -> sr_delta010
dr_delta010        -> sr_delta010
dr_threshold       -> sr_threshold
```

New scripts and generated displays use IR/SR terminology. The older
`ir_dr_compat.py` module and former script entry points remain deprecated v1
compatibility boundaries so immutable evidence and commands stay auditable.
