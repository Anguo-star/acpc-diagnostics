# Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency

Official code and released artifacts for ACPC diagnostics, built on
[LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM).

Action-Conditioned Predictive Consistency (ACPC) encodes a clean history and
a state-preserving visual perturbation of that history, rolls both
representations forward under the same action sequence, and measures the
distance between their predicted futures. The checkpoint-level screen uses
two complementary summaries:

- **Invariance Radius (IR; lower is better):** an upper-tail summary of
  normalized same-history ACPC.
- **Separation Rate (SR; higher is better):** the fraction of eligible nearby
  different-state-coordinate-label pairs whose rollout separation exceeds
  raw IR by a fixed margin.

A checkpoint passes the screen only when both conditions hold. This is a
diagnostic screen for a specified visual shift and state-coordinate labeling;
it is not a robustness certification.

## Repository scope

This release contains:

- the LeWM model and the Paper 1 Gaussian-augmentation training path:
  `jepa.py`, `module.py`, `train.py`, and `config/train/`;
- PLDM baseline training with the same augmentation path: `train_pldm.py`;
- clean and corrupted evaluation: `eval.py` and `config/eval/`;
- ACPC measurement and audit tools under `tools/`;
- frozen protocols and machine-readable results under `paper1/config/` and
  `paper1/results/`;
- deterministic figure/table builders under `paper1/scripts/`;
- released figures and supporting artifacts under `assets/paper1_figs/` and
  `assets/paper1_data/`.

The manuscript source and arXiv packaging files are intentionally excluded
from this code release. See [DATA_MANIFEST.md](DATA_MANIFEST.md) for the
released evidence boundary and [paper1/scripts/README.md](paper1/scripts/README.md)
for the current dependency graph.

## Installation

Python 3.10 is recommended.

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

Download the LeWM HDF5 datasets from the
[Hugging Face collection](https://huggingface.co/collections/quentinll/lewm),
then place the extracted `.h5` files in either of these supported layouts:

```text
$STABLEWM_HOME/<dataset>.h5
$STABLEWM_HOME/datasets/<dataset>.h5
```

For example:

```bash
export STABLEWM_HOME=/path/to/stable-worldmodel-data
```

The four Paper 1 task configs are:

| Task | Hydra data config | HDF5 dataset name |
|---|---|---|
| PushT | `data=pusht` | `pusht_expert_train` |
| TwoRoom | `data=tworoom` | `tworoom` |
| Reacher | `data=dmc` | `reacher` |
| Cube | `data=ogb` | `ogbench/cube_single_expert` |

## Training

The default config reproduces the 10-epoch Paper 1 schedule. A clean LeWM run
and a Gaussian-augmented run can be launched as follows:

```bash
python train.py data=pusht seed=3072 \
  output_model_name=pusht_lewm_baseline_seed3072

python train.py data=pusht seed=3072 \
  output_model_name=pusht_lewm_noise_0to008_p1_seed3072 \
  image_noise.std_min=0.0 image_noise.std_max=0.08 \
  image_noise.noise_prob=1.0
```

`image_noise.std_min` and `std_max` are expressed in display-pixel `[0,1]`
units and are sampled independently per frame. Set both to zero for the clean
baseline. See [BASELINES.md](BASELINES.md) for the PLDM protocol.

## Evaluation

The `policy` value is the checkpoint path relative to `$STABLEWM_HOME`,
without the `_object.ckpt` suffix. Corruption can be applied to the observed
history only or to both history and goal:

```bash
python eval.py --config-name=pusht.yaml \
  policy=pusht_lewm_noise_0to008_p1_seed3072/pusht_lewm_noise_0to008_p1_seed3072_epoch_10 \
  eval.corruption.type=gaussian_noise \
  eval.corruption.std=0.08 \
  eval.corruption.apply_to='[pixels]'
```

`gaussian_noise`, `gaussian_blur`, and downscale/upscale `resize` are
supported. The committed evaluation configs use `eval.save_video=false` for
numerical sweeps; enable it explicitly when videos are needed.

## Rebuild the released displays

The following commands are CPU-only and rebuild the current ACPC/IR/SR
submission assets from committed inputs:

```bash
python -m paper1.scripts.plot_acpc_ir_sr_overview
python -m paper1.scripts.plot_full_sweep_diagnostics
python -m paper1.scripts.build_future_drift_reader_display
python -m paper1.scripts.cross_task_selective_rule
python -m paper1.scripts.plot_pldm_sweep_diagnostics
python -m paper1.scripts.build_cross_stressor_ir_sr_comparison
python -m paper1.scripts.build_acpc_submission_assets
python -m paper1.scripts.plot_gaussian_sensitivity_mechanism
```

The clean and perturbed PushT images displayed in Figure 1 are already
committed. To re-extract the source records from the original dataset and
verify their frozen pixel hashes, run:

```bash
python -m paper1.scripts.build_acpc_overview_inputs \
  --h5 /path/to/pusht_expert_train.h5
python -m paper1.scripts.plot_acpc_ir_sr_overview
```

## Upstream LeWorldModel

This repository retains the LeWM architecture and training objective from
the original project. Environment management, planning, and evaluation use
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), and
training utilities use
[stable-pretraining](https://github.com/galilai-group/stable-pretraining).

Please also cite LeWM when using the underlying model:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
