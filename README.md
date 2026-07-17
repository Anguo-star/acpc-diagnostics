# ACPC Diagnostics
### Action-conditioned predictive diagnostics for JEPA world models (paper companion fork of LeWorldModel)

This repository is a fork of [LeWorldModel](https://github.com/lucas-maes/le-wm), extended with the ACPC/Gaussian-noise robustness diagnostics, evaluation artifacts, and scripts used by the companion diagnostic study. The original LeWM model, citation, and upstream links are retained below for attribution.

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

## Paper 1 Reproduction

This branch also contains the code-facing Paper 1 robustness study release:

- canonical result artifacts: `assets/paper1_data/`
- generated reference figures: `assets/paper1_figs/`
- training, eval, and diagnostics scripts: `run_trainer.sh`, `eval.py`, and `tools/`
- deterministic figure/table rebuild pipeline: `paper1/scripts/`
  (documented in `paper1/scripts/README.md`), frozen protocol contracts in
  `paper1/config/`, and the curated machine-readable inputs in
  `paper1/results/`
- unseen-perturbation reproduction launchers:
  `run_paper1_unseen_origin_vs_std008_eval.sh`,
  `run_paper1_unseen_origin_vs_std008_seeded.sh`, and
  `run_paper1_unseen_phase0_acpc_subset.sh`

The manuscript source, paper-facing documentation, and arXiv packaging files
are intentionally not included in this public code branch.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Rebuild the summary figures and tables (CPU-only):**
```bash
python -m paper1.scripts.build_future_drift_reader_display
python -m paper1.scripts.cross_task_selective_rule
python -m paper1.scripts.build_cross_stressor_selective_transfer
python -m paper1.scripts.build_acpc_submission_assets
```
Artifact provenance (sources, seeds, and hashes) is documented in
`DATA_MANIFEST.md`.

**Unseen perturbation eval reproduction:**
```bash
export DATA_ROOT=/path/to/world_model/quentinll

# Seed-specific std=0.0 vs std=0.08 strongest-only blur/resize eval.
TRAIN_SEED=3073 DRY_RUN=1 bash run_paper1_unseen_origin_vs_std008_seeded.sh
TRAIN_SEED=3074 DRY_RUN=1 bash run_paper1_unseen_origin_vs_std008_seeded.sh

# After running real eval jobs, rebuild the compact review artifact.
python -m tools.build_paper1_unseen_eval_artifact \
  --manifest assets/paper1_data/unseen_origin_vs_std008_strongest_s3073_manifest.json \
  --out assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.json \
  --schema-out assets/paper1_data/unseen_origin_vs_std008_strongest_s3073.schema.json \
  --root "$DATA_ROOT" \
  --allow-missing
```

**Representative unseen Phase-0 ACPC subset:**
```bash
export DATA_ROOT=/path/to/world_model/quentinll
DRY_RUN=1 bash run_paper1_unseen_phase0_acpc_subset.sh

# Real run, then rebuild the joined summary artifact.
bash run_paper1_unseen_phase0_acpc_subset.sh
python -m tools.build_paper1_unseen_phase0_acpc_subset \
  --raw-dir assets/paper1_data/unseen_phase0_acpc_subset_raw \
  --out assets/paper1_data/unseen_phase0_acpc_subset.json \
  --schema-out assets/paper1_data/unseen_phase0_acpc_subset.schema.json \
  --seeds 3073 3074
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME`. Set it explicitly:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
