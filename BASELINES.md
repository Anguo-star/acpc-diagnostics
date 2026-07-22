# PLDM baseline reproduction

The paper's second world-model family is PLDM. `train_pldm.py` is a thin
Hydra entry point around the PLDM implementation shipped by
`stable-worldmodel`; it adds only the same split-level Gaussian image
augmentation and portable HDF5 lookup used by `train.py`.

No external source checkout is required beyond the packages installed by
`requirements.txt`.

## Training

Always set `output_model_name`; it controls both the run directory and the
serialized checkpoint filename.

```bash
# Clean examples
python train_pldm.py data=pusht seed=3072 \
  output_model_name=pusht_pldm_baseline
python train_pldm.py data=tworoom seed=3072 \
  output_model_name=tworoom_pldm_baseline

# Gaussian-augmented example
python train_pldm.py data=pusht seed=3072 \
  output_model_name=pusht_pldm_noise_0to006_p1 \
  image_noise.std_min=0.0 image_noise.std_max=0.06 \
  image_noise.noise_prob=1.0
```

Task-to-config mapping is the same as for LeWM:

| Task | Hydra override |
|---|---|
| PushT | `data=pusht` |
| TwoRoom | `data=tworoom` |
| Reacher | `data=dmc` |
| Cube | `data=ogb` |

The released PLDM sweep contains one training run per task and augmentation
setting: the clean baseline plus `std_max` in
`{0.01, 0.02, ..., 0.08}`. Do not interpret its three evaluation seeds as
three independent PLDM training seeds.

## Evaluation

`eval.py` loads PLDM and LeWM object checkpoints through the same
`stable_worldmodel.policy.AutoCostModel` interface. For example:

```bash
python eval.py --config-name=pusht.yaml \
  policy=pusht_pldm_noise_0to006_p1/pusht_pldm_noise_0to006_p1_epoch_10 \
  eval.corruption.type=gaussian_noise \
  eval.corruption.std=0.08 \
  eval.corruption.apply_to='[pixels,goal]'
```

The Paper 1 behavioral summaries use evaluation seeds 42, 43, and 44 with
100 trajectories per seed. Use the same `world.num_envs`/batch size for rows
that will be compared because CEM sampling can depend on the batch dimension.

## Released PLDM evidence

The frozen row-level input is
`paper1/results/external_validation/pldm_frozen_rows_v2.csv`. Rebuild its
reader-facing display with:

```bash
python -m paper1.scripts.plot_pldm_sweep_diagnostics
python -m paper1.scripts.build_acpc_submission_assets
```

The same ACPC, IR, and SR definitions are used for both model families; the
PLDM results test portability of the diagnostic, not architectural equality.
