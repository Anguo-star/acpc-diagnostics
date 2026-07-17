# Three-Seed Diagnostic Validation

Fixed rule: rank nonzero checkpoints by ACPC-H/trans low, PCC low, CRA high, and MAF low; select the lowest aggregate rank. Robustness endpoint is observation-only `pixels_std0.08_success`.

Protocol: seed 3072 is the development grid used to freeze metric computation and the aggregate-rank rule; seeds 3073/3074 are independent held-out training seeds evaluated after the rule is fixed.

## Split Summary

| Split | seeds | blocks | candidates | exact best | within 5pp | regret mean +/- std | bootstrap 95% CI | top-2 overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| development_seed_3072 | 3072 | 4 | 32 | 2/4 | 3/4 | 2.33 +/- 3.49 | [0.00, 6.25] | 1.00/2 |
| heldout_training_seeds_3073_3074 | 3073,3074 | 8 | 64 | 0/8 | 7/8 | 2.21 +/- 1.83 | [1.04, 3.54] | 0.62/2 |
| all_training_seeds_3072_3073_3074 | 3072,3073,3074 | 12 | 96 | 2/12 | 10/12 | 2.25 +/- 2.51 | [1.00, 3.75] | 0.75/2 |

## Summary

Blocks: 12 task-seed blocks; training seeds [3072, 3073, 3074]; 8 nonzero checkpoints per block.
Exact best hits: 2/12; within-5pp hits: 10/12; mean regret to best: 2.25 +/- 2.51 pp; mean top-2 overlap: 0.75/2.

## Selection Rows

| Task | seed | selected std | selected px08 | best std | best px08 | regret | within 5pp | top-2 overlap |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| TwoRoom | 3072 | 0.08 | 97.67 | 0.08 | 97.67 | 0.00 | yes | 1/2 |
| TwoRoom | 3073 | 0.08 | 97.33 | 0.06 | 97.33 | 0.00 | yes | 1/2 |
| TwoRoom | 3074 | 0.08 | 96.33 | 0.05 | 98.00 | 1.67 | yes | 0/2 |
| PushT | 3072 | 0.08 | 89.00 | 0.08 | 89.00 | 0.00 | yes | 1/2 |
| PushT | 3073 | 0.08 | 81.00 | 0.04 | 86.67 | 5.67 | no | 0/2 |
| PushT | 3074 | 0.06 | 86.33 | 0.08 | 87.33 | 1.00 | yes | 1/2 |
| Reacher | 3072 | 0.02 | 83.67 | 0.07 | 84.67 | 1.00 | yes | 2/2 |
| Reacher | 3073 | 0.08 | 82.33 | 0.07 | 83.33 | 1.00 | yes | 2/2 |
| Reacher | 3074 | 0.08 | 80.33 | 0.05 | 83.33 | 3.00 | yes | 1/2 |
| Cube | 3072 | 0.05 | 60.00 | 0.03 | 68.33 | 8.33 | no | 0/2 |
| Cube | 3073 | 0.08 | 64.67 | 0.05 | 65.67 | 1.00 | yes | 0/2 |
| Cube | 3074 | 0.08 | 61.00 | 0.03 | 65.33 | 4.33 | yes | 0/2 |

## Correlations

| Scope | metric | rho vs px08 | r vs px08 | rho vs -drop | r vs -drop | n |
|---|---|---:|---:|---:|---:|---:|
| all_tasks | ACPC-H/trans | 0.41 | 0.70 | 0.62 | 0.81 | 96 |
| all_tasks | PCC | 0.38 | 0.64 | 0.60 | 0.66 | 96 |
| all_tasks | CRA | 0.15 | 0.58 | 0.54 | 0.60 | 96 |
| all_tasks | MAF | -0.02 | 0.47 | 0.45 | 0.61 | 96 |
| TwoRoom | ACPC-H/trans | 0.34 | 0.52 | 0.18 | 0.55 | 24 |
| TwoRoom | PCC | 0.43 | 0.54 | 0.17 | 0.52 | 24 |
| TwoRoom | CRA | 0.51 | 0.48 | 0.14 | 0.48 | 24 |
| TwoRoom | MAF | 0.38 | 0.50 | 0.11 | 0.45 | 24 |
| PushT | ACPC-H/trans | 0.72 | 0.98 | 0.80 | 0.99 | 24 |
| PushT | PCC | 0.73 | 0.98 | 0.81 | 0.99 | 24 |
| PushT | CRA | 0.71 | 0.97 | 0.79 | 0.98 | 24 |
| PushT | MAF | 0.62 | 0.96 | 0.62 | 0.96 | 24 |
| Reacher | ACPC-H/trans | 0.71 | 0.93 | 0.60 | 0.87 | 24 |
| Reacher | PCC | 0.72 | 0.92 | 0.57 | 0.85 | 24 |
| Reacher | CRA | 0.74 | 0.90 | 0.60 | 0.83 | 24 |
| Reacher | MAF | 0.81 | 0.94 | 0.54 | 0.87 | 24 |
| Cube | ACPC-H/trans | 0.27 | 0.85 | 0.57 | 0.90 | 24 |
| Cube | PCC | 0.25 | 0.85 | 0.58 | 0.89 | 24 |
| Cube | CRA | 0.24 | 0.88 | 0.58 | 0.90 | 24 |
| Cube | MAF | 0.44 | 0.90 | 0.60 | 0.92 | 24 |
