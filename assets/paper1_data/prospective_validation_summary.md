# Prospective Validation Remediation Summary

This artifact separates main completed validation evidence from appendix scope checks and reproducibility details.

## Three-seed unseen score aggregate

Scores include training seeds 3072/3073/3074 and are treated as a bounded unseen-stressor scope check.

| Task | selected stress | baseline stress | std0.08 stress | stress delta | drop improvement |
|---|---|---:|---:|---:|---:|
| TwoRoom | gaussian_blur | 47.67 +/- 5.44 | 90.78 +/- 5.38 | 43.11 +/- 8.90 | 40.89 +/- 8.23 |
| PushT | resize | 63.44 +/- 14.05 | 66.33 +/- 8.38 | 2.89 +/- 17.98 | -3.78 +/- 15.56 |
| Reacher | gaussian_blur | 22.00 +/- 3.78 | 71.22 +/- 1.10 | 49.22 +/- 3.29 | 30.22 +/- 4.16 |
| Cube | resize | 57.00 +/- 1.96 | 56.11 +/- 0.57 | -0.89 +/- 1.40 | 2.78 +/- 3.40 |

## Three-seed fixed-rule Gaussian diagnostic validation

Seed 3072 is the development grid used to freeze the aggregate-rank rule; seeds 3073/3074 are held-out training seeds.

| Split | seeds | blocks | candidates | within 5pp | regret mean +/- std | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| development_seed_3072 | 3072 | 4 | 32 | 3/4 | 2.33 +/- 3.49 | [0.00, 6.25] |
| heldout_training_seeds_3073_3074 | 3073,3074 | 8 | 64 | 7/8 | 2.21 +/- 1.83 | [1.04, 3.54] |
| all_training_seeds_3072_3073_3074 | 3072,3073,3074 | 12 | 96 | 10/12 | 2.25 +/- 2.51 | [1.00, 3.75] |

## Task-state proxy margin pass-rate

| Task | std | pass-rate | same-state radius | state-proxy diff | margin |
|---|---:|---:|---:|---:|---:|
| TwoRoom | 0.0 | 0.44 +/- 0.06 | 17.43 +/- 0.65 | 17.06 +/- 0.11 | -0.71 +/- 0.77 |
| TwoRoom | 0.08 | 1.00 +/- 0.00 | 0.56 +/- 0.02 | 16.40 +/- 0.25 | 15.77 +/- 0.27 |
| PushT | 0.0 | 0.27 +/- 0.14 | 18.35 +/- 1.21 | 15.82 +/- 0.20 | -2.48 +/- 1.66 |
| PushT | 0.08 | 1.00 +/- 0.00 | 0.69 +/- 0.06 | 16.24 +/- 0.02 | 15.06 +/- 0.13 |
| Reacher | 0.0 | 0.58 +/- 0.03 | 15.44 +/- 0.51 | 17.24 +/- 0.13 | 0.83 +/- 0.37 |
| Reacher | 0.08 | 1.00 +/- 0.00 | 0.27 +/- 0.02 | 16.76 +/- 0.21 | 16.50 +/- 0.23 |
| Cube | 0.0 | 0.25 +/- 0.06 | 19.33 +/- 0.15 | 16.46 +/- 0.13 | -3.32 +/- 0.19 |
| Cube | 0.08 | 1.00 +/- 0.00 | 0.49 +/- 0.05 | 16.59 +/- 0.24 | 15.99 +/- 0.28 |

## Matched three-seed unseen diagnostic validation slice

Split: training seeds 3072/3073/3074; selected unseen perturbation cases; fixed comparison std_max 0.0 -> 0.08.

| Metric | rho vs stress delta | r vs stress delta | rho vs drop improvement | r vs drop improvement | n |
|---|---:|---:|---:|---:|---:|
| ACPC-H/trans. delta | 0.92 | 0.85 | 0.80 | 0.66 | 12 |
| PCC delta | 0.94 | 0.95 | 0.86 | 0.89 | 12 |
| CRA delta | 0.94 | 0.94 | 0.94 | 0.91 | 12 |
| MAF delta | 0.85 | 0.92 | 0.71 | 0.78 | 12 |
| Composite signed-rank rule | 0.94 | 0.96 | 0.83 | 0.86 | 12 |

Top-4 agreement: composite signed-rank top-k hits 4/4 for stress-success delta and 2/4 for drop improvement.

## Full blur/resize unseen diagnostic validation slice

Split: training seeds 3072/3073/3074; all task by blur/resize strongest endpoints; fixed comparison std_max 0.0 -> 0.08.

| Metric | rho vs stress delta | r vs stress delta | rho vs drop improvement | r vs drop improvement | n |
|---|---:|---:|---:|---:|---:|
| ACPC-H/trans. delta | 0.92 | 0.85 | 0.77 | 0.67 | 24 |
| PCC delta | 0.94 | 0.90 | 0.84 | 0.90 | 24 |
| CRA delta | 0.94 | 0.89 | 0.87 | 0.91 | 24 |
| MAF delta | 0.90 | 0.91 | 0.77 | 0.77 | 24 |
| Composite signed-rank rule | 0.94 | 0.94 | 0.82 | 0.84 | 24 |

Top-4 agreement: composite signed-rank top-k hits 4/4 for stress-success delta and 2/4 for drop improvement.

## Semantic state proxies

| Task | semantic factor | available source | status |
|---|---|---|---|
| PushT | T-block pose/contact relative to pusher | dataset state column is configured for PushT analysis | reported as a task-state proxy margin pass-rate in semantic_margin_passrate_lewm_three_seed.json; finer oracle labels remain an extension |
| TwoRoom | room/doorway/topology and target-region relation | derive from trajectory position/proprio and map topology | reported as a task-state proxy margin pass-rate from pos_agent geometry in semantic_margin_passrate_lewm_three_seed.json; topology-specific labels remain an extension |
| Reacher | joint/target geometry and end-effector-to-target relation | qpos/goal_qpos are used by eval set-state callables | reported as a task-state proxy margin pass-rate in semantic_margin_passrate_lewm_three_seed.json; finer oracle labels remain an extension |
| Cube | cube pose and gripper-object/goal relation | qpos plus goal block position/quaternion are used by eval callables | reported as a task-state proxy margin pass-rate in semantic_margin_passrate_lewm_three_seed.json; finer oracle labels remain an extension |

## Remaining validation work

- Extend the fixed diagnostic rule to additional perturbation families and method families after this three-seed Gaussian validation.
- Extend state-proxy pair construction to finer oracle contact/topology/goal-relation labels if the claim is expanded beyond matched Gaussian diagnostics.
- Keep training-seed uncertainty as the primary behavior statistic and evaluation-seed variance as the secondary decomposition.
