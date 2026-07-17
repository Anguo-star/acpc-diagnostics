# Paper1 selector baseline audit (2026-07-04)
Generated from `acpc_phase0_lewm_three_seed.json` and `three_seed_diagnostic_validation.json`; no new training or evaluation was run. Random nonzero std is the exact average regret over the eight nonzero Gaussian-grid rows.

## all_three_training_seeds
| Selector | within 5pp | exact best | regret mean +/- std (pp) |
|---|---:|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 10/12 | 3/12 | 2.25 +/- 2.51 |
| Fixed std0.08 | 10/12 | 4/12 | 2.14 +/- 2.18 |
| ACPC only | 9/12 | 3/12 | 3.39 +/- 4.21 |
| PCC only | 10/12 | 3/12 | 2.25 +/- 2.51 |
| CRA only | 10/12 | 4/12 | 2.19 +/- 2.56 |
| MAF only | 10/12 | 5/12 | 1.89 +/- 2.80 |
| Random nonzero std | -- | -- | 7.02 +/- 4.01 |
| Oracle best | 12/12 | 12/12 | 0.00 +/- 0.00 |

## development_seed_3072
| Selector | within 5pp | exact best | regret mean +/- std (pp) |
|---|---:|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 3/4 | 2/4 | 2.33 +/- 3.49 |
| Fixed std0.08 | 3/4 | 2/4 | 2.25 +/- 2.60 |
| ACPC only | 2/4 | 1/4 | 6.00 +/- 5.95 |
| PCC only | 3/4 | 1/4 | 2.58 +/- 3.34 |
| CRA only | 3/4 | 2/4 | 2.33 +/- 3.49 |
| MAF only | 3/4 | 3/4 | 2.42 +/- 4.19 |
| Random nonzero std | -- | -- | 6.60 +/- 3.14 |
| Oracle best | 4/4 | 4/4 | 0.00 +/- 0.00 |

## heldout_training_seeds_3073_3074
| Selector | within 5pp | exact best | regret mean +/- std (pp) |
|---|---:|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 7/8 | 1/8 | 2.21 +/- 1.83 |
| Fixed std0.08 | 7/8 | 2/8 | 2.08 +/- 1.93 |
| ACPC only | 7/8 | 2/8 | 2.08 +/- 1.93 |
| PCC only | 7/8 | 2/8 | 2.08 +/- 1.93 |
| CRA only | 7/8 | 2/8 | 2.12 +/- 1.93 |
| MAF only | 7/8 | 2/8 | 1.62 +/- 1.66 |
| Random nonzero std | -- | -- | 7.23 +/- 4.37 |
| Oracle best | 8/8 | 8/8 | 0.00 +/- 0.00 |

Reading: the aggregate ACPC/PCC/CRA/MAF rule is a no-retraining triage and plateau-localization signal. It is comparable to fixed std0.08 and simple single-metric selectors on the broad Gaussian plateaus, and clearly better than exact random nonzero-std selection.
