# Selector Incremental Audit

This no-retraining audit asks whether the paired ACPC-family readouts explain robustness variation after controlling for `std_max`, `std_max^2`, task fixed effects, and training-seed fixed effects. Rows are the 96 nonzero LeWM Gaussian checkpoints.

| Metric | partial r vs reduced drop | partial R2 | incr. R2 | block p | partial r vs obs0.08 |
|---|---:|---:|---:|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 0.16 | 0.03 | 0.01 | 0.07 | 0.11 |
| ACPC-H/trans. | 0.12 | 0.01 | 0.01 | 0.16 | -0.08 |
| PCC | 0.13 | 0.02 | 0.01 | 0.12 | -0.04 |
| CRA | 0.23 | 0.05 | 0.03 | 0.00 | 0.10 |
| MAF | 0.15 | 0.02 | 0.01 | 0.12 | 0.15 |
| Elite overlap | 0.13 | 0.02 | 0.01 | 0.18 | -0.01 |

Reading: reduced drop is the cleaner residual target than absolute noisy success. The audit tests incremental explanatory signal; it does not establish a superior checkpoint selector.
