# Semantic Margin Pass-Rate

Fixed endpoint rows over LeWM training seeds 3072/3073/3074. Pass-rate is the fraction of samples whose hard semantic-different clean rollout distance exceeds the same-state clean/noisy rollout radius.

| Task | std | pass-rate | discrim. ratio | same radius | semantic diff | margin |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | 0.0 | 0.44 +/- 0.06 | 0.98 +/- 0.04 | 17.43 +/- 0.65 | 17.06 +/- 0.11 | -0.71 +/- 0.77 |
| TwoRoom | 0.08 | 1.00 +/- 0.00 | 29.21 +/- 1.46 | 0.56 +/- 0.02 | 16.40 +/- 0.25 | 15.77 +/- 0.27 |
| PushT | 0.0 | 0.27 +/- 0.14 | 0.87 +/- 0.07 | 18.35 +/- 1.21 | 15.82 +/- 0.20 | -2.48 +/- 1.66 |
| PushT | 0.08 | 1.00 +/- 0.00 | 23.66 +/- 2.28 | 0.69 +/- 0.06 | 16.24 +/- 0.02 | 15.06 +/- 0.13 |
| Reacher | 0.0 | 0.58 +/- 0.03 | 1.12 +/- 0.03 | 15.44 +/- 0.51 | 17.24 +/- 0.13 | 0.83 +/- 0.37 |
| Reacher | 0.08 | 1.00 +/- 0.00 | 61.26 +/- 3.37 | 0.27 +/- 0.02 | 16.76 +/- 0.21 | 16.50 +/- 0.23 |
| Cube | 0.0 | 0.25 +/- 0.06 | 0.85 +/- 0.01 | 19.33 +/- 0.15 | 16.46 +/- 0.13 | -3.32 +/- 0.19 |
| Cube | 0.08 | 1.00 +/- 0.00 | 34.44 +/- 4.02 | 0.49 +/- 0.05 | 16.59 +/- 0.24 | 15.99 +/- 0.28 |
