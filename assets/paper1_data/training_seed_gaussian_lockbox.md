# Training-Seed Gaussian Lockbox

This artifact summarizes the completed LeWM Gaussian sweep over three training seeds: canonical seed 3072 plus lockbox seeds 3073 and 3074.
Each point is the observation-only Gaussian endpoint `pixels_std0.08` mean over eval seeds 42/43/44 with 100 episodes per eval seed.

## Three-seed task summary

| Task | base obs0.08 | std0.08 obs0.08 | gain | best obs0.08 | std0.08 regret | best std range |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | 68.78 +/- 2.46 | 97.11 +/- 0.57 | 28.33 +/- 2.68 | 97.67 +/- 0.27 | 0.56 +/- 0.79 | 0.05--0.08 |
| PushT | 7.22 +/- 5.81 | 85.78 +/- 3.45 | 78.56 +/- 5.18 | 87.67 +/- 0.98 | 1.89 +/- 2.67 | 0.04--0.08 |
| Reacher | 18.22 +/- 0.42 | 81.55 +/- 0.88 | 63.34 +/- 1.25 | 83.78 +/- 0.63 | 2.22 +/- 0.87 | 0.05--0.07 |
| Cube | 43.11 +/- 3.27 | 62.56 +/- 1.55 | 19.45 +/- 4.53 | 66.44 +/- 1.34 | 3.89 +/- 2.20 | 0.03--0.05 |

## Per-seed rows

| Task | seed | baseline obs0.08 | best obs0.08 | best std | std0.08 obs0.08 | std0.08 gain | regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| TwoRoom | 3072 | 65.67 | 97.67 | 0.08 | 97.67 | 32.00 | 0.00 |
| TwoRoom | 3073 | 71.67 | 97.33 | 0.06 | 97.33 | 25.67 | 0.00 |
| TwoRoom | 3074 | 69.00 | 98.00 | 0.05 | 96.33 | 27.33 | 1.67 |
| PushT | 3072 | 4.33 | 89.00 | 0.08 | 89.00 | 84.67 | 0.00 |
| PushT | 3073 | 2.00 | 86.67 | 0.04 | 81.00 | 79.00 | 5.67 |
| PushT | 3074 | 15.33 | 87.33 | 0.08 | 87.33 | 72.00 | 0.00 |
| Reacher | 3072 | 18.33 | 84.67 | 0.07 | 82.00 | 63.67 | 2.67 |
| Reacher | 3073 | 17.67 | 83.33 | 0.07 | 82.33 | 64.67 | 1.00 |
| Reacher | 3074 | 18.67 | 83.33 | 0.05 | 80.33 | 61.67 | 3.00 |
| Cube | 3072 | 47.00 | 68.33 | 0.03 | 62.00 | 15.00 | 6.33 |
| Cube | 3073 | 39.00 | 65.67 | 0.05 | 64.67 | 25.67 | 1.00 |
| Cube | 3074 | 43.33 | 65.33 | 0.03 | 61.00 | 17.67 | 4.33 |
