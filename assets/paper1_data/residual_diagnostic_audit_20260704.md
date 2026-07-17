# Paper1 residual diagnostic audit (2026-07-04)

No retraining. Nonzero LeWM Gaussian-grid checkpoints only: 3 training seeds x 4 tasks x 8 nonzero std levels = 96 rows.

Partial Spearman is computed by rank-transforming metric and outcome, then residualizing both on std_max, task fixed effects, and training-seed fixed effects.

95% CI uses task-seed block bootstrap over 12 blocks, preserving the 8 std rows within each block.

| Metric | Outcome | ordinary rho | partial rho | 95% block CI |
|---|---:|---:|---:|---:|
| ACPC-H/trans. | obs0.08 success | +0.41 | +0.07 | [-0.22, +0.30] |
| ACPC-H/trans. | reduced drop | +0.62 | +0.19 | [+0.06, +0.36] |
| PCC | obs0.08 success | +0.38 | +0.09 | [-0.16, +0.33] |
| PCC | reduced drop | +0.60 | +0.20 | [+0.06, +0.37] |
| CRA | obs0.08 success | +0.15 | +0.23 | [-0.07, +0.47] |
| CRA | reduced drop | +0.54 | +0.29 | [+0.14, +0.45] |
| MAF | obs0.08 success | -0.02 | +0.30 | [+0.07, +0.48] |
| MAF | reduced drop | +0.45 | +0.23 | [+0.11, +0.34] |

Reading: after controlling for training-noise level, task, and training seed, the ACPC-family readouts retain modest residual association with reduced Gaussian drop, while residual association with absolute obs0.08 success is weak or metric-dependent. This supports a residual localization signal, not a standalone selector claim.
