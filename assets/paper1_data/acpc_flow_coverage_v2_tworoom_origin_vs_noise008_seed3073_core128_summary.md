# ACPC-Flow v2 Origin vs Noise008 Summary

TwoRoom seed3073, core128 full-stress audit, same sampling protocol.

## Amplification q90

| Stressor | amp_P origin -> noise008 | amp_R origin -> noise008 | amp_total origin -> noise008 |
|---|---:|---:|---:|
| gaussian_std0.03 | 5.996 -> 1.382 | 1.443 -> 1.445 | 5.476 -> 1.115 |
| gaussian_std0.05 | 5.541 -> 1.353 | 1.360 -> 1.455 | 4.991 -> 1.134 |
| gaussian_std0.08 | 5.192 -> 1.358 | 1.239 -> 1.399 | 3.550 -> 1.064 |
| blur_ks7 | 5.681 -> 1.420 | 1.086 -> 1.393 | 1.880 -> 1.514 |
| resize_factor0.5 | 5.649 -> 1.506 | 1.132 -> 1.433 | 2.063 -> 1.529 |

## Embedding-level rank/crossing

| Stressor | emb wrong_nn origin -> noise008 | rank flip origin -> noise008 | top-k overlap origin -> noise008 | emb decision origin -> noise008 |
|---|---:|---:|---:|---|
| gaussian_std0.03 | 0.135 -> 0.000 | 0.195 -> 0.016 | 0.880 -> 0.991 | no_go -> encoder_projector_small_train |
| gaussian_std0.05 | 0.521 -> 0.000 | 0.445 -> 0.008 | 0.683 -> 0.984 | no_go -> encoder_projector_small_train |
| gaussian_std0.08 | 0.828 -> 0.003 | 0.664 -> 0.016 | 0.534 -> 0.977 | no_go -> no_go |
| blur_ks7 | 0.945 -> 0.372 | 0.828 -> 0.234 | 0.375 -> 0.808 | no_go -> no_go |
| resize_factor0.5 | 0.932 -> 0.164 | 0.789 -> 0.117 | 0.373 -> 0.878 | no_go -> no_go |

## Reading

- Matched Gaussian stressors are strongly repaired by ordinary input-noise training in the v2 diagnostics.
- The dominant origin failure amplifier, `P`, drops from about 5-6x to about 1.35-1.5x under noise training.
- Candidate top-1 flip under Gaussian 0.08 drops from about 0.66 to about 0.02, and top-k overlap rises from about 0.53 to about 0.98.
- Blur/resize improve but remain no-go, especially at `pred_emb`, so the method hypothesis should stay matched/local rather than broad non-Gaussian.
- `t_start` remains non-separable; this does not reopen time-conditioned FM.

## Decision

This is positive evidence that from-scratch training can reshape the `P/R` path. It supports a small P-only or R-only projector-migration MVE for matched Gaussian, not a frozen post-hoc adapter and not a broad blur/resize claim.
