# ACPC-Flow v2 four-task origin-vs-noise008 aligned summary

Primary readout: continuous movement from origin to ordinary noise training, compared with eval and paper diagnostics. The `strict_gate_label` remains an absolute training gate for synthetic local repair only; it is not used here as the paper-facing conclusion.

Cube uses core64 for the v2 audit because the core128 run was too slow; TwoRoom/Reacher/PushT use core128. All rows use seed3073 checkpoints.

## Gaussian 0.08: v2 movement aligned with ATR/SMPR and eval

| Task | eval pixels_std0.08 | ATR q90 | SMPR | amp_P q90 | emb wrongNN | pred wrongNN | top1 flip | top5 overlap | v2 improve count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoRoom | 68.8 -> 97.1 | 1.509 -> 0.111 | 0.339 -> 0.989 | 5.192 -> 1.358 | 0.828 -> 0.003 | 0.906 -> 0.042 | 0.664 -> 0.016 | 0.534 -> 0.977 | 6/6 |
| Reacher | 18.2 -> 81.6 | 2.628 -> 0.082 | 0.733 -> 0.997 | 2.926 -> 1.031 | 0.930 -> 0.003 | 0.927 -> 0.000 | 0.797 -> 0.008 | 0.516 -> 0.980 | 6/6 |
| PushT | 7.2 -> 85.8 | 3.580 -> 0.247 | 0.439 -> 1.000 | 1.545 -> 1.140 | 0.732 -> 0.000 | 0.711 -> 0.003 | 0.867 -> 0.023 | 0.364 -> 0.970 | 6/6 |
| Cube | 43.1 -> 62.6 | 2.320 -> 0.100 | 0.453 -> 1.000 | 2.012 -> 1.207 | 0.932 -> 0.000 | 0.932 -> 0.000 | 0.859 -> 0.000 | 0.413 -> 0.988 | 6/6 |

## Held-out blur/resize: compare v2 movement with eval and Phase-0 diagnostics

| Task | stressor | eval stress success | Phase-0 diag | ACPC-H norm | MAF flip | CRA | PCC abs | amp_P q90 | emb wrongNN | pred wrongNN | top1 flip | top5 overlap | v2 improve count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoRoom | Blur k15 | 47.7 -> 83.7 | 5 | 1.320 -> 0.822 | 0.910 -> 0.650 | 0.213 -> 0.704 | 61.90 -> 22.48 | 5.570 -> 1.362 | 0.935 -> 0.625 | 0.977 -> 0.823 | 0.812 -> 0.578 | 0.386 -> 0.633 | 6/6 |
| TwoRoom | Resize 0.25 | 44.7 -> 84.7 | n/a | n/a | n/a | n/a | n/a | 5.203 -> 1.348 | 0.966 -> 0.518 | 0.971 -> 0.807 | 0.867 -> 0.477 | 0.366 -> 0.705 | 6/6 |
| Reacher | Blur k15 | 19.7 -> 72.0 | 5 | 2.196 -> 0.263 | 0.870 -> 0.220 | 0.354 -> 0.979 | 56.35 -> 5.59 | 3.918 -> 1.096 | 0.823 -> 0.240 | 0.872 -> 0.198 | 0.758 -> 0.148 | 0.516 -> 0.944 | 6/6 |
| Reacher | Resize 0.25 | 38.3 -> 78.3 | n/a | n/a | n/a | n/a | n/a | 4.173 -> 1.117 | 0.484 -> 0.078 | 0.523 -> 0.055 | 0.367 -> 0.016 | 0.830 -> 0.959 | 6/6 |
| PushT | Blur k15 | 51.3 -> 58.7 | n/a | n/a | n/a | n/a | n/a | 1.694 -> 1.316 | 0.120 -> 0.070 | 0.159 -> 0.102 | 0.297 -> 0.156 | 0.825 -> 0.870 | 6/6 |
| PushT | Resize 0.25 | 43.7 -> 68.0 | 5 | 1.181 -> 0.600 | 0.330 -> 0.160 | 0.887 -> 0.944 | 22.08 -> 10.34 | 1.654 -> 1.289 | 0.122 -> 0.026 | 0.161 -> 0.062 | 0.297 -> 0.109 | 0.816 -> 0.919 | 6/6 |
| Cube | Blur k15 | 56.0 -> 58.3 | n/a | n/a | n/a | n/a | n/a | 1.925 -> 1.299 | 0.422 -> 0.500 | 0.365 -> 0.495 | 0.156 -> 0.297 | 0.778 -> 0.688 | 1/6 |
| Cube | Resize 0.25 | 57.7 -> 56.3 | 0 | 0.840 -> 0.949 | 0.330 -> 0.360 | 0.801 -> 0.739 | 22.71 -> 23.35 | 2.014 -> 1.348 | 0.411 -> 0.406 | 0.417 -> 0.391 | 0.094 -> 0.172 | 0.784 -> 0.759 | 3/6 |

Reading: TwoRoom/Reacher blur and resize show eval gains and strong v2 continuous improvements; the available Phase-0 blur diagnostics for TwoRoom/Reacher also improve in all five directional checks. PushT resize is also aligned. Cube resize is the boundary case: eval is slightly down, Phase-0 diagnostics do not improve, and v2 rank metrics are mixed/worse. Cube blur has slight eval gain but mixed/worse rank-side v2 movement, so it should remain uncertain until repeated or paired with stronger diagnostics.

## Strict gate label caveat

The v2 script may still emit `no_go` for many blur/resize rows because its gate asks whether fixed-checkpoint synthetic local noise can safely cover the pixel-induced shift. That gate is stricter than the paper-facing question here: whether ordinary training reshapes P/R and planner-facing rank metrics relative to origin. The summary above therefore treats `no_go` as a method gate, not as evidence that noise training lacks relative robustness improvement.
