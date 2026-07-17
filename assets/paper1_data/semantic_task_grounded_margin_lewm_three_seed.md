# Paper 1 Task-Grounded Semantic Margin Audit

Programmatic near-boundary task-state proxy margin over LeWM training seeds 3072/3073/3074. This is not an oracle hand-labeled semantic proof.

| Task | std | pass-rate | same radius | proxy diff | margin | local state | pair count/seed | label rule |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TwoRoom | 0.0 | 0.34 +/- 0.05 | 17.16 +/- 0.60 | 15.39 +/- 0.11 | -2.16 +/- 0.57 | 0.51 | 61 | agent doorway/room-side split from local x-coordinate |
| TwoRoom | 0.08 | 0.99 +/- 0.01 | 0.77 +/- 0.09 | 15.80 +/- 0.51 | 15.27 +/- 0.54 | 0.51 | 61 | agent doorway/room-side split from local x-coordinate |
| PushT | 0.0 | 0.44 +/- 0.17 | 18.39 +/- 1.24 | 17.59 +/- 0.21 | -0.58 +/- 1.49 | 1.21 | 98 | T-block pose cell from x/y/theta median splits |
| PushT | 0.08 | 1.00 +/- 0.00 | 0.70 +/- 0.06 | 17.54 +/- 0.06 | 16.27 +/- 0.05 | 1.21 | 98 | T-block pose cell from x/y/theta median splits |
| Reacher | 0.0 | 0.73 +/- 0.03 | 15.44 +/- 0.51 | 18.79 +/- 0.07 | 2.71 +/- 0.38 | 1.36 | 100 | target/end-effector relation quadrant plus distance bin |
| Reacher | 0.08 | 1.00 +/- 0.00 | 0.27 +/- 0.02 | 18.82 +/- 0.08 | 18.50 +/- 0.06 | 1.36 | 100 | target/end-effector relation quadrant plus distance bin |
| Cube | 0.0 | 0.45 +/- 0.03 | 19.33 +/- 0.15 | 19.08 +/- 0.12 | -0.27 +/- 0.12 | 3.90 | 100 | cube-pose/goal-relation cell from relation direction and distance |
| Cube | 0.08 | 1.00 +/- 0.00 | 0.49 +/- 0.05 | 19.24 +/- 0.06 | 18.69 +/- 0.13 | 3.90 | 100 | cube-pose/goal-relation cell from relation direction and distance |
