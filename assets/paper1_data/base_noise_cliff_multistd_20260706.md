# LeWM-base Multi-Std Observation Noise Cliff

Reading: this is the exact table source for the main noise-cliff table. The main endpoint perturbs observation pixels only and keeps the goal image clean.

| Task | eval sigma=0 | eval sigma=0.03 | eval sigma=0.05 | eval sigma=0.08 | obs+goal sigma=0.08 |
|---|---:|---:|---:|---:|---:|
| TwoRoom | $94.67 \pm 0.72$ | $92.56 \pm 1.23$ | $81.33 \pm 6.77$ | $68.78 \pm 2.45$ | $53.56 \pm 2.57$ |
| PushT | $81.78 \pm 4.89$ | $53.56 \pm 12.91$ | $19.22 \pm 9.12$ | $7.22 \pm 5.81$ | $4.89 \pm 1.64$ |
| Reacher | $59.22 \pm 1.03$ | $42.56 \pm 2.99$ | $33.56 \pm 1.77$ | $18.22 \pm 0.42$ | $13.78 \pm 1.50$ |
| Cube | $65.22 \pm 1.10$ | $63.00 \pm 1.19$ | $46.67 \pm 1.96$ | $43.11 \pm 3.27$ | $45.78 \pm 0.79$ |

The obs+goal column is auxiliary appendix evidence; it is not a main-table endpoint.
