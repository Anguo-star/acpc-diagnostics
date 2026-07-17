# Plateau membership audit

- Tolerance: 5.0 pp from the block's closed-loop best.
- Screen: top 4 of 8 nonzero Gaussian candidates per task-training-seed block.
- Reading: plateau-entry enrichment, not point-optimal checkpoint ranking.

| Rule | presence hit | screen precision | plateau recall | TP/FP/FN |
|---|---:|---:|---:|---:|
| Aggregate ACPC/PCC/CRA/MAF | 12/12 | 87.5% | 61.8% | 42/6/26 |
| ACPC only | 12/12 | 87.5% | 61.8% | 42/6/26 |
| PCC only | 12/12 | 87.5% | 61.8% | 42/6/26 |
| CRA only | 12/12 | 87.5% | 61.8% | 42/6/26 |
| MAF only | 12/12 | 91.7% | 64.7% | 44/4/24 |
| High-std top-half reference | 12/12 | 87.5% | 61.8% | 42/6/26 |
| Random top-half reference (exact expectation) | 11.96/12 | 70.8% | 50.0% | 34.0/14.0/34.0 |
