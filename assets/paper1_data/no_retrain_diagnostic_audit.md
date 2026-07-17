# No-Retraining Diagnostic Audit

This artifact is computed only from existing Phase-0 paired ACPC JSON files. 
It is a frozen-rule sanity audit, not a true held-out prospective validation.

## Composite diagnostic selection

Rule: among nonzero training-noise checkpoints, select the row with the lowest 
aggregate rank over ACPC-H/transition (low), PCC (low), CRA (high), and MAF (low).

| Method | Task | selected std | selected px08 | closed-loop best std | best px08 | gap pp |
|---|---|---:|---:|---:|---:|---:|
| LeWM | TwoRoom | 0.08 | 98.67 | 0.08 | 98.67 | 0.00 |
| LeWM | PushT | 0.08 | 85.33 | 0.06 | 87.00 | 1.67 |
| LeWM | Reacher | 0.02 | 85.67 | 0.02 | 85.67 | 0.00 |
| LeWM | Cube | 0.05 | 59.67 | 0.07 | 68.00 | 8.33 |
| PLDM | TwoRoom | 0.07 | 96.67 | 0.05 | 98.33 | 1.67 |
| PLDM | PushT | 0.08 | 66.33 | 0.03 | 72.00 | 5.67 |
| PLDM | Reacher | 0.08 | 80.67 | 0.04 | 81.33 | 0.67 |
| PLDM | Cube | 0.08 | 56.33 | 0.08 | 56.33 | 0.00 |

## Spearman correlations

Metric signs are oriented so larger values mean a better diagnostic reading.

| Method | Metric | rho vs px08 success | rho vs -drop | n |
|---|---|---:|---:|---:|
| LeWM | Encoder radius | 0.19 | 0.60 | 36 |
| LeWM | ACPC-H/trans. | 0.46 | 0.79 | 36 |
| LeWM | PCC | 0.46 | 0.77 | 36 |
| LeWM | CRA | 0.31 | 0.68 | 36 |
| LeWM | MAF | 0.21 | 0.65 | 36 |
| PLDM | Encoder radius | 0.14 | 0.35 | 36 |
| PLDM | ACPC-H/trans. | 0.53 | 0.49 | 36 |
| PLDM | PCC | 0.67 | 0.45 | 36 |
| PLDM | CRA | 0.45 | 0.48 | 36 |
| PLDM | MAF | 0.23 | 0.30 | 36 |
