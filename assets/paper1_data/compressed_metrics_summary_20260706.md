# Paper 1 Compressed Selective-ACPC Metrics

ATR/SMPR are paper-facing; PCC/CRA/MAF/CEM/rank/ID remain legacy audits outside the Paper1 PDF.

| Task | ATR base | ATR std0.08 | SMPR base | SMPR std0.08 | obs0.08 base | obs0.08 std0.08 |
|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | 1.509 +/- 0.010 | 0.111 +/- 0.016 | 0.34 +/- 0.05 | 0.99 +/- 0.01 | 68.78 +/- 2.45 | 97.11 +/- 0.57 |
| PushT | 3.580 +/- 0.260 | 0.247 +/- 0.017 | 0.44 +/- 0.17 | 1.00 +/- 0.00 | 7.22 +/- 5.81 | 85.78 +/- 3.45 |
| Reacher | 2.628 +/- 0.048 | 0.082 +/- 0.005 | 0.73 +/- 0.03 | 1.00 +/- 0.00 | 18.22 +/- 0.42 | 81.56 +/- 0.87 |
| Cube | 2.320 +/- 0.045 | 0.100 +/- 0.007 | 0.45 +/- 0.03 | 1.00 +/- 0.00 | 43.11 +/- 3.27 | 62.56 +/- 1.55 |

ATR_q90 uses the per-row p90 same-state ACPC-H distance normalized by the clean transition median; it is not the legacy median R_F basin readout.
SMPR uses task-grounded near-boundary proxy labels and is not an oracle hand-labeled semantic proof.
