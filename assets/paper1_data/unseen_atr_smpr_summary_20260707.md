# Paper1 Unseen ATR/SMPR Summary

Values are population mean over training seeds 3072/3073/3074. ATR and SMPR are reported as raw no-noise -> noise-trained values under the same unseen stressor; lower ATR and higher SMPR are better.

| Task | stressor | no-noise score | noise-trained score | ATR raw | SMPR raw |
|---|---:|---:|---:|---:|---:|
| TwoRoom | gaussian_blur 15 | 47.67 +/- 5.44 | 90.78 +/- 5.38 | 1.61 -> 1.24 | 0.16 -> 0.77 |
| Reacher | gaussian_blur 15 | 22.00 +/- 3.78 | 71.22 +/- 1.10 | 2.81 -> 0.54 | 0.60 -> 0.98 |
| PushT | resize 0.25 | 63.44 +/- 14.05 | 66.33 +/- 8.38 | 1.77 -> 1.53 | 0.93 -> 0.96 |
| Cube | resize 0.25 | 57.00 +/- 1.96 | 56.11 +/- 0.57 | 1.35 -> 1.59 | 0.98 -> 0.95 |

Seed-row correlations are descriptive for this bounded scope check, not a formal transfer theorem.

```json
{
  "pearson_stress_delta_vs_ATR_drop": 0.768142082157261,
  "pearson_stress_delta_vs_SMPR_gain": 0.8705024599334041,
  "seed_rows_n": 12,
  "spearman_stress_delta_vs_ATR_drop": 0.8391608391608392,
  "spearman_stress_delta_vs_SMPR_gain": 0.8721554530812411
}
```
