# Robust CEM Pilot 2026-07-04

Low-concurrency pilot for `paper1/docs/ROBUST_CEM_CODEX.md`. All closed-loop evals used `world.num_envs=1`, `timeout 20m`, reduced CEM budgets, and `record_history=false` for robust runs.

Unit tests: `PYTHONPATH=. pytest -q tests/test_robust_cem.py` -> 9 passed.

## Results

| Task | Condition | Planner | Success | Eval time (s) | Key robust stats |
|---|---|---|---:|---:|---|
| Reacher | gaussian_std0.08 | `cem64_n5_eval3` | 0.0% (0/3) | 9.9 |  |
| Reacher | gaussian_std0.08 | `rcem_meanstd_b05_k2_n64_eval3` | 0.0% (0/3) | 10.4 | score_mode=mean_std; num_views=2; beta=0.5; robust_rescore=all; mean_view_std=2.820491313934326; top1_flip_rate=0.0; robust_changed_top1_rate=0.0 |
| Reacher | gaussian_std0.08 | `rcem_worst_v08_k4_n48_eval3` | 33.3% (1/3) | 8.1 | score_mode=worst; num_views=4; beta=0.5; robust_rescore=all; mean_view_std=28.912654876708984; top1_flip_rate=0.0; robust_changed_top1_rate=0.0 |
| Reacher | gaussian_std0.08 | `cem192_n4_matched_eval3` | 33.3% (1/3) | 7.5 |  |
| Reacher | gaussian_std0.08 | `rcem_mean_v08_k4_n48_eval3` | 0.0% (0/3) | 11.1 | score_mode=mean; num_views=4; beta=0.5; robust_rescore=all; mean_view_std=11.84075927734375; top1_flip_rate=1.0; robust_changed_top1_rate=1.0 |
| Reacher | gaussian_std0.08 | `rcem_meanstd_b10_v08_k4_n48_eval3` | 33.3% (1/3) | 9.5 | score_mode=mean_std; num_views=4; beta=1.0; robust_rescore=all; mean_view_std=29.54024314880371; top1_flip_rate=1.0; robust_changed_top1_rate=1.0 |
| Reacher | clean | `cem48_n4_eval3` | 33.3% (1/3) | 13.5 |  |
| Reacher | clean | `rcem_meanstd_b10_v08_k4_n48_eval3` | 33.3% (1/3) | 10.8 | score_mode=mean_std; num_views=4; beta=1.0; robust_rescore=all; mean_view_std=52.730770111083984; top1_flip_rate=1.0; robust_changed_top1_rate=1.0 |
| PushT | gaussian_std0.08 | `cem48_n4_eval3` | 0.0% (0/3) | 2.7 |  |
| PushT | gaussian_std0.08 | `rcem_worst_v08_k4_n48_eval3` | 0.0% (0/3) | 3.6 | score_mode=worst; num_views=4; beta=0.5; robust_rescore=all; mean_view_std=41.78814697265625; top1_flip_rate=0.3333333432674408; robust_changed_top1_rate=1.0 |
| Reacher | gaussian_std0.08 | `cem48_n4_eval10` | 30.0% (3/10) | 26.4 |  |
| Reacher | gaussian_std0.08 | `cem192_n4_matched_eval10` | 10.0% (1/10) | 26.5 |  |
| Reacher | gaussian_std0.08 | `rcem_worst_v08_k4_n48_eval10` | 10.0% (1/10) | 30.4 | score_mode=worst; num_views=4; beta=0.5; robust_rescore=all; mean_view_std=10.619039535522461; top1_flip_rate=1.0; robust_changed_top1_rate=1.0 |
| Reacher | gaussian_std0.08 | `rcem_elite_meanstd_b10_v08_k4_n192_eval10` | 0.0% (0/10) | 33.5 | score_mode=mean_std; num_views=4; beta=1.0; robust_rescore=elite; mean_view_std=13.505961418151855; top1_flip_rate=1.0; robust_changed_top1_rate=1.0 |
| TwoRoom | gaussian_std0.08 | `cem48_n4_eval10` | 60.0% (6/10) | 8.5 |  |
| TwoRoom | gaussian_std0.08 | `rcem_worst_v08_k4_n48_eval10` | 30.0% (3/10) | 11.9 | score_mode=worst; num_views=4; beta=0.5; robust_rescore=all; mean_view_std=17.36953353881836; top1_flip_rate=0.0; robust_changed_top1_rate=0.0 |
| Reacher | gaussian_std0.08 | `rcem_meanstd_b025_v04_k4_n48_eval10` | 10.0% (1/10) | 31.0 | score_mode=mean_std; num_views=4; beta=0.25; robust_rescore=all; mean_view_std=7.073302745819092; top1_flip_rate=0.6666666865348816; robust_changed_top1_rate=1.0 |

## Interpretation

Go/no-go: **negative for a main-method claim at this budget**.

- Reacher eval10: standard CEM was 30%, compute-matched CEM was 10%, robust worst was 10%, elite mean_std was 0%, and milder mean_std was 10%.
- TwoRoom eval10: standard CEM was 60%, robust worst was 30%.
- PushT eval3: standard and robust worst were both 0%.
- The initial Reacher eval3 robust signal did not survive expansion to eval10.

This pilot validates that the implementation runs and records diagnostics, but it does not validate robust CEM as an effective planner-side method. Full Stage A should not be launched until a cheaper offline ranking analysis finds a regime where robust scoring changes candidate selection in a useful direction.
