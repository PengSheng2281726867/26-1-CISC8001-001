# Action horizon ablation: geometry-oracle ALT-30

Run ID: `action_horizon_ablation_20261023T070258Z`. All conditions use the same 50 frozen environment seeds, the same 30-episode bank, the same 1:1:1 geometry distance, and the official Runner score.

The requested 1, 3, 5, and 7 actions per memory-bank query were evaluated. An 8-action condition was rerun as a control and compared with the previously recorded formal 8-action result.

This directory is the privileged 5D geometry-oracle diagnostic (current pusher and block state), not the image-encoder ALT policy.

| Actions before re-query | Mean score | SD | 95% bootstrap CI | Paired Δ vs 8 (95% CI) | Mean queries/rollout |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.149962 | 0.224396 | [0.092164, 0.215155] | -0.074760 [-0.143604, -0.013607] | 300.00 |
| 3 | 0.156092 | 0.230009 | [0.097482, 0.224679] | -0.068631 [-0.147647, +0.005053] | 100.00 |
| 5 | 0.214395 | 0.309969 | [0.134557, 0.305470] | -0.010327 [-0.081804, +0.064794] | 60.00 |
| 7 | 0.256861 | 0.345723 | [0.166197, 0.354602] | +0.032138 [+0.001564, +0.075667] | 42.92 |
| 8 | 0.224723 | 0.312898 | [0.143386, 0.314001] | +0.000000 [+0.000000, +0.000000] | 37.26 |

The rerun 8-action mean was `0.224722502`; the previous formal run was `0.224722502`. Across matched seeds, mean score delta was `+0.000000000` (maximum absolute per-seed delta `0`).

Official score is the runner’s maximum clipped coverage per episode. Paired intervals resample the same evaluation seeds. Full per-seed scores and retrieval/action traces are saved beside this report and under the ignored artifact directory.

Trace validation passed: all five conditions have 50 unique frozen evaluation seeds; action commands and source indices match the selected demonstration frame; chunks stay within their source episode; executed counts match every re-query interval; and all artifact checksums verify. The 8-action control reproduces the prior per-seed scores exactly.

## Reproduction

From the project root, run `& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/run_horizon_ablation.py` for a new immutable run ID. Validate a run with `& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/validate_horizon_ablation.py <run_id>`. The existing run ID is never overwritten.
