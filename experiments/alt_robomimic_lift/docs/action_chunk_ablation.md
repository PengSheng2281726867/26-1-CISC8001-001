# ALT-Lift action-chunk first-hit ablation

## Protocol

This is a no-retraining ablation of the completed ALT-30 checkpoint and memory bank. Each K means one cosine top-1 query followed by up to K consecutive original 7D incremental OSC_POSE/Panda actions from the matched demonstration episode, then a fresh observation and query. The RoboSuite Lift predicate is checked after every individual action. The evaluator stops immediately the first time cube height is greater than the table offset by 0.04 m. The success signal is evaluator-only; the policy receives no reward or oracle state. Unsuccessful rollouts continue to the unchanged 1,000-step cap. Same-episode terminal padding is retained and counted.

All variants use the same checkpoint, bank, dataset and split fingerprints, reset fixtures, and 50 environment seeds (`100000`–`100049`). Pilot runs on seed `100000` are excluded. K=1 is the existing `evaluation01`: it ran every rollout for 1,000 steps, but its recorded first-success indicator and first-hit step are used as a prefix-equivalent reference. The mean first-hit/cap step count is derived for K=1 by using its recorded first-hit step for successes and 1,000 for failures; this is the counterfactual stopping cost, not a separately run early-stop evaluation. K=1's recorded mean executed steps, retrieval count, and runtime are from the full 1,000-step runs.

The evaluator source file was unchanged between runs. Recorded code commits: K=3: `fe686896c61d62b4248b7ad132b3d360f71d6393`; K=5: `c3d4a47ac04fb6adedb4c2144e545240a51a1df9`; K=7: `c3d4a47ac04fb6adedb4c2144e545240a51a1df9`; K=9: `c3d4a47ac04fb6adedb4c2144e545240a51a1df9`. Evaluator script SHA-256: `647538001d0bd815127a5271c9105dc6ea9413b09afe3ab42fa8a974c6baf53c`; frozen plan SHA-256: `a6916998237dc353446920113958a5fb1a84ad2e74b8ab4368347be669fd8d0a`. The repository HEAD advanced during the long evaluation sequence; all four evaluations record identical evaluator and plan hashes.

## Results

| K actions per query | First-hit success | Wilson 95% interval | Mean steps to success | Median steps to success | Mean first-hit/cap steps per seed | Mean executed steps / rollout | Mean retrieval queries / actual rollout | Total padded actions | Evaluation wall time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37/50 (74%) | 60.4%–84.1% | 81.5 | 56.0 | 320.3 | 1000.0 | 1000.0 | n/a | 3673.0 |
| 3 | 30/50 (60%) | 46.2%–72.4% | 71.4 | 59.5 | 442.8 | 442.8 | 148.0 | 2046 | 1432.0 |
| 5 | 36/50 (72%) | 58.3%–82.5% | 62.6 | 55.0 | 325.1 | 325.1 | 65.2 | 1989 | 1028.0 |
| 7 | 32/50 (64%) | 50.1%–75.9% | 55.0 | 49.5 | 395.2 | 395.2 | 56.8 | 2790 | 1203.0 |
| 9 | 33/50 (66%) | 52.2%–77.6% | 60.2 | 49.0 | 379.8 | 379.8 | 42.8 | 2581 | 1131.6 |

Successful-rollout step quantiles (25th–75th percentile): K=1: 44.0–102.0; K=3: 46.5–81.8; K=5: 43.5–74.0; K=7: 38.8–60.5; K=9: 44.0–57.0.

## Paired outcomes by reset seed

`a_only` means success under the first K but failure under the second; `b_only` means the reverse. The p-value is a two-sided exact McNemar test on discordant seeds; the comparisons are descriptive and exploratory, and no multiplicity correction is applied.

| Comparison | Both succeed | A only | B only | Neither | B − A success rate | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| K=1 vs K=3 | 24 | 13 | 6 | 7 | -14.0% | 0.1671 |
| K=1 vs K=5 | 27 | 10 | 9 | 4 | -2.0% | 1.0000 |
| K=1 vs K=7 | 25 | 12 | 7 | 6 | -10.0% | 0.3593 |
| K=1 vs K=9 | 23 | 14 | 10 | 3 | -8.0% | 0.5413 |
| K=3 vs K=5 | 23 | 7 | 13 | 7 | +12.0% | 0.2632 |
| K=5 vs K=7 | 25 | 11 | 7 | 7 | -8.0% | 0.4807 |
| K=7 vs K=9 | 21 | 11 | 12 | 6 | +2.0% | 1.0000 |

## Interpretation and limits

The best observed point estimate is K=5; the four chunk settings are close and their Wilson intervals overlap substantially. A single trained checkpoint was evaluated, so these intervals describe variation across simulator reset seeds only and do not measure training-seed uncertainty. First-hit completion says that Lift height was reached once; it does not claim the object was held or that the controller learned an endpoint. The action-chunk setting changes both open-loop duration and how often visual retrieval is refreshed, so this is an execution-horizon comparison, not a training-method comparison.

All 50 rollout rows and per-seed first-hit steps are tracked with each evaluation. Videos and detailed per-query action/retrieval traces are local, Git-ignored artifacts. The evaluator reports action-padding totals because repeated final demonstration actions are part of this unchanged retrieval boundary policy.

## Reproduction

From this experiment folder, run `scripts/summarize_action_chunk_ablation.py`. To reproduce the simulator measurements themselves, see the four frozen commands in `README.md`; the original local checkpoint and bank are required. Each evaluation stores its resolved setup and artifact checksums under `experiments/runs/alt_lift_ph30_image_proprio_seed42_v1/evaluations/`.
