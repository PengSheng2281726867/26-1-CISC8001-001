# alt_100_split42_model42_pilot_v1

Status: completed; scope: single_model_seed_scale_pilot.

Official mean score: 0.274523; SD: 0.3023253863341174; 95% bootstrap CI: [0.19474088280574658, 0.35914958482999404]; n=50.

Score = mean over frozen seeds of episode maximum clip(coverage/0.95,0,1). No binary success rate is introduced. Final score is the last actual reward.

Model seed 42; split SHA256 1b786b545a200f8fbbb28fb9385bc1ff80ed8699aba7c14d6fd117af7670322a; code start commit d90d44440c58631a56c732576572bd6ff7a75e0d.

Training time: 1997.52s; evaluation time 43.26s. Bank: 12618 entries; invalid events: 0.

This is a single-model-seed pilot (or untrained integration smoke as labeled), not a multi-seed final comparison. Weights, bank, full traces and bounded videos are local ignored artifacts; their checksums are in artifact_manifest.json. Git alone does not back them up.

## Controlled scale comparison

ALT-30 and ALT-100 use the same pretrained bytes, model/shuffle/augmentation seeds42/43/44, held-out episodes, frozen split hash,50 evaluation seeds and policy settings. Resolved YAML differs only in training.scale. Training/policy/config files at the two start commits69e3517 and d90d444 are identical. Report/diagnostic additions do not change the method.

| Quantity | ALT-30 | ALT-100 |
|---|---:|---:|
| Independent training episodes | 30 | 100 |
| Bank frames | 3806 | 12618 |
| Epochs | 40 | 40 |
| Optimizer updates | 4720 | 15760 |
| Training seconds | 571.28 | 1997.52 |
| Official mean score | 0.274424 | 0.274523 |
| Score SD | 0.283197 | 0.302325 |
| Mean final score | 0.083174 | 0.037305 |
| Simulator-seed bootstrap95% CI | [0.199742,0.354343] | [0.194741,0.359150] |

The paired mean difference (100 minus30) is0.0000988075; paired bootstrap95% CI[-0.0969667,0.0932412],10000 resamples, seed2026. There is no demonstrated improvement in this pilot; this does not prove equivalence or that more demonstrations can never help. The interval conditions on model seed42. Seeds43/44 remain planned, not run. Equal epochs give different optimizer-update budgets; no equal-compute claim is made. Paired raw differences and settings are in results/paired_scale_comparison.json.

## Video and retrieval inspection

All12618 bank self-queries recover equivalent embeddings (12597 exact indices; ties explain the others),300 raw frames re-encode consistently, and every eight-action chunk stays in its demonstration. All1899 actual rollout queries have finite, in-bounds commands. Official scores independently agree with reward telemetry; all four predefined videos have the exact actual action/frame count. There are14995 actual control steps,1261 episode switches and34 padded queries. Mean/median/p95 current-to-matched pusher distance is84.24/68.92/212.90 original world units. These diagnostics are not new policy inputs or geometric corrections.

The predefined seed100000 reaches a maximum score0.572465 but ends at0. Its video shows the block leaving the target area and later stagnating. Queries272,280,288,296 all retrieve episode81/frame68, cosine0.9993016, with the same current pusher=[164.1615,391.1975] and expert pusher=[153.2489,401.9590]. Seed100001 at step296 retrieves episode55/frame19 with cosine0.998448 despite current pusher=[186.7641,141.2155] versus expert=[31.2703,157.1015]. Seed100002 scores0 and alternates the same two matches/positions at steps272/288 and280/296 (68:32 and120:28). The observed failures include unsuitable matches, repeated chunks and cycles; no specific encoder-level causal mechanism is proved.

No split, seed, scoring, action offset/type, augmentation geometry, input, execution horizon, OOD or fallback rule changed after observing these results. Reference source commits/tracked bytes and all dataset bytes remain unchanged. The local checkpoint/bank/videos and full trace inventory carry file sizes and checksums. Git does not contain those heavy artifacts.
