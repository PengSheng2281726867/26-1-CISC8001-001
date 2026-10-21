# B1 ALT-30 measured analysis

See [Chinese interpretation and causal-chain checks](interpretation.md) for the
observed failure modes, distinction between evidence and hypotheses, and limitations.

Completed formal40-epoch training, all2426 heldout-frame diagnostics, and the original
official50-seed Push-T Runner evaluation. Only one model seed42 per method.

| Metric (mean) | Original ALT-30 | B1 |
|---|---:|---:|
| Official max-reward score | 0.274423899 | 0.178184988 |
| Heldout matched pusher xy L2 | 39.847 | 46.628 |
| Heldout matched block xy L2 | 46.417 | 32.201 |
| Heldout matched angle error degrees | 38.312 | 17.273 |
| Eight-action chunk RMSE | 50.028 | 47.136 |
| Numeric-xy swap top1 changed fraction | 0.015625 | 0.023438 |
| Consecutive same-match transition fraction | 0.219178 | 0.620370 |
| Same-match + same-position transition fraction | 0.198356 | 0.577887 |

B1 pose prediction on all heldout frames: block xy L2 mean/median/P95
20.340/14.957/57.025 world units;
wrapped angle error 13.550/3.455/77.906 degrees.
The original baseline has no trained pose-output head, so there is no baseline prediction
error to compare. This directly measures B1 pose accuracy without claiming a measured
prediction improvement over that baseline. Retrieval and rollout outcomes are separate.

Paired B1-minus-baseline official score difference: -0.096238912; paired bootstrap95%CI
[-0.190911620, -0.000688685], using seed2026 and10000 resamples of the same50 simulator seeds.
B1 score SD 0.281800; mean95%CI [0.1066292596720526, 0.2620184695494257].
Changed seeds: 9 improved, 26 worsened,
15 tied. This interval is conditional on these two trained
models and does not estimate variation across training seeds. Official score remains
episode maximum clip(coverage/0.95,0,1), averaged over seeds; final-score mean is
0.139588. No binary success threshold is introduced.

Training: 597.89s; 4720 Adam updates;
max allocated GPU memory 542949376 bytes. Bank entries
3806; checkpoint 45396739 bytes; bank
2042629 bytes. Evaluation 47.01s;
encoder/search/extraction mean latency (ms) 5.160/
0.186/0.626.
Peak-memory and latency depend on hardware/load, and the B1 checkpoint includes its unused head.

All action/source/chunk/episode indices were checked against original absolute commands;
invalid events 0. Actual official scores agree with independent
telemetry. Pose Head forward calls during all50 rollouts:0. Deterministic bank queries,
90 first/middle/last raw-frame reencodings, and all3806 action chunks passed. Exact self
indices may tie and are not unseen-state evidence. No label enters bank matching.

See offline_diagnostics.json for means/medians/P95, identical query-order checksum and
reproduction of the old diagnostic metrics. Position swapping is an artificial inconsistent
input probe. Action chunk difference does not identify a unique correct action. See
heldout_match_comparison.png, training_and_paired_scores.png, and actual decoded
video_trace_contact_sheet.png. Videos are the predefined first four seeds. Full traces
and videos remain local with hashes; artifact_manifest.json is not a Git backup.

Original source/config/split/result preservation guard passed for
93 files; both upstream references and full dataset fingerprint
were revalidated. Checkpoint hash 86d810168e5a501b39040074d7eb3fc474430c32c24855aac28f03bedbab5ec7; bank hash dfa97c7de682ab3429b5f078cdeec44648678398e006e1a2bddd6b24c962944f.
Training-start commit 43ce685a988071e786179a9c9c99edce43476ef1; analysis code hashes saved separately.
