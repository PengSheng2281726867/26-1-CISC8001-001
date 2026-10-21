# alt_030_split42_model42_pilot_v1

Status: completed; scope: single_model_seed_scale_pilot.

Official mean score: 0.274424; SD: 0.2831968418883916; 95% bootstrap CI: [0.19974243617725407, 0.3543433780296769]; n=50.

Score = mean over frozen seeds of episode maximum clip(coverage/0.95,0,1). No binary success rate is introduced. Final score is the last actual reward.

Model seed 42; split SHA256 1b786b545a200f8fbbb28fb9385bc1ff80ed8699aba7c14d6fd117af7670322a; code start commit 69e3517fd53f755e8421e8a2f421858e40d81eb2.

Training time: 571.28s; evaluation time 46.72s. Bank: 3806 entries; invalid events: 0.

This is a single-model-seed pilot (or untrained integration smoke as labeled), not a multi-seed final comparison. Weights, bank, full traces and bounded videos are local ignored artifacts; their checksums are in artifact_manifest.json. Git alone does not back them up.

## Inspection and limitations

The predefined video seed100000 has official score0. Its recorded images and all-query trace show a repeated late retrieval of expert episode67/frame30. At query step192, current pusher=[198.1180,112.2444], matched expert pusher=[295.9028,93.3716], cosine=0.9924289. The same match and current position recur at step256. High latent similarity therefore does not ensure a useful absolute action chunk. The contact sheet is in results/alt_030_split42_model42_pilot_v1_diagnostic.png.

Across all1875 queries, mean current-to-matched pusher distance is63.03 original world units (median46.74,p95=184.85);940 episode switches and14 padded queries were logged. These are diagnostics, not new policy inputs or compensation rules. Four predefined videos were decoded for coordinate/frame-count checks; compressed96px blue-centroid measurements are approximate and are recorded in rollout_inspection.json.

Every bank action chunk and every rollout command passed indexing/bounds checks; raw-frame bank re-encoding, latest-observation selection, original/instrumented transition parity and official-score telemetry checks passed. The evidence points to limitations of state similarity and fixed absolute action reuse; it does not establish a specific encoder modification as a solution. No augmentation, geometry, OOD, horizon, seed or reward change was made in response to the low score.

The validation contrastive loss is diagnostic only and is not a policy-quality metric. This single model seed cannot establish a general scale effect. The one-seed untrained smoke score is not comparable to this50-seed aggregate. Model seeds43/44 remain planned.

Reproduction note: policy/training code at run start is69e3517. Use the frozen split blob from7a13763 or later to preserve the exact recorded SHA256; that Git newline-storage correction did not change the working manifest or the running training method.
