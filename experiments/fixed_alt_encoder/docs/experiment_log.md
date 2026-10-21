# B1 experiment ledger

The only planned formal run is a fresh ALT-30 model42, 40 epochs, lambda_pose=1.0.
Phases: read-only audit; implementation/tests; training/bank; heldout diagnostics;
official fixed-50-seed evaluation; paired report. No new split or scale study.

2026-10-21T08:35Z: Phase 0 passed. Rehashed full dataset; exact 30/20 IDs and 50 seeds
match baseline. Ten first/last RGB reconstructions are exact. Existing RTX5090 environment
works. Full audit and preservation guard saved. Next: implement and validate B1 only.

Phase 0 commit abd17cd was successfully pushed to origin/main.

2026-10-21: Phase 1 passed seven unittest cases (5.471s) and an untrained seed100000
official Runner smoke. Shared initialization tensors equal exactly; image-only pose
prediction ignores numeric xy; pose-only gradients reach conv1/projection/head; loss
weighting, periodic targets, all30 episodes' first/middle/last indexing, inference/bank
compatibility, reset/latest-frame and forbidden-input rejection pass. Formal training
uses the same batching/RNG recipe plus B1 loss. Original evaluator is imported with
explicit B1 model/bank, so no shared baseline registry/config is written. Next: one
prespecified fresh40-epoch run, then bank/heldout diagnostics before formal evaluation.

Phase 1 commit43ce685 was successfully pushed to origin/main.

2026-10-21T08:55Z: Phase 2 completed. Run b1_alt030_split42_model42_pose1_pilot_v1,
fresh pretrained model42, 40epochs, 4720updates, 3806frames; training597.886s,
peak allocated GPU542949376bytes. Final checkpoint86d81016..., bankdfa97c7d...;
full hashes and immutable resolved config are in runs/. No training retry occurred.

2026-10-21T08:58Z: Phase 3 completed after two documented infrastructure failures
(diagnostic output/input path wiring; Windows GBK Git-diff decoding). Both were fixed
inside this subexperiment; failed attempts and traces preserved. All3806 self/action
checks and90 raw-frame reencodings pass. Recomputed original baseline metrics agree
with its saved analysis. All2426 heldout queries per model independently checked in
NumPy FP64; largest FP32 cdist discrepancy0.140312 world units, negligible to reported
means and recorded explicitly. No data/method change or formal rollout yet.

B1 heldout pose mean xy20.3403 world units and angle13.5500deg. Fusion retrieval
block xy46.4169->32.2010 and angle38.3115->17.2725deg improve; pusher xy39.8466->46.6279
worsens. Action chunk RMSE50.0280->47.1362. Numeric-position swap top1 fraction
0.015625->0.0234375 remains low. These metrics do not establish a control benefit.
The first-four-validation-episodes' middle-frame comparison was visually inspected;
it shows improvements and regressions without selecting examples by result.
Seven regression tests and an untrained integration smoke passed again, including
exact equality between training-path z and inherited inference-path z. Next gate:
the original official50-seed closed-loop evaluation, once, without retuning.
