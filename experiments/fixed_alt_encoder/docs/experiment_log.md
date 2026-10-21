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
