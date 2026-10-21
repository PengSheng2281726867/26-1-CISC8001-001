# B1 experiment ledger

The only planned formal run is a fresh ALT-30 model42, 40 epochs, lambda_pose=1.0.
Phases: read-only audit; implementation/tests; training/bank; heldout diagnostics;
official fixed-50-seed evaluation; paired report. No new split or scale study.

2026-10-21T08:35Z: Phase 0 passed. Rehashed full dataset; exact 30/20 IDs and 50 seeds
match baseline. Ten first/last RGB reconstructions are exact. Existing RTX5090 environment
works. Full audit and preservation guard saved. Next: implement and validate B1 only.
