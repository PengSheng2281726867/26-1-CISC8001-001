# B1 training and offline milestone

Formal training completed: 40epochs, 4720updates, 3806 frozen training frames,
597.886seconds. Final epoch selected in advance. Checkpoint and bank full hashes,
sizes, local paths and provenance are saved alongside this file. The original baseline
and split remain unchanged. Two diagnostic infrastructure failures were fixed and retained.

All2426 heldout frames were queried in the same order for original ALT-30 and B1.
Baseline metrics reproduce the earlier saved diagnostics. Independent NumPy FP64
checks pass; values below use the original FP32 diagnostic definition.

| Mean diagnostic | Original ALT-30 | B1 |
|---|---:|---:|
| Matched pusher xy L2 | 39.8466 | 46.6279 |
| Matched block xy L2 | 46.4169 | 32.2010 |
| Matched wrapped angle degrees | 38.3115 | 17.2725 |
| Expert8-action chunk RMSE | 50.0280 | 47.1362 |
| Numeric-position swap top1 change | 1.5625% | 2.34375% |

B1 pose-output xy L2 mean/median/P95:20.3403/14.9566/57.0252 world units.
Angle error:13.5500/3.4551/77.9063 degrees. Baseline has no trained pose head;
there is no direct baseline pose-prediction improvement measurement. B1's block
retrieval improves but pusher matching worsens and numeric-position dependence stays
weak. Expert-action RMSE is not action accuracy or a closed-loop score.

All3806 stored-vector self-queries and action chunks, plus90 raw-frame reencodings,
passed. This validates integration, not unseen-state quality. All7 regression tests
and the official untrained integration smoke passed. Formal B1 closed-loop evaluation
has not run at this milestone. No score claim is made here.
