# ALT-30 run report

**Run:** `alt_lift_ph30_image_proprio_seed42_v1`
**Status:** completed — one training seed, 50 fixed simulator starts
**Split:** `configs/lift_ph_split42.json`, SHA-256 `87ce7bec946683f5434d95c12bac51033ff41ed981247b5eedf7409973921181`

## Outcome

- Training: 40 epochs, final NT-Xent 1.801854, 618.55 seconds. Fresh fine-tuned ImageNet ResNet-18; model seed 42, dataloader seed 43, augmentation seed 44.
- Memory: 1,456 observation embeddings from exactly the 30 train IDs; zero validation IDs. Bank SHA-256 `7f24e98bcb6f406309915b0c1b205fe59bcc1b93f62fa5ed01dec88aa81f1637`.
- Evaluation: seeds 100000–100049, 1,000 control steps each at 20 Hz, 50,000 retrieval queries total. 37 episodes reached the Lift height criterion at least once (74%; Wilson 95% interval 60.4%–84.1%). All 50 final sparse rewards were zero.
- Every rollout reached the explicit step cap. RoboMimic 0.2.0's `EnvRobosuite` forces its wrapped environment to ignore internal done and returns `is_done() == False`; the evaluator applies and records the 1,000-step time limit. No rollout ended early.
- Evaluation used one trained model seed, so the interval does not measure model-training variability.

See [`docs/final_analysis.md`](../../../docs/final_analysis.md) for interpretation, offline validation diagnostics, retrieval behavior, limitations, and artifact retention details. Per-seed rows and aggregate metrics are in `evaluations/evaluation01/`; the checkpoint, bank, detailed query trace, and three predeclared videos remain in ignored local `artifacts/` and are identified by checksums in the adjacent manifests.
