# ALT-30 RoboMimic Lift-PH — final analysis

**Status: completed pilot.** Run `alt_lift_ph30_image_proprio_seed42_v1` trained one ALT encoder on 30 frozen demonstrations and evaluated it on the 50 predeclared simulator initializations. No Diffusion Policy was trained, and no prior Push-T files were changed.

## Frozen method and provenance

- Dataset: local RoboMimic Lift-PH v0.2.0 image dataset, fingerprint `722a35b4b0620b5cc29a07cea33f2b64f3a34e0fbcead2ea666d4f9ccacdd8a1`. The split contract is `configs/lift_ph_split42.json` (SHA-256 `87ce7bec946683f5434d95c12bac51033ff41ed981247b5eedf7409973921181`): 20 held-out validation demos, 30 exact training demos for this run, and evaluation seeds `100000..100049`. The contract records the full ordered IDs.
- Model: shared, fine-tuned ImageNet ResNet-18 for the current agent-view and wrist RGB frames; 9D current EEF pose and gripper proprioception branch; 128D L2-normalized fused embedding. The verified pretrained weights hash is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Training: NT-Xent at temperature 0.4, two independent mild photometric views, batch 32, Adam at `1e-4`, 40 epochs, model seed 42, dataloader seed 43, augmentation seed 44. The ResNet was fine-tuned. No action loss was used; the 20 held-out demos did not enter training, normalization, or the bank.
- Retrieval: deterministic top-1 cosine over every eligible training observation (1,456 rows). The matched source action is the same-index 7D incremental OSC_POSE/Panda command `actions[t]`, applied to the current observation. One command is executed per step, with a new query at each 20 Hz control step; no OOD fallback, blending, or relative correction.
- Simulator: RoboMimic 0.2.0 / RoboSuite 1.2.0 / MuJoCo 2.0.0, 1,000 control steps per episode. The RoboMimic `EnvRobosuite` wrapper forces `ignore_done=True` and returns `is_done() == False`; the evaluator's explicit loop cap ends each rollout and is recorded as `time_limit=true`.

## Results

| Measure | ALT-30 result |
|---|---:|
| Fixed-seed rollouts | 50 / 50 |
| Ever reached Lift success height | 37 / 50 (74.0%) |
| Wilson 95% interval | 60.4%–84.1% |
| Final sparse reward positive | 0 / 50 |
| Steps per rollout | 1,000 for every seed |
| First-success step among the 37 that succeeded | median 56, mean 81.5 |
| Training time | 618.55 s |
| Evaluation wall time | 3,672.98 s total; 72.29 s per rollout on average |
| Mean query latency: preprocessing / encoder / cosine search / action extraction | 0.560 / 14.254 / 0.087 / 0.045 ms |

Success is the RoboSuite Lift task predicate becoming true at least once: the cube rises more than 0.04 m above the table offset. This is an **ever-success** measure, equivalent to whether the maximum sparse reward reached 1. The final sparse reward was zero in every rollout, so none of the 50 episodes ended with the cube still above that criterion. The 74% result therefore demonstrates transient lift-height achievement, not stable completion at the 1,000-step endpoint.

The model-seed uncertainty is not measured: these 50 trials share a single trained model (seed 42). The Wilson interval describes binomial uncertainty over these fixed-condition outcomes, not variation across model training seeds.

Held-out offline retrieval covered 993 frames from all 20 validation demos. Mean top-1 cosine was 0.7962 (SD 0.1081); mean retrieved-versus-query EEF position gap was 0.0220 m, quaternion geodesic gap 7.41°, and gripper qpos gap 0.00238. Mean action-vector discrepancy was 0.5234 and is not an accuracy score. Exact bank-row self-retrieval was 100%, a bank integrity check rather than a generalization result.

The rollout traces show substantial retrieval repetition and switching: the matched frame repeated on 51.4% of consecutive queries on average, with 412.3 demonstration-episode switches per 1,000-step rollout. All 50,000 trace rows were checked: every match belongs to one of the 30 training episodes, every frame index is valid, and each action is finite, 7D, and within the controller's `[-1, 1]` input bounds. The descriptive success and failure groups had similar repeat fractions (51.2% and 52.2%) and episode-switch counts (415.5 and 402.9); these measurements do not explain outcomes causally. Their detailed summary is in `results/failure_diagnostics.json`.

## Audit trail and local artifacts

- Per-seed scores and the full aggregate: `experiments/runs/alt_lift_ph30_image_proprio_seed42_v1/evaluations/evaluation01/eval_rollouts.csv` and `eval_metrics.json`. The compact mirrors are `results/per_seed_scores.csv` and `results/evaluation_summary.json`.
- Run config, exact IDs, training curve, code/data provenance, and checksums: the run directory's `resolved_config.yaml`, `dataset_manifest_ref.json`, `run_metadata.json`, and `artifact_manifest.json`.
- The checkpoint is 45,494,033 bytes, SHA-256 `7eafe7b99b45706fb6a485b9192f982a7cd7284444f12da6e498bd00316d2e4e`. The 1,456-entry bank is 744,890 bytes, SHA-256 `7f24e98bcb6f406309915b0c1b205fe59bcc1b93f62fa5ed01dec88aa81f1637`.
- The 50,000-query trace and videos for predeclared seeds 100000, 100025, and 100049 are in ignored `artifacts/runs/alt_lift_ph30_image_proprio_seed42_v1/evaluations/evaluation01/`. Their sizes and SHA-256 values are in that evaluation's `artifact_manifest.json`; the files are intentionally not in Git.
- Data audit, source provenance, protocol, and known simulator replay drift are documented in `docs/phase0_data_environment_audit.md`, `docs/upstream_provenance.md`, and `docs/experiment_protocol.md`. Expert replay had a maximum full-state divergence of `8.75e-4`; this remains a numerical limitation.

The tracked reports and manifests do not contain the raw HDF5 files, model checkpoint, memory bank, detailed 50,000-step trace, or videos. Reproducing this exact trained policy therefore requires those local ignored artifacts as well as the dataset and compatible simulator environments. The artifact checksums are recorded, but Git does not back them up.

## Conclusion

The ALT-30 pipeline passed the data, model, bank, reset-fixture, and closed-loop integration gates. Under this protocol it lifted the cube above the success threshold at least once in 37 of 50 fixed starts, but all 50 endpoints had zero sparse reward. Treat this as a mixed pilot result: retrieval can cause transient lift, while sustained terminal success was not observed. One model seed is insufficient to establish training-seed robustness, and no data-scale conclusion is available. This study stops at ALT-30; it does not train DP or run Lift-50/100.
