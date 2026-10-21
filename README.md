# Push-T ALT: single RGB frame + current position

ALT-only study of 30/100/300/500 independent demonstrations. The local dataset contains 206 episodes: reserve20, train pool186; 30/100 are feasible, 300/500 are NOT_FEASIBLE_WITH_CURRENT_DATA. No Diffusion Policy model is trained.

## Windows environment

Project: `D:\Code\Course_code\COURSE-HUZI-CODE`. References `D:\Code\Course_code\ACl\alt` and `D:\Code\Course_code\DP\diffusion_policy` remain read-only. Their exact commits, licenses and differences are in docs/upstream_provenance.md and docs/implementation_notes.md. Dataset prerequisite is the existing `DP/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr`; raw data is not included.

The Zarr store has `data/img` (RGB frames), `data/state` (five numbers per frame: pusher x,y, block x,y, block angle in radians), and `data/action` (expert absolute target x,y). ALT reads only the current image and the first two state columns. Current pusher coordinates and expert action targets are different quantities. Actions do not supervise contrastive encoder training; they are retrieved afterward. See docs/data_audit.md for synchronized frame/coordinate examples and results/alignment_debug.png for viewable images.

A verified Python3.10 / Torch2.7.1+cu128 / RTX5090 interpreter already exists. It is reused without installing into or modifying the reference environment. Do not use the default system Python3.14 for old Push-T dependencies.

```powershell
Set-Location D:\Code\Course_code\COURSE-HUZI-CODE
$altPython = 'D:\Code\Course_code\DP\.conda\pusht\python.exe'
& $altPython -B scripts/run.py audit
& $altPython -B scripts/run.py prepare-weights
& $altPython -B scripts/run.py check
```

All paths are in configs/alt_pusht_base.yaml and can be changed using `--config <your-config.yaml>`; library modules contain no machine-specific source/data paths. For a new independent environment, use Python3.10, the official PyTorch cu128 index, and the version snapshot requirements.txt. Gym0.21 requires an old-compatible build toolchain (e.g. pip23/wheel0.38); this snapshot is not a promise of one-command modern pip installation. DP environment dependencies must be present; do not install this project into either reference directory.

## Reproducibility and gates

`audit` checks every data array for finite values/ranges and hashes every dataset file. Five independent episodes reconstruct their stored RGB exactly from current stored state (diagnostic only). It writes docs/data_audit.md, results/data_audit.json and results/alignment_debug.png. Frozen split JSON has exact IDs and seeds plus SHA256 sidecar; an existing changed split is rejected, never overwritten. Every experiment must validate this contract. ImageNet weights are downloaded from the official torchvision URL into ignored artifacts/pretrained, verified by hash prefix, and their full SHA256 is recorded. Random-backbone fallback is prohibited.

`.gitattributes` disables Git newline conversion for the frozen split JSON and checksum files. Their checksums cover exact bytes; the original Windows CRLF bytes are intentionally preserved in Git. This storage rule changes neither IDs/order nor the frozen manifest SHA256. Reconstructing a manifest from a Git blob must reproduce its advertised checksum.

`check` verifies split leakage, action alignment and all episode boundaries, independent augmentation, an analytic NT-Xent reference, real pretrained loading/backbone gradient update, normalized128D embeddings, and deterministic bank/query/save/load using real Push-T images.

The first commit includes verified audit, frozen split and working core. Seven gates now also check latest-frame policy selection, reset/API/absolute action units, and exact transition parity between the instrumented and original environments. See docs/experiment_log.md and docs/failure_log.md for measured progress.

## Training and official closed-loop evaluation

Run the cheap untrained integration smoke before starting a fresh trained pilot. The smoke uses only the first frozen test seed and is explicitly not a trained-policy benchmark. Each new train command starts from the same verified ImageNet weights, never a previous scale's checkpoint. Use a new run ID for every attempt; existing configs and artifacts are never overwritten.

```powershell
$smokeRun = 'alt_030_split42_model42_smoke_reproduction_v1'
$pilotRun = 'alt_030_split42_model42_reproduction_v1'
& $altPython -B scripts/run.py smoke --run-id $smokeRun
& $altPython -B scripts/run.py train --run-id $pilotRun --scale 30 --model-seed 42
& $altPython -B scripts/run.py bank --run-id $pilotRun
& $altPython -B scripts/diagnose_bank.py --run-id $pilotRun
& $altPython -B scripts/run.py evaluate --run-id $pilotRun
& $altPython -B scripts/inspect_rollout.py --run-id $pilotRun
& $altPython -B scripts/run.py report
```

The names above are fresh reproduction IDs, separate from the saved pilot records. Choose another unused suffix when repeating them. To reproduce scale100, use a new run ID with `--scale 100` in `train`, then run the same bank/diagnostic/evaluation commands with that ID. Keep model seed42 and the frozen split/config unchanged for this paired comparison.

Training uses Adam, 40 epochs, batch32, lr1e-4, tau0.4, full backbone fine-tuning and two independent per-observation photometric views. Final-epoch checkpoint selection is frozen. Held-out contrastive loss is computed once at the final epoch for diagnosis only. Evaluation uses the **unmodified official PushTImageRunner.run** method with n_train=0, n_test=50, seeds100000..100049, legacy_test=True, n_obs_steps=1 and eight-step execution. n_envs=1 gives simple, exact per-episode telemetry. A subclass records base termination/actual steps; a proxy records rewards and timing. A parity test proves the subclass preserves original transitions. Telemetry and official Runner scores are independently cross-checked.

Run directories contain immutable resolved YAML, split/config references, event history, training/validation metrics, per-rollout CSV, two full query samples per evaluated seed, aggregate metrics, and artifact checksums. Full query traces (all eight commands and source indices), checkpoints, memory banks and the predefined first four videos are under ignored artifacts/runs/. `bank` and `evaluate` load the run's immutable resolved config. `--scale` and `--model-seed` apply when creating a fresh run. Failed training requires a new ID.

`diagnose_bank.py` queries every stored embedding, re-encodes first/middle/last observations of each selected demonstration with deterministic preprocessing, and verifies every stored frame's eight-action source indexing and episode boundary. Exact self indices can differ for duplicate or floating-point-tied embeddings; the maximum similarity and embedding equivalence are checked and recorded explicitly in bank_validation.json. The smoke bank passes3806 self-queries,90 raw-frame re-encodings and3806 action-chunk checks;3801 self indices are exact and all3806 returned embeddings are equivalent within L2<1e-3.

`report` writes results/scale_summary.csv, training-loss/official-score figures and a diagnostic contact sheet showing actual rollout video frames above the retrieved expert images. These figures describe completed saved runs only. Simulator-seed bootstrap intervals do not measure uncertainty across independently trained model seeds. Begin with one model seed per scale; model seeds42/43/44 are planned for a later controlled comparison if resources permit.

Once both scales are completed, `report` checks that their resolved settings differ only in scale and pairs the same50 simulator seeds to write results/paired_scale_comparison.json and results/paired_scale_scores_model42.png. The difference interval resamples paired simulator seeds. Both scales use40 epochs; larger datasets therefore receive more optimizer updates, so this is not a comparison at equal compute.

## Measured milestone

ALT-30 model42 has completed40 epochs and all50 fixed-seed official rollouts: mean score0.2744, SD0.2832, simulator-seed bootstrap95%CI[0.1997,0.3543]. This is a weak single-model-seed pilot, not a final claim that ALT reliably solves Push-T. The predefined failure video and traces show high-similarity retrievals with mismatched pusher positions and repeated action chunks. Full results/limitations are in experiments/runs/alt_030_split42_model42_pilot_v1/report.md; primary metrics are eval_metrics.json and eval_rollouts.csv. ALT-100 and additional model seeds are separate fresh runs, never continuations of this checkpoint.

ALT-100 model42 also completed40 epochs and the same50 official rollouts: mean score0.2745, SD0.3023,95%CI[0.1947,0.3591]. Its paired difference from ALT-30 is+0.0001, paired95%CI[-0.0970,0.0932]; this pilot does not demonstrate a scale benefit. Some simulator seeds improve while others regress. Training took1997.52s versus571.28s; bank entries grow12618 versus3806. All bank/action/trace checks pass, yet predefined videos still show wrong matches and stagnation. See experiments/runs/alt_100_split42_model42_pilot_v1/report.md and results/paired_scale_comparison.json. This is one trained model seed per feasible scale; replications43/44 remain unperformed, and300/500 remain unavailable with current data.

Architecture: ResNet18 image128 + position MLP2->64->32, fusion160->256->128, L2 normalization. Backbone fine-tuned; NT-Xent tau0.4; photometric changes only. Retrieval top1 cosine, no OOD/fallback/blending, eight original absolute target positions per query, same-episode last-action padding.

Heavy checkpoints, banks, videos and caches remain in ignored artifacts/. Git stores small manifests, metrics and plots only; it is not a backup of heavy local artifacts. Official score is mean over fixed seeds of each episode's maximum clip(coverage/0.95,0,1), rather than contrastive loss or an invented success rate.

## Forward-only failure analysis

The saved models were inspected on all2426 frozen heldout frames and256 artificial image/position sensitivity probes, without new training or simulator rollouts. Numeric-position replacement changes only1.56%/2.73% of the trained30/100 top1 matches; the measured representation is strongly image-dominated and has substantial geometric retrieval errors. Full-state geometry is used solely as an explicitly privileged diagnostic comparator, never as a policy input. See docs/retrieval_failure_analysis.md for evidence, caveats and unimplemented next steps.

```powershell
& $altPython -B scripts/diagnose_representation.py
```

Settings and selection rules are in configs/diagnostics/retrieval_failure_v1.json; summary/figure/independent metric checks are in results/diagnostics/. Existing analysis IDs are immutable: choose a new analysis_id in a copied config and pass `--config` when repeating the command. Full per-frame CSVs remain under ignored artifacts/diagnostics/ with sizes/checksums in the summary. Further training and closed-loop experiments remain paused for discussion.
