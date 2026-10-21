# Push-T ALT: single RGB frame + current position

ALT-only study of 30/100/300/500 independent demonstrations. The local dataset contains 206 episodes: reserve20, train pool186; 30/100 are feasible, 300/500 are NOT_FEASIBLE_WITH_CURRENT_DATA. No Diffusion Policy model is trained.

## Windows environment

Project: `D:\Code\Course_code\COURSE-HUZI-CODE`. References `D:\Code\Course_code\ACl\alt` and `D:\Code\Course_code\DP\diffusion_policy` remain read-only. Their exact commits, licenses and differences are in docs/upstream_provenance.md and docs/implementation_notes.md. Dataset prerequisite is the existing `DP/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr`; raw data is not included.

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

`check` verifies split leakage, action alignment and all episode boundaries, independent augmentation, an analytic NT-Xent reference, real pretrained loading/backbone gradient update, normalized128D embeddings, and deterministic bank/query/save/load using real Push-T images.

The first commit includes verified audit, frozen split and working core. Training/Runner integration follows as a separate validated gate. Training/evaluation commands and actual run IDs will be added when available; there are no claimed rollout results yet. See docs/experiment_log.md and docs/failure_log.md.

Architecture: ResNet18 image128 + position MLP2->64->32, fusion160->256->128, L2 normalization. Backbone fine-tuned; NT-Xent tau0.4; photometric changes only. Retrieval top1 cosine, no OOD/fallback/blending, eight original absolute target positions per query, same-episode last-action padding.

Heavy checkpoints, banks, videos and caches remain in ignored artifacts/. Git stores small manifests, metrics and plots only; it is not a backup of heavy local artifacts. Official score is mean over fixed seeds of each episode's maximum clip(coverage/0.95,0,1), rather than contrastive loss or an invented success rate.
