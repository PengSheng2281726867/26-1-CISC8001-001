# ALT on RoboMimic Lift-PH

This folder contains a Lift-specific Action Lookup Table adaptation. It is isolated from the earlier Push-T work. This study does not train Diffusion Policy.

## Phase 0 status

Phase 0 passed with a documented numerical replay limitation. The local RoboMimic 0.2.0 Lift-PH image file contains 200 independent demonstrations and is already usable; no image rendering conversion was needed. The dataset's recorded environment is an older RoboSuite `offline_study` setup, and the matching local simulator is RoboSuite 1.2.0 / MuJoCo 2.0.0. The source metadata does not identify a collection commit. See [the Phase 0 audit](docs/phase0_data_environment_audit.md) and [source provenance](docs/upstream_provenance.md).

The evaluation protocol fixes 50 seeds (`100000` through `100049`) and preserves verified initial MuJoCo states and hashes in `results/evaluation_initial_states.json`. Those states are reset fixtures only; the policy receives two current RGB images and current robot proprioception. Phase 1 has frozen 20 held-out demos and the nested 30/50/100 episode lists in `configs/lift_ph_split42.json`; only ALT-30 is planned in this run.

## Local prerequisites

- The repository's existing `artifacts/datasets/robomimic/lift/ph/image.hdf5` and `low_dim.hdf5` files. Their full SHA-256 values are checked by the audit script.
- WSL Ubuntu and the repository's prepared `artifacts/environments/robomimic_lift_legacy` Python 3.8 environment for RoboSuite validation and rollouts.
- The prepared legacy environment needs its MuJoCo path set. `scripts/with_legacy_env.sh` does that and places Python caches under this Lift folder.
- A Windows PowerShell session and the local `wsl.exe` distribution named `Ubuntu`.

The default data paths are relative in `configs/data_paths.json`. Override them with `--image-dataset` and `--low-dim-dataset`; Windows drive paths and WSL paths are accepted.

## Phase 0 commands

From PowerShell, set the working directory to this folder and run:

```powershell
$lift = '/mnt/d/Code/Course_code/COURSE-HUZI-CODE/experiments/alt_robomimic_lift'
wsl.exe -d Ubuntu --cd $lift -- /bin/bash -lc 'bash scripts/with_legacy_env.sh scripts/audit_dataset.py'
wsl.exe -d Ubuntu --cd $lift -- /bin/bash -lc 'bash scripts/with_legacy_env.sh scripts/validate_environment.py'
wsl.exe -d Ubuntu --cd $lift -- /bin/bash -lc 'bash scripts/with_legacy_env.sh scripts/build_eval_initial_states.py'
```

`audit_dataset.py` writes the verified HDF5 schema and hashes to `results/dataset_audit.json`; `validate_environment.py` checks state restoration, camera alignment, and continuous expert-action replay; `build_eval_initial_states.py` freezes and repeat-checks the evaluation reset fixtures. Visual contact sheets and compressed replay traces stay under ignored local directories.

Freeze and test the experiment split contract with:

```powershell
$lift = '/mnt/d/Code/Course_code/COURSE-HUZI-CODE/experiments/alt_robomimic_lift'
wsl.exe -d Ubuntu --cd $lift -- /bin/bash -lc 'bash scripts/with_legacy_env.sh scripts/freeze_split.py'
wsl.exe -d Ubuntu --cd $lift -- /bin/bash -lc 'bash scripts/with_legacy_env.sh -m unittest discover -s tests -v'
```

Phase 1 is frozen and its data/split tests pass. Phase 2 adds the Lift-specific shared-ResNet encoder, NT-Xent training, deterministic memory bank, and one-step retrieval policy. Only ALT-30 is run. No DP, Lift-50/100, or action-chunk ablation is part of this study.

## Training environment and pretrained weights

Use the experiment-local Python 3.10 virtual environment. It reuses the existing CUDA-enabled Torch/TorchVision packages through `--system-site-packages`; no package was installed into or changed in the Push-T reference environment. The current validated versions are Python 3.10.22, Torch 2.7.1+cu128, TorchVision 0.22.1+cu128, h5py 3.7.0, and NumPy 1.23.5 on the RTX 5090. The separate WSL Python 3.8 environment remains dedicated to the older RoboSuite 1.2.0 simulator.

The required ImageNet ResNet-18 weights are read from `../../artifacts/pretrained/resnet18-f37072fd.pth`, relative to this experiment folder. SHA-256 is checked against `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` before training; the classifier is replaced by the 128D feature head after loading. If your copy is elsewhere, update `configs/runtime_paths.json` or pass another paths config. Do not continue with random initialization if the verified weights are unavailable.

Create the experiment-local venv once if needed:

```powershell
$trainPython = 'artifacts/environments/alt_lift_train/Scripts/python.exe'
if (-not (Test-Path $trainPython)) {
  D:\Code\Course_code\DP\.conda\pusht\python.exe -m venv --system-site-packages artifacts/environments/alt_lift_train
}
```

The dataset paths, pretrained checkpoint, tracked run directory, and ignored artifact directory are configured in `configs/runtime_paths.json`. The frozen experiment YAML and split contract are hash-checked and must not be regenerated or edited.

## Phase 2 training and bank commands

From PowerShell in this folder, run the one-batch smoke check, then start the single scheduled 40-epoch ALT-30 run:

```powershell
$trainPython = 'artifacts/environments/alt_lift_train/Scripts/python.exe'
& $trainPython scripts/train_alt_lift.py --smoke
& $trainPython scripts/train_alt_lift.py
& $trainPython scripts/build_memory_bank.py --run-dir experiments/runs/alt_lift_ph30_image_proprio_seed42_v1
& $trainPython scripts/diagnose_offline.py --run-dir experiments/runs/alt_lift_ph30_image_proprio_seed42_v1
```

After training, the cross-version checkpoint loader can be checked in the WSL simulator runtime with `scripts/check_checkpoint.py --checkpoint artifacts/runs/alt_lift_ph30_image_proprio_seed42_v1/checkpoint_final.pt`.

The output under `experiments/runs/<run-id>/` contains the resolved config, exact split reference, software and seed metadata, per-epoch loss, checkpoint/bank checksums, and small run reports. The checkpoint, bank, full query trace, and videos are kept under the ignored `artifacts/runs/<run-id>/`. A run directory is immutable; use `--run-id` with a new name for a documented retry.

Training uses the exact 30 train IDs in `configs/lift_ph_split42.json`; the 20 validation demos are not used for normalization, contrastive pairs, or the action bank. Two independently photometrically augmented views preserve camera pairing and proprioception. The ResNet is shared by both cameras and fine-tuned. The only loss is NT-Xent at temperature 0.4. At inference, the top-1 match supplies one original 7D incremental OSC_POSE/Panda command; the policy queries again at the next control step. Chunk lengths 4 and 8 are unit-tested for same-episode padding but are not evaluated.

## Phase 3 fixed-seed evaluation

After the bank and offline diagnostics pass, run the fixed 50 conditions once in the legacy simulator. The script first checks a frozen seed's reset observation and the 7D policy API, then proceeds over all seeds `100000..100049`; do not use evaluation output to modify the model or protocol.

```powershell
$lift = '/mnt/d/Code/Course_code/COURSE-HUZI-CODE/experiments/alt_robomimic_lift'
wsl.exe -d Ubuntu --cd $lift -- /bin/bash -lc 'bash scripts/with_legacy_env.sh scripts/evaluate_alt_lift.py --run-dir experiments/runs/alt_lift_ph30_image_proprio_seed42_v1'
```

The evaluation writes the 50 rollout rows and aggregate Lift success rate with Wilson 95% interval to `experiments/runs/<run-id>/evaluations/evaluation01/eval_metrics.json` and `results/evaluation_summary.json`. It writes per-step retrieval matches/actions to the ignored artifact directory. Three predeclared videos are saved there for seeds 100000, 100025, and 100049. Success means RoboSuite Lift's task check became true at least once in an episode; each episode continues to termination or the frozen 1,000-step limit.

## Outputs

- `configs/`: path and evaluation protocol; frozen split contract is added in Phase 1.
- `scripts/`: audit, training, bank creation, offline diagnostics, and fixed-seed evaluation tools.
- `experiments/runs/`: versioned per-run configs, metadata, loss, checksums, and aggregate evaluation rows.
- `results/`: small audit and aggregate metric files; HDF5 data, checkpoints, memory banks, detailed traces, and videos stay ignored under `artifacts/`.
- `docs/`: audit, method and experiment notes, failure log, and final analysis.

Only verified, small protocol records are committed. A reproduction of the trained policy also needs the local ignored checkpoint and memory-bank files; their sizes and SHA-256 hashes are recorded in the tracked run manifest.
