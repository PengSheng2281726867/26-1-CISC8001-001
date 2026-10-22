# ALT on RoboMimic Lift-PH

This folder contains a Lift-specific Action Lookup Table adaptation. It is isolated from the earlier Push-T work. This study does not train Diffusion Policy.

## Phase 0 status

Phase 0 passed with a documented numerical replay limitation. The local RoboMimic 0.2.0 Lift-PH image file contains 200 independent demonstrations and is already usable; no image rendering conversion was needed. The dataset's recorded environment is an older RoboSuite `offline_study` setup, and the matching local simulator is RoboSuite 1.2.0 / MuJoCo 2.0.0. The source metadata does not identify a collection commit. See [the Phase 0 audit](docs/phase0_data_environment_audit.md) and [source provenance](docs/upstream_provenance.md).

The evaluation protocol fixes 50 seeds (`100000` through `100049`) and preserves verified initial MuJoCo states and hashes in `results/evaluation_initial_states.json`. Those states are reset fixtures only; the policy receives two current RGB images and current robot proprioception.

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

The remaining phases are gated: freeze the Lift-30 split, implement and validate the ALT core, train one 40-epoch Lift-30 encoder from verified ImageNet initialization, build its memory bank, run offline diagnostics, then perform the fixed 50-condition closed-loop evaluation. Training and evaluation commands will be recorded here when the corresponding scripts are validated. No DP, Lift-50/100, or action-chunk ablation is run in this study.

## Outputs

- `configs/`: path and evaluation protocol; frozen split contract is added in Phase 1.
- `scripts/`: audit and environment commands, followed by training and evaluation tools.
- `results/`: small JSON manifests and metrics; raw data, checkpoints, memory banks, replay traces, and videos stay ignored under `artifacts/`.
- `docs/`: audit, method and experiment notes, failure log, and final analysis.

Only verified, small protocol records are committed. A reproduction of the final model will also need the local ignored checkpoint and memory-bank files whose checksums will be listed in the run manifest.
