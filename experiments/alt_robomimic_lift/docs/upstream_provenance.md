# Upstream and dataset provenance

## ALT method reference

- Paper: He et al., *Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives* (2025), arXiv:2505.05787.
- Official source: <https://github.com/StanfordMSL/alt>.
- The supplied read-only snapshot is `D:\Code\Course_code\ACl\alt`. It has no Git metadata, so its commit cannot be identified. SHA-256 hashes for `training.py`, `rollout.py`, `data_creation.py`, `LICENSE.md`, and `README.md` are recorded in `results/source_manifest.json`.
- The official GitHub repository reports an MIT license. The local snapshot also includes an MIT text, but its copyright header is unrelated to the ALT authors; no source files are copied into this adaptation. The implementation here is independently written from the paper's contrastive-nearest-neighbor method and the verified Lift interface. Attribution is retained in documentation and module headers.
- The local main-branch `training.py` uses an ImageNet-pretrained ResNet-18, two image inputs, NT-Xent, batch size 32, 40 epochs, learning rate 1e-4, and temperature 0.5. Its pose MLP and pose-fusion path are commented out in the forward pass. The Lift baseline intentionally enables a proprioception branch and fuses two current camera views plus 9D current proprioception; it is not a line-for-line port. Lift uses temperature 0.4 and only mild photometric augmentation, with no horizontal flips or geometric image changes.

## RoboMimic and RoboSuite

- RoboMimic source and dataset version: 0.2.0. The official v0.2.0 GitHub release identifies commit prefix `62ed2de`; the locally installed package reports exactly 0.2.0. The original collection commit is absent from the dataset metadata.
- RoboSuite: the local immutable vendor snapshot is `offline_study`, commit `d9baa17d5328d7b0ec15ecda54c76c61bb983124`, version 1.2.0. This is the compatibility line used by the 0.2.0 dataset and the existing validated environment.
- MuJoCo: 2.0.0 engine, `mujoco-py` 2.0.2.9. Runtime and downloaded binary hashes are in the existing environment validation records under `artifacts/environments/robomimic_lift_legacy/`; no environment package or source was changed for this experiment.

## Dataset

The source HDF5 URLs, byte sizes, and full-file SHA-256 hashes are in `results/dataset_audit.json`. Image and low-dimensional files share the exact 200 episode IDs and identical expert action arrays. The original HDF5 files remain outside this experiment directory and are not copied or committed.

## Policy and training provenance

The only allowed policy modalities are one current `agentview_image`, one current `robot0_eye_in_hand_image`, and current `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`. Encoder initialization must be the verified torchvision ImageNet ResNet-18 weights, fine-tuned with NT-Xent. No pretrained ALT task checkpoint is used. Run-level configuration, checkpoint hashes, memory-bank hashes, and the exact data/evaluation manifest references will be added when the first training run is made.
