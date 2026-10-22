# Lift experiment log

| Date (UTC) | Phase / run | Status | Reason and decision |
|---|---|---|---|
| 2026-10-22 | Phase 0 dataset/environment audit | PASS with replay drift noted | Verified both PH HDF5 files and hashes, 200 episodes / 9,666 frames, RGB/proprio/action alignment, controller semantics, image reconstruction, action replay on three selected demos, and 50 repeatable evaluation reset conditions. Full-state replay drift peaks at 8.75e-4. No training was performed. Commit `703da7c`. |
| 2026-10-22 | Phase 1 split contract | PASS | Frozen seed-42 permutation, 20 offline-validation episodes, exact ordered pool of 180, nested lists for 30/50/100, evaluation seeds `100000..100049`, and the ALT-30 resolved config hash. Five data/split/config contract tests pass. Commit `6d5f53b`; push succeeded. |
| 2026-10-22 | Phase 2 ALT core smoke | PASS | Six ALT-core tests plus five data/split tests pass in both the Python 3.10 CUDA training venv and Python 3.8 legacy simulator env. Pretrained checkpoint SHA-256 verified; one 32-sample CUDA forward/backward smoke produced finite NT-Xent (`3.309222`), normalized 128D embeddings, and gradients in image, proprio, and fusion branches. A Torch 2.7.1 state dict loaded strictly in Torch 1.10.2/TorchVision 0.11.3. The legacy `imageio`/FFmpeg backend encoded a 168x84 test frame. No formal training yet. |
| 2026-10-22 | `alt_lift_ph30_image_proprio_seed42_v1` | PLANNED | Exactly one fresh ResNet-18 ImageNet initialization, 40 epochs, NT-Xent temperature 0.4; Phase 2 implementation is validated and ready to commit before launch. |
