# Lift experiment log

| Date (UTC) | Phase / run | Status | Reason and decision |
|---|---|---|---|
| 2026-10-22 | Phase 0 dataset/environment audit | PASS with replay drift noted | Verified both PH HDF5 files and hashes, 200 episodes / 9,666 frames, RGB/proprio/action alignment, controller semantics, image reconstruction, action replay on three selected demos, and 50 repeatable evaluation reset conditions. Full-state replay drift peaks at 8.75e-4. No training was performed. Commit `703da7c`. |
| 2026-10-22 | Phase 1 split contract | PASS | Frozen seed-42 permutation, 20 offline-validation episodes, exact ordered pool of 180, nested lists for 30/50/100, evaluation seeds `100000..100049`, and the ALT-30 resolved config hash. Five data/split/config contract tests pass. Commit pending. |
| 2026-10-22 | `alt_lift_ph30_image_proprio_seed42_v1` | PLANNED | One fresh ResNet-18 ImageNet initialization, 40 epochs, NT-Xent temperature 0.4; not started until the ALT core, pretrained weights, and integration checks pass. |
