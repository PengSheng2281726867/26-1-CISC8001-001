# Experiment ledger

2026-10-20: Phase 0 source/data audit and Phase 1 implementation in progress. No training or rollout results yet. ALT main uses one RGB image plus current 2D pusher position; no Diffusion Policy training. Freeze 20 validation episodes before nested train-30/train-100; 300/500 are unavailable. Checkpoint selection is final epoch, independent of test rewards.

Phase 0 completed: full-byte integrity audited, 20 heldout + 186 train-pool episodes, exact nested30/100 and fixed50 test seeds frozen. Phase 1 completed: all five core gates passed on RTX5090 with verified ImageNet weights. No closed-loop score yet. Initial commit includes attributed functional core and all audit/split evidence.

2026-10-21: Initial core commit ad54817 successfully created and pushed to origin/main after user-side ownership repair and supported per-command approval. Current project .git owner verified as yc67001; no reference repo permissions changed.

2026-10-21: Phase 2 integration gate: all seven real-data/interface tests passed. Run alt_030_split42_model42_smoke_v1 completed official PushTImageRunner on the first predefined seed100000 with 300 actual actions, 38 queries, 3806 bank frames, zero invalid events and a real saved video. Official score0.39167831266120906 is **untrained ImageNet integration smoke only**, not the trained ALT-30 pilot and not a comparable 50-seed benchmark. Instrumented transitions and separately recorded per-episode/aggregate rewards agree exactly with the original environment/Runner. Contact sheet inspected; RGB, pusher coordinates, retrieved frames and absolute commands are coherent. No method, split or test seeds changed.

Next gate: commit the validated training/Runner integration, then create a new ALT-30 model from ImageNet initialization with model/shuffle/augmentation seeds42/43/44, fixed40 epochs and final-epoch checkpoint, build a fresh exact-subset bank, and evaluate all50 frozen seeds. The smoke checkpoint is not reused for training.
