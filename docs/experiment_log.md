# Experiment ledger

2026-10-20: Phase 0 source/data audit and Phase 1 implementation in progress. No training or rollout results yet. ALT main uses one RGB image plus current 2D pusher position; no Diffusion Policy training. Freeze 20 validation episodes before nested train-30/train-100; 300/500 are unavailable. Checkpoint selection is final epoch, independent of test rewards.

Phase 0 completed: full-byte integrity audited, 20 heldout + 186 train-pool episodes, exact nested30/100 and fixed50 test seeds frozen. Phase 1 completed: all five core gates passed on RTX5090 with verified ImageNet weights. No closed-loop score yet. Initial commit includes attributed functional core and all audit/split evidence.
