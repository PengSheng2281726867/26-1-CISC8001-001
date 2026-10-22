# Lift-PH ALT-30 protocol

## Frozen split

`configs/lift_ph_split42.json` is the authoritative immutable split contract. It records the exact 20 held-out demo IDs, full ordered training pool, nested 30/50/100 IDs, source HDF5 hashes, evaluation seeds, and evaluation-initial-state manifest hash. It is separate from every Push-T split. The source HDF5's original 180/20 masks remain recorded in the audit, but the experiment uses the requested seed-42 permutation over all 200 unique episodes. This avoids silently substituting a different split algorithm while preserving a disjoint 20-demo validation set.

Only `train_subsets["30"]` is trained in this study. The 50 and 100 subsets are reserved; they are not additional experiments in the current authorization. Validation episodes are for offline retrieval diagnostics only and are excluded from normalizer fitting and memory-bank construction.

## Random seeds and initialization

| Purpose | Frozen value |
|---|---:|
| Split order | 42 |
| Model initialization | 42 |
| Data-loader shuffle | 43 |
| Image augmentation | 44 |
| Simulator initializations | 100000–100049 |

The actual 50 MuJoCo starting states and per-state hashes are in `results/evaluation_initial_states.json`. The states support exact repeatable initialization; they are not passed to the policy.

## ALT-Lift-30 baseline

The policy input at time `t` is exactly one current `agentview_image`, one current `robot0_eye_in_hand_image`, and current EEF position, quaternion, and gripper joint positions. The two camera calls share a single ImageNet-pretrained ResNet-18. The proprioception MLP produces 32 features; the image branches produce 128 features each; fusion outputs a 128D L2-normalized embedding. The ResNet and all heads are fine-tuned.

Training uses two independent mild photometric views of the same synchronized observation, with identical proprioception values in both. NT-Xent temperature is 0.4. The baseline uses Adam at 1e-4, batch size 32, and 40 epochs. No action, reward, object-state, or success supervision is used. The final epoch is saved without test-seed checkpoint selection.

At inference, the policy selects the top-1 cosine match in the bank and returns its original `actions[t]` command. In this controller those values are normalized, incremental 6D OSC pose controls plus the original Panda gripper command; the action is not an absolute target. The baseline replans every control step (chunk length 1). There is no OOD cutoff, fallback, blend, or coordinate compensation.

## Simulator evaluation

The matched simulator is Lift with the dataset's Panda `OSC_POSE` controller, 20 Hz control, 84×84 RGB observations, sparse reward, and a fixed 1,000-step horizon. Success is RoboSuite's task check: cube height above table height by 0.04 m. The same 50 reset fixtures are used in order. Per rollout, save success, maximum and final reward, steps, termination, wall time, and a per-query retrieval/action trace. Produce local videos only for seeds 100000, 100025, and 100049.

The Phase 0 expert replay succeeds on the three checked source episodes, but full-state numerical drift reaches 8.75e-4. Do not call that replay bitwise exact. If the learned policy has low success, inspect videos and traces while preserving this limitation.
