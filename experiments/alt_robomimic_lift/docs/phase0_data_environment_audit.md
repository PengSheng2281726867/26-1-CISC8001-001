# Phase 0: Lift dataset and environment audit

**Decision: PASS, with a quantified action-replay numerical limitation.** Dataset semantics, image alignment, controller meaning, environment task signal, and deterministic evaluation reset conditions were checked before model training. No model was trained in Phase 0.

## Dataset identity and integrity

The local source is RoboMimic 0.2.0 **Lift / PH / image**. The image HDF5 was already present, so no image conversion or episode recreation was performed. Its low-dimensional companion file has the same 200 episode IDs and every stored 7D action array is byte-value-identical by element comparison.

| File | Bytes | SHA-256 | Episodes | Frames |
|---|---:|---|---:|---:|
| `image.hdf5` | 837,217,792 | `87f813fa5118eaaac520f9933dc34e03336e307929c5f1f5fc53fb9609e1789e` | 200 | 9,666 |
| `low_dim.hdf5` | 18,548,952 | `18106c8c072020672fecdd2a9ffbfe7dc656823cb877f972ade7fad34d2a5b74` | 200 | 9,666 |

Episode lengths are 36 / 48 / 64 (minimum / median / maximum). All 200 image episodes have equal-length `states`, `actions`, and `obs` arrays. Both 84×84 camera arrays are `uint8` HWC RGB. Image value ranges across the full dataset are 8–255 for `agentview_image` and 6–255 for `robot0_eye_in_hand_image`. The contact sheet was inspected: it shows the table, red cube, robot, and the two expected camera views. It is kept locally at `results/figures/dataset_contact_sheet.png` and is not committed.

The official file-level split is `/mask/train` = 180 episodes and `/mask/valid` = 20 episodes. This leaves a real pool for 30, 50, and 100 training demonstrations while preserving a separate 20-episode offline-validation set. Split IDs are frozen separately in Phase 1.

The image file contains `actions`, `states`, and `obs`; it has no `actions_abs` field and no stored `next_obs` sequence. Observation fields include `robot0_eef_pos` (3), `robot0_eef_quat` (4), and `robot0_gripper_qpos` (2), plus other low-dimensional observations and privileged object state. The policy contract uses only the first three fields (9 numeric values) and the two images. `states`, `object`, reward/done fields, validation episodes, and future values are excluded from policy input.

## Action meaning and temporal alignment

The dataset metadata specifies Panda with `OSC_POSE`, control frequency 20 Hz, normalized input bounds [-1, 1], translation output scales ±0.05 m per axis, rotation output scales ±0.5 rad per axis, fixed impedance, and `control_delta=true`. The seven controls are six incremental end-effector pose commands and one Panda gripper command. In this RoboSuite implementation the gripper mapping is `-1` open and `+1` closed. The stored action ranges per dimension are recorded in `results/dataset_audit.json`; all are within [-1, 1].

At index `t`, `obs[t]` is the current pre-action observation and `actions[t]` is the command applied from it. The resulting simulator observation is compared with stored `obs[t+1]`. Arrays have equal length, but only `T-1` actions have a recorded successor state; the final stored action is not included in the replay divergence calculation. This matches the one-step policy's query/action indexing.

RoboSuite exposes the end-effector quaternion in **xyzw** order. The future encoder will normalize the quaternion and canonicalize its sign using the `w` component, without changing the represented rotation. The 9D proprioception normalizer will be fit only on the selected 30 training episodes and then frozen for bank creation and rollout.

## Camera and environment checks

RoboMimic's `EnvRobosuite` wrapper vertically flips raw simulator RGB arrays before returning observations. Re-rendering fixed recorded states and comparing them with the image HDF5 gave direct pixel MAE from 0.57 to 1.63 across the sampled 3 episodes × 3 frames × 2 cameras; vertically flipping again raised MAE to 69–90, and reversing channels raised it above the direct RGB comparison. Live and stored frames therefore use the same orientation, color order, and 84×84 size. The observation wrapper must classify camera keys as RGB and numeric keys as low-dimensional, and the ALT pipeline will preserve raw uint8 observations until its common ImageNet preprocessing stage.

The verified legacy runtime is Python 3.8.20, robomimic 0.2.0, robosuite 1.2.0 (`offline_study` commit `d9baa17d5328d7b0ec15ecda54c76c61bb983124`), MuJoCo 2.0.0, and `mujoco-py` 2.0.2.9. The dataset metadata is from the older robomimic 0.2.0 generation path and does not provide the exact collection commit. Although the initial request described compatibility with RoboSuite 1.5, the actual dataset and prepared, replay-validated environment are the older 1.2.0 / `offline_study` combination; this verified compatibility takes precedence. A claim that the demonstrations were collected by a specific RoboSuite commit is not made.

The success signal is the task's own Lift check: cube height must exceed table height by 0.04 m. The environment's class default horizon is 1,000 steps; evaluation explicitly freezes that horizon, `control_freq=20`, and sparse reward (`reward_shaping=false`). The dataset metadata's `ignore_done=true` is overridden to `false` for evaluation so the official success termination is respected.

## State restoration and expert action replay

Three preselected demonstrations (`demo_0`, `demo_1`, `demo_2`) were state-restored and replayed continuously from each recorded initial state using original `actions[t]`; the simulator was not reset between actions. State restoration reproduced every low-dimensional observation within 1e-6. All three action replays ended with the official task success signal.

Continuous full-state divergence peaked at `8.74596e-4` in `demo_1` (a velocity component); the corresponding peak for `demo_2` was `2.64935e-4`, and `demo_0` was `3.02697e-8`. The strict 1e-5 full-state replay gate therefore does not pass. Object position discrepancies stayed sub-millimeter, the selected action sequences still completed Lift, and image reconstruction matched. The audit records this as a small numerical drift consistent with the earlier legacy-runtime diagnosis, not exact trajectory identity. The environment/task gate passes with this limitation; it must remain visible when interpreting a low closed-loop score.

## Frozen evaluation reset candidate

The 50 planned seeds are `100000..100049`. The environment was reset for all 50 and produced 50 distinct MuJoCo state hashes. Repeats at seeds 100000, 100025, and 100049 reproduced both simulator-state and current policy-observation hashes exactly. The full initial simulator states are saved as compact reset fixtures in `results/evaluation_initial_states.json`; they are used only to initialize evaluation, never as policy inputs. The future runner will verify its initial policy-observation hash before each rollout.

## Phase 0 artifacts

- Dataset schema and full-byte hashes: `results/dataset_audit.json`
- State restoration, camera reconstruction, and action replay: `results/action_replay_validation.json`
- Reproducible evaluation conditions: `results/evaluation_initial_states.json`
- Source versions and snapshot hashes: `results/source_manifest.json`
- Local-only visual diagnostics and compressed traces: `results/figures/` and `artifacts/phase0/`

See `docs/failure_log.md` for the invocation and validator issues found and corrected during this audit.
