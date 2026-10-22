# AGENTS.md — ALT Adaptation to RoboMimic Lift (Image)

> **Scope:** This file applies only to `experiments/alt_robomimic_lift/` inside the existing `COURSE-HUZI-CODE` repository. 这是新的 Lift 专项研究，不是覆盖既有 Push-T 研究。遵守根目录 `AGENTS.md` 的 Git、安全、来源与可复现性规则；根文件中只针对 Push-T 的任务、数据、动作和评测参数**不得套用到 Lift**。本次用户明确授权开发 RoboMimic Lift 的 ALT 适配，暂不实现 Diffusion Policy。

## 0. Your role and scientific objective

You are Codex, responsible for investigating the dataset and simulator, implementing an isolated ALT-Lift adaptation, autonomously diagnosing reproducibility/engineering failures, documenting experiments, and maintaining Git.

**Research questions**

1. 在小规模 RoboMimic Lift-PH 图像示范（首轮 **30 条独立轨迹**）下，ALT 风格的视觉表征检索 + 专家动作回放，能否完成闭环抓取？
2. 是否可以在完全不使用未来信息的情况下，用两路图像及机器人当前本体状态学到可检索的状态表征？
3. 后续如何在相同训练 Episode 和测试初始条件下，与 DP 比较数据规模敏感性？**本轮不训练 DP、不比较 50/100 规模，也不增加新 Loss 或 Oracle 方法。**

**Success criteria:** a documented and tested dataset-to-training-to-memory-to-closed-loop evaluation pipeline, with genuine, reproducible Lift success-rate measurements. A negative result is acceptable; never tune the evaluation to make ALT look good.

## 1. Workspace, references, and Git

- **Writable repository root:** `D:\Code\Course_code\COURSE-HUZI-CODE`
- **This experiment's folder:** `D:\Code\Course_code\COURSE-HUZI-CODE\experiments\alt_robomimic_lift`
- **Existing ALT reference (read-only):** inspect `D:\Code\Course_code\ACl` and its actual `alt` subfolder.
- **Existing Push-T/DP reference (read-only):** `D:\Code\Course_code\DP`; this is only a reference, **not** the Lift dataset/environment.
- **Existing remote:** `https://github.com/PengSheng2281726867/26-1-CISC8001-001.git`
- **Official upstreams (record exact versions/commits):**
  - ALT: `https://github.com/StanfordMSL/alt`
  - RoboMimic: `https://github.com/ARISE-Initiative/robomimic`
  - RoboSuite: `https://github.com/ARISE-Initiative/robosuite`

Rules:

1. Read the root and this local `AGENTS.md`; inspect branch, `git status`, remote, commits, existing files. Do not overwrite user changes or initialize nested Git.
2. Place **all new Lift experiment code/config/tests/notes/small result artifacts** inside this experiment folder. Prefer importing reusable utilities rather than modifying existing Push-T/B1/Oracle code.
3. Never edit reference repositories, the frozen Push-T split, old experiment results, external raw HDF5 files, or the original ALT code.
4. Store paths in configuration; support Windows paths and CLI overrides. Do not assume the user's download directory. First search obvious, explicit local paths/configs; if an indispensable dataset path is unavailable, report the exact blocker rather than inventing one.
5. Use a dedicated compatible Python environment, not a destructive reinstall of the existing Push-T environment. Record `python`, Torch, CUDA, robomimic, robosuite, mujoco, h5py and driver versions.
6. No force push, history rewrite, fake commits, deletion of old research artifacts, or automatic merge of unrelated user changes. Commit and push validated milestones to the existing remote; on push conflict, fetch and reconcile safely.
7. Do **not** commit raw/processed HDF5 datasets, copies of video frames, all-frame data dumps, learned weights, checkpoints, large videos, downloaded model weights, `.env`, tokens, caches or logs. Add experiment-local `.gitignore`. Save large artifacts in Git-ignored `artifacts/` and commit small configs, manifests, summaries and selected non-sensitive diagnostic figures.
8. Explicitly distinguish **verified results / hypotheses / pending / blocked / negative results**. No invented success metrics or provenance.

## 2. Phase 0 — dataset, controller and simulator audit (mandatory gate)

**Do not train until this phase passes.**

### 2.1 Find and validate the Lift-PH v1.5 dataset

The desired source is the **RoboMimic Lift / Proficient Human (PH)** dataset, compatible with a RoboSuite v1.5 environment. Use the user's existing local `.hdf5` file if it contains the necessary states/actions and environment metadata. The v1.5 download registry may offer `raw` and `low_dim` files while a pre-generated `image` file is unavailable; do not mistake that for missing image support.

- Inspect HDF5 groups and attributes (`data`, `data/demo_*`, `env_args`, `model_file` if present, per-episode `states`, `actions`, `obs`, `next_obs`, etc.). Record real keys, shapes, dtypes, image color/layout, trajectory counts/lengths, reward/success annotations, control frequency, dataset provenance and hash/fingerprint. **Never assume all demonstrations have equal length or 200 episodes without auditing.**
- If an appropriate image HDF5 already exists, verify it was produced from the same source demonstrations and is internally consistent; do not regenerate it needlessly.
- Otherwise use RoboMimic's documented `robomimic/scripts/dataset_states_to_obs.py` to **render RGB images from stored simulator states**, preserving original episode IDs and action arrays. Example *after locating compatible versions/paths*:

  ```powershell
  python robomimic/scripts/dataset_states_to_obs.py `
    --dataset "<existing Lift-PH v1.5 HDF5 with states>" `
    --output_name image_v15.hdf5 `
    --done_mode 2 `
    --camera_names agentview robot0_eye_in_hand `
    --camera_height 84 --camera_width 84
  ```

  Avoid reusing a literal example path. Log the conversion command, input/output hashes, environment/software versions, warnings and image generation settings. The output is a **processed image dataset**, not newly collected independent demonstrations.
- Default cameras: `agentview` and `robot0_eye_in_hand`. Inspect actual HDF5 keys, normally `obs/agentview_image` and `obs/robot0_eye_in_hand_image` with 84×84 RGB pixels. The dataset's exact schema is authoritative.
- Create a **small contact sheet and a short 2-camera video** from predetermined demo/frame IDs to check alignment, orientation, flip, RGB-vs-BGR, wrist view, gripper visibility and cube visibility. Videos stay local unless tiny summary images are justified.
- If rendering/setup fails, investigate version, GPU/offscreen OpenGL, robosuite controller and the HDF5 environment metadata. Record failures and fixes. Do not replace real images with stock pictures or invent synthetic observations as if they came from Lift-PH.

**Critical:** The raw `states` used for rendering and dataset audits must never become a policy input or retrieval feature in the visual baseline.

### 2.2 Audit control and temporal alignment

Lift's demonstrations commonly include a 7D action, but its **meaning is controlled by the exact stored RoboSuite controller configuration**. Determine and document:

- The robot model, controller type/config, action dimension and numerical range, EEF translation/orientation command convention, gripper sign/convention, controller frequency, action clipping and environment horizon.
- Whether `actions[t]` is an **incremental/delta end-effector command** rather than an absolute Cartesian target. Do **not** reuse Push-T's absolute-action interpretation, normalization or coordinate transforms.
- Whether the image `obs[t]`, proprioception `obs[t]`, simulator `states[t]` and `actions[t]` refer to the same moment; verify several transitions and initial/final frames. `next_obs[t]` and `states[t+1]` must not leak into the live policy input.
- Whether `actions_abs` exists and, if so, how it differs. **Main ALT uses `actions` only**; do not silently substitute `actions_abs` or call the former absolute.
- Which current proprioception fields are really available online: planned `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (usually 2) = **nominal 9D**. Confirm the true dimension/order and online/offline units, quaternion convention and gripper behavior. Do not fill missing fields with arbitrary zeros without documenting and resolving the issue.
- Verify if the dataset contains object pose only for diagnostics. `object` fields, cube position, ground-truth simulator state, rewards/dones and future actions are **not allowed policy inputs**.

### 2.3 Action replay / environment reproducibility gate

Before training, reset the matching RoboSuite environment to a few **preselected recorded demonstration initial states** and replay expert actions open-loop. Compare at least: observed images/key geometry, state trajectory divergence, gripper behavior and Lift task success. Use official RoboMimic playback utilities where practical.

- Clearly distinguish **state-forced rendering** (each recorded simulator state rendered separately) from **action replay** (start once, step the recorded commands). Successful rendering does not prove that the 7D controller and dynamics reproduce actions.
- If action playback diverges severely, diagnose the RoboSuite/MuJoCo/controller versions and alignment; **do not proceed to interpret learned policy failure** until the environment has a credible validation record. If a perfect match is impossible, document discrepancy quantitatively and establish what is and isn't valid before running pilot learning.

**Required outputs:** `docs/phase0_data_environment_audit.md`, `results/dataset_audit.json`, `results/action_replay_validation.json`, visual samples, and a version/source manifest. End Phase 0 with a clear `PASS / BLOCKED` decision and a Git commit/push.

## 3. Phase 1 — freeze Lift-30 data and experiment contract

**This Lift split is separate from all Push-T manifests.** Do not reuse Push-T Episode IDs.

- From the audited Lift-PH episodes, with split seed **42**, first reserve **20 distinct held-out demo episodes** for **offline validation only**, if the dataset permits. Then choose **30 distinct training episodes** from the remaining pool; save actual IDs, order, source path/hash and the deterministic selection algorithm in `configs/lift_ph_split42.json`. Do not select favorable/short/easy demos after seeing evaluation results.
- Keep a deterministic ordered training pool so future nested **30 ⊂ 50 ⊂ 100** data-scale comparisons can use the exact same split if counts permit. **Only train 30 now.** Never duplicate a trajectory to inflate the sample count.
- Freeze initial simulator evaluation conditions independently of training demonstrations: plan **50 evaluation episodes** with fixed seeds `100000..100049` as a project protocol **only if** the actual RoboSuite/RoboMimic evaluation wrapper supports deterministic, reproducible reset. Record actual initial-state hashes/configuration for each seed, and verify repeats. If seeds alone do not reproduce resets, implement and freeze a documented valid initial-state reset contract; don't claim reproducibility from seeds alone.
- Do not choose/tune model variants or hyperparameters using the final closed-loop test seeds. Validation Episodes may be used for diagnostic retrieval metrics, not to populate the memory bank.
- Main experiment ID: `alt_lift_ph30_image_proprio_seed42_v1`; first train seed 42, shuffle seed 43, augmentation seed 44. Preserve all resolved values and model/dataset fingerprints.

**Required outputs:** frozen episode-ID and evaluation-condition manifests; `docs/experiment_protocol.md`; basic schema and split-disjointness tests.

## 4. Phase 2 — implement ALT-Lift-30 with minimal changes to the ALT principle

### 4.1 Fixed baseline network

The adaptation's **primary variant** uses two current RGB views **plus current robot proprioception**. It deliberately differs from the public ALT main branch, whose pose branch is commented out. Document this explicitly. Do not claim it is an exact line-for-line upstream reproduction.

```text
current agentview RGB (84×84) --------┐
                                     ├── SAME shared pretrained ResNet18 -> 128D each
current wrist RGB (84×84) ------------┘
current robot proprio (nominal 9D) ------- MLP -> 32D
concat [128 + 128 + 32] ------------------ Fusion MLP -> 128D L2-normalized z
```

- **One ResNet18 instance used twice, with shared weights**; NOT two independently trained ResNets. Add a test asserting actual parameter sharing.
- Load verified ImageNet-pretrained ResNet18. Replace final classifier by a 128D feature projection. Fine-tune it along with the proprio MLP and Fusion MLP; don't silently replace missing pretrained weights with random initialization.
- Normalize robot numeric features with statistics computed **from training Episodes only** (or predeclared physically meaningful scales), and apply the identical transform at bank construction and rollout. Handle quaternion order and sign carefully, and check all feature scales. Keep normalization parameters in the checkpoint/config.
- Inputs at time `t` are exactly **one current synchronized image per camera + current robot state**. No object ground truth, future frames/actions, rewards, privileged simulator state or validation/test demos.
- Keep code/config support for a future `image_only` ablation (two images, no proprio) corresponding more closely to the published main-branch implementation, but **do not run a second formal training automatically**. No B1 pose head, B2 geometry-aware Loss, action auxiliary loss, optical flow, pretrained robotics foundation model or novel fusion weighting in this first baseline.

### 4.2 Original ALT-style representation learning

Use **NT-Xent / SimCLR** with two independently augmented views of the **same time-aligned joint observation**. Two camera images belong to the same example, not to two different training samples. Keep numeric robot state corresponding to the same original frame in both augmented views.

Initial recipe based on ALT public training defaults:

- batch size **32**, epochs **40**, Adam learning rate **1e-4**, NT-Xent temperature **0.4**, model seed **42**.
- Training objective **only NT-Xent**. No expert action labels, task success or object state supervision during Encoder training.
- Use conservative, label-preserving **photometric augmentation** for Lift's small 84×84 images; avoid aggressive crop, translation and horizontal flip in the baseline because they can remove/move cube or gripper. Log deviation from upstream ALT's exact resize/crop policy.
- Same dataset, observation normalization, fixed seeds and deterministic inference transforms across train/bank/test. No random augmentation when building bank or evaluating.
- Train a **fresh model**, not a warm start from Push-T ALT/B1. Save epoch-wise contrastive Loss, optimizer settings/step counts, pretrained hash, reproducibility metadata and final-epoch checkpoint. Do not choose a checkpoint using test performance.

Run tiny integration smoke and unit tests before the one planned 40-epoch training. Stop/report if numerics are invalid, branch gradients disappear or pretrained weights are unverified.

### 4.3 Memory bank and retrieval

After training, build the bank from **all time steps of exactly the 30 training Episodes**. Store at least:

- final normalized `128D` embedding;
- `episode_id`, local time index and HDF5/source index;
- verified access to `actions[t:]` without crossing episode boundaries;
- model/checkpoint/dataset/split fingerprints.

The live policy computes the embedding from two online images + proprioception, performs **Top-1 cosine** nearest-neighbor search, and executes the corresponding stored original **7D expert action**.

**Initial closed-loop baseline: action chunk length = 1.** At every control step:

1. Read current observation and process both cameras/proprio identically to the bank.
2. Retrieve the highest cosine similarity training frame.
3. Apply its **one original 7D action** via RoboSuite's verified controller.
4. Receive the next observation, repeat.

No action translation, coordinate compensation, grasp heuristic, action mixing, similarity threshold/fallback, simulator oracle feature, privileged label, success-conditioned lookup, or resetting simulator to a retrieved training state.

Reason for 1-step: incremental 7D action semantics and previous Push-T chunk-stagnation results. This is a **conscious adaptation**, not the original real-robot ALT's 30-step rollout. Report the distinction. Prepare `action_chunk_length` as a configurable variable and unit-test length 4/8 with end-of-episode padding, but **do not run additional formal 4/8 experiments without later authorization**. No need to pad for chunk=1; always forbid cross-episode indexing.

Required safety assertions: exact 7D shape, finite controls, permitted action range, valid gripper command, synchronized observation timestamps, deterministic query tie-break, zero validation/test episodes in bank, no future labels in policy input.

## 5. Phase 3 — offline checks and full closed-loop evaluation

### 5.1 Offline diagnostics first

On held-out Lift demo frames (not in bank), report:

- Top-1 cosine distribution, retrieved episode/time/source, temporal or repeated matches;
- differences in **current EEF position, quaternion orientation and gripper state** between the query and retrieved demo state; use a correct quaternion geodesic angular distance (sign-equivalent q and −q);
- differences between the query expert 7D action and retrieved 7D action, clearly labeled **action discrepancy, not accuracy/correctness**, since multiple actions can be valid;
- qualitative contact sheets with fixed, prespecified held-out query examples and matched bank frames from **both** cameras;
- bank self-retrieval/inference re-encoding checks without confusing these with out-of-sample retrieval accuracy.

If offline matches are poor, keep the negative result and still evaluate the frozen pilot unless a correctness/safety gate fails. Do not retune from test results.

### 5.2 Formal RoboSuite Lift simulation

- Evaluate ALT-Lift-30 in the **matched Lift simulation environment** with the established, reproducible 50 initial conditions. Same robot, controller, observation cameras, initial-state distribution, task-success definition, horizon and action rate as the validated data/evaluation contract.
- Use **Lift success rate** (successful lift episodes / total), not Push-T coverage. Record each episode's success, max/final reward if defined, steps-to-success, termination/time-limit and seed/initial-state hash. Confirm the precise success signal from the actual simulator; do not invent success thresholds.
- Freeze evaluation horizon from the compatible RoboMimic/RoboSuite task configuration **before** formal testing; do not silently inherit `max_steps=300` from Push-T. Record the real horizon in config and run metadata.
- For pilot: one fresh trained encoder seed, same fixed evaluation initializations; report binomial uncertainty for success fraction (e.g. Wilson 95% interval) and state that model-seed uncertainty has **not** been measured. Plan multiple training seeds for a later study, not an unapproved extra run.
- Log query matches, similarities, action/gripper values, control step, eventual success, repeated same-frame retrieval, stagnation/failure patterns. Save at least a few **predeclared** seed videos locally, and a small reproducible multi-view comparison figure for the report.
- If environment success is near zero, analyze whether experts can succeed when replayed, whether gripper actions work, observations are aligned, and whether retrieval loops. **Do not call a low score an implementation success or conceal it.**

## 6. Future DP comparison (design only; DO NOT EXECUTE)

Preserve everything necessary so an eventual DP-Lift comparison can use the **same train-30 Episode IDs**, same two RGB cameras, same available proprio inputs (or explicitly declare differing observation history), same controller/action semantics, and the **same frozen test initial conditions**.

Do not import Push-T's action format, train DP now, or present cross-task numerical scores as directly comparable. Future scale experiments can use nested 30/50/100 episodes, independently trained encoders and paired test initializations.

## 7. Required project organization and records

Create/maintain inside `experiments/alt_robomimic_lift/`:

```text
AGENTS.md
README.md
.gitignore
configs/
  lift_ph_split42.json
  alt_lift30.yaml
  evaluation_initial_states.json       # if compact; never raw giant dumps
src/
  dataset.py
  encoder.py
  loss.py
  memory.py
  policy.py
  rollout.py
  train.py
  diagnose.py
  evaluate.py
  repro.py
tests/
docs/
  phase0_data_environment_audit.md
  experiment_protocol.md
  implementation_notes.md
  experiment_log.md
  failure_log.md
  final_analysis.md
results/
  dataset_audit.json
  action_replay_validation.json
  offline_diagnostics.json
  evaluation_summary.json
  per_seed_scores.csv
  figures/
artifacts/                          # Git-ignored weights, HDF5, videos, caches, traces
```

Names can differ where justified, but keep the same separation and describe changes in `README.md`.

For every formal run record:

- run ID, Git commit and working-tree status at launch;
- exact source/data version & hashes, source camera names/resolution, environment/controller config;
- frozen train/validation Episode IDs and test initializations;
- all model/augmentation/optimizer/seeds/epoch parameters;
- checkpoint and bank hashes, training duration, GPU/runtime environment;
- full output summary, failure/success per evaluation episode and diagnostic log location.

Any edit to method code after training should create a provenance entry; never silently replace an earlier run or claim later evaluation code was the same as training code without checking.

## 8. Autonomous execution and phase gates

**Phase 0:** inspect dependencies and HDF5 → render/verify camera images → check state/action alignment → validate expert action playback. Commit/push audit, code and limitations. **If incompatible, stop and report exact blocker; do not manufacture an apparent experiment.**

**Phase 1:** freeze Lift-30 selection, verification/test reset manifest; implement schema/split/repro tests. Commit/push.

**Phase 2:** implement shared-ResNet Image+Proprio ALT, NT-Xent training, bank and 1-step Top-1 policy; run unit/integration tests → exactly one scheduled 40-epoch ALT-Lift-30 pilot → checkpoint/bank validations. Commit/push.

**Phase 3:** held-out offline analysis → once-per-config fixed 50-condition closed-loop Lift evaluation → independent score/control checks → bilingual or Chinese final research report. Commit/push.

When any phase fails:

1. Diagnose using real traces and source evidence.
2. Fix a genuine implementation/environment problem within this isolated folder and document it in `docs/failure_log.md`.
3. Rerun relevant tests; record all retries and reasons. Never change the scientific protocol invisibly or run repeated evaluations to cherry-pick a good seed.
4. If the obstacle requires changing split, observation/action semantics, controller, success criteria or scientific method, **stop, document the proposed change, and request user approval** instead of silently proceeding.

Before each push check `git diff`, `git status`, `.gitignore`, file sizes and secrets. Use descriptive commits (for example `audit(lift): ...`, `feat(lift): ...`, `exp(lift): ...`). Push to the existing configured remote; record whether the push was actually successful.

## 9. Final report to user

End by reporting, in concise Chinese:

1. Whether dataset conversion, image inspection and expert action replay passed;
2. Actual frozen train/val counts, environment/controller version and action semantics;
3. ALT-Lift-30 architecture, exact input modalities, training completion and bank size;
4. Held-out retrieval diagnostics and official closed-loop **Lift success rate**, including uncertainty and observed failures;
5. Whether the result is truly comparable to a future DP setup, what remains incomplete;
6. Key output paths, Git commit SHA(s) and verified push status.

**Do not automatically start DP, expand to Lift-50/100, try action chunks 4/8 formally, alter the encoder/loss, or add Geometry-Oracle runs. Stop after the single ALT-Lift-30 pipeline and report.**
