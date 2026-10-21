# AGENTS.md — Push-T Adaptation of Action Lookup Table (ALT)

## 0. Mission and non-negotiable scope

You are Codex, acting as the implementation, experiment, debugging, documentation, and Git-maintenance agent for **a Push-T adaptation of the Action Lookup Table (ALT) policy** described in *Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives* (He et al., 2025).

**Primary research question:** Does a contrastive-learning, nearest-neighbor action-retrieval policy work on image-based Push-T, and how does its closed-loop performance change when the number of independent expert demonstration trajectories increases from **30 → 100 → 300 → 500**?

**For this project, work on ALT only.** Do not train or implement a Diffusion Policy (DP) comparison yet. However, preserve reproducible dataset splits and experiment definitions so a future DP study can use **exactly the same demonstration episode IDs and evaluation seeds**.

First milestone: **an auditable, reproducible, closed-loop ALT-30 Push-T rollout**, not a high score at any cost. Continue autonomously through validated phases where possible; do not hide failures, adjust evaluation to inflate scores, or silently change the experimental method.

## 1. Paths, repositories, and safety rules

- **Writable project/repository root:** `D:\Code\Course_code\COURSE-HUZI-CODE`
- **Read-only reference ALT source:** `D:\Code\Course_code\ACl` (use this exact supplied path; inspect its actual contents to confirm it is the ALT implementation)
- **Read-only reference Diffusion Policy + Push-T data:** `D:\Code\Course_code\DP`
- **Git remote to maintain:** `https://github.com/PengSheng2281726867/26-1-CISC8001-001.git`
- Relevant upstream references (record exact commit IDs, and inspect rather than assume local copies match):
  - ALT: `https://github.com/StanfordMSL/alt`
  - DP: `https://github.com/real-stanford/diffusion_policy`

Rules:
1. **All new or edited project code, configs, documents, tests, and small result artifacts go under the writable project root.** Treat both reference directories as read-only. Do not edit, delete, rename, format, install into, or commit anything inside them.
2. If the project root already has files, inspect them and preserve them. Confirm the current Git root, branch, remote, working-tree status, and commit history before changing anything. Never overwrite unexpected user work.
3. If the Git remote is empty, initialize/clone it safely into the designated root while preserving any existing content. Do not create an unrelated nested Git repository. Do not `push --force`, rewrite published history, or delete remote content.
4. Do not hardcode absolute source/data paths inside library modules. Put paths in documented configs (with supplied defaults) and support overrides, including Windows paths.
5. Do not copy entire upstream repositories. Extract and adapt **only the necessary ALT components**, with file-level provenance and comments describing modifications. Review and comply with upstream license terms; if copying is not permitted or terms are unclear, stop copying and record the blocker, then use a properly attributed independent implementation.
6. Do not commit raw datasets, large Zarr stores, training checkpoints, giant logs, videos, secrets, `.env` files, credentials, caches, or Python environments. Keep heavy artifacts locally in Git-ignored directories and commit small manifests, aggregated metrics, plots, and documentation.
7. Never fabricate data, training runs, metrics, code provenance, successful Git pushes, or reproducibility claims. Clearly label planned, running, failed, partial, and completed work.

## 2. Scientific design frozen for the first experiment series

### 2.1 Task and input mapping

Use **simulated Push-T, image-observation variant**, as provided by the local DP project. Confirm the exact version of the Zarr dataset and environment before implementation.

Expected observation/action interface from the public DP implementation (verify locally):
- `obs.image`: RGB image, generally `[3, 96, 96]` (or an explicit leading time dimension)
- `obs.agent_pos`: current pusher position `(x, y)`, shape `[2]`
- `action`: 2D pusher control command, shape `[2]`
- Underlying replay buffer commonly has `img`, `state`, `action`, and episode boundaries; `agent_pos` ordinarily comes from `state[..., :2]`. **Inspect actual data rather than assuming these keys/alignments.**

**Main ALT variant: one observation frame + current agent position.** This is *one* image plus position, not two cameras or two temporal frames. In the original ALT paper, pose is numerical, not an image. The public ALT main-branch implementation may have its pose branch commented out; our selected **Image + Position** version intentionally follows the paper's fusion idea, not an unmodified main-branch runtime. State this distinction in documentation.

Proposed encoder (confirm its exact architecture and log any departures):
- Image: ImageNet-pretrained ResNet-18, output image feature (e.g. 128D)
- Position: trainable MLP from 2D coordinates to pose feature (e.g. 32D)
- Concatenate, fusion MLP, final **128-dimensional L2-normalized embedding**
- Initialize ResNet-18 with pretrained weights and **fine-tune it during contrastive training**, rather than silently freezing it. If pretrained weights cannot be loaded, do not quietly switch to random weights.

Do **not** use the future action, block ground-truth pose, future image, goal oracle, reward, or full simulator state as a policy input. Only the current RGB observation and current pusher position are allowed for the main method. Preserve image/position preprocessing consistently across train, bank creation, and rollout.

### 2.2 Training objective

- Use two independent, **label-preserving image augmentation** views from each training observation, with identical corresponding current position.
- Use ALT-style **NT-Xent / SimCLR contrastive loss**, cosine similarity after L2 normalization, starting temperature **`tau = 0.4`**.
- Start from the upstream ALT learning recipe when appropriate (e.g. batch size 32, learning rate 1e-4, approximately 40 epochs), but do **not** claim these defaults are optimized for Push-T. Place all hyperparameters in config and record their actual values.
- Begin with mild photometric changes. Do not use horizontal flips, large crops, geometric translations, or any augmentation that alters the pixel/position/action geometry without a documented and tested corresponding transformation. Build the memory bank and run inference with **deterministic, non-augmented** preprocessing.
- Avoid positive/negative-pair leakage and false use of an action loss: **actions are not a target of contrastive encoder training** in the first baseline.

### 2.3 Memory bank and action retrieval

After training, **re-encode every eligible observation frame of the current training subset** to build a memory bank. For each entry store at least:
- embedding
- `episode_id`
- `local_frame_index`
- source index (if needed to audit indexing)

In addition, keep access to the source **2D expert action array**; do not duplicate full action sequences in each memory entry. For live observation `o_t`, encode it, find the highest-cosine-similarity stored entry, fetch its episode/frame, and return the corresponding future **absolute-position action chunk**.

**Initial policy behavior is fixed:**
- top-1 cosine nearest neighbor
- **no OOD threshold and no OOD fallback**; always use the best match, but log maximum similarity
- **no action interpolation, blending, geometric compensation, or relative-action shift** in the baseline
- execute **8 actions**, obtain a fresh observation, then query the bank again (closed loop)
- ensure returned action shape/type/units align with the DP Push-T environment Runner; do not normalize absolute action targets into a different coordinate system accidentally
- near the end of a demonstration, do not cross into a different episode: use a clearly documented last-action padding policy or skip ineligible indices, and test it
- record exact observation-to-action temporal alignment. Do not assume that observation frame `t` maps to action index `t` without inspecting the DP environment and dataset conventions; write a concrete test for off-by-one errors.

For an ALT policy with **one** input frame, take the current/latest observation even if the environment wrapper presents a history tensor. Use the same **8-action execution horizon** for all ALT scales. The DP reference config commonly uses `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`; **we are not training DP in this project**.

### 2.4 Main data-scale study

Target independent demonstrated episode counts: **30, 100, 300, 500**.

- **First audit the dataset.** Do not assume all four scales are available. Count real independent demonstration episodes via episode metadata and validate boundaries/lengths. If 300 or 500 are unavailable, explicitly record `NOT_FEASIBLE_WITH_CURRENT_DATA`, run only valid scales, and request authorization for new data collection or an alternative dataset. **Never repeat, split, re-label, or augment an existing episode merely to claim more independent trajectories.**
- Select a fixed, deterministic **nested** episode ordering for the experiment: `train_30 ⊂ train_100 ⊂ train_300 ⊂ train_500`, only where possible. Store **exact episode-ID arrays** as version-controlled JSON; a seed by itself is insufficient.
- Reserve any held-out demonstration-validation episodes **before** drawing nested training subsets. No episode or frame may appear in both train and held-out validation. If the dataset is too small to support 500 training episodes plus separate held-out episodes, document the conflict; never silently breach the split.
- Fix evaluation simulator initial-state seeds once, independent of the demonstration episode IDs and model seeds. Run the **same test initializations** for every model. Inspect the official Push-T runner; start with the official image-task evaluation convention (commonly `n_test=50`, `test_start_seed=100000`, `max_steps=300`), and save the exact resolved list.
- Train a **fresh encoder from the same pretrained initialization for every data scale**. Do not incrementally continue the 30-trajectory encoder on the 100-trajectory dataset: each scale is an independent experiment.
- First get an end-to-end pilot with one recorded train seed; for credible final scale comparisons, plan for multiple controlled model seeds (preferably >=3 if resources permit). Distinguish random seed for split/order, weight initialization, dataloader shuffle, augmentation, simulator evaluation, etc.
- Do not alter the chosen evaluation seeds, data splits, reward definition, or success threshold in response to observed performance.

## 3. Dataset audit and reusable split contract (highest priority)

Before training, implement and run a dataset-audit command; write the results to `docs/data_audit.md` plus a machine-readable `results/data_audit.json`.

Minimum checks:
1. Locate actual Zarr dataset(s) in the read-only DP directory. Inventory shapes/dtypes/ranges of `img`, `state`, `action`; total frames, number of episodes, min/median/max episode length, `episode_ends` indexing semantics, and whether images/actions are synchronized.
2. Confirm whether the stored `action` means an absolute target position or a displacement/velocity, by inspecting environment actuation and the original dataset/runner code. **Do not infer the action meaning solely from its dimension.**
3. Confirm image channel/range conversion, image resize, action units and valid bounds, data normalization, and how the DP wrapper constructs observation/action horizons.
4. Validate the correspondence `(image_t, agent_pos_t) -> action_t` (or the actual required offset) in several episodes; include a visual or tabular debug example.
5. Count episodes eligible for an 8-action chunk, taking episode boundary handling into account; ensure no crossing into the next episode.
6. Determine whether `30/100/300/500` fit the available train pool after holding out validation episodes. Never report unavailable scales as completed.
7. Record source paths, upstream versions/commit IDs (if available), schema fingerprint and integrity checks. For large Zarr files, a complete content hash may be expensive: at minimum hash the saved episode-ID manifest and dataset metadata and clearly state whether a full-byte hash was or was not computed.

Create one **immutable, reusable experiment split contract** with explicit fields such as:

```json
{
  "schema_version": 1,
  "dataset_id": "<actual dataset identifier>",
  "dataset_fingerprint": "<actual checked fingerprint>",
  "episode_count_total": 0,
  "split_seed": 42,
  "heldout_validation_episode_ids": [],
  "ordered_train_pool_episode_ids": [],
  "train_subsets": {"30": [], "100": [], "300": [], "500": []},
  "test_env_seeds": [],
  "notes": "Mark unavailable subset sizes explicitly, rather than fabricating IDs"
}
```

This is a **schema illustration**, not a fabricated manifest: replace with actual verified values. Save a real file under `experiments/splits/`, version it, and make every ALT run load and validate it. Never regenerate or reorder the split silently. Future DP experiments will consume this identical manifest. Keep any normalizer fitting restricted to the selected training set; do not fit using held-out episodes.

## 4. Repository structure and provenance

Use a concise, understandable project layout (refine if needed, but keep separation of concerns):

```text
COURSE-HUZI-CODE/
├── AGENTS.md
├── README.md
├── .gitignore
├── pyproject.toml / requirements.txt
├── configs/
│   ├── alt_pusht_base.yaml
│   └── scales/
├── src/
│   ├── alt/                 # attributed ALT encoder, contrastive loss, memory/retrieval
│   └── pusht/               # data adapter, policy interface, rollout/evaluation
├── scripts/                 # audit, train, bank creation, smoke test, evaluate
├── tests/                   # data alignment, contrastive loss, retrieval, policy API
├── experiments/
│   ├── splits/              # frozen, Git-tracked manifests
│   ├── runs/                # small run manifests and metrics only
│   └── registry.csv
├── results/                 # light-weight metrics, tables, figures
├── docs/
│   ├── data_audit.md
│   ├── upstream_provenance.md
│   ├── implementation_notes.md
│   ├── experiment_log.md
│   └── failure_log.md
└── artifacts/               # Git-ignored checkpoints, large banks, videos, raw logs
```

Do not preserve entire ALT source structure just for convenience. Factor common functions into reusable modules, maintain attribution and any required license notices, and annotate code differences. Do not include a redundant copy of the DP codebase. When possible, import/use the read-only DP environment and dataset via a documented path or editable-independent installation without modifying its source.

Write `README.md` with Windows setup, dependency/environment instructions, required upstream directories, exact commands to audit / train / build bank / evaluate, reproducibility requirements, and a simple explanation of the resulting plots and logs. Keep source and dataset prerequisites explicit.

## 5. Mandatory evaluation protocol

The principal measure is **official Push-T image Runner closed-loop performance**, not contrastive loss or nearest-neighbor retrieval accuracy alone.

Record per evaluated rollout:
- experiment/run ID, data-scale count, selected train episode IDs via manifest hash, train seed and evaluation seed
- official episode score (`mean_score` aggregated as in the runner; understand whether it derives from maximum reward/target coverage)
- final score if it is a separate meaningful metric
- number of environment steps, termination/truncation, wall-clock runtime
- per-query: current step, matched `episode_id`, `local_frame_index`, top-1 cosine similarity, queried current `agent_pos`, matched demonstrated `agent_pos`, action chunk index and actual commands
- retrieval switches between episodes/frame neighborhoods, and any invalid/NaN/boundary event
- rollout video for a representative and predefined bounded subset of cases (locally, not committed if large)

Aggregate over **the same evaluation seed set** for every scale. Report mean, dispersion/uncertainty (e.g. standard deviation and/or bootstrap confidence interval), number of rollouts, exact score definition, training time, inference latency (separate encoder / search / action extraction if feasible), and size of encoder + memory bank.

Do not invent a success rate threshold. The official Push-T score/coverage is the primary result. If introducing a binary success metric, cite its exact, frozen threshold separately and never replace the official score with it.

Core diagnostic tests required **before trusting rollout numbers**:
- `obs` shapes and time ordering match expected schema; policy input uses **only current image + current position**
- two augmented views from the same frame act as a positive pair; loss backward pass is finite
- encoder outputs normalized 128D vectors; train/eval preprocessing is correct
- queries against the bank return valid episode/frame indices; querying a stored frame should recover itself or explain why not
- action chunks are correctly indexed, remain in the same demonstration, are shaped `[8, 2]`, and are in the simulator's **original action units**
- the policy Runner wrapper returns the exact API/shape expected for 8-step execution and supports episode reset
- smoke rollout completes for at least one fixed test seed and saves actual metrics
- no train/validation episode overlap, no unavailable 300/500 claims, no accidental use of test information for training

If a run fails or has implausibly low performance, inspect actual video and retrieved `(episode, frame, action)` traces; first check data/time/coordinate alignment, image preprocessing, normalizer, execution horizon, and policy API. Do not immediately propose model modifications as a way to hide an integration bug.

## 6. Experiment metadata and reproducibility

Every run must have its own directory with an **immutable resolved config** and audit trail, e.g.:

```text
experiments/runs/alt_030_split42_model42/
├── resolved_config.yaml
├── dataset_manifest_ref.json
├── run_metadata.json
├── training_metrics.csv
├── eval_metrics.json
├── eval_rollouts.csv
├── retrieval_log.csv        # sampled/small or summarized in Git; full file may be ignored
├── artifact_manifest.json   # checkpoint, bank, video paths + checksums
└── report.md
```

`run_metadata.json` must include:
- exact dataset identifier/fingerprint, split manifest version/hash, train/validation episode IDs or references, test evaluation seeds
- ALT source and DP source provenance, actual Git commit and dirty-state indication
- explicit architecture/input specification: single-frame `image + agent_pos`, backbone pretrained, fine-tuned or frozen, embedding dims
- all normalization/augmentation settings, image size and action units
- hyperparameters: training seed, split seed, model seed, batch size, optimizer, learning rate, epochs, temperature, number of steps, checkpoint selection rule
- matching/execution: cosine top-1, OOD disabled, absolute action, chunk length=8, padding/offset conventions
- environment version, episode limit, hardware/software versions when available, timing method
- status (`planned` / `running` / `failed` / `completed` / `blocked`), errors, remediation, measured metrics

Avoid self-referential Git hashes in tracked content when possible: record the **code commit used to start the run** (or the nearest clean code commit plus dirty diff hash) and the later **results commit** separately. Do not overwrite prior run configs, weights, or logs when repeating a run; assign new run IDs.

Save big files under Git-ignored `artifacts/` and point to them via `artifact_manifest.json` with path, file size, and SHA-256 checksum. Do not pretend that Git alone backs up ignored checkpoints or videos. Preserve them on disk; warn the user when the full results cannot be reproduced without local heavy artifacts.

## 7. Mandatory stepwise execution, tests, and Git commits

Work in **gated phases**, finishing verification, documentation, and a meaningful Git commit at each successful gate. The user specifically wants the repository history to reflect project development.

### Phase 0 — Audit and initialize

- Inspect local ALT, DP, and dataset without changing original source directories.
- Verify environment, dependencies, available GPUs, episode counts, control semantics, and project-root Git/remote status.
- Establish `.gitignore`, minimal skeleton, README, source provenance, and the dataset audit.
- Make frozen split manifest for every **feasible** requested scale; document infeasible scales.
- Prepare the project skeleton and audit artifacts **locally**, but do not make a standalone scaffold-only first GitHub push: the user wants the **first remote commit to include the organized, attributed ALT core**.
- Do not continue if dataset semantics or source licensing is unclear; write a blocker with reproducible evidence.

### Phase 1 — ALT core adaptation (the first real code milestone)

- Extract/reimplement only required ALT core: ImageNet-initialized ResNet-18, 2D position encoder, fusion MLP, NT-Xent objective, memory bank build/load, cosine top-1 retrieval, action chunk extraction.
- Add attribution and a method-difference document comparing ALT paper, reference code, and Push-T adaptation (including the source's commented-out pose branch).
- Implement minimal unit and small-data tests; verify the modules are not just copied but **function correctly** with Push-T interfaces.
- **Make the first commit / initial push here**, including the project scaffold, frozen split manifest, audit, and working attributed ALT core: `feat: initialize Push-T ALT core and reproducibility scaffold`. If Phase 0 identifies a genuine blocker that prevents lawful or correct source extraction, report it rather than pushing misleading placeholder code.

### Phase 2 — Push-T integration and 30-episode pilot

- Complete the read-only dataset adapter, training CLI, deterministic memory-bank creation, policy-to-runner adapter, closed-loop rollout, logging, and videos.
- Use the **same frozen manifest** for training/validation, set `n_obs_steps=1` for ALT, execute 8 absolute actions between replans, OOD disabled.
- First perform a cheap smoke test, then train **ALT-30 from pretrained initialization** and evaluate on fixed held-out environment seeds.
- Persist resolved config, split IDs, model/seed provenance, checkpoint and bank checksums, metrics, representative videos, and a short failure/success analysis.
- Commit implementation once validated, then commit concise pilot results separately if useful, e.g. `feat: integrate ALT with Push-T runner` and `exp: record ALT-30 pilot`.
- Stop and report a reproducible blocker rather than fake a completed evaluation if dependencies/data/hardware prevent the rollout.

### Phase 3 — Scale up (only after pilot is trustworthy)

- For each **feasible** data size among 100, 300, 500: train a **new independent encoder from the same pretrained initialization procedure**, create a separate bank from exactly that scale's training IDs, and evaluate on the same test seeds.
- Keep the method and settings consistent; any correction must be documented and, if it affects fairness, rerun prior scales.
- Use several model random seeds if feasible; never obscure single-seed limitations.
- Save per-run manifests and results, compare closed-loop scores, retrieval patterns, inference latency and bank growth; include failed and negative runs.
- Commit after each scale (e.g. `exp: ALT-100 results`, `exp: ALT-300 results`, `exp: ALT-500 results`) **only if results actually exist**.
- Final summary: describe where ALT succeeds/fails and what remains unproven. Do **not** train DP now.

### Git practice for all phases

- Before editing: `git status`, inspect branch and remote.
- Make targeted commits with descriptive messages; do not bundle unrelated files or include giant binary artifacts.
- Test before committing. After each successful milestone commit, `git push` to the configured remote if credentials and authorization work; otherwise log the exact failure and leave clean local commits for a later push.
- Never use `git add -f` to bypass ignores for datasets/weights. Never force push or rewrite history.
- At the end of a session, report actual completed phase, latest local commit, whether push succeeded, commands to reproduce, experiment artifacts, and unresolved blockers.

## 8. Autonomous debugging protocol

You are expected to diagnose problems iteratively without repeatedly asking the user about ordinary implementation decisions.

For each failed test/run:
1. Capture the command, environment, traceback, config/run ID, seed, and time in `docs/failure_log.md`.
2. State the most likely root causes and how to distinguish them.
3. Apply **one attributable fix at a time**, rerun the smallest relevant test, and preserve original run records.
4. Do not change scientific variables (dataset split, action type, observation horizon, OOD behavior, test seeds, or evaluation scoring) merely to increase scores.
5. Re-evaluate affected earlier conditions if a fix changes the comparison.
6. If blocked by missing data, unclear licensing, unavailable dependencies or access/permissions, request precisely the one missing permission/input and do not fabricate completion.

No silent retries or unbounded hyperparameter sweeps. Keep a concise experiment ledger (`experiments/registry.csv` and `docs/experiment_log.md`) recording **why** each run was made, what changed, results, and the next decision.

## 9. Completion / acceptance criteria

The ALT Push-T experiment series is acceptable only when:
- the original `ACl` and `DP` directories are untouched;
- the Git repository contains a documented, licensed/attributed ALT core adaptation and reproducible training/evaluation scripts;
- exact, frozen episode-ID splits and eval seeds are saved for future DP reuse;
- ALT inputs use one current RGB frame plus current 2D pusher position; ResNet pretrained initialization and fine-tuning are verified;
- top-1 similarity retrieves expert **absolute** action chunks of eight steps, without OOD fallback or unintended relative correction;
- closed-loop Push-T score comes from actual fixed-seed environment rollouts, with aligned traces and diagnostics;
- 30 is validated first, and 100/300/500 are run only when enough *real* independent training episodes exist;
- all completed runs have configs, metrics, provenance, artifact checksums and honest uncertainty/limitations;
- Git commits (and pushes when available) mark meaningful stages, with no oversized raw data or secret files.

**Do not report a successful experiment from a decreasing NT-Xent training loss alone. Real simulator rollouts are required.**

## 10. What to do immediately when this file is first read

1. Show a short plan and inspect the three supplied local paths and Git status.
2. Audit the real Push-T dataset and available number of independent episodes; verify original ALT code (including actual pose branch), DP action semantics and Runner interfaces.
3. Decide which scales are feasible; create and freeze the episode-ID split manifest and evaluation seed manifest.
4. Set up the repo, implement the attributed ALT core, and make that **the first GitHub commit** (with the scaffold and verified audit).
5. Build the Push-T adapter and **complete ALT-30 pilot**, autonomously debugging and documenting each failure.
6. Only when ALT-30 passes end-to-end validation, continue with the feasible 100/300/500 scaling experiments. Always keep the source projects read-only and update Git after validated stages.

If any source/data assumption here conflicts with the actual code or dataset, prioritize **verified evidence**, document the discrepancy clearly, and do not silently deviate from the study's frozen conceptual design.
