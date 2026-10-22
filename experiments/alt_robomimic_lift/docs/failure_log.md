# Lift failure and correction log

## 2026-10-22 — Initial MuJoCo import lacked runtime variables

- **Command:** imported `robosuite` directly in the prepared Lift Python 3.8 environment.
- **Observed:** `mujoco_py` reported that MuJoCo was missing from `/home/yc67001/.mujoco/mujoco200`.
- **Cause:** the local legacy environment requires `MUJOCO_PY_MUJOCO_PATH`, `MUJOCO_PY_MJKEY_PATH`, and `LD_LIBRARY_PATH`; the command bypassed the repository wrapper.
- **Correction:** added `scripts/with_legacy_env.sh`, which points at the already installed read-only simulator and redirects Python caches into this experiment directory. No dependency was reinstalled and no prior environment files were changed.
- **Status:** corrected; environment import and validation are pending.

## 2026-10-22 — Inline Python passed through WSL lost nested quotes

- **Command:** used `wsl.exe ... python -c` with a multiline HDF5 inspection expression.
- **Observed:** Bash/Python received stripped quote characters and raised a syntax error.
- **Cause:** nested PowerShell, WSL, Bash, and Python command parsing made the inline expression ambiguous.
- **Correction:** moved reusable inspection into `scripts/audit_dataset.py` and invoke the file directly in the legacy environment.
- **Status:** corrected; the audit completed and hashes match both existing download manifests.

## 2026-10-22 — First audit pass looked for split masks under `data/`

- **Command:** ran the initial `scripts/audit_dataset.py` implementation.
- **Observed:** episode masks were empty even though the dataset manifest lists train/valid masks.
- **Cause:** this HDF5 stores masks at the file root (`/mask`), not under `/data/mask`.
- **Correction:** the audit now checks both locations and records root keys. The rerun found 180 official train demos and 20 official validation demos.
- **Status:** corrected; split selection is not frozen until Phase 0 finishes.

## 2026-10-22 — State-replay validator classified camera keys as low-dimensional

- **Command:** `bash scripts/with_legacy_env.sh scripts/validate_environment.py`.
- **Observed:** simulator construction succeeded and reported Lift with a 7D action, then the validator raised `KeyError: 'agentview_image'`.
- **Cause:** the initial observation specification put every HDF5 key under `low_dim`. RoboMimic's wrapper only exposes camera keys classified under the `rgb` modality.
- **Correction:** initialize the observation spec with camera keys under `rgb` and numeric keys under `low_dim`; disable the wrapper's optional image scaling so both stored and live values remain uint8 RGB after the same vertical flip.
- **Status:** correction applied; replay validation is being rerun. The failed JSON is retained as `action_replay_validation_attempt01_failed.json`.

## 2026-10-22 — WSL terminal returned an environment without RGB observations

- **Command:** `bash scripts/with_legacy_env.sh scripts/validate_environment.py`.
- **Observed:** RoboSuite started with the expected Lift task and 7D action, but the validator raised `KeyError: 'agentview_image'`.
- **Cause:** the validator's observation spec initially registered camera keys as low-dimensional fields. RoboMimic only includes them in returned observations when they are registered as RGB.
- **Correction:** split the observation spec into `rgb` and `low_dim`, and disable optional image postprocessing so replay comparisons use the stored uint8 pixels.
- **Verification:** the rerun reproduced state observations within 1e-6, matched both cameras directly (sampled maximum pixel MAE 1.63), and replayed all three demonstrations to Lift success. Full-state drift is documented separately.

## 2026-10-22 — First Phase 1 schema test expected `next_obs` to be absent

- **Command:** `bash scripts/with_legacy_env.sh -m unittest discover -s tests -v`.
- **Observed:** 4 checks passed, while `test_audited_schema_and_alignment` failed because the dataset has a `next_obs` group.
- **Cause:** the Phase 0 audit had not summarized that group; the assertion repeated an unsupported assumption from the first audit draft.
- **Correction:** the audit now records all `next_obs` fields and sequence lengths and checks `next_obs[t]` against `obs[t+1]` for both camera images and all three policy proprioception fields across three predetermined demos. The policy-input exclusion list remains explicit.
- **Status:** corrected; rerun passed all five data/split/config contract tests in both Python 3.10 and the legacy Python 3.8 runtime.

## 2026-10-22 — ALT core first unit-test pass had a dimensional fixture error

- **Command:** `artifacts/environments/alt_lift_train/Scripts/python.exe -m unittest discover -s tests -p test_alt_core.py -v`; Python 3.10.22, Torch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA RTX 5090.
- **Observed:** five of six new core checks passed. The cosine-search test supplied a 2D query while the bank contract correctly requires 128D embeddings. A separate `py_compile` invocation also passed an unexpanded PowerShell wildcard as a literal filename.
- **Cause:** test-fixture dimensionality and shell wildcard expansion, not a model or retrieval defect.
- **Correction:** pad synthetic retrieval fixtures to 128 dimensions and use Python `compileall` for recursive syntax validation. The encoder factory also uses the modern torchvision `weights=None` argument and falls back to `pretrained=False` for the legacy 0.11 simulator runtime.
- **Status:** corrected; all 11 tests pass under Python 3.10.22/Torch 2.7.1+cu128 and Python 3.8.20/Torch 1.10.2+cpu. Recursive `compileall` passes. The CUDA pretrained one-batch smoke passes with finite gradients and unit 128D embeddings.

## 2026-10-22 — First cross-runtime fixture creation command lost Python quotes

- **Command:** a PowerShell `python.exe -c` command tried to create a temporary ResNet state dict before loading it in WSL.
- **Observed:** PowerShell stripped the embedded string quotes and Python raised `NameError: name 'src' is not defined`; no model file was created.
- **Cause:** nested command-line quoting, matching the earlier WSL inline-code issue.
- **Correction:** moved fixture creation into `scripts/create_checkpoint_compat_fixture.py`, which also refuses to overwrite an existing fixture. The legacy loader check is rerun from that file's output.
- **Status:** corrected. A Python 3.10 state-dict fixture loads strictly in Torch 1.10.2/TorchVision 0.11.3 with no missing or unexpected keys.

## 2026-10-22 — Video smoke encoded but reported a missing v2 version attribute

- **Command:** run `scripts/check_video_backend.py` in the legacy WSL environment with `imageio` 2.31.6 and `imageio-ffmpeg` 0.4.9.
- **Observed:** FFmpeg produced the two-frame MP4, then the helper raised `AttributeError` because `imageio.v2` has no `__version__` attribute. FFmpeg also padded the 168x84 frame to 176x96 by default.
- **Cause:** package version was read from the `v2` API module rather than the top-level package; default macro-block padding changed only the diagnostic video dimensions.
- **Correction:** read the version from top-level `imageio` and set `macro_block_size=1` for both smoke and evaluation writers. The failed local output is retained and a new unique path is used for rerun.
- **Status:** corrected; the second local output encoded and decoded at exactly 168x84 with FFmpeg 4.2.2, `imageio` 2.31.6, and `imageio-ffmpeg` 0.4.9.

## 2026-10-22 — First formal evaluator invocations stopped before scoring

- **Command:** `bash scripts/with_legacy_env.sh scripts/evaluate_alt_lift.py --run-dir experiments/runs/alt_lift_ph30_image_proprio_seed42_v1` in WSL after model/bank completion; Python 3.8.20, Torch 1.10.2+cpu.
- **Observed:** the first invocation exited before simulator creation on a `FileNotFoundError` for a Windows-backslash relative artifact path. After fixing that resolver and pushing `fbcf173`, the next invocation created the environment but stopped at the pre-step frozen-observation check for seed 100000. No policy action was applied in either attempt.
- **Cause:** two integration issues: Windows-serialized relative paths were treated as POSIX literal backslashes; and the second attempt restored the fixture state using `demo_0`'s recorded `model_file`, while the frozen fixtures were generated from the environment's seeded reset model. The model-based `reset_to` path therefore did not reproduce the fixture's policy observation.
- **Correction:** normalize relative Windows separators on POSIX; for every evaluation condition, reseed Python and NumPy and call the exact validated `env.reset()` path, then compare both the full simulator-state hash and policy-observation hash to the frozen fixture before any action. The reset path has a focused regression test.
- **Status:** the two failed attempts advanced no action and produced no score. The `env.reset()` state-and-observation hash guard now passes focused tests in both runtimes; the full 50-condition retry remains pending.
