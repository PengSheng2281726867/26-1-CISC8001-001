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
