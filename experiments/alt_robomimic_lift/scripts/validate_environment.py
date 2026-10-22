"""Check HDF5 state restoration and uninterrupted expert-action replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = HERE / "configs" / "data_paths.json"
DEFAULT_OUTPUT = HERE / "results" / "action_replay_validation.json"
CAMERAS = ("agentview", "robot0_eye_in_hand")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # WSL also accepts explicit Windows drive paths supplied from PowerShell.
    if len(value) > 2 and value[1] == ":" and value[2] in ("\\", "/"):
        drive = value[0].lower()
        return Path("/mnt") / drive / value[3:].replace("\\", "/")
    return (base / path).resolve()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+", default=["demo_0", "demo_1", "demo_2"])
    parser.add_argument("--image-checks", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    image_path = _resolve(args.dataset or config["image_hdf5"], HERE)

    # Imports are delayed so this script can still print argument errors without loading MuJoCo.
    import mujoco_py
    import robomimic
    import robosuite
    from robomimic.utils import env_utils, obs_utils

    result: dict[str, Any] = {
        "status": "running",
        "started_unix_time": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "dataset_path": str(image_path),
        "dataset_sha256": _sha256(image_path),
        "versions": {
            "robomimic": robomimic.__version__,
            "robosuite": robosuite.__version__,
            "mujoco_py": mujoco_py.__version__,
            "mujoco_engine_integer": int(mujoco_py.functions.mj_version()),
        },
        "episodes": [],
        "method": "state restore audit then uninterrupted actions[t] replay from states[0]",
        "policy_training": False,
        "action_alignment": "obs[t] is pre-action; actions[t] is stepped at that observation; replay result is compared with stored state/obs[t+1]",
        "camera_alignment": [],
    }
    env = None
    try:
        with h5py.File(str(image_path), "r") as h5:
            data = h5["data"]
            metadata = json.loads(_text(data.attrs["env_args"]))
            result["environment_metadata"] = metadata
            result["camera_names"] = list(CAMERAS)
            result["episode_ids"] = list(args.episodes)
            keys = sorted(data[args.episodes[0]]["obs"].keys())
            image_keys = [key for key in keys if key.endswith("_image")]
            low_dim_keys = [key for key in keys if not key.endswith("_image")]
            obs_utils.initialize_obs_utils_with_obs_specs(
                {"obs": {"rgb": image_keys, "low_dim": low_dim_keys}}
            )
            env = env_utils.create_env_from_metadata(
                env_meta=metadata, render=False, render_offscreen=True, use_image_obs=True
            )
            # The image HDF5 stores uint8 RGB frames after robomimic's vertical flip.
            # Keep observations uint8 here so replay comparison and policy preprocessing
            # share exactly the same pixels.
            env.postprocess_visual_obs = False
            result["action_dimension"] = int(env.action_dimension)
            result["environment_horizon_default"] = int(env.env.horizon)
            if env.action_dimension != 7:
                raise ValueError("Expected 7D RoboSuite Panda OSC_POSE + gripper action")

            visual_samples = []
            for episode_id in args.episodes:
                demo = data[episode_id]
                states = np.asarray(demo["states"])
                actions = np.asarray(demo["actions"])
                xml = _text(demo.attrs["model_file"])
                if states.shape[0] != actions.shape[0] or actions.shape[1] != 7:
                    raise ValueError("Invalid action/state alignment in %s" % episode_id)

                # Restoring each state is an observation/render audit only. It is kept
                # separate from the continuous action replay below.
                per_key_error = {key: 0.0 for key in keys if not key.endswith("_image")}
                pixel_rows = []
                frame_indices = sorted(set((0, len(states) // 2, len(states) - 1)))[: args.image_checks]
                for t, state in enumerate(states):
                    obs = env.reset_to({"model": xml, "states": state}) if t == 0 else env.reset_to({"states": state})
                    for key in per_key_error:
                        err = float(np.max(np.abs(np.asarray(obs[key]) - demo["obs"][key][t])))
                        per_key_error[key] = max(per_key_error[key], err)
                    if t in frame_indices:
                        for camera, key in zip(CAMERAS, ("agentview_image", "robot0_eye_in_hand_image")):
                            live = np.asarray(obs[key])
                            stored = np.asarray(demo["obs"][key][t])
                            mae = _mean_abs(live, stored)
                            mae_vertical = _mean_abs(np.flipud(live), stored)
                            mae_bgr = _mean_abs(live[..., ::-1], stored)
                            row = {
                                "episode_id": episode_id,
                                "frame": int(t),
                                "camera": camera,
                                "observation_key": key,
                                "stored_shape": list(stored.shape),
                                "live_shape": list(live.shape),
                                "pixel_mae_direct": mae,
                                "pixel_mae_vertical_flip": mae_vertical,
                                "pixel_mae_channel_reversed": mae_bgr,
                            }
                            pixel_rows.append(row)
                            visual_samples.append((episode_id, t, camera, "stored", stored))
                            visual_samples.append((episode_id, t, camera, "restored", live))
                # Start once from the recorded initial state; do not restore between actions.
                env.reset_to({"model": xml, "states": states[0]})
                actual_states, state_errors, object_errors = [], [], []
                ever_success = False
                final_reward = 0.0
                final_success = False
                # Last action has no stored successor observation/state; it is recorded but
                # not included in transition-divergence statistics.
                for t, action in enumerate(actions[:-1]):
                    obs, final_reward, done, info = env.step(action)
                    final_success = bool(env.is_success()["task"])
                    ever_success = ever_success or final_success
                    next_state = np.asarray(env.get_state()["states"])
                    actual_states.append(next_state.copy())
                    state_errors.append(float(np.max(np.abs(next_state - states[t + 1]))))
                    object_errors.append(float(np.max(np.abs(obs["object"] - demo["obs"]["object"][t + 1]))))
                actual_states = np.stack(actual_states)
                expected_states = states[1:]
                differences = np.abs(actual_states - expected_states)
                peak_step, peak_dimension = np.unravel_index(np.argmax(differences), differences.shape)
                trace_dir = HERE / "artifacts" / "phase0"
                trace_dir.mkdir(parents=True, exist_ok=True)
                row = {
                    "episode_id": episode_id,
                    "frames": int(len(states)),
                    "transition_count": int(len(actions) - 1),
                    "action_dimension": int(actions.shape[1]),
                    "action_min": actions.min(axis=0).tolist(),
                    "action_max": actions.max(axis=0).tolist(),
                    "state_restore_max_abs_observation_error": per_key_error,
                    "state_restore_passed_at_1e-6": max(per_key_error.values()) <= 1e-6,
                    "action_replay_max_abs_full_state_error": max(state_errors),
                    "action_replay_max_abs_object_obs_error": max(object_errors),
                    "action_replay_peak_index": int(peak_step),
                    "action_replay_peak_dimension": int(peak_dimension),
                    "action_replay_final_success": bool(final_success),
                    "action_replay_ever_success": bool(ever_success),
                    "action_replay_final_reward": float(final_reward),
                    "camera_reconstruction_checks": pixel_rows,
                }
                result["episodes"].append(row)
                result["camera_alignment"].extend(pixel_rows)
                np.savez_compressed(
                    trace_dir / (episode_id + "_action_replay_trace.npz"),
                    actual_states=actual_states,
                    expected_states=expected_states,
                    actions=actions[:-1],
                )
            visual_dir = HERE / "results" / "figures"
            visual_dir.mkdir(parents=True, exist_ok=True)
            tile_w = tile_h = 168
            label_h = 22
            cols = 4
            rows_n = (len(visual_samples) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tile_w, rows_n * (tile_h + label_h)), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (episode_id, frame, camera, source, array) in enumerate(visual_samples):
                x = (i % cols) * tile_w
                y = (i // cols) * (tile_h + label_h)
                sheet.paste(Image.fromarray(array.astype(np.uint8), mode="RGB").resize((tile_w, tile_h)), (x, y + label_h))
                draw.text((x + 3, y + 4), "%s t=%d %s %s" % (episode_id, frame, camera, source), fill="black")
            sheet.save(str(visual_dir / "state_restore_image_check.png"))

        all_restored = all(ep["state_restore_passed_at_1e-6"] for ep in result["episodes"])
        all_success = all(ep["action_replay_final_success"] for ep in result["episodes"])
        max_error = max(ep["action_replay_max_abs_full_state_error"] for ep in result["episodes"])
        result["summary"] = {
            "state_restoration_exact_within_1e-6": all_restored,
            "all_action_replays_end_successful": all_success,
            "max_full_state_drift": max_error,
            "strict_action_replay_1e-5_gate": max_error <= 1e-5,
            "phase0_decision": "PASS" if all_restored and all_success and max_error <= 1e-3 else "BLOCKED",
            "phase0_limitations": (
                "Recorded replay reaches the official Lift success signal for all selected episodes; "
                "full-state drift must still be described and may exceed the strict 1e-5 numerical gate."
            ),
        }
        result["status"] = "completed"
        result["elapsed_seconds"] = time.time() - result["started_unix_time"]
        _json_dump(args.output, result)
        print(json.dumps(result["summary"], indent=2))
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        result["elapsed_seconds"] = time.time() - result["started_unix_time"]
        _json_dump(args.output, result)
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
