"""Freeze and verify deterministic Lift reset conditions for future paired runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = HERE / "configs" / "data_paths.json"
DEFAULT_EVAL = HERE / "configs" / "evaluation.json"
DEFAULT_OUTPUT = HERE / "results" / "evaluation_initial_states.json"
POLICY_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
POLICY_PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str, base: Path) -> Path:
    if len(value) > 2 and value[1] == ":" and value[2] in ("\\", "/"):
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _policy_obs_hash(obs: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in POLICY_IMAGE_KEYS + POLICY_PROPRIO_KEYS:
        arr = np.ascontiguousarray(obs[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(json.dumps(list(arr.shape)).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_config = json.loads(args.data_config.read_text(encoding="utf-8"))
    eval_config = json.loads(args.eval_config.read_text(encoding="utf-8"))
    image_path = _resolve(data_config["image_hdf5"], HERE)
    seeds = [int(seed) for seed in eval_config["test_seeds"]]

    import robomimic
    import robosuite
    import mujoco_py
    from robomimic.utils import env_utils, obs_utils

    with h5py.File(str(image_path), "r") as h5:
        metadata = json.loads(h5["data"].attrs["env_args"])
        keys = sorted(h5["data"]["demo_0"]["obs"].keys())
    rgb_keys = [key for key in keys if key.endswith("_image")]
    low_dim_keys = [key for key in keys if not key.endswith("_image")]
    obs_utils.initialize_obs_utils_with_obs_specs({"obs": {"rgb": rgb_keys, "low_dim": low_dim_keys}})
    metadata = json.loads(json.dumps(metadata))
    metadata["env_kwargs"].update(
        {
            "ignore_done": False,
            "horizon": int(eval_config["horizon"]),
            "control_freq": int(eval_config["control_freq"]),
            "reward_shaping": bool(eval_config["reward_shaping"]),
        }
    )
    env = env_utils.create_env_from_metadata(
        env_meta=metadata, render=False, render_offscreen=True, use_image_obs=True
    )
    env.postprocess_visual_obs = False
    started = time.time()
    fixtures = []
    model_hash = None
    try:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            obs = env.reset()
            sim_state = np.asarray(env.get_state()["states"], dtype=np.float64)
            model_hash = hashlib.sha256(env.get_state()["model"].encode("utf-8")).hexdigest()
            fixture = {
                "seed": seed,
                "sim_state": sim_state.tolist(),
                "sim_state_sha256": _array_hash(sim_state),
                "policy_observation_sha256": _policy_obs_hash(obs),
                "success_at_reset": bool(env.is_success()["task"]),
            }
            fixtures.append(fixture)

        repeat_results = []
        fixture_by_seed = {item["seed"]: item for item in fixtures}
        for seed in eval_config["validation_repeat_seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            obs = env.reset()
            sim_state = np.asarray(env.get_state()["states"], dtype=np.float64)
            expected = fixture_by_seed[int(seed)]
            repeated_state_hash = _array_hash(sim_state)
            repeated_obs_hash = _policy_obs_hash(obs)
            repeat_results.append(
                {
                    "seed": int(seed),
                    "state_hash_matches": repeated_state_hash == expected["sim_state_sha256"],
                    "policy_observation_hash_matches": repeated_obs_hash == expected["policy_observation_sha256"],
                    "repeated_state_sha256": repeated_state_hash,
                    "repeated_policy_observation_sha256": repeated_obs_hash,
                }
            )
        payload = {
            "schema_version": 1,
            "status": "completed",
            "dataset_sha256": _sha256(image_path),
            "environment": {
                "env_name": metadata["env_name"],
                "robomimic_version": robomimic.__version__,
                "robosuite_version": robosuite.__version__,
                "robosuite_source_commit": "d9baa17d5328d7b0ec15ecda54c76c61bb983124",
                "mujoco_py_version": mujoco_py.__version__,
                "mujoco_engine_integer": int(mujoco_py.functions.mj_version()),
                "model_xml_sha256": model_hash,
                "controller": metadata["env_kwargs"]["controller_configs"],
                "cameras": list(POLICY_IMAGE_KEYS),
                "camera_shape_hwc": [84, 84, 3],
                "control_freq": int(eval_config["control_freq"]),
                "horizon": int(eval_config["horizon"]),
                "ignore_done": False,
                "reward_shaping": False,
                "success_signal": eval_config["success_signal"],
            },
            "seed_contract": {
                "method": "seed Python random and NumPy global RNG, then env.reset(); retain full MuJoCo state as a compact reset fixture",
                "seeds": seeds,
                "repeat_seeds": [int(x) for x in eval_config["validation_repeat_seeds"]],
                "repeat_checks": repeat_results,
                "all_repeat_checks_passed": all(
                    row["state_hash_matches"] and row["policy_observation_hash_matches"]
                    for row in repeat_results
                ),
                "unique_initial_state_hashes": len(set(row["sim_state_sha256"] for row in fixtures)),
                "states_are_reset_fixtures_only_not_policy_inputs": True,
            },
            "initial_conditions": fixtures,
            "elapsed_seconds": time.time() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(payload["seed_contract"], indent=2))
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
