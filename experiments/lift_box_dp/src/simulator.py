"""Internal WSL subprocess for the existing RoboSuite Lift simulator."""
import os
import pickle
import random

import numpy as np
import sys

import h5py

from common import IMAGE_KEYS, PROPRIO_KEYS, add_references, contract, read_json, sha256


def serve(input_stream, output_stream):
    cfg, alt, _, split, image = contract()
    add_references(alt)
    sys.path.insert(0, str(alt / "scripts"))
    from evaluate_alt_lift import _array_hash, _policy_obs_hash, reset_to_frozen_seed
    from robomimic.utils import env_utils, obs_utils

    eval_path = alt / split["evaluation_config"]["path"]
    fixtures_path = alt / split["evaluation_initial_state_contract"]["path"]
    if sha256(eval_path) != split["evaluation_config"]["sha256"]:
        raise ValueError("Evaluation configuration mismatch")
    if sha256(fixtures_path) != split["evaluation_initial_state_contract"]["sha256"]:
        raise ValueError("Initial-state manifest mismatch")
    settings = read_json(eval_path)
    fixtures = {row["seed"]: row for row in read_json(fixtures_path)["initial_conditions"]}
    if list(fixtures) != settings["test_seeds"]:
        raise ValueError("Evaluation seeds mismatch")
    import json
    with h5py.File(str(image), "r") as h5:
        data = h5["data"]
        meta = json.loads(data.attrs["env_args"])
        keys = sorted(data["demo_0"]["obs"].keys())
    rgb = [key for key in keys if key.endswith("_image")]
    obs_utils.initialize_obs_utils_with_obs_specs({"obs": {
        "rgb": rgb, "low_dim": [key for key in keys if key not in rgb],
    }})
    meta["env_kwargs"].update(horizon=settings["horizon"], control_freq=settings["control_freq"],
                              reward_shaping=settings["reward_shaping"])
    env = env_utils.create_env_from_metadata(env_meta=meta, render=False,
                                             render_offscreen=True, use_image_obs=True)
    env.postprocess_visual_obs = False
    try:
        while True:
            command, value = pickle.load(input_stream)
            if command == "close":
                return
            if command == "reset":
                obs = reset_to_frozen_seed(env, value, fixtures[value])
                result = {key: obs[key] for key in IMAGE_KEYS + PROPRIO_KEYS}
            elif command == "validation_fixture":
                if value in fixtures:
                    raise ValueError("Validation seed overlaps the frozen test set")
                random.seed(int(value))
                np.random.seed(int(value))
                obs = env.reset()
                state = np.asarray(env.get_state()["states"], dtype=np.float64)
                result = {"seed": value, "sim_state_sha256": _array_hash(state),
                          "policy_observation_sha256": _policy_obs_hash(obs)}
            elif command == "reset_validation":
                seed, fixture = value
                if seed in fixtures or seed != fixture["seed"]:
                    raise ValueError("Invalid independent validation seed")
                obs = reset_to_frozen_seed(env, seed, fixture)
                result = {key: obs[key] for key in IMAGE_KEYS + PROPRIO_KEYS}
            elif command == "step":
                obs, reward, done, _ = env.step(value)
                result = ({key: obs[key] for key in IMAGE_KEYS + PROPRIO_KEYS},
                          bool(env.is_success()["task"]), float(reward), bool(done))
            else:
                raise ValueError("Unknown simulator command")
            pickle.dump(result, output_stream, protocol=4)
            output_stream.flush()
    finally:
        if hasattr(env.env, "close"):
            env.env.close()


if __name__ == "__main__":
    # Keep legacy simulator diagnostics out of the private binary subprocess pipe.
    with os.fdopen(os.dup(1), "wb") as responses:
        os.dup2(2, 1)
        serve(sys.stdin.buffer, responses)
