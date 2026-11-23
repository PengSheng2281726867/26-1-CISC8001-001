"""Simulation-only phase annotations based on bilateral finger-pad contact and cube height."""
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lift_box_dp/src"))
from simulator import serve_stdio


def phase(env):
    lift = env.env
    if lift._check_success():
        return 2
    return int(lift._check_grasp(lift.robots[0].gripper, lift.cube))


def observe(env):
    return {"oracle_phase": phase(env)}


def label_demonstrations(env, image, split, command, episode_ids):
    if command != "label_demonstrations" or episode_ids != split["train_subsets"]["100"]:
        raise ValueError("Unexpected phase-audit request")
    result = {}
    max_height_error = 0.0
    with h5py.File(image, "r") as data:
        for key in episode_ids:
            demo = data["data"][key]
            states = demo["states"][:]
            heights = demo["obs"]["object"][:, 2]
            model = demo.attrs["model_file"]
            if isinstance(model, bytes):
                model = model.decode("utf-8")
            env.reset_to({"model": model, "states": states[0]})
            lift = env.env
            labels = []
            for state, height in zip(states, heights):
                lift.sim.set_state_from_flattened(state)
                lift.sim.forward()
                restored = lift.sim.data.body_xpos[lift.cube_body_id][2]
                max_height_error = max(max_height_error, abs(float(restored - height)))
                labels.append(phase(env))
            result[key] = labels
    if max_height_error > 0.001:
        raise ValueError("Restored cube heights disagree with the recorded object observation")
    return {"episodes": result, "max_cube_height_error_m": max_height_error}


if __name__ == "__main__":
    serve_stdio(observation_fields=observe, extra_command=label_demonstrations)
