"""Describe lift durations recorded in the frozen expert demonstrations."""
import argparse

import h5py
import numpy as np

from common import ROOT, contract, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/exp_001_lift30/demo_hold.json")
    args = parser.parse_args()
    cfg, _, _, split, image = contract()
    image_hash = sha256(image)
    if image_hash != split["source_files"]["image_hdf5"]["sha256"]:
        raise ValueError("Image HDF5 checksum mismatch")
    rows = []
    with h5py.File(str(image), "r") as h5:
        for episode in split["train_subsets"]["30"]:
            demo = h5["data"][episode]
            rewards = np.asarray(demo["rewards"]).reshape(-1)
            if len(rewards) != len(demo["actions"]) or not np.isin(rewards, [0, 1]).all():
                raise ValueError(f"Unexpected sparse rewards in {episode}")
            current = longest = 0
            for reward in rewards:
                current = current + 1 if reward == 1 else 0
                longest = max(current, longest)
            rows.append({"episode": episode, "steps": len(rewards), "max_consecutive_positive_rewards": longest,
                         "terminal_positive_reward_steps": current})
    durations = [row["max_consecutive_positive_rewards"] for row in rows]
    result = {"image_sha256": image_hash, "split_sha256": cfg["split_sha256"],
              "hold_steps": cfg["hold_steps"], "n": len(rows), "ever_positive": sum(x > 0 for x in durations),
              "recorded_at_least_hold_steps": sum(x >= cfg["hold_steps"] for x in durations),
              "median_consecutive_positive": float(np.median(durations)),
              "max_consecutive_positive": max(durations), "episodes": rows}
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    print({key: value for key, value in result.items() if key != "episodes"})


if __name__ == "__main__":
    main()
