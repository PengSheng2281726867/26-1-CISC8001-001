"""Inspect the chosen head on the held-out demos, including deployed action clipping."""
import argparse
from pathlib import Path

import numpy as np
import torch

from study import (DESIGN, LIFT, Head, LiftFrameStore, MemoryBank, correct, load_base,
                   pairs, read_json, sha256, write_json)


def diagnose(run):
    policy, _, split, image, _ = load_base()
    training = read_json(run / "training.json")
    epoch = read_json(run / "selection.json")["epoch"]
    ckpt = run / f"epoch{epoch}.pt"
    if sha256(ckpt) != training["checkpoints"][str(epoch)]:
        raise ValueError("Selected checkpoint changed")
    held = LiftFrameStore(image, split["heldout_validation_episode_ids"])
    query = MemoryBank.load(run / "heldout_queries.npz")
    np.testing.assert_array_equal(query.source_indices, held.source_indices)
    np.testing.assert_array_equal(query.episode_ids, held.episode_ids_per_frame)
    np.testing.assert_array_equal(query.local_frame_indices, held.local_indices)
    expert = {k: v.actions for k, v in held.episodes.items()}
    x, _, index = pairs(query, policy, expert)
    head = Head().eval()
    head.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        delta = head(x).numpy()
    prior = np.stack([policy.actions_by_episode[policy.bank.episode_ids[i]][policy.bank.local_frame_indices[i]]
                      for i in index])
    pred = correct(prior, delta)
    target = np.stack([expert[k][i] for k, i in zip(query.episode_ids, query.local_frame_indices)])
    np.testing.assert_array_equal(prior[:, 6], pred[:, 6])
    metrics = {}
    for name, columns in (("translation", slice(0, 3)), ("rotation", slice(3, 6)), ("pose6", slice(0, 6))):
        metrics[name] = {label: float(np.sqrt(np.mean((values[:, columns] - target[:, columns]) ** 2)))
                         for label, values in (("alt", prior), ("residual_after_clip", pred))}
    result = dict(heldout_frames=len(x), epoch=epoch, rmse_in_controller_units=metrics,
                  gripper_sign_disagreement_frames=int(np.count_nonzero(np.sign(prior[:, 6]) != np.sign(target[:, 6]))),
                  frames_with_any_motion_clipping=int(np.any(np.abs(prior[:, :6]+delta) > 1, axis=1).sum()),
                  selection=DESIGN["selection"])
    write_json(run / "offline_actions.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    run_id = parser.parse_args().run_id
    if Path(run_id).name != run_id or run_id in (".", ".."):
        parser.error("Use a single directory name for the run")
    diagnose(LIFT / "artifacts/residual" / run_id)
