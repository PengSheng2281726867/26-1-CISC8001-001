"""Evaluate ALT-50 checkpoints on the same 24 Lift validation conditions."""
import argparse

import torch

from common import ROOT, add_references, read_json, sha256, write_json
from epoch_study import SCALE50_PATH, study
from evaluate_epochs import run_evaluation
from extend_epoch_validation import (
    LABEL, MANIFEST, NEW_SEEDS, ORIGINAL, SAMPLING_SEED, counts, verified_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("phase", choices=["evaluate", "summarize"])
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    cfg, alt, dp, _, image = study(scale=50)
    add_references(alt, dp)
    original = read_json(ORIGINAL)
    extra = read_json(MANIFEST)
    base_cfg, *_ = study(supplement=True)
    original_sha = sha256(ORIGINAL)
    extra_sha = sha256(MANIFEST)
    if (original["study_config_sha256"] != cfg["study_config_sha256"]
            or extra["original_sha256"] != original_sha
            or extra["sampling_seed"] != SAMPLING_SEED
            or extra["supplement_sha256"] != base_cfg["supplement_config_sha256"]):
        raise ValueError("Frozen validation manifests changed")
    old = {row["seed"]: row for row in original["initial_conditions"]}
    new = {row["seed"]: row for row in extra["initial_conditions"]}
    if list(old) != cfg["validation_seeds"] or list(new) != NEW_SEEDS or set(old) & set(new):
        raise ValueError("Invalid validation seed membership")
    run = ROOT / "artifacts" / args.run_id
    if read_json(run / "config.json") != cfg:
        raise ValueError("Training run does not match the ALT-50 configuration")
    checkpoints = cfg["alt_epochs"]
    if args.phase == "evaluate":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        epochs = checkpoints if args.epoch is None else [args.epoch]
        if any(epoch not in checkpoints for epoch in epochs):
            parser.error("Epoch must be one of the frozen checkpoints")
        for epoch in epochs:
            run_evaluation("alt", run, epoch, cfg, alt, image, list(old), cfg["validation_horizon"],
                           old, original_sha, "validation")
            run_evaluation("alt", run, epoch, cfg, alt, image, list(new), cfg["validation_horizon"],
                           new, extra_sha, LABEL, sampling_seed=SAMPLING_SEED)
        return
    if args.epoch is not None:
        parser.error("--epoch applies only to evaluate")
    recorded = read_json(run / "checkpoints.json")
    epochs = {}
    for epoch in checkpoints:
        a = verified_rows(run, "alt", epoch, cfg, list(old), original_sha, "validation")
        b = verified_rows(run, "alt", epoch, cfg, list(new), extra_sha, LABEL)
        epochs[str(epoch)] = {"artifact": recorded[str(epoch)], "original8": counts(a),
                              "new16": counts(b), "combined24": counts(a + b)}
    output = SCALE50_PATH.parent / f"validation_{args.run_id}.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, {"config": cfg, "training": read_json(run / "training.json"),
                        "validation_original_sha256": original_sha,
                        "validation_extra_sha256": extra_sha, "epochs": epochs})
    print(output, flush=True)


if __name__ == "__main__":
    main()
