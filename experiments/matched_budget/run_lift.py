"""Train and evaluate fixed-budget Lift policies on frozen validation starts."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "lift_box_dp/src"))

from common import ROOT as LIFT, add_references, read_json, sha256, write_json
from evaluate_epochs import run_evaluation
from matched_budget import PROTOCOL_PATH, fixtures, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("alt", "dp"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    locks = ROOT / "artifacts/matched_budget/locks"
    locks.mkdir(parents=True, exist_ok=True)
    label = "smoke" if args.smoke else "runs"
    lock = locks / f"lift_{args.method}_{label}.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        print(f"A {label} process already owns {lock}", flush=True)
        return
    try:
        run_campaign(args)
    finally:
        lock.unlink()


def run_campaign(args):
    protocol = read_json(PROTOCOL_PATH)
    scales, seeds = protocol["lift"]["scales"], protocol["model_seeds"]
    if args.smoke:
        scales, seeds = scales[:1], seeds[:1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    entrypoint_hash = sha256(__file__)
    for scale in scales:
        for seed in seeds:
            cfg, alt, dp, split, image = matched(scale, seed, args.smoke)
            design, validation_seeds, records, hashes = fixtures(split, alt)
            run_id = f"matched_lift_{args.method}_{scale}_{seed}"
            if args.smoke:
                run_id += "_smoke"
            run = LIFT / "artifacts" / run_id
            if (run / "completed.json").is_file():
                if read_json(run / "completed.json")["protocol_sha256"] != sha256(PROTOCOL_PATH):
                    raise ValueError(f"Existing run uses a different protocol: {run}")
                continue
            source = "train_alt10.py" if args.method == "alt" else "train.py"
            command = [sys.executable, "-B", str(LIFT / "src" / source), run_id,
                       "--scale", str(scale), "--matched-seed", str(seed)]
            if args.smoke:
                command.append("--matched-smoke")
            print(run_id, flush=True)
            if not (run / "training.json").is_file():
                subprocess.run(command, cwd=ROOT, check=True)
            training = read_json(run / "training.json")
            if read_json(run / "config.json") != cfg or training["steps"] != cfg["max_updates"]:
                raise ValueError(f"Resolved config or training budget differs: {run}")
            epoch, = map(int, read_json(run / "checkpoints.json"))
            horizon = design["validation_horizon"]
            if args.smoke:
                validation_seeds, horizon = validation_seeds[:2], 25
            add_references(alt, dp)
            import torch
            torch.set_num_threads(4)
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
            run_evaluation(args.method, run, epoch, cfg, alt, image, validation_seeds, horizon,
                           records, hashes, "fixed_validation")
            write_json(run / "completed.json", {
                "task": "lift", "method": args.method, "scale": scale, "model_seed": seed,
                "smoke": args.smoke, "optimizer_steps": training["steps"], "batch_size": cfg["batch_size"],
                "train_episode_ids": cfg["train_ids"], "split_manifest_sha256": cfg["split_sha256"],
                "dataset_fingerprint": split["source_files"]["image_hdf5"]["sha256"],
                "protocol_sha256": sha256(PROTOCOL_PATH), "base_commit": revision,
                "entrypoint_sha256": entrypoint_hash, "train_source_sha256": sha256(LIFT / "src" / source),
                "checkpoint_selection": protocol["checkpoint_selection"],
                "checkpoint": read_json(run / "checkpoints.json")[str(epoch)],
                "checkpoint_batch_epoch": epoch, "training_seconds": training["wall_seconds"],
                "torch": training["torch"], "gpu": training["gpu"],
                "evaluation_seeds": validation_seeds, "fixture_hashes": hashes,
            })


if __name__ == "__main__":
    main()
