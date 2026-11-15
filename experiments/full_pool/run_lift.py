"""Validate full-pool Lift checkpoints and test the validation-selected policy."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lift_box_dp/src"))
from common import ROOT as LIFT, add_references, read_json, sha256, write_json
from evaluate_epochs import run_evaluation
from full_pool import full_pool
from matched_budget import fixtures
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("alt", "dp"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-available", action="store_true",
                        help="Only validate the saved checkpoints that are currently available.")
    args = parser.parse_args()
    cfg, alt, dp, split, image = full_pool(args.smoke)
    _, validation_seeds, records, fixture_hashes = fixtures(split, alt, cfg)
    add_references(alt, dp)
    tag = "smoke_v1" if args.smoke else "v1"
    run = LIFT / "artifacts" / f"full_pool_lift_{args.method}_{tag}"
    if read_json(run / "config.json") != cfg:
        raise ValueError("Saved training config differs from the frozen full Lift pool")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    validation_horizon = cfg["validation_horizon"]
    if args.smoke:
        validation_seeds, validation_horizon = validation_seeds[:2], 25
    expected_epochs = cfg[f"{args.method}_epochs"]
    saved_path = run / "checkpoints.json"
    saved = read_json(saved_path) if saved_path.exists() else {}
    available = [epoch for epoch in expected_epochs if str(epoch) in saved]
    if len(available) != len(expected_epochs) and not args.validate_available:
        raise ValueError(f"Expected checkpoints at {expected_epochs}, currently available {available}")
    scores = {}
    for epoch in available:
        scores[epoch] = run_evaluation(args.method, run, epoch, cfg, alt, image, validation_seeds,
                                       validation_horizon, records, fixture_hashes, "validation")
    if args.validate_available:
        return
    criteria = cfg["selection_order"][:-1]
    selected = max(scores, key=lambda epoch: tuple(scores[epoch][metric]["count"] for metric in criteria) + (-epoch,))
    selection = {"method": args.method, "selected_epoch": selected, "selection_order": cfg["selection_order"],
                 "validation_seeds": validation_seeds, "fixture_hashes": fixture_hashes,
                 "validation_scores": scores, "checkpoints": saved}
    write_json(run / "selection.json", selection)
    print(f"Lift {args.method} selected epoch {selected}", flush=True)
    if args.smoke:
        return
    settings = read_json(alt / split["evaluation_config"]["path"])
    if set(validation_seeds) & set(settings["test_seeds"]):
        raise ValueError("Lift validation and test starts overlap")
    metrics = run_evaluation(args.method, run, selected, cfg, alt, image, settings["test_seeds"],
                             settings["horizon"], None, split["evaluation_initial_state_contract"]["sha256"], "test")
    write_json(run / "completed.json", {"status": "completed", "method": args.method,
               "eval_source_sha256": sha256(__file__), "selection_sha256": sha256(run / "selection.json"),
               "training": read_json(run / "training.json"), "test": metrics})


if __name__ == "__main__":
    main()
