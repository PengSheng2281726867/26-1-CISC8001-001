"""Extend the six frozen Lift-10 checkpoints to 24 paired validation conditions."""
import argparse

import torch

from common import ROOT, add_references, read_json, sha256, write_json
from epoch_study import STUDY_DIR, study
from evaluate import Simulator, wilson
from evaluate_epochs import check_trace, run_evaluation


NEW_SEEDS = list(range(90008, 90024))
SAMPLING_SEED = 46
LABEL = "validation_extra16"
MANIFEST = STUDY_DIR / "validation_extra16_initial_states.json"
ORIGINAL = STUDY_DIR / "validation_initial_states.json"
METRICS = ("ever_success", "hold_success", "final_success")


def freeze(cfg, alt, split):
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    test_path = alt / split["evaluation_initial_state_contract"]["path"]
    if sha256(test_path) != split["evaluation_initial_state_contract"]["sha256"]:
        raise ValueError("Frozen test manifest checksum mismatch")
    old_rows = read_json(ORIGINAL)["initial_conditions"]
    test_rows = read_json(test_path)["initial_conditions"]
    occupied = old_rows + test_rows
    if set(NEW_SEEDS) & {r["seed"] for r in occupied}:
        raise ValueError("New validation seeds overlap an existing set")
    logs = ROOT / "artifacts/lift10_extra16_fixtures"
    logs.mkdir(parents=True, exist_ok=False)
    simulator = Simulator(alt, logs / "simulator.log")
    try:
        rows = [simulator.send("validation_fixture", seed) for seed in NEW_SEEDS]
        for row in rows:
            simulator.send("reset_validation", (row["seed"], row))
    finally:
        simulator.close()
    hashes = {r["sim_state_sha256"] for r in rows}
    if len(hashes) != len(rows) or hashes & {r["sim_state_sha256"] for r in occupied}:
        raise ValueError("New validation states are duplicated or overlap a frozen set")
    write_json(MANIFEST, {"initial_conditions": rows, "original_sha256": sha256(ORIGINAL),
                          "supplement_sha256": cfg["supplement_config_sha256"],
                          "sampling_seed": SAMPLING_SEED})
    print("Frozen and repeat-verified 16 additional validation states", flush=True)


def verified_rows(run, method, epoch, cfg, seeds, fixture_sha, label):
    output = run / f"{label}_e{epoch:03d}"
    protocol = read_json(output / "protocol.json")
    recorded = read_json(run / "checkpoints.json")[str(epoch)]
    expected = {"policy": method, "epoch": epoch, "config": cfg, "seeds": seeds,
                "horizon": cfg["validation_horizon"], "initial_states_sha256": fixture_sha,
                "checkpoint": recorded}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected evaluation protocol: {output}")
    if label == LABEL and protocol.get("sampling_seed") != SAMPLING_SEED:
        raise ValueError("Additional sampling seed mismatch")
    return check_trace(output, protocol)


def counts(rows):
    result = {}
    for key in METRICS:
        count = sum(row[key] == "True" for row in rows)
        result[key] = {"count": count, "n": len(rows), "wilson95": wilson(count, len(rows))}
    return result


def summarize(cfg, new_sha):
    results = {}
    for method in ("alt", "dp"):
        run = ROOT / "artifacts" / f"{method}10_supplement_v1"
        epochs = {}
        for epoch in cfg[f"{method}_epochs"]:
            old = verified_rows(run, method, epoch, cfg, cfg["validation_seeds"], sha256(ORIGINAL), "validation")
            new = verified_rows(run, method, epoch, cfg, NEW_SEEDS, new_sha, LABEL)
            epochs[str(epoch)] = {"original8": counts(old), "new16": counts(new),
                                  "combined24": counts(old + new)}
        results[method] = epochs
    output = STUDY_DIR / "validation_extra16_summary.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, {"initial_states_sha256": new_sha, "config": cfg, "results": results})
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "alt", "dp", "summarize"])
    args = parser.parse_args()
    cfg, alt, dp, split, image = study(supplement=True)
    add_references(alt, dp)
    if args.phase == "freeze":
        freeze(cfg, alt, split)
        return
    manifest = read_json(MANIFEST)
    fixtures = {row["seed"]: row for row in manifest["initial_conditions"]}
    if (list(fixtures) != NEW_SEEDS or manifest["original_sha256"] != sha256(ORIGINAL)
            or manifest["sampling_seed"] != SAMPLING_SEED
            or manifest["supplement_sha256"] != cfg["supplement_config_sha256"]):
        raise ValueError("Additional validation manifest mismatch")
    fixture_sha = sha256(MANIFEST)
    if args.phase == "summarize":
        summarize(cfg, fixture_sha)
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    method = args.phase
    run = ROOT / "artifacts" / f"{method}10_supplement_v1"
    for epoch in cfg[f"{method}_epochs"]:
        run_evaluation(method, run, epoch, cfg, alt, image, NEW_SEEDS, cfg["validation_horizon"],
                       fixtures, fixture_sha, LABEL, sampling_seed=SAMPLING_SEED)


if __name__ == "__main__":
    main()
