"""Select checkpoint epochs on frozen validation fixtures and evaluate once on test."""
import argparse
import csv

import h5py
import numpy as np
import torch

from common import ROOT, add_references, read_json, sha256, write_json
from compare import verify_actions
from epoch_study import STUDY_DIR, study
from evaluate import Simulator, run_rollouts
from model import batch_observations, make_model


def freeze_fixtures(cfg, alt, split):
    path = STUDY_DIR / "validation_initial_states.json"
    if path.exists():
        raise FileExistsError(path)
    logs = ROOT / "artifacts/lift10_fixtures"
    logs.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(alt, logs / "simulator.log")
    try:
        rows = [simulator.send("validation_fixture", seed) for seed in cfg["validation_seeds"]]
        for row in rows:
            simulator.send("reset_validation", (row["seed"], row))
    finally:
        simulator.close()
    if len({row["sim_state_sha256"] for row in rows}) != len(rows):
        raise ValueError("Validation simulator states are duplicated")
    # The holdout also excludes any state shared with the historical test fixtures.
    original = read_json(alt / split["evaluation_initial_state_contract"]["path"])["initial_conditions"]
    if {r["sim_state_sha256"] for r in original} & {r["sim_state_sha256"] for r in rows}:
        raise ValueError("Validation initial state overlaps test")
    write_json(path, {"study_config_sha256": cfg["study_config_sha256"], "initial_conditions": rows})
    print(f"Frozen and repeat-verified {len(rows)} validation initial states", flush=True)


def load_act(method, run, epoch, cfg, image, device):
    record = read_json(run / "checkpoints.json")[str(epoch)]
    path = run / f"epoch{epoch:03d}.pt"
    if sha256(path) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != epoch or checkpoint["config"] != cfg or checkpoint.get("smoke", False):
        raise ValueError("Checkpoint protocol mismatch")
    if method == "dp":
        model = make_model(cfg)
        model.load_state_dict(checkpoint["weights"])
        model.to(device).eval()

        def act(obs):
            return model.predict_action(batch_observations(obs, device))["action"][:, 0].cpu().numpy()
    else:
        from alt_lift.data import ProprioNormalizer
        from alt_lift.encoder import VisualProprioEncoder
        from alt_lift.memory import MemoryBank
        from alt_lift.policy import ActionRetrievalPolicy

        bank_path = run / f"epoch{epoch:03d}_bank.npz"
        if sha256(bank_path) != record["bank_sha256"]:
            raise ValueError("Memory bank hash mismatch")
        bank = MemoryBank.load(bank_path)
        if set(bank.episode_ids) != set(cfg["train_ids"]):
            raise ValueError("Bank includes unexpected episode IDs")
        model = VisualProprioEncoder(checkpoint["variant"], pretrained_path=None)
        model.load_state_dict(checkpoint["model_state_dict"])
        normalizer = ProprioNormalizer(checkpoint["proprio_mean"].numpy(), checkpoint["proprio_std"].numpy())
        with h5py.File(str(image), "r") as h5:
            actions = {key: np.asarray(h5["data"][key]["actions"]) for key in cfg["train_ids"]}
        policy = ActionRetrievalPolicy(model, bank, normalizer, actions, device)

        def act(obs):
            return np.stack([policy.act(item, chunk_length=1)[0] for item in obs])
    return act, record


def check_trace(output, protocol):
    with (output / "rollouts.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["seed"]) for row in rows] != protocol["seeds"]:
        raise ValueError("Incomplete or misordered seeds")
    verify_actions(output / "actions.csv", rows, protocol)
    return rows


def run_evaluation(method, run, epoch, cfg, alt, image, seeds, horizon, fixtures, fixture_sha, label, sampling_seed=None):
    output = run / f"{label}_e{epoch:03d}"
    protocol = {"policy": method, "epoch": epoch, "config": cfg, "seeds": seeds,
                "horizon": horizon, "initial_states_sha256": fixture_sha, "split": label}
    if sampling_seed is not None:
        protocol["sampling_seed"] = sampling_seed
    if output.exists():
        saved = read_json(output / "protocol.json")
        if any(saved.get(key) != value for key, value in protocol.items()):
            raise ValueError(f"Existing evaluation protocol differs: {output}")
        metrics = read_json(output / "metrics.json")
        check_trace(output, protocol)
        return metrics
    act, record = load_act(method, run, epoch, cfg, image, torch.device("cuda"))
    protocol["checkpoint"] = record
    output.mkdir()
    write_json(output / "protocol.json", protocol)
    rollout_cfg = cfg if sampling_seed is None else {**cfg, "sampling_seed": sampling_seed}
    metrics = run_rollouts(act, rollout_cfg, alt, seeds, horizon, min(cfg["test_workers"], len(seeds)),
                           output, validation_fixtures=fixtures)
    check_trace(output, protocol)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "validation", "test"])
    parser.add_argument("method", nargs="?", choices=["dp", "alt"])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    cfg, alt, dp, split, image = study(supplement=args.supplement)
    add_references(alt, dp)
    if args.supplement and args.phase != "validation":
        parser.error("Supplement only evaluates the existing validation fixtures")
    if args.phase == "freeze":
        freeze_fixtures(cfg, alt, split)
        return
    if not args.method or not args.run_id:
        parser.error("A policy and training run ID are required")
    run = ROOT / "artifacts" / args.run_id
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    fixtures_path = STUDY_DIR / "validation_initial_states.json"
    validation = read_json(fixtures_path)
    fixtures_sha = sha256(fixtures_path)
    fixtures = {row["seed"]: row for row in validation["initial_conditions"]}
    if validation["study_config_sha256"] != cfg["study_config_sha256"] or list(fixtures) != cfg["validation_seeds"]:
        raise ValueError("Validation manifest differs from the frozen configuration")
    result_path = run / "selection.json"
    if args.phase == "validation":
        if result_path.exists():
            raise FileExistsError(result_path)
        scores = {}
        for epoch in cfg[f"{args.method}_epochs"]:
            scores[str(epoch)] = run_evaluation(
                args.method, run, epoch, cfg, alt, image, cfg["validation_seeds"], cfg["validation_horizon"],
                fixtures, fixtures_sha, "validation",
            )
        priority = cfg["selection_order"][:-1]
        selected = max(scores, key=lambda epoch: tuple(scores[epoch][metric]["count"] for metric in priority)
                       + (-int(epoch),))
        write_json(result_path, {"policy": args.method, "selected_epoch": int(selected), "scores": scores,
                                 "validation_sha256": fixtures_sha, "selection_order": cfg["selection_order"]})
        print(f"{args.method} selected epoch {selected}", flush=True)
    else:
        selected = read_json(result_path)
        if selected["policy"] != args.method or selected["validation_sha256"] != fixtures_sha:
            raise ValueError("Test selection does not match the frozen validation states")
        settings = read_json(alt / split["evaluation_config"]["path"])
        run_evaluation(args.method, run, selected["selected_epoch"], cfg, alt, image, settings["test_seeds"],
                       settings["horizon"], None, split["evaluation_initial_state_contract"]["sha256"], "test")


if __name__ == "__main__":
    main()
