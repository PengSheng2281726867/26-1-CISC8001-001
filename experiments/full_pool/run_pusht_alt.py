"""Train full-pool visual ALT and select an epoch on held-out Push-T rollouts."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "pusht_dp")]
from pusht.repro import configure_runtime, git, load_config, load_split, save_json, sha256

configure_runtime(ROOT)
import torch
from alt.encoder import FusionEncoder
from alt.memory import MemoryBank, build_bank
from pusht.data import PushTData
from pusht.experiment import train_model, validate_method
from pusht.policy import ALTPolicy
from run import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    design_path = HERE / "pusht_alt.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    dp = json.loads((HERE / "pusht_dp.json").read_text(encoding="utf-8"))
    if any(design[key] != dp[key] for key in ("validation_seed_start", "validation_count")):
        raise ValueError("ALT and DP validation starts must match")
    config_path = ROOT / "configs/alt_pusht_base.yaml"
    cfg = load_config(config_path, ROOT)
    sys.path.insert(0, cfg["paths"]["dp_source"])
    manifest = load_split(cfg["split"]["manifest"], cfg["paths"]["dataset"])
    validate_method(cfg, manifest)
    scale = design["train_count"]
    episodes = manifest["ordered_train_pool_episode_ids"]
    if len(episodes) != scale or dp["scales"] != [scale]:
        raise ValueError("Unexpected Push-T full training pool size")
    manifest["train_subsets"][str(scale)] = episodes
    seeds = list(range(design["validation_seed_start"], design["validation_seed_start"] + design["validation_count"]))
    if set(seeds) & set(manifest["test_env_seeds"]):
        raise ValueError("Push-T validation and test starts overlap")
    epochs = [1] if args.smoke else design["checkpoint_epochs"]
    cfg["training"].update(scale=scale, epochs=epochs[-1], num_workers=design["loader_workers"],
                           checkpoint_epochs=epochs, checkpoint_selection="validation_peak_then_earliest")
    if args.smoke:
        cfg["training"]["max_updates"] = 2
        seeds = seeds[:2]
    output = ROOT / "artifacts/full_pool" / ("pusht_alt_smoke_v1" if args.smoke else "pusht_alt_v1")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", cfg, immutable=True)
    metadata = {"scale": scale, "smoke": args.smoke, "train_episodes": episodes, "validation_seeds": seeds,
                "test_seeds": manifest["test_env_seeds"], "dataset_fingerprint": manifest["dataset_fingerprint"],
                "split_manifest_sha256": sha256(cfg["split"]["manifest"]),
                "pretrained_sha256": sha256(cfg["paths"]["pretrained"]), "protocol_sha256": sha256(design_path),
                "base_commit": git(ROOT, "rev-parse", "HEAD"), "source_sha256": sha256(__file__),
                "train_sha256": sha256(ROOT / "src/pusht/experiment.py"), "base_config_sha256": sha256(config_path),
                "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name()}
    if metadata["pretrained_sha256"] != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
        raise ValueError("Unexpected ImageNet weights")
    runner_cfg = {**dp, "obs_steps": cfg["evaluation"]["n_obs_steps"]}
    evaluations = {}

    def checkpoint(model, data, epoch, steps):
        stem = f"epoch{epoch:03d}"
        weights = output / f"{stem}.pt"
        bank_path = output / f"{stem}_bank.pt"
        torch.save(model.state_dict(), weights)
        digest = sha256(weights)
        bank = build_bank(model, data, episodes, {"checkpoint_sha256": digest, "train_episode_ids": episodes,
                                                   "split_manifest_sha256": metadata["split_manifest_sha256"]})
        if set(bank.episode_ids.tolist()) != set(episodes):
            raise ValueError("Push-T checkpoint bank membership differs")
        bank.save(bank_path)
        policy = ALTPolicy(model, bank, data)
        scores = rollout(policy, runner_cfg, cfg["evaluation"], seeds, output / f"validation_{stem}", 0)
        evaluations[epoch] = {**scores, "optimizer_steps": steps, "checkpoint_sha256": digest,
                              "bank_sha256": sha256(bank_path), "bank_count": len(bank.embeddings)}
        save_json(output / "validation.json", evaluations)
        print(f"ALT-{scale} epoch {epoch}: {evaluations[epoch]}", flush=True)

    train_model(cfg, manifest, output, output, metadata, on_checkpoint=checkpoint)
    selected = max(evaluations, key=lambda epoch: (evaluations[epoch]["mean_score"], -epoch))
    metadata.update(selected_epoch=selected, selection_order=["mean_score", "earlier_epoch"])
    if not args.smoke:
        stem = f"epoch{selected:03d}"
        weights, bank_path = output / f"{stem}.pt", output / f"{stem}_bank.pt"
        expected = evaluations[selected]
        if sha256(weights) != expected["checkpoint_sha256"] or sha256(bank_path) != expected["bank_sha256"]:
            raise ValueError("Selected ALT checkpoint or bank has changed")
        device = cfg["training"]["device"]
        model = FusionEncoder().to(device)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        policy = ALTPolicy(model, MemoryBank.load(bank_path), PushTData(cfg["paths"]["dataset"]))
        metadata["test"] = rollout(policy, runner_cfg, cfg["evaluation"], manifest["test_env_seeds"], output / "test", 0)
        save_json(output / "test/retrieval_sample.json", policy.query_records[:32])
        print(f"ALT-{scale} epoch {selected} test: {metadata['test']}", flush=True)
    metadata.update(status="smoke_completed" if args.smoke else "completed")
    save_json(output / "run.json", metadata, immutable=True)


if __name__ == "__main__":
    main()
