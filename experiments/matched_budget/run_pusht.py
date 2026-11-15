"""Evaluate ALT and DP on the same Push-T episodes at a fixed update budget."""
import argparse
import json
from pathlib import Path
import sys

from study_protocol import PROTOCOL_PATH, compatible_hashes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "pusht_dp")]
from pusht.repro import configure_runtime, git, load_config, load_split, save_json, sha256

configure_runtime(ROOT)
import torch
from pusht.experiment import create_bank, train_model, validate_method
from pusht.policy import ALTPolicy
from run import fit, rollout, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("alt", "dp"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hashes = compatible_hashes("pusht", protocol)
    design = protocol["pusht"]
    base = load_config(ROOT / "configs/alt_pusht_base.yaml", ROOT)
    sys.path.insert(0, base["paths"]["dp_source"])
    manifest = load_split(base["split"]["manifest"], base["paths"]["dataset"])
    validate_method(base, manifest)
    split_hash = sha256(base["split"]["manifest"])
    pretrained_hash = sha256(base["paths"]["pretrained"])
    if pretrained_hash != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
        raise ValueError("Unexpected ImageNet ResNet-18 weights")
    dp_config = json.loads((HERE.parent / "pusht_dp/config.json").read_text(encoding="utf-8"))
    updates = 2 if args.smoke else design["updates"]
    seeds = protocol["model_seeds"][:1] if args.smoke else protocol["model_seeds"]
    scales = design["scales"][:1] if args.smoke else design["scales"]
    revision = git(ROOT, "rev-parse", "HEAD")
    for scale in scales:
        for seed in seeds:
            tag = "smoke" if args.smoke else "runs"
            output = ROOT / "artifacts/matched_budget" / tag / f"pusht_{args.method}_{scale}_{seed}"
            if (output / "completed.json").is_file():
                previous = json.loads((output / "completed.json").read_text(encoding="utf-8"))
                if previous["protocol_sha256"] not in protocol_hashes:
                    raise ValueError(f"Existing run uses a different protocol: {output}")
                continue
            output.mkdir(parents=True, exist_ok=False)
            metadata = {"task": "pusht", "method": args.method, "scale": scale, "model_seed": seed,
                        "smoke": args.smoke, "optimizer_steps": updates, "batch_size": protocol["batch_size"],
                        "train_episode_ids": manifest["train_subsets"][str(scale)],
                        "split_manifest_sha256": split_hash, "protocol_sha256": sha256(protocol_path),
                        "dataset_fingerprint": manifest["dataset_fingerprint"], "base_commit": revision,
                        "entrypoint_sha256": sha256(__file__), "torch": str(torch.__version__),
                        "gpu": torch.cuda.get_device_name(), "checkpoint_selection": protocol["checkpoint_selection"]}
            print(output.name, flush=True)
            if args.method == "alt":
                settings = {**base["training"], "scale": scale, "model_seed": seed, "shuffle_seed": seed + 1,
                            "augmentation_seed": seed + 2, "batch_size": protocol["batch_size"],
                            "max_updates": updates, "checkpoint_selection": protocol["checkpoint_selection"]}
                cfg = {**base, "training": settings}
                save_json(output / "config.json", cfg, immutable=True)
                metadata.update(pretrained_sha256=pretrained_hash,
                                train_source_sha256=sha256(ROOT / "src/pusht/experiment.py"))
                model, data = train_model(cfg, manifest, output, output, metadata)
                model, data, bank = create_bank(cfg, manifest, output, output, metadata, model, data)
                runner_cfg = {**dp_config, "obs_steps": 1}
                policy = ALTPolicy(model, bank, data)
                eval_seeds = manifest["test_env_seeds"][:2] if args.smoke else manifest["test_env_seeds"]
                rollout(policy, runner_cfg, base["evaluation"], eval_seeds, output / "test", 0)
                save_json(output / "retrieval_sample.json", policy.query_records[:32])
                del policy, model, data, bank
            else:
                cfg = {**dp_config, "model_seed": seed, "batch_size": protocol["batch_size"], "checkpoints": [updates]}
                config_path = output / "config.json"
                save_json(config_path, cfg, immutable=True)
                fit(scale, cfg, config_path, base, manifest, output, args.smoke)
                if not args.smoke:
                    test(scale, cfg, {**base, "evaluation": {**base["evaluation"], "video_count": 0}}, manifest, output)
                validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))[str(updates)]
                metadata.update({key: validation[key] for key in ("checkpoint_sha256", "training_seconds", "windows_seen")})
                metadata["run_record_sha256"] = sha256(output / "run.json")
            if metadata["optimizer_steps"] != updates:
                raise ValueError("Optimizer count differs from protocol")
            save_json(output / "completed.json", metadata, immutable=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
