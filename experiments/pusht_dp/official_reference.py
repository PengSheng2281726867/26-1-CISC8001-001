"""Reevaluate one identified upstream visual CNN checkpoint on this project's seeds."""
import argparse
from pathlib import Path
import sys

from run import ROOT, load_config, load_split, now, rollout, save_json, sha256, torch

ASSET = Path("experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/"
             "epoch=0500-test_mean_score=0.884.ckpt")
DIGEST = "bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/alt_pusht_base.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/pusht_dp") or output.exists():
        parser.error("Use a new output directory inside artifacts/pusht_dp")
    base = load_config(args.base_config, ROOT)
    dp_source = Path(base["paths"]["dp_source"])
    sys.path.insert(0, str(dp_source))
    manifest = load_split(base["split"]["manifest"])
    checkpoint = dp_source.parent / ASSET
    if sha256(checkpoint) != DIGEST:
        raise ValueError("Expected the verified official train_0 checkpoint")
    import dill
    import hydra
    # This one publisher asset is pinned above before pickle loading.
    torch.set_num_threads(4)
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, mmap=True,
                         weights_only=False)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"], assign=True)
    del payload
    assert cfg.training.use_ema and policy.num_inference_steps == 100
    rollout_config = {"obs_steps": policy.n_obs_steps, "action_steps": policy.n_action_steps,
                      "eval_noise_seed": 1242}
    assert rollout_config["obs_steps"] == 2 and rollout_config["action_steps"] == 8
    metrics = rollout(policy.to("cuda:0"), rollout_config, base["evaluation"],
                      manifest["test_env_seeds"], output, 0)
    save_json(output / "reference.json", {
        "checkpoint": ASSET.as_posix(), "checkpoint_sha256": DIGEST, "source_epoch": 500,
        "checkpoint_training_seed": int(cfg.training.seed),
        "source_train_cap": int(cfg.task.dataset.max_train_episodes),
        "source_test_start_seed": int(cfg.task.env_runner.test_start_seed),
        "test_seeds": manifest["test_env_seeds"], "evaluation_noise_seed": 1242,
        "inference_steps": int(policy.num_inference_steps),
        "scheduler": type(policy.noise_scheduler).__name__,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "metrics": metrics, "finished": now(),
    })
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
