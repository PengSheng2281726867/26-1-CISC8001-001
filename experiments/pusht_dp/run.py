"""Fit the upstream Push-T image DP on the frozen ALT demonstration split."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pusht.repro import configure_runtime, load_config, load_split, now, save_json, sha256

configure_runtime(ROOT)
temp_dir = ROOT / "artifacts" / "pusht_dp" / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temp_dir)
import numpy as np
import torch


class WindowsSync:
    """Delay the upstream import until the configured DP source is on sys.path."""
    def __new__(cls, functions):
        from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
        return SyncVectorEnv(functions)


class PushTWindows:
    """Cache only two visual observations per sampled 16-action window."""
    def __init__(self, root, ends, episodes, cfg, device):
        from diffusion_policy.common.sampler import create_indices

        selected = np.zeros(len(ends), dtype=bool)
        selected[episodes] = True
        horizon = cfg["horizon"]
        obs_steps = cfg["obs_steps"]
        windows = create_indices(ends, horizon, selected, obs_steps - 1, cfg["action_steps"] - 1)
        starts, stops, left, _ = windows.T
        frame = np.clip(starts[:, None] - left[:, None] + np.arange(horizon),
                        starts[:, None], stops[:, None] - 1)
        first = np.r_[0, ends[:-1]]
        for_start = np.searchsorted(ends, frame[:, 0], side="right")
        for_end = np.searchsorted(ends, frame[:, -1], side="right")
        assert np.array_equal(for_start, for_end) and selected[for_start].all()
        ids = np.concatenate([np.arange(first[i], ends[i]) for i in sorted(episodes)])
        image = np.concatenate([root["img"][first[i]:ends[i]] for i in sorted(episodes)])
        positions = root["state"][:, :2].astype(np.float32)
        actions = root["action"][:].astype(np.float32)
        self.frames = len(ids)
        self.obs_index = torch.as_tensor(np.searchsorted(ids, frame[:, :obs_steps]), device=device)
        self.images = torch.from_numpy(image).to(device)
        self.positions = torch.as_tensor(positions[frame[:, :obs_steps]], device=device)
        self.actions = torch.as_tensor(actions[frame], device=device)
        self.stats = {"action": actions[ids], "agent_pos": positions[ids]}
        assert np.array_equal(frame[0, :obs_steps], np.repeat(first[min(episodes)], obs_steps))
        assert frame[0, obs_steps] == first[min(episodes)] + 1
        self.count = len(frame)

    def batch(self, indices):
        rgb = self.images[self.obs_index[indices]].permute(0, 1, 4, 2, 3).float() / 255
        return {"obs": {"image": rgb, "agent_pos": self.positions[indices]},
                "action": self.actions[indices]}

    def normalizer(self):
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.fit(self.stats, last_n_dims=1, mode="limits")
        normalizer["image"] = get_image_range_normalizer()
        return normalizer


def make_policy(cfg, normalizer):
    from diffusers import DDIMScheduler
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
    shape = {"action": {"shape": [2]}, "obs": {
        "agent_pos": {"shape": [2], "type": "low_dim"},
        "image": {"shape": [3, 96, 96], "type": "rgb"}}}
    scheduler = DDIMScheduler(num_train_timesteps=cfg["diffusion_steps"],
                             beta_schedule="squaredcos_cap_v2", clip_sample=True,
                             prediction_type="epsilon")
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape, noise_scheduler=scheduler, horizon=cfg["horizon"],
        n_action_steps=cfg["action_steps"], n_obs_steps=cfg["obs_steps"],
        num_inference_steps=cfg["inference_steps"], crop_shape=(84, 84),
        diffusion_step_embed_dim=128, down_dims=cfg["down_dims"], kernel_size=5,
        obs_encoder_group_norm=True, eval_fixed_crop=True)
    policy.set_normalizer(normalizer)
    return policy.to(cfg["device"])


def rollout(policy, cfg, evaluation, seeds, path, videos):
    from pusht.rollout import ObservedImageEnv, runner_module

    class ObserveLastStep(ObservedImageEnv):
        def get_attr(self, name):
            return getattr(self, name)

    class TraceRewards(WindowsSync):
        def __new__(cls, functions):
            env = super().__new__(cls, functions)
            call = env.call

            def traced_call(name, *args, **kwargs):
                output = call(name, *args, **kwargs)
                if name == "get_attr" and args == ("reward",):
                    for rewards, terminal in zip(output, call("get_attr", "base_terminated")):
                        if len(records) == len(seeds):
                            break
                        records.append({"evaluation_seed": seeds[len(records)],
                                        "official_episode_score": float(max(rewards)),
                                        "final_score": float(rewards[-1]), "steps": len(rewards),
                                        "terminated": bool(terminal),
                                        "truncated": not terminal and len(rewards) == evaluation["max_steps"]})
                return output

            env.call = traced_call
            return env

    assert seeds == list(range(seeds[0], seeds[0] + len(seeds)))
    path.mkdir(parents=True, exist_ok=False)
    records = []
    with patch.object(runner_module, "AsyncVectorEnv", TraceRewards), \
         patch.object(runner_module, "PushTImageEnv", ObserveLastStep):
        runner = runner_module.PushTImageRunner(
            output_dir=str(path), n_train=0, n_train_vis=0, n_test=len(seeds),
            n_test_vis=min(videos, len(seeds)), test_start_seed=seeds[0],
            legacy_test=evaluation["legacy_test"], max_steps=evaluation["max_steps"],
            n_obs_steps=cfg["obs_steps"], n_action_steps=cfg["action_steps"],
            n_envs=min(25, len(seeds)), fps=evaluation["fps"], past_action=False)
    assert runner.env_seeds == seeds
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.manual_seed(cfg["eval_noise_seed"])
            official = runner.run(policy.eval())
    finally:
        runner.env.close()
    assert len(records) == len(seeds)
    for row in records:
        assert row["official_episode_score"] == official[f'test/sim_max_reward_{row["evaluation_seed"]}']
        assert row["terminated"] or row["truncated"]
    scores = [r["official_episode_score"] for r in records]
    assert np.isclose(np.mean(scores), official["test/mean_score"], rtol=0, atol=1e-12)
    with (path / "rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"n": len(records), "mean_score": float(np.mean(scores)),
               "sd": float(np.std(scores, ddof=1)) if len(records) > 1 else 0.0,
               "mean_final_score": float(np.mean([r["final_score"] for r in records])),
               "successes": sum(r["terminated"] for r in records),
               "runtime_seconds": time.perf_counter() - started}
    save_json(path / "metrics.json", summary)
    return summary


def fit(scale, cfg, base, manifest, output, smoke):
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    import zarr

    random.seed(cfg["model_seed"])
    np.random.seed(cfg["model_seed"])
    torch.manual_seed(cfg["model_seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    episodes = manifest["train_subsets"][str(scale)]
    root = zarr.open(base["paths"]["dataset"], mode="r")["data"]
    data = PushTWindows(root, np.array(manifest["episode_ends"]), episodes, cfg, cfg["device"])
    policy = make_policy(cfg, data.normalizer())
    ema = EMAModel(copy.deepcopy(policy), power=cfg["ema_power"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"],
                                  betas=(0.95, 0.999), weight_decay=1e-6, fused=True)
    checkpoints = [2] if smoke else cfg["checkpoints"]
    valid_seeds = list(range(cfg["validation_seed_start"],
                             cfg["validation_seed_start"] + (1 if smoke else cfg["validation_count"])))
    assert not set(valid_seeds) & set(manifest["test_env_seeds"])
    metadata = {"status": "running", "started": now(), "smoke": smoke, "config": cfg,
                "split_sha256": sha256(base["split"]["manifest"]), "scale": scale,
                "dataset_fingerprint": manifest["dataset_fingerprint"], "train_episodes": episodes,
                "heldout_episodes": manifest["heldout_validation_episode_ids"],
                "training_frames": data.frames, "training_windows": data.count,
                "validation_seeds": valid_seeds, "test_seeds": manifest["test_env_seeds"],
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(cfg["device"]),
                "code_sha256": sha256(__file__), "config_sha256": sha256(HERE / "config.json"),
                "policy_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                "upstream_files": {file: sha256(Path(base["paths"]["dp_source"]) / file) for file in (
                    "diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py",
                    "diffusion_policy/model/diffusion/conditional_unet1d.py",
                    "diffusion_policy/env_runner/pusht_image_runner.py")}}
    save_json(output / "run.json", metadata)
    generator = torch.Generator(device=cfg["device"]).manual_seed(cfg["model_seed"] + 1)
    validation = {}
    train_seconds = 0.0
    started = time.perf_counter()
    update = 0
    seen = 0
    losses = []
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["update", "mean_loss", "windows_seen", "training_seconds"])
        while update < checkpoints[-1]:
            indices = torch.randperm(data.count, generator=generator, device=cfg["device"])
            for batch_ids in indices.split(cfg["batch_size"]):
                update += 1
                seen += len(batch_ids)
                warmup = cfg["warmup_updates"]
                fraction = max(0, update - warmup) / max(1, checkpoints[-1] - warmup)
                optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * (
                    min(update / warmup, 1.0) * (0.5 + 0.5 * math.cos(math.pi * fraction)))
                optimizer.zero_grad(set_to_none=True)
                batch = data.batch(batch_ids)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = policy.compute_loss(batch)
                loss.backward()
                if update == 1:
                    assert all(torch.isfinite(p.grad).all().item() for p in policy.parameters()
                               if p.grad is not None)
                optimizer.step()
                ema.step(policy)
                losses.append(loss.detach())
                if update % 50 == 0 or update in checkpoints:
                    average = float(torch.stack(losses).mean())
                    if not math.isfinite(average):
                        raise ValueError(f"Non-finite training loss at {update}")
                    seconds = train_seconds + time.perf_counter() - started
                    writer.writerow([update, average, seen, seconds])
                    stream.flush()
                    print(f"DP-{scale} update {update}: loss={average:.5f}, train={seconds:.1f}s", flush=True)
                    losses.clear()
                if update in checkpoints:
                    train_seconds = seconds
                    checkpoint = output / f"ema_{update}.pt"
                    torch.save(ema.averaged_model.state_dict(), checkpoint)
                    result = rollout(ema.averaged_model, cfg, base["evaluation"], valid_seeds,
                                     output / f"validation_{update}", 0)
                    validation[update] = {**result, "checkpoint_sha256": sha256(checkpoint),
                                          "training_seconds": seconds, "windows_seen": seen,
                                          "equivalent_epochs": seen / data.count}
                    save_json(output / "validation.json", validation)
                    print(f"DP-{scale} validation {update}: {result}", flush=True)
                    started = time.perf_counter()
                if update == checkpoints[-1]:
                    break
    selected = max(validation, key=lambda u: (validation[u]["mean_score"], -u))
    metadata.update(status="validated", selected_update=selected, total_training_seconds=train_seconds)
    save_json(output / "run.json", metadata)


def test(scale, cfg, base, manifest, output):
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "validated" and not run["smoke"]
    assert run["config"] == cfg and run["code_sha256"] == sha256(__file__)
    assert run["split_sha256"] == sha256(base["split"]["manifest"])
    step = run["selected_update"]
    checks = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    checkpoint = output / f"ema_{step}.pt"
    assert sha256(checkpoint) == checks[str(step)]["checkpoint_sha256"]
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    torch.set_num_threads(4)
    policy = make_policy(cfg, LinearNormalizer())
    policy.load_state_dict(torch.load(checkpoint, map_location=cfg["device"], weights_only=True))
    metrics = rollout(policy, cfg, base["evaluation"], manifest["test_env_seeds"],
                      output / "test", base["evaluation"]["video_count"])
    run.update(status="completed", finished=now(), test=metrics)
    save_json(output / "run.json", run)
    print(f"DP-{scale} test: {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "train", "test", "study"))
    parser.add_argument("--scale", type=int, choices=(30, 100))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/alt_pusht_base.yaml")
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("Run ID must contain ASCII letters, digits, hyphens or underscores")
    if (args.phase == "study") == (args.scale is not None):
        parser.error("Use --scale for smoke, train and test; study runs both scales")
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    base = load_config(args.base_config, ROOT)
    sys.path.insert(0, base["paths"]["dp_source"])
    manifest = load_split(base["split"]["manifest"], base["paths"]["dataset"], args.scale)
    run_root = ROOT / "artifacts/pusht_dp" / args.run_id
    scales = cfg["scales"] if args.phase == "study" else [args.scale]
    if args.phase != "test":
        for scale in scales:
            output = run_root / str(scale)
            output.mkdir(parents=True, exist_ok=False)
            fit(scale, cfg, base, manifest, output, args.phase == "smoke")
            torch.cuda.empty_cache()
    if args.phase in ("test", "study"):
        for scale in scales:
            test(scale, cfg, base, manifest, run_root / str(scale))


if __name__ == "__main__":
    main()
