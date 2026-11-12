"""Train only the frozen 30-episode DP-CNN and select final EMA weights."""
import argparse
import csv
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import CONFIG_PATH, ROOT, add_references, contract, sha256, write_json
from model import make_model, set_normalizer


class ActionWindows(Dataset):
    def __init__(self, frames, horizon):
        self.frames = frames
        self.offsets = np.arange(horizon)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, row):
        frames = self.frames
        from common import IMAGE_KEYS, PROPRIO_KEYS

        episode = frames.episodes[frames.episode_ids_per_frame[row]]
        local = frames.local_indices[row]
        indices = np.minimum(local + self.offsets, len(episode.actions) - 1)
        proprio = frames.proprio[row:row + 1]
        images = (frames.agentview[row], frames.wrist[row])
        obs = {key: torch.from_numpy(np.moveaxis(image, -1, 0).copy())[None].float() / 255
               for key, image in zip(IMAGE_KEYS, images)}
        obs.update({key: torch.from_numpy(proprio[:, columns])
                    for key, columns in zip(PROPRIO_KEYS, (slice(0, 3), slice(3, 7), slice(7, 9)))})
        return {"obs": obs, "action": episode.actions[indices].astype(np.float32)}


def check_windows(dataset):
    frames = dataset.frames
    for row in range(len(frames)):
        episode = frames.episodes[frames.episode_ids_per_frame[row]]
        index = frames.local_indices[row]
        item = dataset[row]
        expected_end = min(index + len(dataset.offsets) - 1, len(episode.actions) - 1)
        np.testing.assert_array_equal(item["action"][0], episode.actions[index].astype(np.float32))
        np.testing.assert_array_equal(item["action"][-1], episode.actions[expected_end].astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--scale", type=int, choices=(10, 50, 100), default=10)
    args = parser.parse_args()
    if args.scale != 10 and not args.study:
        parser.error("Nondefault --scale requires --study")
    if args.supplement and not args.study:
        parser.error("--supplement requires --study")
    if args.study:
        from epoch_study import study
        cfg, alt, dp, split, image = study(supplement=args.supplement, scale=args.scale, method="dp")
    else:
        cfg, alt, dp, split, image = contract()
    add_references(alt, dp)
    from alt_lift.data import LiftFrameStore
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusers.optimization import get_cosine_schedule_with_warmup

    output = ROOT / "artifacts" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    start = time.perf_counter()
    assert sha256(image) == split["source_files"]["image_hdf5"]["sha256"]
    random.seed(cfg["model_seed"])
    np.random.seed(cfg["model_seed"])
    torch.manual_seed(cfg["model_seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("This training configuration requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    frames = LiftFrameStore(image, cfg.get("train_ids", split["train_subsets"]["30"]))
    data = ActionWindows(frames, cfg["horizon"])
    check_windows(data)
    batch_size = 4 if args.smoke else cfg["batch_size"]
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(cfg["shuffle_seed"]))
    validation = ActionWindows(LiftFrameStore(image, split["heldout_validation_episode_ids"]), cfg["horizon"])
    val_loader = DataLoader(validation, batch_size=batch_size, pin_memory=True)
    model = make_model(cfg)
    set_normalizer(model, frames.proprio)
    model = model.to(device)
    ema_model = make_model(cfg).to(device)
    ema_model.normalizer.load_state_dict(model.normalizer.state_dict())
    ema_model.load_state_dict(model.state_dict())
    ema = EMAModel(ema_model, power=cfg["ema_power"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], betas=(0.95, 0.999),
                                  weight_decay=1e-6, fused=True)
    epochs = 1 if args.smoke else cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0 if args.smoke else cfg.get("dp_warmup_steps", 500),
                                                1 if args.smoke else epochs * len(loader))

    def move(batch):
        return {"obs": {key: value.to(device, non_blocking=True) for key, value in batch["obs"].items()},
                "action": batch["action"].to(device, non_blocking=True)}

    saved = {}
    steps = 0
    with (output / "training.csv").open("x", newline="", encoding="utf-8") as stream:
        metrics = csv.writer(stream)
        metrics.writerow(["epoch", "steps", "train_loss", "validation_loss", "wall_seconds"])
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model.compute_loss(move(batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite diffusion loss")
                loss.backward()
                if steps == 0 and not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Non-finite first-step gradients")
                optimizer.step()
                scheduler.step()
                ema.step(model)
                steps += 1
                losses.append(loss.item())
                if args.smoke:
                    break
            val_loss = None
            if args.smoke or epoch % 10 == 0 or epoch == epochs:
                ema_model.eval()
                with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(45)
                    val_losses = []
                    for batch in val_loader:
                        val_losses.extend([ema_model.compute_loss(move(batch)).item()] * len(batch["action"]))
                        if args.smoke:
                            break
                    val_loss = float(np.mean(val_losses))
            elapsed = time.perf_counter() - start
            metrics.writerow([epoch, steps, float(np.mean(losses)), val_loss, elapsed])
            stream.flush()
            print(f"epoch {epoch}/{epochs} loss={np.mean(losses):.5f} val={val_loss} elapsed={elapsed:.0f}s", flush=True)
            if args.study and epoch in cfg["dp_epochs"]:
                snapshot = output / f"epoch{epoch:03d}.pt"
                torch.save({"weights": ema_model.state_dict(), "config": cfg, "smoke": args.smoke,
                            "epoch": epoch}, snapshot)
                saved[str(epoch)] = {"steps": steps, "checkpoint_sha256": sha256(snapshot)}
                write_json(output / "checkpoints.json", saved)
    checkpoint = output / ("epoch%03d.pt" % epochs if args.study and not args.smoke else "final_ema.pt")
    if not args.study or args.smoke:
        torch.save({"weights": ema_model.state_dict(), "config": cfg, "smoke": args.smoke}, checkpoint)
    write_json(output / "training.json", {
        "run_id": args.run_id, "smoke": args.smoke, "epochs": epochs, "steps": steps,
        "train_ids": frames.episode_ids, "train_frames": len(data), "validation_frames": len(validation),
        "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "wall_seconds": time.perf_counter() - start, "torch": torch.__version__,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_sha256": {p.name: sha256(p) for p in (ROOT / "src").glob("*.py")},
    })


if __name__ == "__main__":
    main()
