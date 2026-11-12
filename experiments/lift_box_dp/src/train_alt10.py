"""Train one ALT trajectory and rebuild a bank at each frozen epoch."""
import argparse
import csv
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from common import ROOT, add_references, read_json, sha256, write_json
from epoch_study import study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--scale", type=int, choices=(10, 50, 100), default=10)
    args = parser.parse_args()
    cfg, alt, dp, split, image = study(supplement=args.supplement, scale=args.scale)
    add_references(alt, dp)
    from alt_lift.data import ContrastiveViews, LiftFrameStore, ProprioNormalizer
    from alt_lift.encoder import VisualProprioEncoder
    from alt_lift.loss import nt_xent_loss
    from alt_lift.memory import build_memory_bank
    from alt_lift.train import _seed_everything

    original_cfg = alt / "configs/alt_lift30.yaml"
    if sha256(original_cfg) != split["run_config"]["sha256"]:
        raise ValueError("Original ALT configuration hash mismatch")
    base = yaml.safe_load(original_cfg.read_text(encoding="utf-8"))
    training = base["training"]
    paths = read_json(alt / "configs/runtime_paths.json")
    pretrained = (alt / paths["pretrained_resnet18"]).resolve()
    if sha256(pretrained) != paths["pretrained_resnet18_sha256"]:
        raise ValueError("Pretrained ResNet-18 checksum mismatch")
    if sha256(image) != split["source_files"]["image_hdf5"]["sha256"]:
        raise ValueError("Source image HDF5 checksum mismatch")
    output = ROOT / "artifacts" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    _seed_everything(training["model_seed"])
    device = torch.device("cuda")
    store = LiftFrameStore(image, cfg["train_ids"])
    normalizer = ProprioNormalizer.fit(store.proprio)
    dataset = ContrastiveViews(store, normalizer, base["augmentation"])
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                        drop_last=training["drop_last"], pin_memory=True,
                        generator=torch.Generator().manual_seed(training["dataloader_shuffle_seed"]))
    model = VisualProprioEncoder(base["model"]["variant"], pretrained_path=pretrained).to(device)
    torch.manual_seed(training["augmentation_seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    start = time.perf_counter()
    steps = 0
    saved = {}
    epochs = cfg["alt_epochs"]
    with (output / "training.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "steps", "train_loss", "wall_seconds"])
        for epoch in range(1, epochs[-1] + 1):
            losses = []
            model.train()
            for a, b, prop in loader:
                optimizer.zero_grad(set_to_none=True)
                z_a, z_b = model.encode_pair(a.to(device), b.to(device), prop.to(device))
                loss = nt_xent_loss(z_a, z_b, base["objective"]["temperature"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite contrastive loss")
                loss.backward()
                if steps == 0 and not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Non-finite first-step gradients")
                optimizer.step()
                steps += 1
                losses.append(loss.item())
            writer.writerow([epoch, steps, float(np.mean(losses)), time.perf_counter() - start])
            stream.flush()
            if epoch in epochs:
                checkpoint = output / f"epoch{epoch:03d}.pt"
                bank_path = output / f"epoch{epoch:03d}_bank.npz"
                torch.save({"model_state_dict": model.state_dict(), "variant": base["model"]["variant"],
                            "proprio_mean": torch.from_numpy(normalizer.mean),
                            "proprio_std": torch.from_numpy(normalizer.std), "epoch": epoch,
                            "config": cfg}, checkpoint)
                bank = build_memory_bank(model, store, normalizer, bank_path, device)
                if len(bank.embeddings) != len(store) or set(bank.episode_ids) != set(cfg["train_ids"]):
                    raise ValueError("ALT bank coverage differs from frozen train episodes")
                saved[str(epoch)] = {"steps": steps, "checkpoint_sha256": sha256(checkpoint),
                                     "bank_sha256": sha256(bank_path)}
                write_json(output / "checkpoints.json", saved)
            if epoch % 10 == 0:
                print(f"ALT epoch={epoch} steps={steps} loss={np.mean(losses):.5f}", flush=True)
    write_json(output / "training.json", {
        "train_ids": cfg["train_ids"], "train_frames": len(store), "steps": steps,
        "wall_seconds": time.perf_counter() - start, "checkpoint_epochs": epochs,
        "pretrained_sha256": sha256(pretrained), "original_config_sha256": sha256(original_cfg),
        "source_sha256": sha256(__file__), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    })


if __name__ == "__main__":
    main()
