"""Train the single scheduled ALT-Lift-30 encoder from verified ImageNet weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from .data import ContrastiveViews, LiftFrameStore, ProprioNormalizer
from .encoder import VisualProprioEncoder
from .loss import nt_xent_loss
from .repro import HERE, git_value, json_write, resolve_config_path, sha256_file


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_contract(split: dict, config_path: Path, data_path: Path, low_dim_path: Path, config: dict):
    expected_cfg = split.get("run_config", {}).get("sha256")
    actual_cfg = sha256_file(config_path)
    if expected_cfg and expected_cfg != actual_cfg:
        raise ValueError("Resolved ALT config changed after the split contract was frozen")
    for key, path in (("image_hdf5", data_path), ("low_dim_hdf5", low_dim_path)):
        record = split["source_files"][key]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError("Dataset file missing or hash differs from frozen split: %s" % path)
    ids = list(split["train_subsets"][str(config["training"]["train_scale"])])
    validation = set(split["heldout_validation_episode_ids"])
    if len(ids) != len(set(ids)) or validation.intersection(ids):
        raise ValueError("Invalid train/validation split overlap or duplicate IDs")
    return ids


def _load_store(data_path: Path, episode_ids):
    return LiftFrameStore(data_path, episode_ids)


def _seed_everything(model_seed: int):
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def run(config_path: Path, paths_path: Path, smoke: bool = False, run_id_override=None):
    config_path = Path(config_path).resolve()
    paths_path = Path(paths_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = _load_json(paths_path)
    split_path = resolve_config_path(config["dataset"]["split_manifest"])
    data_path = resolve_config_path(paths["image_hdf5"])
    low_dim_path = resolve_config_path(paths["low_dim_hdf5"])
    pretrained_path = resolve_config_path(paths["pretrained_resnet18"])
    split = _load_json(split_path)
    train_ids = _validate_contract(split, config_path, data_path, low_dim_path, config)
    pretrained_sha = sha256_file(pretrained_path)
    if pretrained_sha.lower() != paths["pretrained_resnet18_sha256"].lower():
        raise ValueError("ImageNet checkpoint hash differs from the verified torchvision file")

    _seed_everything(int(config["training"]["model_seed"]))
    store = _load_store(data_path, train_ids)
    normalizer = ProprioNormalizer.fit(store.proprio)
    dataset = ContrastiveViews(store, normalizer, config["augmentation"])
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(config["training"]["dataloader_shuffle_seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=bool(config["training"]["drop_last"]),
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=shuffle_generator,
    )
    if len(loader) == 0:
        raise ValueError("No full training batch; frozen batch size cannot be satisfied")

    model = VisualProprioEncoder(config["model"]["variant"], pretrained_path=pretrained_path).to(device)
    # Augmentation randomness is independent of weight initialization and loader order.
    torch.manual_seed(int(config["training"]["augmentation_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["training"]["augmentation_seed"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if smoke:
        model.train()
        a, b, prop = next(iter(loader))
        a, b, prop = a.to(device), b.to(device), prop.to(device)
        optimizer.zero_grad(set_to_none=True)
        z_a, z_b = model.encode_pair(a, b, prop)
        loss = nt_xent_loss(z_a, z_b, float(config["objective"]["temperature"]))
        loss.backward()
        gradient_checks = {
            "image_encoder": model.image_encoder.layer1[0].conv1.weight.grad is not None,
            "proprio_encoder": model.proprio_encoder[0].weight.grad is not None,
            "fusion": model.fusion[0].weight.grad is not None,
            "finite_gradients": all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
            ),
        }
        result = {
            "status": "passed" if all(gradient_checks.values()) and torch.isfinite(loss) else "failed",
            "batch_size": int(len(prop)),
            "z_a_shape": list(z_a.shape),
            "z_b_shape": list(z_b.shape),
            "embedding_norm_min": float(torch.linalg.vector_norm(z_a, dim=1).min().item()),
            "embedding_norm_max": float(torch.linalg.vector_norm(z_a, dim=1).max().item()),
            "loss": float(loss.item()),
            "gradient_checks": gradient_checks,
            "pretrained_sha256": pretrained_sha,
            "device": str(device),
        }
        return result

    run_id = run_id_override or config["experiment_id"]
    run_root = resolve_config_path(paths["run_root"])
    run_dir = run_root / run_id
    artifact_dir = resolve_config_path(paths["artifact_root"]) / run_id
    if run_dir.exists() or artifact_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing run or artifact directory")
    run_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    resolved_config = dict(config)
    resolved_config["resolved_paths"] = {
        "image_hdf5": str(data_path),
        "low_dim_hdf5": str(low_dim_path),
        "pretrained_resnet18": str(pretrained_path),
        "split_manifest": str(split_path),
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    json_write(run_dir / "dataset_manifest_ref.json", {
        "contract_id": split["contract_id"],
        "dataset_id": split["dataset_id"],
        "dataset_fingerprint": split["dataset_fingerprint"],
        "split_manifest_sha256": sha256_file(split_path),
        "train_episode_ids": train_ids,
        "heldout_validation_episode_ids": split["heldout_validation_episode_ids"],
        "evaluation_seed_manifest_sha256": split["evaluation_initial_state_contract"]["sha256"],
        "train_episode_count": len(train_ids),
    })
    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        "run_id": run_id,
        "artifact_root": str(artifact_dir.relative_to(HERE)),
        "status": "running",
        "started_unix_time": time.time(),
        "code_commit": git_value("rev-parse", "HEAD"),
        "git_worktree_status_at_launch": git_value("status", "--short"),
        "alt_config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(HERE / "results" / "source_manifest.json"),
        "dataset_id": split["dataset_id"],
        "dataset_fingerprint": split["dataset_fingerprint"],
        "split_manifest_sha256": sha256_file(split_path),
        "train_episode_ids": train_ids,
        "heldout_validation_episode_ids": split["heldout_validation_episode_ids"],
        "evaluation_seeds": split["test_env_seeds"],
        "evaluation_seed_manifest_sha256": split["evaluation_initial_state_contract"]["sha256"],
        "architecture": {
            "inputs": ["current agentview RGB", "current wrist RGB", "current XYZW EEF pose", "current gripper qpos"],
            "shared_resnet18_two_cameras": True,
            "pretrained": "ImageNet torchvision ResNet18",
            "pretrained_sha256": pretrained_sha,
            "fine_tuned": True,
            "proprio_mlp": [9, 64, 32],
            "fusion_mlp": [288, 256, 128],
            "embedding_dim": 128,
        },
        "normalization": {"images": "RGB uint8 /255 then ImageNet mean/std", "proprio": normalizer.to_dict()},
        "augmentation": config["augmentation"],
        "hyperparameters": config["training"],
        "objective": config["objective"],
        "optimizer": {"name": "Adam", "learning_rate": config["training"]["learning_rate"], "weight_decay": config["training"]["weight_decay"]},
        "matching_execution": {"retrieval": "cosine top-1", "ood_threshold": None, "fallback": None, "action_source": "original actions[t]", "action_semantics": "incremental 7D OSC_POSE + Panda gripper command", "chunk_length": 1},
        "software": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "numpy": np.__version__, "h5py": h5py.__version__},
        "status_note": "Training metrics use NT-Xent only; final epoch checkpoint, no test-seed selection.",
    }
    json_write(metadata_path, metadata)
    metrics_path = run_dir / "training_metrics.csv"
    wall_start = time.perf_counter()
    epoch_losses = []
    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean_nt_xent", "batches", "seconds"])
            writer.writeheader()
            for epoch in range(1, int(config["training"]["epochs"]) + 1):
                model.train()
                epoch_start = time.perf_counter()
                losses = []
                for a, b, prop in loader:
                    a, b, prop = a.to(device, non_blocking=True), b.to(device, non_blocking=True), prop.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    z_a, z_b = model.encode_pair(a, b, prop)
                    loss = nt_xent_loss(z_a, z_b, float(config["objective"]["temperature"]))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite NT-Xent at epoch %d" % epoch)
                    loss.backward()
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError("Non-finite gradient at epoch %d" % epoch)
                    optimizer.step()
                    losses.append(float(loss.detach().item()))
                mean_loss = float(np.mean(losses))
                epoch_losses.append(mean_loss)
                row = {"epoch": epoch, "mean_nt_xent": mean_loss, "batches": len(losses), "seconds": time.perf_counter() - epoch_start}
                writer.writerow(row)
                f.flush()
                print("epoch %02d/%02d nt_xent=%.6f seconds=%.1f" % (epoch, config["training"]["epochs"], mean_loss, row["seconds"]), flush=True)
        checkpoint_path = artifact_dir / "checkpoint_final.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "proprio_mean": torch.from_numpy(normalizer.mean),
            "proprio_std": torch.from_numpy(normalizer.std),
            "variant": config["model"]["variant"],
            "embedding_dim": int(config["model"]["embedding_dim"]),
            "pretrained_sha256": pretrained_sha,
            "split_manifest_sha256": sha256_file(split_path),
            "run_id": run_id,
            "epoch": int(config["training"]["epochs"]),
        }, checkpoint_path)
        metadata.update({
            "status": "completed",
            "completed_unix_time": time.time(),
            "training_wall_seconds": time.perf_counter() - wall_start,
            "optimizer_steps": len(loader) * int(config["training"]["epochs"]),
            "final_epoch_nt_xent": epoch_losses[-1],
        "checkpoint_path": str(checkpoint_path.relative_to(HERE)),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        json_write(metadata_path, metadata)
        return {"run_dir": str(run_dir.relative_to(HERE)), "status": "completed", "final_epoch_nt_xent": epoch_losses[-1], "checkpoint_sha256": metadata["checkpoint_sha256"]}
    except Exception as exc:
        metadata.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "failed_unix_time": time.time()})
        json_write(metadata_path, metadata)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "alt_lift30.yaml")
    parser.add_argument("--paths-config", type=Path, default=HERE / "configs" / "runtime_paths.json")
    parser.add_argument("--smoke", action="store_true", help="Run one finite forward/backward batch without creating a formal run")
    parser.add_argument("--run-id", type=str, help="Optional new immutable run directory name for a documented retry")
    args = parser.parse_args()
    result = run(args.config, args.paths_config, smoke=args.smoke, run_id_override=args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] not in ("completed", "passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
