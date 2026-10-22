#!/usr/bin/env python
"""Build and validate the deterministic all-frame bank for a completed run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.data import LiftFrameStore, ProprioNormalizer
from alt_lift.encoder import VisualProprioEncoder
from alt_lift.memory import build_memory_bank
from alt_lift.repro import json_write, resolve_config_path, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "alt_lift30.yaml")
    parser.add_argument("--paths-config", type=Path, default=HERE / "configs" / "runtime_paths.json")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed":
        raise ValueError("Refusing to build a bank from an incomplete training run")
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    paths = json.loads(args.paths_config.read_text(encoding="utf-8"))
    image_path = resolve_config_path(paths["image_hdf5"])
    low_dim_path = resolve_config_path(paths["low_dim_hdf5"])
    split_path = resolve_config_path(config["dataset"]["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha256_file(split_path) != metadata["split_manifest_sha256"]:
        raise ValueError("Split manifest changed since training")
    if sha256_file(image_path) != split["source_files"]["image_hdf5"]["sha256"] or sha256_file(low_dim_path) != split["source_files"]["low_dim_hdf5"]["sha256"]:
        raise ValueError("Dataset source files changed since training")
    train_ids = list(split["train_subsets"]["30"])
    if train_ids != metadata["train_episode_ids"]:
        raise ValueError("Training run episode IDs do not exactly match the frozen manifest")
    split_sha = sha256_file(split_path)
    artifact_dir = resolve_config_path(metadata["artifact_root"])
    checkpoint_path = artifact_dir / "checkpoint_final.pt"
    if sha256_file(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum changed after training")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if checkpoint["split_manifest_sha256"] != split_sha:
        raise ValueError("Checkpoint is bound to a different split contract")
    store = LiftFrameStore(image_path, train_ids)
    normalizer = ProprioNormalizer(checkpoint["proprio_mean"].numpy(), checkpoint["proprio_std"].numpy())
    model = VisualProprioEncoder(config["model"]["variant"], pretrained_path=None).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    bank_path = artifact_dir / "memory_bank.npz"
    bank = build_memory_bank(model, store, normalizer, bank_path, device=device)
    if set(bank.episode_ids.tolist()) != set(train_ids) or set(bank.episode_ids.tolist()).intersection(split["heldout_validation_episode_ids"]):
        raise RuntimeError("Memory bank contains an unexpected or held-out episode")
    # Self-retrieval should return each stored row for deterministic re-encoding.
    bank_embeddings = torch.from_numpy(bank.embeddings).to(device)
    scores = bank_embeddings @ bank_embeddings.T
    diagonal = torch.arange(len(bank.embeddings), device=device)
    self_rows = torch.argmax(scores, dim=1)
    self_retrieval_rate = float((self_rows == diagonal).float().mean().item())
    manifest = {
        "status": "completed",
        "run_id": metadata["run_id"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "split_manifest_sha256": split_sha,
        "train_episode_ids": train_ids,
        "validation_episode_ids_in_bank": [],
        "entry_count": int(len(bank.embeddings)),
        "embedding_dim": 128,
        "episode_count": int(len(set(bank.episode_ids.tolist()))),
        "self_retrieval_rate": self_retrieval_rate,
        "self_retrieval_checks": int(len(bank.embeddings)),
        "self_retrieval_is_diagnostic_not_a_pass_gate": True,
        "action_source": "source HDF5 actions[episode_id][local_frame_index], never duplicated per bank row",
        "bank_path": str(bank_path.relative_to(HERE)),
        "artifact_root": str(artifact_dir.relative_to(HERE)),
        "bank_bytes": bank_path.stat().st_size,
        "bank_sha256": sha256_file(bank_path),
        "checkpoint_path": str(checkpoint_path.relative_to(HERE)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "proprio_normalizer": normalizer.to_dict(),
        "device": str(device),
    }
    json_write(run_dir / "artifact_manifest.json", manifest)
    metadata.update({"memory_bank": {k: manifest[k] for k in ("entry_count", "episode_count", "bank_bytes", "bank_sha256", "self_retrieval_rate")}})
    json_write(run_dir / "run_metadata.json", metadata)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
