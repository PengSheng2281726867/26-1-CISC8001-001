#!/usr/bin/env python
"""Frozen held-out retrieval diagnostics; validation data never enters the bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.data import LiftFrameStore, ProprioNormalizer, deterministic_image_tensor
from alt_lift.encoder import VisualProprioEncoder
from alt_lift.memory import CosineTop1, MemoryBank
from alt_lift.repro import json_write, resolve_config_path, sha256_file


def quat_angle_deg(q1, q2):
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = min(1.0, max(0.0, abs(float(np.dot(a, b)))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "alt_lift30.yaml")
    parser.add_argument("--paths-config", type=Path, default=HERE / "configs" / "runtime_paths.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "offline_diagnostics.json")
    parser.add_argument("--figure", type=Path, default=HERE / "results" / "figures" / "alt_lift30_offline_matches.png")
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("Refusing to overwrite offline diagnostics; choose new --output and --figure paths")
    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    artifact = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    paths = json.loads(args.paths_config.read_text(encoding="utf-8"))
    image_path = resolve_config_path(paths["image_hdf5"])
    low_dim_path = resolve_config_path(paths["low_dim_hdf5"])
    split_path = resolve_config_path(config["dataset"]["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha256_file(split_path) != meta["split_manifest_sha256"] or sha256_file(image_path) != split["source_files"]["image_hdf5"]["sha256"] or sha256_file(low_dim_path) != split["source_files"]["low_dim_hdf5"]["sha256"]:
        raise ValueError("Split or dataset source changed after training")
    val_ids = list(split["heldout_validation_episode_ids"])
    train_ids = set(split["train_subsets"]["30"])
    if train_ids.intersection(val_ids) or artifact["validation_episode_ids_in_bank"]:
        raise ValueError("Validation episodes overlap with the training memory bank")
    artifact_dir = resolve_config_path(meta["artifact_root"])
    bank_path = artifact_dir / "memory_bank.npz"
    if sha256_file(bank_path) != artifact["bank_sha256"]:
        raise ValueError("Memory bank checksum mismatch")
    if artifact["split_manifest_sha256"] != meta["split_manifest_sha256"]:
        raise ValueError("Memory bank and training metadata reference different split contracts")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(artifact_dir / "checkpoint_final.pt"), map_location="cpu", weights_only=True)
    normalizer = ProprioNormalizer(checkpoint["proprio_mean"].numpy(), checkpoint["proprio_std"].numpy())
    model = VisualProprioEncoder(config["model"]["variant"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    bank = MemoryBank.load(bank_path)
    search = CosineTop1(bank, device)
    store = LiftFrameStore(image_path, val_ids)
    train_store = LiftFrameStore(image_path, list(split["train_subsets"]["30"]))
    entries = []
    z_values = []
    with torch.inference_mode():
        for start in range(0, len(store), 64):
            stop = min(start + 64, len(store))
            agent = torch.stack([deterministic_image_tensor(x, device) for x in store.agentview[start:stop]], dim=0)
            wrist = torch.stack([deterministic_image_tensor(x, device) for x in store.wrist[start:stop]], dim=0)
            images = torch.stack((agent, wrist), dim=1)
            prop = torch.from_numpy(normalizer.transform(store.proprio[start:stop])).to(device)
            z_values.append(model.encode(images, prop).cpu())
    queries = torch.cat(z_values, dim=0).numpy()
    entries = []
    with h5py_file(image_path) as h5:
      for i, z in enumerate(queries):
        row, similarity = search.query(torch.from_numpy(z).to(device))
        query_id = str(store.episode_ids_per_frame[i])
        query_t = int(store.local_indices[i])
        matched_id = str(bank.episode_ids[row])
        matched_t = int(bank.local_frame_indices[row])
        qprop = store.proprio[i]
        mprop = bank.proprio[row]
        q_action = store.episodes[query_id].actions[query_t]
        # This diagnostic-only read uses the held-out expert action as a comparator.
        matched_action = np.asarray(h5["data"][matched_id]["actions"][matched_t])
        entries.append({
            "query_episode_id": query_id,
            "query_local_frame_index": query_t,
            "query_source_index": int(store.source_indices[i]),
            "matched_episode_id": matched_id,
            "matched_local_frame_index": matched_t,
            "matched_source_index": int(bank.source_indices[row]),
            "top1_cosine_similarity": float(similarity),
            "eef_position_l2_m": float(np.linalg.norm(qprop[:3] - mprop[:3])),
            "eef_quaternion_geodesic_degrees": quat_angle_deg(qprop[3:7], mprop[3:7]),
            "gripper_qpos_l2": float(np.linalg.norm(qprop[7:9] - mprop[7:9])),
            "action_discrepancy_l2_not_accuracy": float(np.linalg.norm(q_action - matched_action)),
            "query_action": q_action.tolist(),
            "retrieved_action": matched_action.tolist(),
        })
    # Prespecified qualitative examples: first six held-out IDs, frames 0/mid/last.
    sample_keys = []
    for episode_id in val_ids[:6]:
        episode = store.episodes[episode_id]
        for frame in sorted(set((0, len(episode.actions) // 2, len(episode.actions) - 1))):
            sample_keys.append((episode_id, frame))
    by_key = {(x["query_episode_id"], x["query_local_frame_index"]): x for x in entries}
    tile_w, tile_h, label_h = 168, 168, 30
    cols = 4
    rows = len(sample_keys)
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, key in enumerate(sample_keys):
        record = by_key[key]
        qepisode, qt = key
        mepisode, mt = record["matched_episode_id"], record["matched_local_frame_index"]
        qdata = store.episodes[qepisode]
        # Training frames are retrieved by their exact episode/local index.
        mdata = train_store.episodes[mepisode]
        images = [qdata.agentview[qt], mdata.agentview[mt], qdata.wrist[qt], mdata.wrist[mt]]
        labels = ["query agentview", "match agentview", "query wrist", "match wrist"]
        for c, (arr, label) in enumerate(zip(images, labels)):
            x, y = c * tile_w, r * (tile_h + label_h)
            tile = Image.fromarray(arr).resize((tile_w, tile_h))
            sheet.paste(tile, (x, y))
            draw.text((x + 2, y + tile_h + 2), "%s %s:%d / sim %.3f" % (label, qepisode if "query" in label else mepisode, qt if "query" in label else mt, record["top1_cosine_similarity"]), fill="black")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.figure)
    sims = np.asarray([x["top1_cosine_similarity"] for x in entries], dtype=np.float64)
    same_episode = np.asarray([x["query_episode_id"] == x["matched_episode_id"] for x in entries])
    output = {
        "status": "completed",
        "run_id": meta["run_id"],
        "split_manifest_sha256": sha256_file(split_path),
        "training_episode_ids": sorted(train_ids),
        "heldout_validation_episode_ids": val_ids,
        "validation_frame_count": int(len(entries)),
        "validation_episodes_in_bank": [],
        "top1_cosine_similarity": {"mean": float(sims.mean()), "std": float(sims.std()), "min": float(sims.min()), "median": float(np.median(sims)), "max": float(sims.max())},
        "same_episode_retrieval_fraction": float(same_episode.mean()),
        "eef_position_l2_m_mean": float(np.mean([x["eef_position_l2_m"] for x in entries])),
        "eef_quaternion_geodesic_degrees_mean": float(np.mean([x["eef_quaternion_geodesic_degrees"] for x in entries])),
        "gripper_qpos_l2_mean": float(np.mean([x["gripper_qpos_l2"] for x in entries])),
        "action_discrepancy_l2_not_accuracy_mean": float(np.mean([x["action_discrepancy_l2_not_accuracy"] for x in entries])),
        "bank_self_retrieval_rate": artifact["self_retrieval_rate"],
        "qualitative_example_rule": "first six held-out manifest IDs, frames 0, midpoint, final",
        "qualitative_figure": str(args.figure),
        "entries": entries,
    }
    json_write(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != "entries"}, indent=2, sort_keys=True))


def h5py_file(path):
    import h5py
    return h5py.File(str(path), "r")


if __name__ == "__main__":
    main()
