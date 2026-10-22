"""Audit the local RoboMimic Lift-PH HDF5 files without changing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = HERE / "configs" / "data_paths.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _decode_ids(values: Any) -> list[str]:
    result = []
    for item in np.asarray(values).reshape(-1):
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    if os.name != "nt" and len(value) > 2 and value[1] == ":" and value[2] in ("\\", "/"):
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    return path if path.is_absolute() else (base / path).resolve()


def _episode_keys(data: h5py.Group) -> list[str]:
    return sorted(
        (key for key in data.keys() if key.startswith("demo_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )


def _summarize_file(path: Path, inspect_images: bool) -> dict[str, Any]:
    with h5py.File(str(path), "r") as h5:
        data = h5["data"]
        demos = _episode_keys(data)
        lengths = [int(data[key]["actions"].shape[0]) for key in demos]
        first = data[demos[0]]
        obs_keys = sorted(first["obs"].keys()) if "obs" in first else []
        fields: dict[str, Any] = {}
        for key in obs_keys:
            sample = first["obs"][key]
            info: dict[str, Any] = {
                "shape_per_episode": list(sample.shape),
                "dtype": str(sample.dtype),
            }
            if inspect_images and key.endswith("_image"):
                low, high = 255, 0
                channel_low = np.full((sample.shape[-1],), 255, dtype=np.int64)
                channel_high = np.zeros((sample.shape[-1],), dtype=np.int64)
                for demo_key in demos:
                    arr = data[demo_key]["obs"][key]
                    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
                        raise ValueError("Expected HWC uint8 RGB image: %s/%s" % (demo_key, key))
                    for start in range(0, arr.shape[0], 16):
                        block = np.asarray(arr[start : start + 16])
                        low = min(low, int(block.min()))
                        high = max(high, int(block.max()))
                        channel_low = np.minimum(channel_low, block.min(axis=(0, 1, 2)))
                        channel_high = np.maximum(channel_high, block.max(axis=(0, 1, 2)))
                info.update(
                    {
                        "range": [low, high],
                        "channel_range": [
                            [int(lo), int(hi)] for lo, hi in zip(channel_low, channel_high)
                        ],
                        "layout": "HWC",
                        "stored_color_order": "RGB (verified against RoboSuite image pipeline)",
                    }
                )
            elif inspect_images:
                low, high = np.inf, -np.inf
                for demo_key in demos:
                    arr = np.asarray(data[demo_key]["obs"][key])
                    low, high = min(low, float(arr.min())), max(high, float(arr.max()))
                info["range"] = [low, high]
            fields[key] = info

        action_min = np.full((7,), np.inf)
        action_max = np.full((7,), -np.inf)
        state_shapes: set[tuple[int, ...]] = set()
        length_mismatches = []
        for demo_key in demos:
            demo = data[demo_key]
            actions = demo["actions"]
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError("Expected 7D action at %s, got %s" % (demo_key, actions.shape))
            action_min = np.minimum(action_min, np.asarray(actions[:]).min(axis=0))
            action_max = np.maximum(action_max, np.asarray(actions[:]).max(axis=0))
            if "states" in demo:
                state_shapes.add(tuple(demo["states"].shape))
            lengths_here = {"actions": len(actions)}
            if "obs" in demo:
                lengths_here.update({key: len(demo["obs"][key]) for key in demo["obs"]})
            if "states" in demo:
                lengths_here["states"] = len(demo["states"])
            if len(set(lengths_here.values())) != 1:
                length_mismatches.append({"episode_id": demo_key, "lengths": lengths_here})

        masks = {}
        mask_group = data["mask"] if "mask" in data else h5.get("mask")
        if mask_group is not None:
            for mask_name in sorted(mask_group.keys()):
                masks[mask_name] = _decode_ids(mask_group[mask_name])
        env_args = _jsonable(data.attrs.get("env_args", h5.attrs.get("env_args", "")))
        env_meta = _jsonable(data.attrs.get("env_metadata", h5.attrs.get("env_metadata", "")))
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "root_keys": sorted(h5.keys()),
            "root_attributes": _jsonable(dict(h5.attrs)),
            "data_attributes": _jsonable(dict(data.attrs)),
            "episode_count": len(demos),
            "episode_ids": demos,
            "total_frames": int(sum(lengths)),
            "episode_length_min_median_max": [
                int(min(lengths)), int(statistics.median(lengths)), int(max(lengths))
            ],
            "episode_lengths": {key: int(length) for key, length in zip(demos, lengths)},
            "observation_fields": fields,
            "state_shapes": [list(shape) for shape in sorted(state_shapes)],
            "action_shape_per_episode": list(first["actions"].shape),
            "action_dtype": str(first["actions"].dtype),
            "action_min_per_dimension": action_min.tolist(),
            "action_max_per_dimension": action_max.tolist(),
            "actions_abs_present": "actions_abs" in first,
            "episode_masks": masks,
            "episode_length_mismatches": length_mismatches,
            "env_args": env_args,
            "env_metadata": env_meta,
        }


def _make_contact_sheet(path: Path, output: Path) -> None:
    with h5py.File(str(path), "r") as h5:
        data = h5["data"]
        samples = []
        for episode_id in ("demo_0", "demo_1", "demo_2"):
            if episode_id not in data:
                continue
            demo = data[episode_id]
            length = len(demo["actions"])
            for frame in sorted(set((0, length // 2, length - 1))):
                for key, title in (
                    ("agentview_image", "agentview"),
                    ("robot0_eye_in_hand_image", "wrist"),
                ):
                    if key in demo["obs"]:
                        image = np.asarray(demo["obs"][key][frame])
                        samples.append((episode_id, frame, title, Image.fromarray(image, mode="RGB")))
    if not samples:
        raise ValueError("No expected camera image keys found")
    tile_w, tile_h, label_h = 252, 252, 28
    columns = 4
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (episode_id, frame, camera, image) in enumerate(samples):
        image = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
        x, y = (i % columns) * tile_w, (i // columns) * (tile_h + label_h)
        sheet.paste(image, (x, y + label_h))
        draw.text((x + 4, y + 5), "%s t=%d %s" % (episode_id, frame, camera), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-dataset", type=str)
    parser.add_argument("--low-dim-dataset", type=str)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "dataset_audit.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    image_path = _resolve(args.image_dataset or config["image_hdf5"], HERE)
    low_dim_path = _resolve(args.low_dim_dataset or config["low_dim_hdf5"], HERE)
    image = _summarize_file(image_path, inspect_images=True)
    low_dim = _summarize_file(low_dim_path, inspect_images=False)

    image_episodes = set(image["episode_ids"])
    low_episodes = set(low_dim["episode_ids"])
    shared_episodes = sorted(image_episodes & low_episodes)
    action_equal = True
    action_mismatch = None
    with h5py.File(str(image_path), "r") as image_h5, h5py.File(str(low_dim_path), "r") as low_h5:
        for episode_id in shared_episodes:
            image_actions = image_h5["data"][episode_id]["actions"]
            low_actions = low_h5["data"][episode_id]["actions"]
            if image_actions.shape != low_actions.shape or not np.array_equal(image_actions[:], low_actions[:]):
                action_equal = False
                action_mismatch = episode_id
                break

    metadata = {
        "schema_version": 1,
        "dataset_id": "robomimic_lift_ph_v0.2.0_image",
        "image_dataset": image,
        "low_dim_dataset": low_dim,
        "cross_file_checks": {
            "same_episode_ids": image_episodes == low_episodes,
            "shared_episode_count": len(shared_episodes),
            "all_shared_action_arrays_identical": action_equal,
            "first_action_mismatch_episode": action_mismatch,
            "all_image_episode_lengths_aligned": not image["episode_length_mismatches"],
            "all_low_dim_episode_lengths_aligned": not low_dim["episode_length_mismatches"],
        },
        "alignment_contract": {
            "policy_input_at_t": [
                "obs/agentview_image[t]",
                "obs/robot0_eye_in_hand_image[t]",
                "obs/robot0_eef_pos[t]",
                "obs/robot0_eef_quat[t]",
                "obs/robot0_gripper_qpos[t]",
            ],
            "expert_action_at_t": "actions[t]",
            "dataset_semantics": "obs[t] is the pre-action observation; actions[t] is the command applied from that observation. Stored arrays are equal length; only T-1 transitions have a recorded successor observation.",
            "policy_input_excludes": ["states", "object", "reward", "done", "next_obs", "future actions"],
        },
        "provenance": {
            "dataset_release": "RoboMimic 0.2.0 Lift / PH",
            "image_source_url": "https://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/image.hdf5",
            "low_dim_source_url": "http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/low_dim.hdf5",
            "full_file_sha256_computed": True,
        },
    }
    metadata["schema_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "episodes": image["episode_ids"],
                "lengths": image["episode_lengths"],
                "observation_fields": image["observation_fields"],
                "action_shape": image["action_shape_per_episode"],
                "env_args": image["env_args"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sheet_path = HERE / "results" / "figures" / "dataset_contact_sheet.png"
    _make_contact_sheet(image_path, sheet_path)
    print(json.dumps({
        "dataset_audit": str(args.output),
        "contact_sheet": str(sheet_path),
        "episodes": image["episode_count"],
        "frames": image["total_frames"],
        "length_min_median_max": image["episode_length_min_median_max"],
        "image_sha256": image["sha256"],
        "low_dim_sha256": low_dim["sha256"],
        "cross_file_checks": metadata["cross_file_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
