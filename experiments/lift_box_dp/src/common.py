"""Frozen Lift-30 data and cross-platform paths."""
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/exp_001_lift30/config.json"
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def contract(config=None):
    cfg = read_json(CONFIG_PATH) if config is None else config
    alt = (CONFIG_PATH.parent.parent / cfg["alt_root"]).resolve()
    dp = (CONFIG_PATH.parent.parent / cfg["dp_root"]).resolve()
    bundled_alt = ROOT.parent / "alt_robomimic_lift"
    if (bundled_alt / "configs/lift_ph_split42.json").is_file():
        alt = bundled_alt
        dp = alt.parents[2] / "DP/diffusion_policy"
    split_path = alt / "configs/lift_ph_split42.json"
    if sha256(split_path) != cfg["split_sha256"]:
        raise ValueError("Frozen Lift split checksum mismatch")
    split = read_json(split_path)
    image = alt.parents[1] / split["source_files"]["image_hdf5"]["path_from_repo_root"]
    train = split["train_subsets"]["30"]
    val = split["heldout_validation_episode_ids"]
    if len(train) != 30 or len(val) != 20 or len(set(train + val)) != 50:
        raise ValueError("Invalid frozen training and validation membership")
    return cfg, alt, dp, split, image


def add_references(alt, dp=None):
    sys.path.insert(0, str(alt / "src"))
    if dp is not None:
        sys.path.insert(0, str(dp))


def wsl_path(path):
    drive, tail = os.path.splitdrive(str(Path(path).resolve()))
    return "/mnt/" + drive[0].lower() + tail.replace("\\", "/") if drive else tail
