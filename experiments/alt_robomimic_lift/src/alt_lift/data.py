"""RoboMimic Lift-PH HDF5 adapter and fixed preprocessing.

This is an independent adapter for the audited Lift dataset. It intentionally
does not use simulator state, object observations, rewards, next_obs, or labels
as encoder or policy inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
ACTION_DIM = 7
IMAGE_SIZE = (84, 84)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def numeric_demo_key(episode_id: str) -> int:
    if not episode_id.startswith("demo_"):
        raise ValueError("Unexpected episode id: %s" % episode_id)
    return int(episode_id.split("_", 1)[1])


def canonicalize_proprio(raw: np.ndarray) -> np.ndarray:
    """Return XYZW proprioception with a unit, sign-canonical quaternion."""
    values = np.asarray(raw, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError("Expected proprioception shaped [N, 9], got %r" % (values.shape,))
    quat = values[:, 3:7]
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-8):
        raise ValueError("Invalid zero or non-finite EEF quaternion")
    quat /= norms
    flip = quat[:, 3] < 0.0  # audited XYZW order: the final coordinate is w
    quat[flip] *= -1.0
    if not np.isfinite(values).all():
        raise ValueError("Non-finite Lift proprioception")
    return values


@dataclass
class ProprioNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train_proprio: np.ndarray) -> "ProprioNormalizer":
        values = np.asarray(train_proprio, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 9 or values.shape[0] == 0:
            raise ValueError("Normalizer requires non-empty training proprioception [N,9]")
        if not np.isfinite(values).all():
            raise ValueError("Cannot fit proprioception normalizer to non-finite values")
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, proprio: np.ndarray) -> np.ndarray:
        values = canonicalize_proprio(proprio)
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fit_scope": "exact_train_episode_ids_only"}


@dataclass
class EpisodeFrames:
    episode_id: str
    agentview: np.ndarray
    wrist: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    source_indices: np.ndarray


class LiftFrameStore:
    """Load only the explicitly selected episodes and their policy observations."""

    def __init__(self, hdf5_path: Path, episode_ids: Sequence[str]):
        self.hdf5_path = Path(hdf5_path)
        self.episode_ids = tuple(episode_ids)
        if len(self.episode_ids) != len(set(self.episode_ids)):
            raise ValueError("Episode selection contains duplicates")
        self.episodes: Dict[str, EpisodeFrames] = {}
        with h5py.File(str(self.hdf5_path), "r") as h5:
            data = h5["data"]
            all_ids = sorted(data.keys(), key=numeric_demo_key)
            offsets: Dict[str, int] = {}
            cursor = 0
            for episode_id in all_ids:
                offsets[episode_id] = cursor
                cursor += int(data[episode_id]["actions"].shape[0])
            for episode_id in self.episode_ids:
                if episode_id not in data:
                    raise KeyError("Episode missing from HDF5: %s" % episode_id)
                demo = data[episode_id]
                obs = demo["obs"]
                agentview = np.asarray(obs[IMAGE_KEYS[0]], dtype=np.uint8)
                wrist = np.asarray(obs[IMAGE_KEYS[1]], dtype=np.uint8)
                raw_proprio = np.concatenate(
                    [np.asarray(obs[key], dtype=np.float32) for key in PROPRIO_KEYS], axis=1
                )
                proprio = canonicalize_proprio(raw_proprio)
                # Preserve the source HDF5 action precision and command values.
                actions = np.asarray(demo["actions"])
                n = int(actions.shape[0])
                if agentview.shape != (n, 84, 84, 3) or wrist.shape != agentview.shape:
                    raise ValueError("Unexpected two-camera image shape in %s" % episode_id)
                if proprio.shape != (n, 9) or actions.shape != (n, ACTION_DIM):
                    raise ValueError("Observation/action alignment failed for %s" % episode_id)
                if not np.isfinite(actions).all():
                    raise ValueError("Non-finite action in %s" % episode_id)
                if np.any(np.abs(actions[:, :6]) > 1.00001) or np.any(np.abs(actions[:, 6]) > 1.00001):
                    raise ValueError("Action outside the audited normalized controller range")
                source_indices = np.arange(offsets[episode_id], offsets[episode_id] + n, dtype=np.int64)
                self.episodes[episode_id] = EpisodeFrames(
                    episode_id, agentview, wrist, proprio, actions, source_indices
                )

        self.agentview = np.concatenate([self.episodes[e].agentview for e in self.episode_ids], axis=0)
        self.wrist = np.concatenate([self.episodes[e].wrist for e in self.episode_ids], axis=0)
        self.proprio = np.concatenate([self.episodes[e].proprio for e in self.episode_ids], axis=0)
        self.source_indices = np.concatenate([self.episodes[e].source_indices for e in self.episode_ids])
        self.episode_ids_per_frame = np.concatenate(
            [np.repeat(e, len(self.episodes[e].actions)) for e in self.episode_ids]
        )
        self.local_indices = np.concatenate(
            [np.arange(len(self.episodes[e].actions), dtype=np.int32) for e in self.episode_ids]
        )

    def __len__(self) -> int:
        return int(self.proprio.shape[0])


def deterministic_image_tensor(image: np.ndarray, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """HWC RGB uint8 -> ImageNet-normalized CHW float tensor, with no augmentation."""
    image = np.asarray(image)
    if image.shape != (84, 84, 3):
        raise ValueError("Expected current RGB image [84,84,3], got %r" % (image.shape,))
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    return (tensor - mean) / std


class ContrastiveViews(Dataset):
    """Two independent photometric views of one synchronized two-camera frame."""

    def __init__(
        self,
        store: LiftFrameStore,
        normalizer: ProprioNormalizer,
        augmentation: Mapping[str, float],
    ):
        self.store = store
        self.normalizer = normalizer
        self.jitter = transforms.ColorJitter(
            brightness=float(augmentation["brightness"]),
            contrast=float(augmentation["contrast"]),
            saturation=float(augmentation["saturation"]),
            hue=float(augmentation["hue"]),
        )
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.store)

    def _view(self, index: int) -> torch.Tensor:
        cameras = []
        for array in (self.store.agentview[index], self.store.wrist[index]):
            image = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255.0)
            image = self.jitter(image)
            cameras.append((image - self.mean) / self.std)
        return torch.stack(cameras, dim=0)

    def __getitem__(self, index: int):
        prop = self.normalizer.transform(self.store.proprio[index : index + 1])[0]
        return self._view(index), self._view(index), torch.from_numpy(prop)


def action_chunk(actions: np.ndarray, local_index: int, chunk_length: int) -> np.ndarray:
    """Extract a same-episode action chunk; pad only by repeating this episode's last action."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError("Expected one episode of 7D actions")
    if chunk_length < 1 or local_index < 0 or local_index >= len(values):
        raise ValueError("Invalid local action index or chunk length")
    stop = min(local_index + chunk_length, len(values))
    chunk = values[local_index:stop].copy()
    if len(chunk) < chunk_length:
        pad = np.repeat(values[-1:, :], chunk_length - len(chunk), axis=0)
        chunk = np.concatenate((chunk, pad), axis=0)
    if chunk.shape != (chunk_length, ACTION_DIM) or not np.isfinite(chunk).all():
        raise ValueError("Invalid action chunk")
    return chunk
