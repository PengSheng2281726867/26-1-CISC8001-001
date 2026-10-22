"""Deterministic all-frame memory-bank construction and top-1 cosine search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .data import LiftFrameStore, ProprioNormalizer, deterministic_image_tensor


@dataclass
class MemoryBank:
    embeddings: np.ndarray
    episode_ids: np.ndarray
    local_frame_indices: np.ndarray
    source_indices: np.ndarray
    proprio: np.ndarray

    def validate(self, expected_dim: int = 128) -> None:
        n = len(self.embeddings)
        if self.embeddings.shape != (n, expected_dim) or n == 0:
            raise ValueError("Invalid or empty memory bank embedding matrix")
        if self.episode_ids.shape != (n,) or self.local_frame_indices.shape != (n,) or self.source_indices.shape != (n,):
            raise ValueError("Memory bank index arrays are not aligned")
        if self.proprio.shape != (n, 9):
            raise ValueError("Memory bank proprioception is not aligned")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.isfinite(self.embeddings).all() or not np.allclose(norms, 1.0, atol=2e-4):
            raise ValueError("Memory bank embeddings must be finite and unit-normalized")

    def save(self, path: Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np.float32),
            episode_ids=self.episode_ids.astype(str),
            local_frame_indices=self.local_frame_indices.astype(np.int32),
            source_indices=self.source_indices.astype(np.int64),
            proprio=self.proprio.astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "MemoryBank":
        with np.load(str(path), allow_pickle=False) as data:
            bank = cls(*(data[k].copy() for k in ("embeddings", "episode_ids", "local_frame_indices", "source_indices", "proprio")))
        bank.validate()
        return bank


def build_memory_bank(
    model: torch.nn.Module,
    store: LiftFrameStore,
    normalizer: ProprioNormalizer,
    path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> MemoryBank:
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(store), batch_size):
            stop = min(start + batch_size, len(store))
            first = torch.stack(
                [deterministic_image_tensor(x, device=device) for x in store.agentview[start:stop]], dim=0
            )
            second = torch.stack(
                [deterministic_image_tensor(x, device=device) for x in store.wrist[start:stop]], dim=0
            )
            images = torch.stack((first, second), dim=1)
            proprio = torch.from_numpy(normalizer.transform(store.proprio[start:stop])).to(device)
            z = model.encode(images, proprio)
            embeddings.append(z.cpu().numpy().astype(np.float32))
    bank = MemoryBank(
        embeddings=np.concatenate(embeddings, axis=0),
        episode_ids=store.episode_ids_per_frame.astype(str),
        local_frame_indices=store.local_indices.astype(np.int32),
        source_indices=store.source_indices.astype(np.int64),
        proprio=store.proprio.astype(np.float32),
    )
    bank.validate()
    bank.save(path)
    return bank


class CosineTop1:
    def __init__(self, bank: MemoryBank, device: torch.device):
        bank.validate()
        self.bank = bank
        self.device = device
        self.embeddings = torch.from_numpy(bank.embeddings).to(device=device, dtype=torch.float32)

    def query(self, embedding: torch.Tensor):
        query = torch.nn.functional.normalize(embedding.reshape(1, -1).to(self.device), dim=1)
        scores = torch.mv(self.embeddings, query.reshape(-1))
        index = int(torch.argmax(scores).item())  # first row wins exact ties
        return index, float(scores[index].item())
