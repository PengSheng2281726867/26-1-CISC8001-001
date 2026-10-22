"""Online top-1 retrieval policy for the audited 7D OSC_POSE action interface."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np
import torch

from .data import IMAGE_KEYS, PROPRIO_KEYS, ProprioNormalizer, action_chunk, canonicalize_proprio, deterministic_image_tensor
from .memory import CosineTop1, MemoryBank


class ActionRetrievalPolicy:
    def __init__(self, model, bank: MemoryBank, normalizer: ProprioNormalizer, actions_by_episode: Mapping[str, np.ndarray], device: torch.device):
        self.model = model.to(device).eval()
        self.bank = bank
        self.search = CosineTop1(bank, device)
        self.normalizer = normalizer
        self.actions_by_episode = actions_by_episode
        self.device = device
        self.query_count = 0

    def reset(self) -> None:
        self.query_count = 0

    def act(self, observation: Mapping[str, np.ndarray], chunk_length: int = 1):
        # Read only the current two RGB images and the current three proprio keys.
        preprocess_start = time.perf_counter()
        images = torch.stack(
            [deterministic_image_tensor(observation[key], self.device) for key in IMAGE_KEYS], dim=0
        ).unsqueeze(0)
        raw_prop = np.concatenate([np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in PROPRIO_KEYS])
        raw_prop = canonicalize_proprio(raw_prop[None, :])
        normalized_prop = self.normalizer.transform(raw_prop)
        prop = torch.from_numpy(normalized_prop).to(self.device)
        preprocess_seconds = time.perf_counter() - preprocess_start
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            embedding = self.model.encode(images, prop)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        encoder_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        bank_row, similarity = self.search.query(embedding[0])
        search_seconds = time.perf_counter() - t1
        episode_id = str(self.bank.episode_ids[bank_row])
        local_index = int(self.bank.local_frame_indices[bank_row])
        if episode_id not in self.actions_by_episode:
            raise KeyError("Retrieved episode is absent from the selected train-only action pool")
        action_start = time.perf_counter()
        actions = action_chunk(self.actions_by_episode[episode_id], local_index, chunk_length)
        action_seconds = time.perf_counter() - action_start
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.00001):
            raise ValueError("Retrieved controller action is invalid")
        self.query_count += 1
        info = {
            "bank_row": bank_row,
            "episode_id": episode_id,
            "local_frame_index": local_index,
            "source_index": int(self.bank.source_indices[bank_row]),
            "top1_cosine_similarity": similarity,
            "query_agentview_image": observation[IMAGE_KEYS[0]],
            "query_wrist_image": observation[IMAGE_KEYS[1]],
            "query_proprio": raw_prop[0].copy(),
            "matched_proprio": self.bank.proprio[bank_row].copy(),
            "action_chunk": actions.copy(),
            "encoder_seconds": encoder_seconds,
            "search_seconds": search_seconds,
            "preprocess_seconds": preprocess_seconds,
            "action_extraction_seconds": action_seconds,
        }
        return actions[0].copy(), info
