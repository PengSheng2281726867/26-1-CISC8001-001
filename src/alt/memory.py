"""Cosine top-1 memory; no OOD or action compensation. Independent Push-T code.
Concept: He et al. (2025), ALT data_creation.py / rollout.py. Stores frame indices,
not duplicated action sequences. See licenses/ALT-MIT.txt and provenance notes.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pusht.data import ObservationFrames


class MemoryBank:
    def __init__(self, embeddings, episode_ids, local_frames, source_indices, metadata):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.local_frames = np.asarray(local_frames, dtype=np.int64)
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.metadata = metadata
        n = len(self.embeddings)
        if (self.embeddings.ndim != 2 or self.embeddings.shape[1] != 128 or not n
                or any(len(x) != n for x in [self.episode_ids, self.local_frames, self.source_indices])
                or not torch.isfinite(self.embeddings).all()
                or not torch.allclose(self.embeddings.norm(dim=-1), torch.ones(n), atol=1e-5)):
            raise ValueError('Invalid bank shape, indices or normalization')

    def query(self, embeddings):
        scores = embeddings @ self.embeddings.to(embeddings.device).T
        similarity, index = scores.max(dim=-1)
        return index, similarity  # deterministic first index for exact ties

    def save(self, path):
        torch.save({'embeddings': self.embeddings.cpu(),
                    'episode_ids': torch.from_numpy(self.episode_ids),
                    'local_frames': torch.from_numpy(self.local_frames),
                    'source_indices': torch.from_numpy(self.source_indices),
                    'metadata': self.metadata}, path)

    @classmethod
    def load(cls, path):
        v = torch.load(path, map_location='cpu', weights_only=True)
        return cls(v['embeddings'], v['episode_ids'].numpy(), v['local_frames'].numpy(),
                   v['source_indices'].numpy(), v['metadata'])


@torch.inference_mode()
def build_bank(model, data, episodes, metadata, batch_size=128):
    frames = ObservationFrames(data, episodes)
    device = next(model.parameters()).device
    model.eval()
    vectors = []
    for image, pos in DataLoader(frames, batch_size=batch_size, shuffle=False):
        vectors.append(model(image.to(device), pos.to(device)).cpu())
    indices = frames.indices
    eps = np.searchsorted(data.ends, indices, side='right')
    local = indices - data.starts[eps]
    return MemoryBank(torch.cat(vectors), eps, local, indices, metadata)
