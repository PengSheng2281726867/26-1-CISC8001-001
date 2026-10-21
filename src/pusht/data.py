"""Read-only adapter. See data audit for observation_t -> action_t evidence.
No DP normalizer: its default fits the entire replay. Position uses fixed 512x512
bounds; expert actions stay in original absolute target coordinates.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class PushTData:
    def __init__(self, path):
        self.root = zarr.open(str(path), mode='r')
        self.ends = self.root['meta/episode_ends'][:]
        self.starts = np.r_[0, self.ends[:-1]]
        if not (np.all(np.diff(np.r_[0, self.ends]) > 0)
                and all(len(self.root['data'][k]) == self.ends[-1] for k in ['img', 'state', 'action'])):
            raise ValueError('Invalid replay boundaries/schema')
        self.positions = self.root['data/state'][:, :2].astype(np.float32)
        self.actions = self.root['data/action'][:].astype(np.float32)

    def index(self, episode, frame):
        if not 0 <= episode < len(self.ends):
            raise IndexError('Invalid episode ID')
        if not 0 <= frame < self.ends[episode] - self.starts[episode]:
            raise IndexError('Frame outside demonstration')
        return int(self.starts[episode] + frame)

    def action_chunk(self, episode, frame, length=8):
        episode, frame = int(episode), int(frame)
        start = self.index(episode, frame)
        idx = np.minimum(np.arange(start, start + length), self.ends[episode] - 1)
        chunk = self.actions[idx].copy()
        if chunk.shape != (length, 2) or not np.isfinite(chunk).all():
            raise ValueError('Invalid action chunk')
        if np.any(chunk < 0) or np.any(chunk > 512):
            raise ValueError('Action outside original environment units')
        return chunk, idx

    def frames(self, episodes):
        indices = np.concatenate([np.arange(self.starts[e], self.ends[e]) for e in episodes])
        images = np.concatenate([self.root['data/img'][self.starts[e]:self.ends[e]]
                                 for e in episodes]).astype(np.uint8)
        return images, self.positions[indices], indices


class ObservationFrames(Dataset):
    def __init__(self, data, episodes, augmentation=None):
        self.images, self.positions, self.indices = data.frames(episodes)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image = torch.from_numpy(self.images[i].copy()).permute(2, 0, 1).float() / 255.0
        pos = torch.from_numpy(self.positions[i].copy())
        if self.augmentation is not None:
            return self.augmentation(image), self.augmentation(image), pos
        return image, pos
