"""Same global frame indexing and augmentation as ObservationFrames; extra labels."""
import numpy as np
import torch
from pusht.data import ObservationFrames


def pose_target(pose):
    pose = torch.as_tensor(pose, dtype=torch.float32)
    return torch.stack([pose[..., 0]/256-1, pose[..., 1]/256-1,
                        pose[..., 2].sin(), pose[..., 2].cos()], dim=-1)


class PoseFrames(ObservationFrames):
    def __init__(self, data, episodes, augmentation=None):
        super().__init__(data, episodes, augmentation)
        self.block_pose = data.root['data/state'][:][self.indices, 2:5].astype(np.float32)
        self.targets = pose_target(self.block_pose)

    def __getitem__(self, i):
        # Base adapter supplies identical HWC->CHW /255 preprocessing and two
        # independent per-sample ColorJitter calls when augmentation is enabled.
        return (*super().__getitem__(i), self.targets[i].clone())
