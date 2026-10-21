"""B1 extends the attributed baseline FusionEncoder without altering inference.

Baseline: src/alt/encoder.py (He et al. 2025 ALT concept; root ALT MIT notice).
Only addition: image_feature -> Linear(128,64), ReLU, Linear(64,4).
The inherited forward never calls this head. Constructing it after super().__init__
preserves every shared initial tensor under the same seed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from alt.encoder import FusionEncoder


class B1FusionEncoder(FusionEncoder):
    def __init__(self, image_dim=128, position_dim=32, embedding_dim=128, pose_hidden_dim=64):
        super().__init__(image_dim, position_dim, embedding_dim)
        self.pose_head = nn.Sequential(nn.Linear(image_dim, pose_hidden_dim), nn.ReLU(), nn.Linear(pose_hidden_dim, 4))

    def forward_with_pose(self, image, agent_pos):
        if image.ndim != 4 or image.shape[1:] != (3, 96, 96):
            raise ValueError('Expected [B,3,96,96] current RGB images')
        if agent_pos.shape != (len(image), 2):
            raise ValueError('Only current pusher xy is an input')
        image_feature = self.image_encoder((image-self.image_mean)/self.image_std)
        position_feature = self.position_encoder(agent_pos.to(image.dtype)/256-1)
        z = F.normalize(self.fusion(torch.cat([image_feature, position_feature], -1)), dim=-1, eps=1e-12)
        return z, self.pose_head(image_feature)
