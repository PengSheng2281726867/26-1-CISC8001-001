"""Shared visual and proprioception encoder for the Image+Proprio ALT variant.

The public ALT main branch has its pose path commented out. This implementation
follows the paper's fusion idea with current Lift proprioception and uses one
shared ResNet-18 for both cameras; it is an adaptation, not a source-code copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models import resnet18


class VisualProprioEncoder(nn.Module):
    def __init__(self, variant: str = "image_proprio", pretrained_path: Optional[Path] = None):
        super().__init__()
        if variant not in ("image_proprio", "image_only"):
            raise ValueError("Unknown encoder variant: %s" % variant)
        self.variant = variant
        # Load the verified torchvision ImageNet state into the standard ResNet18
        # before replacing its 1000-class classifier with the 128D feature head.
        try:
            self.image_encoder = resnet18(weights=None)
        except TypeError:  # torchvision 0.11 in the read-only RoboSuite runtime
            self.image_encoder = resnet18(pretrained=False)
        if pretrained_path is not None:
            path = Path(pretrained_path)
            if not path.is_file():
                raise FileNotFoundError("Required ImageNet checkpoint missing: %s" % path)
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            self.image_encoder.load_state_dict(state, strict=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 128)
        self.proprio_encoder = nn.Sequential(nn.Linear(9, 64), nn.ReLU(inplace=True), nn.Linear(64, 32), nn.ReLU(inplace=True))
        fusion_input_dim = 256 + (32 if variant == "image_proprio" else 0)
        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 128))

    def encode(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Encode B x 2 x 3 x H x W current camera observations."""
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("Expected images [B,2,3,H,W], got %r" % (tuple(images.shape),))
        batch = images.shape[0]
        image_features = self.image_encoder(images.reshape(batch * 2, *images.shape[2:]))
        image_features = image_features.reshape(batch, 2, 128)
        fused_parts = [image_features[:, 0], image_features[:, 1]]
        if self.variant == "image_proprio":
            if proprio.shape != (batch, 9):
                raise ValueError("Expected normalized proprioception [B,9]")
            fused_parts.append(self.proprio_encoder(proprio))
        embedding = self.fusion(torch.cat(fused_parts, dim=1))
        return torch.nn.functional.normalize(embedding, p=2, dim=1, eps=1e-12)

    def encode_pair(self, images_a: torch.Tensor, images_b: torch.Tensor, proprio: torch.Tensor):
        """Jointly encode two augmented observations while sharing both camera weights."""
        if images_a.shape != images_b.shape or images_a.ndim != 5:
            raise ValueError("Augmented image views must have matching [B,2,3,H,W] shapes")
        batch = images_a.shape[0]
        stacked = torch.stack((images_a, images_b), dim=1).reshape(batch * 4, *images_a.shape[2:])
        features = self.image_encoder(stacked).reshape(batch, 2, 2, 128)
        parts_a = [features[:, 0, 0], features[:, 0, 1]]
        parts_b = [features[:, 1, 0], features[:, 1, 1]]
        if self.variant == "image_proprio":
            pose = self.proprio_encoder(proprio)
            parts_a.append(pose)
            parts_b.append(pose)
        z_a = torch.nn.functional.normalize(self.fusion(torch.cat(parts_a, dim=1)), p=2, dim=1, eps=1e-12)
        z_b = torch.nn.functional.normalize(self.fusion(torch.cat(parts_b, dim=1)), p=2, dim=1, eps=1e-12)
        return z_a, z_b
