"""Based on He et al. (2025), StanfordMSL/alt training.py; see licenses/ALT-MIT.txt.
Independent implementation: one 96x96 image, active 2D position, fixed coordinate
scaling, strict ImageNet initialization, no geometric augmentation.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.transforms import ColorJitter
from pusht.repro import sha256


class FusionEncoder(nn.Module):
    def __init__(self, image_dim=128, position_dim=32, embedding_dim=128):
        super().__init__()
        # Construction only. Fresh training MUST use from_pretrained; inference
        # must load a complete checkpoint. No random-weight training fallback.
        self.image_encoder = resnet18(weights=None)
        self.image_encoder.fc = nn.Linear(512, image_dim)
        self.position_encoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                              nn.Linear(64, position_dim))
        self.fusion = nn.Sequential(nn.Linear(image_dim + position_dim, 256), nn.ReLU(),
                                    nn.Linear(256, embedding_dim))
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('image_std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.pretrained_sha256 = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        path = Path(path)
        if not path.is_file() or not sha256(path).startswith('f37072fd'):
            raise ValueError('Verified ImageNet ResNet18 weights required; run prepare-weights')
        model = cls(**kwargs)
        state = torch.load(path, map_location='cpu', weights_only=True)
        state.pop('fc.weight')
        state.pop('fc.bias')
        missing, unexpected = model.image_encoder.load_state_dict(state, strict=False)
        if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
            raise ValueError('Unexpected pretrained backbone keys')
        model.pretrained_sha256 = sha256(path)
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    def forward(self, image, agent_pos):
        if image.ndim != 4 or image.shape[1:] != (3, 96, 96):
            raise ValueError(f'Expected one RGB frame [B,3,96,96], got {image.shape}')
        if agent_pos.shape != (len(image), 2):
            raise ValueError('Only current 2D agent position is supported')
        image = (image - self.image_mean) / self.image_std
        pos = agent_pos.to(image.dtype) / 256.0 - 1.0
        features = torch.cat([self.image_encoder(image), self.position_encoder(pos)], dim=-1)
        return F.normalize(self.fusion(features), dim=-1, eps=1e-12)


def make_augmentation(settings):
    return ColorJitter(**settings)  # called independently for every sample/view
