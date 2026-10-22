"""ALT-style NT-Xent objective; actions are not training targets."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.4) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("NT-Xent temperature must be positive")
    if z_a.shape != z_b.shape or z_a.ndim != 2 or z_a.shape[0] < 2:
        raise ValueError("Expected two matching [B,D] views with batch size >= 2")
    z = F.normalize(torch.cat((z_a, z_b), dim=0), p=2, dim=1)
    similarities = torch.matmul(z, z.T) / temperature
    batch = z_a.shape[0]
    diagonal = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    similarities = similarities.masked_fill(diagonal, torch.finfo(similarities.dtype).min)
    targets = torch.cat(
        (torch.arange(batch, 2 * batch, device=z.device), torch.arange(0, batch, device=z.device))
    )
    return F.cross_entropy(similarities, targets)
