"""ALT/SimCLR NT-Xent mathematical objective, independently implemented.
Positive of i is i+B modulo 2B; only self-comparisons are excluded. No actions.
"""
import torch
from torch.nn import functional as F


def nt_xent(z1, z2, temperature=0.4):
    if z1.shape != z2.shape or z1.ndim != 2 or len(z1) < 2 or temperature <= 0:
        raise ValueError('Matching batches >=2 and positive temperature required')
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    b = len(z1)
    targets = (torch.arange(2 * b, device=z.device) + b) % (2 * b)
    return F.cross_entropy(logits, targets)
