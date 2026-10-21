"""Frozen B1 objective; no action supervision or geometric retrieval weighting."""
from torch.nn import functional as F
from alt.loss import nt_xent


def losses(z1, z2, p1, p2, target, temperature=0.4, lambda_pose=1.0):
    contrast = nt_xent(z1, z2, temperature)
    xy = .5 * (F.mse_loss(p1[:, :2], target[:, :2]) + F.mse_loss(p2[:, :2], target[:, :2]))
    angle = .5 * (F.mse_loss(p1[:, 2:], target[:, 2:]) + F.mse_loss(p2[:, 2:], target[:, 2:]))
    pose = xy + angle
    return {'contrast': contrast, 'xy': xy, 'angle': angle, 'pose': pose,
            'total': contrast + lambda_pose*pose}
