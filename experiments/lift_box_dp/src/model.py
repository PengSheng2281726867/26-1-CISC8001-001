"""Adapt the MIT-licensed local DiffusionUnetHybridImagePolicy to current-frame Lift."""
import numpy as np
import torch

from common import IMAGE_KEYS, PROPRIO_KEYS


def make_model(cfg):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy

    shape = {"action": {"shape": [7]}, "obs": {
        **{key: {"shape": [3, 84, 84], "type": "rgb"} for key in IMAGE_KEYS},
        **{key: {"shape": [dim]} for key, dim in zip(PROPRIO_KEYS, (3, 4, 2))},
    }}
    scheduler = DDIMScheduler(num_train_timesteps=cfg["diffusion_steps"], beta_start=0.0001,
                              beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, prediction_type="epsilon")
    return DiffusionUnetHybridImagePolicy(
        shape_meta=shape, noise_scheduler=scheduler, horizon=cfg["horizon"],
        n_obs_steps=1, n_action_steps=1, num_inference_steps=cfg["inference_steps"],
        crop_shape=None, diffusion_step_embed_dim=128, down_dims=cfg["down_dims"],
        obs_encoder_group_norm=True,
    )


def set_normalizer(model, proprio):
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer

    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    for key in IMAGE_KEYS:
        normalizer[key] = get_image_range_normalizer()
    for key, columns in zip(PROPRIO_KEYS, (slice(0, 3), slice(3, 7), slice(7, 9))):
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(proprio[:, columns], mode="gaussian")
    model.set_normalizer(normalizer)


def batch_observations(observations, device):
    from alt_lift.data import canonicalize_proprio

    result = {}
    for key in IMAGE_KEYS:
        images = np.stack([obs[key] for obs in observations])
        result[key] = torch.from_numpy(np.moveaxis(images, -1, 1).copy()).to(device).float()[:, None] / 255
    proprio = canonicalize_proprio(np.stack([
        np.concatenate([obs[key] for key in PROPRIO_KEYS]) for obs in observations
    ]))
    for key, columns in zip(PROPRIO_KEYS, (slice(0, 3), slice(3, 7), slice(7, 9))):
        result[key] = torch.from_numpy(proprio[:, None, columns]).to(device)
    return result
