"""Camera-observation validation, scaling, cropping, and augmentation."""

from __future__ import annotations

import torch


def validate_images(images: torch.Tensor, observation_horizon: int) -> torch.Tensor:
    """Validate camera_rgb shape/dtype and return float32 CHW images in [0, 1]."""

    if images.ndim != 5 or images.shape[1] != observation_horizon:
        raise ValueError(
            "camera_rgb must have shape [batch, observation_horizon, C, H, W] "
            "or [batch, observation_horizon, H, W, C]"
        )
    if images.shape[0] == 0:
        raise ValueError("training batches cannot be empty")
    if images.shape[2] == 3:
        pass
    elif images.shape[-1] == 3:
        images = images.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError("camera_rgb must contain exactly three RGB channels")
    if images.dtype == torch.uint8:
        images = images.float().div_(255)
    elif images.is_floating_point():
        images = images.float()
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError(
                "floating-point camera_rgb values must be finite in [0, 1]"
            )
    else:
        raise TypeError("camera_rgb must be uint8 or floating point")
    return images


def augment_images(
    images: torch.Tensor, *, crop_fraction: float, color_jitter: float, augment: bool
) -> torch.Tensor:
    """Crop to ``crop_fraction``; augment to also jitter brightness/contrast."""

    batch, horizon, channels, height, width = images.shape
    if crop_fraction < 1:
        crop_height = round(height * crop_fraction)
        crop_width = round(width * crop_fraction)
        max_top = height - crop_height
        max_left = width - crop_width
        if augment:
            # One offset per sample, shared across that sample's history frames.
            # Generated on CPU and listed to avoid a device sync per sample.
            top = torch.randint(0, max_top + 1, (batch,)).tolist()
            left = torch.randint(0, max_left + 1, (batch,)).tolist()
        else:
            top = [max_top // 2] * batch
            left = [max_left // 2] * batch
        cropped = torch.empty(
            (batch, horizon, channels, crop_height, crop_width),
            dtype=images.dtype,
            device=images.device,
        )
        for sample in range(batch):
            sample_top, sample_left = top[sample], left[sample]
            cropped[sample] = images[
                sample,
                :,
                :,
                sample_top : sample_top + crop_height,
                sample_left : sample_left + crop_width,
            ]
        images = cropped

    if augment and color_jitter > 0:
        # Factors are per-sample (shared across history); the mean is per-frame.
        factors = torch.rand((2, batch, 1, 1, 1, 1), device=images.device)
        brightness, contrast = 1 - color_jitter + 2 * color_jitter * factors
        mean = images.mean(dim=(2, 3, 4), keepdim=True)
        images = (((images - mean) * contrast + mean) * brightness).clamp_(0, 1)
    return images
