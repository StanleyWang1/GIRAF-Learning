"""Neural network components for the conditional diffusion policy."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _group_count(channels: int) -> int:
    """Return the largest group size in (8, 4, 2, 1) that divides ``channels``."""

    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


def _normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Normalize a [0, 1] image batch with ImageNet mean and std."""

    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _replace_batchnorm_with_groupnorm(module: nn.Module) -> None:
    """Recursively swap every BatchNorm2d child for a GroupNorm(16, channels)."""

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(16, child.num_features))
        else:
            _replace_batchnorm_with_groupnorm(child)


class ConvEncoder(nn.Module):
    """Small resolution-independent RGB encoder."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in (32, 64, 128):
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                    ),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.Mish(),
                )
            )
            input_channels = output_channels
        layers.extend(
            (nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, output_dim))
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a [N, 3, H, W] image batch to [N, output_dim]."""

        return self.encoder(images)


class SpatialSoftmax(nn.Module):
    """Reduce a feature map to expected keypoint coordinates on a [-1, 1] grid."""

    def __init__(self, in_channels: int, num_keypoints: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map a [N, C, H, W] feature map to [N, 2 * num_keypoints] coordinates."""

        heatmaps = self.project(features)
        batch, keypoints, height, width = heatmaps.shape
        weights = F.softmax(heatmaps.flatten(2), dim=-1).view(
            batch, keypoints, height, width
        )
        xs = torch.linspace(-1, 1, width, device=features.device)
        ys = torch.linspace(-1, 1, height, device=features.device)
        expected_x = (weights * xs.view(1, 1, 1, width)).sum(dim=(2, 3))
        expected_y = (weights * ys.view(1, 1, height, 1)).sum(dim=(2, 3))
        return torch.stack((expected_x, expected_y), dim=-1).flatten(1)


class ResNetEncoder(nn.Module):
    """ResNet-18 backbone (GroupNorm) with a spatial-softmax keypoint head."""

    def __init__(self, output_dim: int, num_keypoints: int = 32) -> None:
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        _replace_batchnorm_with_groupnorm(backbone)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.spatial_softmax = SpatialSoftmax(512, num_keypoints)
        self.project = nn.Linear(2 * num_keypoints, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a [N, 3, H, W] image batch in [0, 1] to [N, output_dim]."""

        features = self.backbone(_normalize_imagenet(images))
        return self.project(self.spatial_softmax(features))


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 backbone with a spatial-softmax keypoint head."""

    def __init__(
        self,
        output_dim: int,
        *,
        backbone: nn.Module | None = None,
        model_name: str = "dinov2_vits14",
        patch_size: int = 14,
        num_keypoints: int = 32,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.patch_size = patch_size
        self.spatial_softmax = SpatialSoftmax(backbone.embed_dim, num_keypoints)
        self.project = nn.Linear(2 * num_keypoints, output_dim)

    def train(self, mode: bool = True) -> DinoV2Encoder:
        """Set training mode on this module while keeping the backbone in eval mode."""

        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a [N, 3, H, W] image batch in [0, 1] to [N, output_dim]."""

        height, width = images.shape[-2:]
        target_height = math.ceil(height / self.patch_size) * self.patch_size
        target_width = math.ceil(width / self.patch_size) * self.patch_size
        if (target_height, target_width) != (height, width):
            images = F.interpolate(
                images,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        images = _normalize_imagenet(images)
        tokens = self.backbone.forward_features(images)["x_norm_patchtokens"]
        batch, num_patches, embed_dim = tokens.shape
        grid = math.isqrt(num_patches)
        if grid * grid != num_patches:
            raise ValueError(
                f"DINOv2 returned {num_patches} patch tokens, expected a square grid"
            )
        features = tokens.transpose(1, 2).reshape(batch, embed_dim, grid, grid)
        return self.project(self.spatial_softmax(features))


def build_encoder(name: str, output_dim: int) -> nn.Module:
    """Construct the named image encoder ("conv", "resnet18", or "dinov2")."""

    if name == "conv":
        return ConvEncoder(output_dim)
    if name == "resnet18":
        return ResNetEncoder(output_dim)
    if name == "dinov2":
        return DinoV2Encoder(output_dim)
    raise ValueError(
        f"unknown encoder {name!r}, expected one of: conv, resnet18, dinov2"
    )


class SinusoidalEmbedding(nn.Module):
    """Encode integer diffusion timesteps."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("timestep embedding dimension must be even and at least 4")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1d(nn.Module):
    """Residual temporal block with FiLM conditioning."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.first = Conv1dBlock(input_channels, output_channels, kernel_size)
        self.second = Conv1dBlock(output_channels, output_channels, kernel_size)
        self.condition = nn.Sequential(
            nn.Mish(), nn.Linear(condition_dim, output_channels * 2)
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.first(value)
        scale, bias = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.second(hidden * (1 + scale) + bias)
        return hidden + self.residual(value)


class ConditionalUnet1d(nn.Module):
    """Temporal U-Net that predicts noise in an action trajectory."""

    def __init__(
        self,
        *,
        action_dim: int,
        observation_dim: int,
        down_dims: tuple[int, ...],
        timestep_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        condition_dim = timestep_dim + observation_dim
        self.timestep_encoder = nn.Sequential(
            SinusoidalEmbedding(timestep_dim),
            nn.Linear(timestep_dim, timestep_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_dim * 4, timestep_dim),
        )

        dimensions = (action_dim, *down_dims)
        transitions = list(pairwise(dimensions))
        self.downs = nn.ModuleList()
        for index, (input_dim, output_dim) in enumerate(transitions):
            downsample: nn.Module = (
                nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)
                if index < len(transitions) - 1
                else nn.Identity()
            )
            self.downs.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1d(
                            input_dim, output_dim, condition_dim, kernel_size
                        ),
                        ConditionalResidualBlock1d(
                            output_dim, output_dim, condition_dim, kernel_size
                        ),
                        downsample,
                    )
                )
            )

        middle_dim = down_dims[-1]
        self.middle = nn.ModuleList(
            (
                ConditionalResidualBlock1d(
                    middle_dim, middle_dim, condition_dim, kernel_size
                ),
                ConditionalResidualBlock1d(
                    middle_dim, middle_dim, condition_dim, kernel_size
                ),
            )
        )

        self.ups = nn.ModuleList()
        for input_dim, output_dim in reversed(transitions[1:]):
            self.ups.append(
                nn.ModuleList(
                    (
                        ConditionalResidualBlock1d(
                            output_dim * 2, input_dim, condition_dim, kernel_size
                        ),
                        ConditionalResidualBlock1d(
                            input_dim, input_dim, condition_dim, kernel_size
                        ),
                        nn.ConvTranspose1d(
                            input_dim,
                            input_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                    )
                )
            )

        self.output = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], action_dim, kernel_size=1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(noisy_actions.shape[0])
        condition = torch.cat((self.timestep_encoder(timesteps), observation), dim=-1)
        value = noisy_actions.transpose(1, 2)
        skips: list[torch.Tensor] = []
        for first, second, downsample in self.downs:
            value = first(value, condition)
            value = second(value, condition)
            skips.append(value)
            value = downsample(value)
        for block in self.middle:
            value = block(value, condition)
        for first, second, upsample in self.ups:
            value = torch.cat((value, skips.pop()), dim=1)
            value = first(value, condition)
            value = second(value, condition)
            value = upsample(value)
        return self.output(value).transpose(1, 2)


class DiffusionNetwork(nn.Module):
    """Encode observation histories and denoise action trajectories."""

    def __init__(
        self,
        *,
        observation_horizon: int,
        state_dim: int,
        action_dim: int,
        vision_features: int,
        down_dims: tuple[int, ...],
        timestep_dim: int,
        kernel_size: int,
        encoder: str = "conv",
    ) -> None:
        super().__init__()
        self.observation_horizon = observation_horizon
        self.state_dim = state_dim
        self.image_encoder = build_encoder(encoder, vision_features)
        observation_dim = observation_horizon * (vision_features + state_dim)
        self.noise_predictor = ConditionalUnet1d(
            action_dim=action_dim,
            observation_dim=observation_dim,
            down_dims=down_dims,
            timestep_dim=timestep_dim,
            kernel_size=kernel_size,
        )

    def encode_observation(
        self, images: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, horizon = images.shape[:2]
        if horizon != self.observation_horizon:
            raise ValueError(
                f"received {horizon} observations, expected {self.observation_horizon}"
            )
        if states.shape != (batch_size, horizon, self.state_dim):
            raise ValueError(
                f"state shape is {tuple(states.shape)}, expected "
                f"({batch_size}, {horizon}, {self.state_dim})"
            )
        features = self.image_encoder(images.flatten(0, 1)).reshape(
            batch_size, horizon, -1
        )
        return torch.cat((features, states), dim=-1).flatten(1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        return self.noise_predictor(noisy_actions, timesteps, observation)
