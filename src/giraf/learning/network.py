"""Neural network components for the conditional diffusion policy."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn


def _group_count(channels: int) -> int:
    """Return the largest group size in (8, 4, 2, 1) that divides ``channels``."""

    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


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
    ) -> None:
        super().__init__()
        self.observation_horizon = observation_horizon
        self.state_dim = state_dim
        self.image_encoder = ConvEncoder(vision_features)
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
