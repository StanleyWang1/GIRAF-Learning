"""Conditional DDPM policy for image and robot-state observations."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch import nn

from giraf.data.schema import ACTION_DIM, GRASP_INDEX, STATE_DIM

from .environment import Observation
from .network import DiffusionNetwork
from .policy import Batch, Metrics, Tensor

_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class DiffusionPolicyConfig:
    observation_horizon: int = 2
    prediction_horizon: int = 16
    action_horizon: int = 8
    diffusion_steps: int = 100
    inference_steps: int = 100
    vision_features: int = 128
    down_dims: tuple[int, ...] = (64, 128, 256)
    timestep_features: int = 128
    kernel_size: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    device: str = "auto"

    def __post_init__(self) -> None:
        integer_values = (
            self.observation_horizon,
            self.prediction_horizon,
            self.action_horizon,
            self.diffusion_steps,
            self.inference_steps,
            self.vision_features,
            self.timestep_features,
            self.kernel_size,
            *self.down_dims,
        )
        if not self.down_dims or min(integer_values) <= 0:
            raise ValueError("diffusion configuration values must be positive")
        if self.action_horizon + self.observation_horizon - 1 > self.prediction_horizon:
            raise ValueError("the executable action window exceeds prediction_horizon")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("inference_steps cannot exceed diffusion_steps")
        if self.timestep_features < 4 or self.timestep_features % 2:
            raise ValueError("timestep_features must be even and at least 4")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        downsample_factor = 2 ** (len(self.down_dims) - 1)
        if self.prediction_horizon % downsample_factor:
            raise ValueError(
                "prediction_horizon must be divisible by the network downsample factor"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DiffusionPolicy:
    """Predict action chunks with a conditional temporal U-Net and DDPM.

    TODO: the observation/action contract (15D command-derived state, 7D twist
    plus grasp) is provisional and may change once real demonstrations exist.

    TODO: actions are required to lie in [-1, 1] but the collector stores raw
    twists in m/s and rad/s. A dataset normalizer (per-dimension bounds saved
    in the checkpoint) is needed before training on collected data.
    """

    def __init__(self, config: DiffusionPolicyConfig | None = None) -> None:
        self.config = config or DiffusionPolicyConfig()
        self.device = _resolve_device(self.config.device)
        self.model = DiffusionNetwork(
            observation_horizon=self.config.observation_horizon,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            vision_features=self.config.vision_features,
            down_dims=self.config.down_dims,
            timestep_dim=self.config.timestep_features,
            kernel_size=self.config.kernel_size,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config.diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )
        self._images: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._states: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._actions: deque[np.ndarray] = deque()

    def reset(self) -> None:
        """Clear observation and action history at an episode boundary."""

        self._images.clear()
        self._states.clear()
        self._actions.clear()

    def train_step(self, batch: Batch) -> Metrics:
        """Perform one epsilon-prediction update and return scalar metrics."""

        images, states = self._prepare_observations(batch.observations)
        actions = self._prepare_actions(batch.actions, images.shape[0])
        self.model.train()
        condition = self.model.encode_observation(images, states)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.config.diffusion_steps,
            (actions.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        predicted_noise = self.model(noisy_actions, timesteps, condition)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }

    @torch.no_grad()
    def act(self, observation: Observation) -> np.ndarray:
        """Return one action while replanning in configurable action chunks."""

        image, state = self._validate_current_observation(observation)
        self._images.append(image)
        self._states.append(state)
        if not self._actions:
            images = list(self._images)
            states = list(self._states)
            while len(images) < self.config.observation_horizon:
                images.insert(0, images[0])
                states.insert(0, states[0])
            prepared_images, prepared_states = self._prepare_observations(
                {
                    "camera_rgb": np.stack(images)[None],
                    "state": np.stack(states)[None],
                }
            )
            self._actions.extend(self._sample(prepared_images, prepared_states))
        return self._actions.popleft().copy()

    def save(self, path: str | Path) -> None:
        """Atomically save model, optimizer, and configuration."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        payload = {
            "version": _CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        try:
            torch.save(payload, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> DiffusionPolicy:
        """Restore a checkpoint, optionally overriding its execution device."""

        resolved_device = _resolve_device(device or "auto")
        payload: dict[str, Any] = torch.load(
            Path(path), map_location=resolved_device, weights_only=True
        )
        if payload.get("version") != _CHECKPOINT_VERSION:
            raise ValueError("unsupported diffusion-policy checkpoint version")
        raw_config = dict(payload["config"])
        raw_config["down_dims"] = tuple(raw_config["down_dims"])
        config = DiffusionPolicyConfig(**raw_config)
        config = replace(config, device=str(resolved_device))
        policy = cls(config)
        policy.model.load_state_dict(payload["model"])
        policy.optimizer.load_state_dict(payload["optimizer"])
        return policy

    def _prepare_observations(
        self, observations: dict[str, Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = {"camera_rgb", "state"}.difference(observations)
        if missing:
            raise KeyError(f"observations are missing keys: {sorted(missing)}")

        images = torch.as_tensor(observations["camera_rgb"], device=self.device)
        if images.ndim != 5 or images.shape[1] != self.config.observation_horizon:
            raise ValueError(
                "camera_rgb must have shape [batch, observation_horizon, C, H, W] "
                "or [batch, observation_horizon, H, W, C]"
            )
        if images.shape[0] == 0:
            raise ValueError("training batches cannot be empty")
        expected_prefix = (images.shape[0], self.config.observation_horizon)
        if images.shape[2] == 3:
            pass
        elif images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError("camera_rgb must contain exactly three RGB channels")
        if min(images.shape[-2:]) < 8:
            raise ValueError("camera_rgb height and width must be at least 8 pixels")
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

        states = torch.as_tensor(
            observations["state"], dtype=torch.float32, device=self.device
        )
        expected_state_shape = (*expected_prefix, STATE_DIM)
        if states.shape != expected_state_shape:
            raise ValueError(
                f"state shape is {tuple(states.shape)}, expected {expected_state_shape}"
            )
        if not torch.isfinite(states).all():
            raise ValueError("state values must be finite")
        return images.contiguous(), states

    def _prepare_actions(self, value: Tensor, batch_size: int) -> torch.Tensor:
        actions = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        expected_shape = (batch_size, self.config.prediction_horizon, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(
                f"action shape is {tuple(actions.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("action values must be finite")
        if actions.min() < -1 or actions.max() > 1:
            raise ValueError("action values must be in [-1, 1]")
        return actions

    @staticmethod
    def _validate_current_observation(
        observation: Observation,
    ) -> tuple[np.ndarray, np.ndarray]:
        missing = {"camera_rgb", "state"}.difference(observation)
        if missing:
            raise KeyError(f"observation is missing keys: {sorted(missing)}")
        image = np.asarray(observation["camera_rgb"])
        state = np.asarray(observation["state"], dtype=np.float32)
        if image.ndim != 3 or 3 not in (image.shape[0], image.shape[-1]):
            raise ValueError("camera_rgb must be a single CHW or HWC RGB image")
        height, width = image.shape[1:] if image.shape[0] == 3 else image.shape[:2]
        if min(height, width) < 8:
            raise ValueError("camera_rgb height and width must be at least 8 pixels")
        if image.dtype == np.uint8:
            pass
        elif np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(
                    "floating-point camera_rgb values must be finite in [0, 1]"
                )
        else:
            raise TypeError("camera_rgb must be uint8 or floating point")
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state shape is {state.shape}, expected ({STATE_DIM},)")
        if not np.isfinite(state).all():
            raise ValueError("state values must be finite")
        return image.copy(), state.copy()

    def _sample(self, images: torch.Tensor, states: torch.Tensor) -> list[np.ndarray]:
        was_training = self.model.training
        self.model.eval()
        try:
            condition = self.model.encode_observation(images, states)
            sample = torch.randn(
                (images.shape[0], self.config.prediction_horizon, ACTION_DIM),
                device=self.device,
            )
            self.scheduler.set_timesteps(
                self.config.inference_steps, device=self.device
            )
            for timestep in self.scheduler.timesteps:
                predicted_noise = self.model(sample, timestep, condition)
                sample = self.scheduler.step(
                    predicted_noise, timestep, sample
                ).prev_sample
        finally:
            self.model.train(was_training)

        start = self.config.observation_horizon - 1
        stop = start + self.config.action_horizon
        actions = np.clip(sample[0, start:stop].cpu().numpy(), -1, 1).astype(np.float32)
        actions[:, GRASP_INDEX] = (actions[:, GRASP_INDEX] >= 0.5).astype(np.float32)
        return list(actions)
