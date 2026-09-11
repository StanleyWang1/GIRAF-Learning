"""Per-dimension affine normalization fitted on a dataset and stored in checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from giraf.data.schema import ACTION_DIM, GRASP_INDEX, STATE_DIM

CONSTANT_EPS = 1e-6


def _affine(value, center: np.ndarray, half_range: np.ndarray, *, inverse: bool):
    if isinstance(value, torch.Tensor):
        center = torch.as_tensor(center, dtype=value.dtype, device=value.device)
        half_range = torch.as_tensor(half_range, dtype=value.dtype, device=value.device)
    if inverse:
        return value * half_range + center
    return (value - center) / half_range


@dataclass(frozen=True)
class Normalizer:
    """Map raw actions and states to [-1, 1] using per-dimension bounds.

    Grasp is forced to bounds (0, 1) so {0, 1} maps to {-1, 1} and the policy's
    0.5 threshold after denormalization keeps working. Constant dimensions get a
    unit half-range so they map to 0 and stay bounded if inference drifts.
    """

    action_low: np.ndarray
    action_high: np.ndarray
    state_low: np.ndarray
    state_high: np.ndarray
    imu_low: np.ndarray | None = None
    imu_high: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name, dim in (("action", ACTION_DIM), ("state", STATE_DIM)):
            for bound in ("low", "high"):
                value = np.asarray(getattr(self, f"{name}_{bound}"), dtype=np.float32)
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f"{name}_{bound} must be a finite vector of {dim}")
                object.__setattr__(self, f"{name}_{bound}", value)
        if (self.action_high < self.action_low).any() or (
            self.state_high < self.state_low
        ).any():
            raise ValueError("normalizer bounds must satisfy low <= high")
        if self.imu_low is not None or self.imu_high is not None:
            low = np.asarray(self.imu_low, dtype=np.float32)
            high = np.asarray(self.imu_high, dtype=np.float32)
            if (
                low.shape not in ((6,), (10,))
                or high.shape != low.shape
                or not np.isfinite(low).all()
                or not np.isfinite(high).all()
                or (high < low).any()
            ):
                raise ValueError("IMU bounds must be finite ordered vectors of 6 or 10")
            object.__setattr__(self, "imu_low", low)
            object.__setattr__(self, "imu_high", high)

    @classmethod
    def fit(
        cls, actions: np.ndarray, states: np.ndarray, imu: np.ndarray | None = None
    ) -> Normalizer:
        actions = np.asarray(actions, dtype=np.float32)
        states = np.asarray(states, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
            raise ValueError(f"actions must have shape [N, {ACTION_DIM}] with N > 0")
        if states.ndim != 2 or states.shape[1] != STATE_DIM or states.shape[0] == 0:
            raise ValueError(f"states must have shape [N, {STATE_DIM}] with N > 0")
        action_low, action_high = actions.min(axis=0), actions.max(axis=0)
        action_low[GRASP_INDEX], action_high[GRASP_INDEX] = 0.0, 1.0
        imu_low = imu_high = None
        if imu is not None:
            imu = np.asarray(imu, dtype=np.float32)
            if (
                imu.ndim != 2
                or imu.shape[1] not in (6, 10)
                or not len(imu)
                or not np.isfinite(imu).all()
            ):
                raise ValueError("IMU must be finite [N, 6] or [N, 10] with N > 0")
            imu_low, imu_high = imu.min(axis=0), imu.max(axis=0)
            # Quaternion components already have a physical [-1, 1] scale.
            if imu.shape[1] == 10:
                imu_low[6:], imu_high[6:] = -1.0, 1.0
        return cls(
            action_low, action_high, states.min(axis=0), states.max(axis=0),
            imu_low, imu_high,
        )

    @staticmethod
    def _center_half(
        low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        center = (low + high) / 2
        half_range = (high - low) / 2
        half_range = np.where(half_range < CONSTANT_EPS, 1.0, half_range)
        return center.astype(np.float32), half_range.astype(np.float32)

    def normalize_actions(self, actions):
        center, half = self._center_half(self.action_low, self.action_high)
        return _affine(actions, center, half, inverse=False)

    def denormalize_actions(self, actions):
        center, half = self._center_half(self.action_low, self.action_high)
        return _affine(actions, center, half, inverse=True)

    def normalize_states(self, states):
        center, half = self._center_half(self.state_low, self.state_high)
        return _affine(states, center, half, inverse=False)

    def normalize_imu(self, imu):
        if self.imu_low is None:
            raise ValueError("normalizer has no IMU bounds")
        center, half = self._center_half(self.imu_low, self.imu_high)
        return _affine(imu, center, half, inverse=False)

    def to_dict(self) -> dict[str, list[float]]:
        payload = {
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "state_low": self.state_low.tolist(),
            "state_high": self.state_high.tolist(),
        }
        if self.imu_low is not None:
            payload.update(
                imu_low=self.imu_low.tolist(), imu_high=self.imu_high.tolist()
            )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> Normalizer:
        return cls(
            np.asarray(payload["action_low"], dtype=np.float32),
            np.asarray(payload["action_high"], dtype=np.float32),
            np.asarray(payload["state_low"], dtype=np.float32),
            np.asarray(payload["state_high"], dtype=np.float32),
            payload.get("imu_low"),
            payload.get("imu_high"),
        )
