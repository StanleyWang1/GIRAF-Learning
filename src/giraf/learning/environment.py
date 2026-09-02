"""Environment boundary shared by simulation and robot rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

type Observation = dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@runtime_checkable
class Environment(Protocol):
    """Small Gym-compatible contract required by the learning pipeline."""

    def reset(self, *, seed: int | None = None) -> Observation: ...

    def step(self, action: np.ndarray) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class GymEnvironment(Protocol):
    """Subset of the Gym/Gymnasium API accepted by SimEnvironment."""

    def reset(self, *, seed: int | None = None) -> Any: ...

    def step(self, action: np.ndarray) -> tuple[Any, ...]: ...

    def close(self) -> None: ...


class SimEnvironment:
    """Adapt a Gym-style simulator to the Environment contract.

    Accepts both the five-value Gymnasium step result and the legacy four-value
    Gym result. Observations must be mappings with ``camera_rgb`` and ``state``.

    TODO: the project's MuJoCo backend is not in this repository yet, so this
    adapter has only been checked against the contract, not a real simulator.
    """

    def __init__(self, backend: GymEnvironment) -> None:
        self.backend = backend

    @staticmethod
    def _observation(value: Any) -> Observation:
        if not isinstance(value, Mapping):
            raise TypeError("simulator observations must be mappings")
        observation = {str(key): np.asarray(item) for key, item in value.items()}
        missing = {"camera_rgb", "state"}.difference(observation)
        if missing:
            raise KeyError(f"simulator observation is missing keys: {sorted(missing)}")
        return observation

    def reset(self, *, seed: int | None = None) -> Observation:
        result = self.backend.reset() if seed is None else self.backend.reset(seed=seed)
        observation = (
            result[0] if isinstance(result, tuple) and len(result) == 2 else result
        )
        return self._observation(observation)

    def step(self, action: np.ndarray) -> StepResult:
        result = self.backend.step(np.asarray(action, dtype=np.float32))
        if not isinstance(result, tuple):
            raise TypeError("simulator step() must return a tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, terminated, info = result
            truncated = False
        else:
            raise ValueError("simulator step() must return four or five values")
        return StepResult(
            observation=self._observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def close(self) -> None:
        self.backend.close()
