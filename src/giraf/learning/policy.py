"""Policy and training-batch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .environment import Observation

type Metrics = dict[str, float]
type Tensor = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class Batch:
    """One batch of observation histories and target action sequences."""

    observations: dict[str, Tensor]
    actions: Tensor


@runtime_checkable
class Policy(Protocol):
    """Contract used by training and rollout code."""

    def act(self, observation: Observation) -> np.ndarray: ...

    def train_step(self, batch: Batch) -> Metrics: ...

    def save(self, path: str | Path) -> None: ...
