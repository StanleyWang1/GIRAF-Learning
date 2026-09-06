"""Running sum/count buffer for temporally ensembling overlapping policy plans."""

from __future__ import annotations

from collections import deque

import numpy as np


class PlanBuffer:
    """Accumulate per-future-step predictions from overlapping plans for ensembling."""

    def __init__(self) -> None:
        self._sum: deque[np.ndarray] = deque()
        self._count: deque[int] = deque()
        self._steps_since_plan = 0

    def reset(self) -> None:
        """Clear all pending steps and the replan countdown."""

        self._sum.clear()
        self._count.clear()
        self._steps_since_plan = 0

    def ready_to_replan(self, action_horizon: int) -> bool:
        """Return True when the buffer is empty or action_horizon steps have elapsed."""

        return not self._sum or self._steps_since_plan >= action_horizon

    def add(self, plan: np.ndarray, *, ensemble: bool) -> None:
        """Merge plan into the buffer, or replace it outright when not ensembling."""

        if not ensemble:
            self._sum.clear()
            self._count.clear()
        for offset, prediction in enumerate(plan):
            if offset < len(self._sum):
                self._sum[offset] += prediction
                self._count[offset] += 1
            else:
                self._sum.append(prediction.copy())
                self._count.append(1)
        self._steps_since_plan = 0

    def pop(self) -> np.ndarray:
        """Return and remove the averaged prediction for the next step."""

        return self._sum.popleft() / self._count.popleft()

    def step(self) -> None:
        """Advance the replan countdown after executing one action."""

        self._steps_since_plan += 1
