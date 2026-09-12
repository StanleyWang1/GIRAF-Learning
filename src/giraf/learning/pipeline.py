"""Framework-independent learning loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .policy import Batch, Metrics, Policy


def train(
    policy: Policy,
    batches: Iterable[Batch],
    *,
    epochs: int,
    on_step: Callable[[Metrics], None] | None = None,
) -> list[Metrics]:
    """Run a minimal training loop and return step metrics.

    ``giraf.learning.train_cli`` wraps this with checkpoints and logging.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if epochs > 1 and iter(batches) is batches:
        raise ValueError("multiple epochs require a re-iterable batch source")
    history: list[Metrics] = []
    for _ in range(epochs):
        for batch in batches:
            metrics = policy.train_step(batch)
            history.append(metrics)
            if on_step is not None:
                on_step(metrics)
    return history


def mean_metrics(history: list[Metrics], weights: list[float] | None = None) -> Metrics:
    """Return each metric's mean over history, optionally weighted per entry."""

    if not history:
        raise ValueError("mean_metrics requires at least one entry")
    keys = history[0].keys()
    if weights is None:
        return {key: float(np.mean([step[key] for step in history])) for key in keys}
    return {
        key: float(np.average([step[key] for step in history], weights=weights))
        for key in keys
    }


def evaluate(policy: Policy, batches: Iterable[Batch]) -> Metrics:
    """Run ``policy.evaluate`` over batches and return the batch-size-weighted mean."""

    evaluate_batch = getattr(policy, "evaluate", None)
    if not callable(evaluate_batch):
        raise TypeError(f"{type(policy).__name__} does not implement evaluate")
    history: list[Metrics] = []
    weights: list[int] = []
    for batch in batches:
        history.append(evaluate_batch(batch))
        weights.append(len(batch.actions))
    if not history:
        raise ValueError("evaluate requires at least one batch")
    return mean_metrics(history, weights=weights)
