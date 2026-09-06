"""Framework-independent learning loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .environment import Environment
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
    TODO: evaluation rollouts once a simulator backend exists.
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


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    reward: float
    steps: int
    terminated: bool
    truncated: bool


def rollout(
    policy: Policy,
    environment: Environment,
    *,
    max_steps: int,
    seed: int | None = None,
) -> RolloutSummary:
    """Evaluate a policy through the shared environment contract."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    observation = environment.reset(seed=seed)
    reward = 0.0
    terminated = truncated = False
    steps = 0
    for steps in range(1, max_steps + 1):
        result = environment.step(policy.act(observation))
        reward += float(result.reward)
        observation = result.observation
        terminated, truncated = result.terminated, result.truncated
        if terminated or truncated:
            break
    return RolloutSummary(reward, steps, terminated, truncated)
