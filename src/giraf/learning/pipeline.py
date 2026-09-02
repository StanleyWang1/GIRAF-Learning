"""Framework-independent learning loops."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .environment import Environment
from .policy import Batch, Metrics, Policy


def train(policy: Policy, batches: Iterable[Batch], *, epochs: int) -> list[Metrics]:
    """Run a minimal training loop and return step metrics.

    TODO: no Zarr -> Batch windowing loader exists yet; callers must build
    ``batches`` themselves from the collected ReplayBuffer.
    TODO: experiment tracking (wandb), periodic checkpointing, and evaluation
    rollouts belong in a training script around this loop; none exist yet.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if epochs > 1 and iter(batches) is batches:
        raise ValueError("multiple epochs require a re-iterable batch source")
    history: list[Metrics] = []
    for _ in range(epochs):
        for batch in batches:
            history.append(policy.train_step(batch))
    return history


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
