from __future__ import annotations

import unittest

import numpy as np
import torch

from giraf.data.schema import ACTION_DIM, STATE_DIM
from giraf.learning.diffusion import DiffusionPolicy, DiffusionPolicyConfig
from giraf.learning.policy import Batch


def _tiny_batch(
    batch_size: int = 3, observation_horizon: int = 2, prediction_horizon: int = 16
) -> Batch:
    """Build a small in-range batch that needs no normalizer."""

    rng = np.random.default_rng(0)
    images = rng.integers(
        0, 256, size=(batch_size, observation_horizon, 8, 8, 3), dtype=np.uint8
    )
    states = rng.normal(size=(batch_size, observation_horizon, STATE_DIM)).astype(
        np.float32
    )
    actions = rng.uniform(
        -1, 1, size=(batch_size, prediction_horizon, ACTION_DIM)
    ).astype(np.float32)
    return Batch(observations={"camera_rgb": images, "state": states}, actions=actions)


class DiffusionPolicyEvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DiffusionPolicy(
            DiffusionPolicyConfig(
                down_dims=(8, 16),
                diffusion_steps=4,
                inference_steps=4,
                device="cpu",
                eval_seed=123,
                crop_fraction=1.0,
            )
        )
        self.batch = _tiny_batch()

    def test_evaluate_is_deterministic_across_calls(self) -> None:
        first = self.policy.evaluate(self.batch)
        second = self.policy.evaluate(self.batch)
        self.assertEqual(first, second)

    def test_evaluate_action_mse_is_finite_and_non_negative(self) -> None:
        metrics = self.policy.evaluate(self.batch)
        self.assertIn("action_mse", metrics)
        self.assertTrue(np.isfinite(metrics["action_mse"]))
        self.assertGreaterEqual(metrics["action_mse"], 0.0)

    def test_evaluate_does_not_change_model_parameters(self) -> None:
        before = [parameter.clone() for parameter in self.policy.model.parameters()]
        self.policy.evaluate(self.batch)
        after = list(self.policy.model.parameters())
        for old, new in zip(before, after, strict=True):
            torch.testing.assert_close(old, new)

    def test_evaluate_restores_prior_training_mode(self) -> None:
        self.policy.model.train()
        self.policy.evaluate(self.batch)
        self.assertTrue(self.policy.model.training)

        self.policy.model.eval()
        self.policy.evaluate(self.batch)
        self.assertFalse(self.policy.model.training)


if __name__ == "__main__":
    unittest.main()
