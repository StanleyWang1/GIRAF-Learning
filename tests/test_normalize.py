from __future__ import annotations

import unittest

import numpy as np
import torch

from giraf.learning.normalize import Normalizer


class NormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.actions = rng.uniform(-0.5, 0.5, size=(50, 7)).astype(np.float32)
        self.actions[:, 6] = 0.0  # grasp never pressed in this dataset
        self.actions[:, 2] = 0.3  # constant vz
        self.states = rng.uniform(-2.0, 3.0, size=(50, 15)).astype(np.float32)
        self.normalizer = Normalizer.fit(self.actions, self.states)

    def test_fit_maps_data_into_unit_interval_and_round_trips(self) -> None:
        normalized = self.normalizer.normalize_actions(self.actions)
        self.assertLessEqual(float(np.abs(normalized).max()), 1.0 + 1e-6)
        np.testing.assert_allclose(
            self.normalizer.denormalize_actions(normalized), self.actions, atol=1e-5
        )
        states = self.normalizer.normalize_states(self.states)
        self.assertLessEqual(float(np.abs(states).max()), 1.0 + 1e-6)

    def test_grasp_bounds_are_fixed_regardless_of_data(self) -> None:
        self.assertEqual(float(self.normalizer.action_low[6]), 0.0)
        self.assertEqual(float(self.normalizer.action_high[6]), 1.0)
        grasp = np.zeros((2, 7), np.float32)
        grasp[1, 6] = 1.0
        normalized = self.normalizer.normalize_actions(grasp)
        np.testing.assert_allclose(normalized[:, 6], [-1.0, 1.0])

    def test_constant_dimension_maps_to_zero_and_stays_bounded(self) -> None:
        normalized = self.normalizer.normalize_actions(self.actions)
        np.testing.assert_allclose(normalized[:, 2], 0.0, atol=1e-6)
        drifted = self.actions[:1].copy()
        drifted[0, 2] = 0.8
        self.assertAlmostEqual(
            float(self.normalizer.normalize_actions(drifted)[0, 2]), 0.5, places=5
        )

    def test_torch_tensors_keep_device_and_dtype(self) -> None:
        actions = torch.as_tensor(self.actions[:4])
        normalized = self.normalizer.normalize_actions(actions)
        self.assertIsInstance(normalized, torch.Tensor)
        self.assertEqual(normalized.dtype, torch.float32)
        torch.testing.assert_close(
            self.normalizer.denormalize_actions(normalized), actions, atol=1e-5, rtol=0
        )

    def test_dict_round_trip(self) -> None:
        restored = Normalizer.from_dict(self.normalizer.to_dict())
        np.testing.assert_array_equal(restored.action_low, self.normalizer.action_low)
        np.testing.assert_array_equal(restored.state_high, self.normalizer.state_high)

    def test_rejects_wrong_shapes(self) -> None:
        with self.assertRaises(ValueError):
            Normalizer.fit(self.actions[:, :6], self.states)
        with self.assertRaises(ValueError):
            Normalizer(np.zeros(7), np.zeros(7), np.zeros(14), np.zeros(14))


if __name__ == "__main__":
    unittest.main()
