"""State selection and checkpoint compatibility using a small CPU policy."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from giraf.learning.diffusion import DiffusionPolicy, DiffusionPolicyConfig
from giraf.learning.normalize import Normalizer
from giraf.learning.policy import Batch
from giraf.learning.train_cli import _ignored_resume_flags, parse_config


class StateInputTests(unittest.TestCase):
    def setUp(self):
        self.config = DiffusionPolicyConfig(
            device="cpu", prediction_horizon=4, action_horizon=2,
            down_dims=(8, 16), vision_features=8, timestep_features=8,
            diffusion_steps=4, inference_steps=2, crop_fraction=1.0, color_jitter=0.0,
        )
        self.normalizer = Normalizer(
            -np.ones(7), np.ones(7), np.arange(15), np.arange(15) + 4,
        )
        self.observations = {
            "camera_rgb": np.zeros((2, 2, 16, 16, 3), dtype=np.uint8),
            "state": np.broadcast_to(
                np.arange(15) + np.linspace(0.1, 3.9, 15), (2, 2, 15)
            ).astype(np.float32).copy(),
        }
        self.batch = Batch(self.observations, np.zeros((2, 4, 7), np.float32))

    def test_cli_defaults_choices_and_resume_warning(self):
        default = parse_config(["--dataset", "unused.zarr"]).policy
        self.assertEqual(default.state_input, "full")
        selected = parse_config([
            "--dataset", "unused.zarr", "--state-input", "joint_angles",
        ]).policy
        self.assertEqual(selected.state_input, "joint_angles")
        self.assertEqual(_ignored_resume_flags(selected, default), ["state_input"])
        with self.assertRaisesRegex(ValueError, "state_input"):
            replace(self.config, state_input="unsupported")

    def test_preprocessing_keeps_selected_normalized_fields(self):
        normalized = self.normalizer.normalize_states(self.observations["state"])
        original = self.observations["state"].copy()
        for mode, indices in (
            ("full", list(range(15))), ("joint_angles", [0, 1, 3, 4, 5]),
        ):
            policy = DiffusionPolicy(
                replace(self.config, state_input=mode), normalizer=self.normalizer,
            )
            for augment in (False, True):
                with self.subTest(mode=mode, augment=augment):
                    _, states = policy._prepare_observations(
                        self.observations, augment=augment,
                    )
                    np.testing.assert_allclose(states.numpy(), normalized[..., indices])
                    self.assertEqual(policy.model.state_dim, len(indices))
            np.testing.assert_array_equal(self.observations["state"], original)

    def test_excluded_values_do_not_affect_evaluation(self):
        policy = DiffusionPolicy(
            replace(self.config, state_input="joint_angles"),
            normalizer=self.normalizer,
        )
        expected = policy.evaluate(self.batch)
        changed = self.observations["state"].copy()
        changed[..., [2, *range(6, 15)]] += 100
        altered = Batch({**self.observations, "state": changed}, self.batch.actions)
        self.assertEqual(policy.evaluate(altered), expected)

    def test_training_checkpoint_and_inference_both_modes(self):
        for mode in ("full", "joint_angles"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                policy = DiffusionPolicy(
                    replace(self.config, state_input=mode), normalizer=self.normalizer,
                )
                metrics = policy.train_step(self.batch)
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                path = Path(directory) / "policy.pt"
                policy.save(path)
                restored = DiffusionPolicy.load(path, device="cpu")
                self.assertEqual(restored.config.state_input, mode)
                self.assertEqual(
                    restored.evaluate(self.batch), policy.evaluate(self.batch),
                )
                observation = {
                    key: value[0, 0] for key, value in self.observations.items()
                }
                torch.manual_seed(7)
                expected = policy.act(observation)
                torch.manual_seed(7)
                actual = restored.act(observation)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (7,))
                self.assertTrue(np.isfinite(actual).all())

    def test_legacy_checkpoint_defaults_to_full(self):
        policy = DiffusionPolicy(self.config, normalizer=self.normalizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            policy.save(path)
            payload = torch.load(path, weights_only=True)
            del payload["config"]["state_input"]
            torch.save(payload, path)
            restored = DiffusionPolicy.load(path, device="cpu")
            self.assertEqual(restored.config.state_input, "full")
            self.assertEqual(restored.model.state_dim, 15)
            self.assertEqual(restored.evaluate(self.batch), policy.evaluate(self.batch))


if __name__ == "__main__":
    unittest.main()
