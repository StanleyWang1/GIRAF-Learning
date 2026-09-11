"""Optional IMU conditioning, valid windows, and checkpoint compatibility."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import zarr

from giraf.learning.dataset import ReplayDataset
from giraf.learning.diffusion import DiffusionPolicy, DiffusionPolicyConfig
from giraf.learning.normalize import Normalizer
from giraf.learning.policy import Batch
from giraf.learning.train_cli import (
    TrainConfig,
    _ignored_resume_flags,
    parse_config,
    run,
)


class ImuInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "replay.zarr"
        self.root = zarr.open_group(str(self.path), mode="w")
        self.root.create_dataset(
            "data/camera_rgb", data=np.zeros((12, 16, 16, 3), np.uint8)
        )
        self.root.create_dataset("data/state", data=np.zeros((12, 15), np.float32))
        self.root.create_dataset("data/action", data=np.zeros((12, 7), np.float32))
        self.root.create_dataset("data/alignment_valid", data=np.ones(12, np.uint8))
        self.root.create_dataset("meta/episode_ends", data=[6, 12])
        imu = np.zeros((12, 10), np.float32)
        imu[:, :6] = np.arange(12)[:, None]
        imu[:, 9] = 1
        imu[2, :6] = 1e6  # Invalid training row must not affect scaling.
        imu[4, 6:] = np.nan  # Irrelevant to accel_gyro.
        imu[6:, :6] += 100  # Validation split must not affect scaling.
        flags = np.ones((12, 3), np.uint8)
        flags[2, 1] = 0
        flags[4, 2] = 0
        self.root.create_dataset("data/imu", data=imu)
        self.root.create_dataset("data/imu_sensor_valid", data=flags)
        self.config = DiffusionPolicyConfig(
            device="cpu", prediction_horizon=4, action_horizon=2,
            down_dims=(8, 16), vision_features=8, timestep_features=8,
            diffusion_steps=4, inference_steps=2, crop_fraction=1.0, color_jitter=0.0,
        )

    def dataset(self, mode="none", **kwargs):
        return ReplayDataset(
            self.path, batch_size=8, observation_horizon=2, prediction_horizon=4,
            shuffle=False, imu_input=mode, **kwargs,
        )

    def test_cli_modes_and_legacy_checkpoint(self):
        default = parse_config(["--dataset", "unused.zarr"]).policy
        self.assertEqual(default.imu_input, "none")
        for mode in ("accel_gyro", "full"):
            config = parse_config([
                "--dataset", "unused.zarr", "--imu-input", mode,
            ]).policy
            self.assertEqual(config.imu_input, mode)
            self.assertEqual(_ignored_resume_flags(config, default), ["imu_input"])
        with self.assertRaisesRegex(ValueError, "imu_input"):
            replace(default, imu_input="bad")
        dataset = self.dataset()
        policy = DiffusionPolicy(self.config, normalizer=dataset.fit_normalizer())
        batch = next(iter(dataset))
        path = self.directory / "legacy.pt"
        policy.save(path)
        payload = torch.load(path, weights_only=True)
        del payload["config"]["imu_input"]
        self.assertNotIn("imu_low", payload["normalizer"])
        torch.save(payload, path)
        restored = DiffusionPolicy.load(path, device="cpu")
        self.assertEqual(restored.config.imu_input, "none")
        self.assertEqual(restored.evaluate(batch), policy.evaluate(batch))

    def test_windows_and_training_only_scaling(self):
        for mode, anchors in (
            ("none", [0, 1, 2, 3, 4, 5]),
            ("accel_gyro", [0, 1, 4, 5]),
            ("full", [0, 1]),
        ):
            with self.subTest(mode=mode):
                dataset = self.dataset(mode, episodes=[0])
                np.testing.assert_array_equal(dataset.obs_idx[:, -1], anchors)
                batch = next(iter(dataset))
                normalizer = dataset.fit_normalizer()
                if mode == "none":
                    self.assertNotIn("imu", batch.observations)
                    self.assertIsNone(normalizer.imu_low)
                    continue
                np.testing.assert_array_equal(
                    dataset.obs_idx, [[max(0, a - 1), a] for a in anchors]
                )
                # Future invalid IMU does not discard an otherwise valid window.
                self.assertIn(2, dataset.act_idx[0])
                self.assertEqual(batch.observations["imu"].shape[-1], 10)
                np.testing.assert_array_equal(normalizer.imu_low[:6], np.zeros(6))
                np.testing.assert_array_equal(
                    normalizer.imu_high[:6], np.full(6, anchors[-1])
                )
                restored = Normalizer.from_dict(normalizer.to_dict())
                self.assertEqual(restored.to_dict(), normalizer.to_dict())
                if mode == "full":
                    np.testing.assert_array_equal(normalizer.imu_low[6:], -np.ones(4))
                    np.testing.assert_array_equal(normalizer.imu_high[6:], np.ones(4))

    def test_missing_invalid_and_unused_imu(self):
        self.root["data/imu_sensor_valid"][:] = 0
        self.assertEqual(self.dataset().n_windows, 12)
        with self.assertRaisesRegex(ValueError, "no usable training windows"):
            self.dataset("full")
        del self.root["data/imu"]
        self.assertEqual(self.dataset().n_windows, 12)
        with self.assertRaisesRegex(ValueError, "requires data/imu"):
            self.dataset("accel_gyro")

    def test_policy_train_checkpoint_history_and_reset(self):
        for state_mode in ("full", "joint_angles"):
            for imu_mode, dim in (("none", 0), ("accel_gyro", 6), ("full", 10)):
                with self.subTest(state=state_mode, imu=imu_mode):
                    dataset = self.dataset(imu_mode, episodes=[0])
                    batch = next(iter(dataset))
                    config = replace(
                        self.config, state_input=state_mode, imu_input=imu_mode
                    )
                    policy = DiffusionPolicy(
                        config, normalizer=dataset.fit_normalizer()
                    )
                    state_dim = 15 if state_mode == "full" else 5
                    self.assertEqual(policy.model.state_dim, state_dim + dim)
                    metrics = policy.train_step(batch)
                    self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                    path = self.directory / "policy.pt"
                    policy.save(path)
                    restored = DiffusionPolicy.load(path, device="cpu")
                    self.assertEqual(restored.config.imu_input, imu_mode)
                    self.assertEqual(restored.evaluate(batch), policy.evaluate(batch))
                    observation = {k: v[0, 0] for k, v in batch.observations.items()}
                    for _ in range(2):
                        for step in range(4):
                            torch.manual_seed(step)
                            expected = policy.act(observation)
                            torch.manual_seed(step)
                            np.testing.assert_array_equal(
                                restored.act(observation), expected
                            )
                        if dim:
                            self.assertEqual(len(policy._imus), 2)
                        policy.reset()
                        restored.reset()
                        self.assertEqual(len(policy._imus), 0)

    def test_selection_validation_and_quaternion_scale(self):
        dataset = self.dataset("accel_gyro", episodes=[0])
        batch = next(iter(dataset))
        policy = DiffusionPolicy(
            replace(self.config, imu_input="accel_gyro"),
            normalizer=dataset.fit_normalizer(),
        )
        expected = policy.evaluate(batch)
        changed = batch.observations["imu"].copy()
        changed[..., 6:] = np.nan
        altered = Batch({**batch.observations, "imu": changed}, batch.actions)
        self.assertEqual(policy.evaluate(altered), expected)
        observation = {k: v[0, 0] for k, v in batch.observations.items()}
        del observation["imu"]
        with self.assertRaisesRegex(KeyError, "imu"):
            policy.act(observation)
        self.assertEqual(len(policy._images), 0)
        for values in (np.zeros(6), np.full(10, np.nan)):
            with self.assertRaises(ValueError):
                policy.act({**observation, "imu": values})
        full = self.dataset("full", episodes=[0])
        full_policy = DiffusionPolicy(
            replace(self.config, imu_input="full"), normalizer=full.fit_normalizer(),
        )
        full_batch = next(iter(full))
        _, features = full_policy._prepare_observations(
            full_batch.observations, augment=False
        )
        np.testing.assert_array_equal(
            features[..., -4:].numpy(), full_batch.observations["imu"][..., 6:]
        )
        with self.assertRaisesRegex(ValueError, "quaternion norm"):
            full_policy.act({**observation, "imu": np.zeros(10)})
        with self.assertRaisesRegex(ValueError, "bounds must match"):
            DiffusionPolicy(
                replace(self.config, imu_input="full"),
                normalizer=dataset.fit_normalizer(),
            )

    def test_training_cli_and_resume_use_checkpoint_mode(self):
        config = TrainConfig(
            dataset=self.path, output_dir=self.directory / "train", epochs=1,
            batch_size=8, checkpoint_every=1, val_fraction=0.5, warmup_steps=0,
            policy=replace(self.config, imu_input="full"),
        )
        checkpoint = run(config)
        # Default CLI mode must not disable IMU when resuming an IMU checkpoint.
        resumed = run(replace(
            config, output_dir=self.directory / "resume", resume=checkpoint,
            start_epoch=1, epochs=2, policy=self.config,
        ))
        restored = DiffusionPolicy.load(resumed, device="cpu")
        self.assertEqual(restored.config.imu_input, "full")
        self.assertEqual(len(restored.normalizer.imu_low), 10)


if __name__ == "__main__":
    unittest.main()
