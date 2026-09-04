from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from giraf.learning import DiffusionPolicy
from giraf.learning.train_cli import main, parse_config
from tests.test_dataset import write_dataset


class TrainCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.dataset = write_dataset(
            root / "buffer.zarr", episode_ends=(12, 24), image_size=(8, 8)
        )
        self.run_dir = root / "run"
        self.args = [
            "--dataset",
            str(self.dataset),
            "--output-dir",
            str(self.run_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--checkpoint-every",
            "2",
            "--device",
            "cpu",
            "--down-dims",
            "8",
            "16",
            "--diffusion-steps",
            "4",
        ]

    def test_parse_config_applies_model_overrides(self) -> None:
        config = parse_config(self.args)
        self.assertEqual(config.policy.down_dims, (8, 16))
        self.assertEqual(config.policy.diffusion_steps, 4)
        self.assertEqual(config.policy.inference_steps, 4)
        self.assertEqual(config.epochs, 2)

    def test_training_writes_checkpoints_metrics_and_normalizer(self) -> None:
        with patch(
            "giraf.learning.diffusion.nn.utils.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip_grad_norm:
            self.assertEqual(main(self.args), 0)
        self.assertTrue(clip_grad_norm.called)
        self.assertTrue(
            all(call.kwargs["foreach"] is False for call in clip_grad_norm.call_args_list)
        )
        self.assertTrue((self.run_dir / "policy.pt").is_file())
        self.assertTrue((self.run_dir / "policy_epoch_0002.pt").is_file())
        self.assertFalse((self.run_dir / "policy_epoch_0001.pt").exists())
        self.assertTrue((self.run_dir / "config.json").is_file())
        records = [
            json.loads(line)
            for line in (self.run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([record["epoch"] for record in records], [1, 2])
        self.assertIn("loss", records[0])

        policy = DiffusionPolicy.load(self.run_dir / "policy.pt", device="cpu")
        saved = json.loads((self.run_dir / "normalizer.json").read_text())
        self.assertEqual(policy.normalizer.to_dict(), saved)

        # act() returns physical units: twist within the fitted bounds, binary grasp.
        action = policy.act(
            {
                "camera_rgb": np.zeros((8, 8, 3), np.uint8),
                "state": np.zeros(15, np.float32),
            }
        )
        self.assertEqual(action.shape, (7,))
        self.assertTrue((action[:6] >= policy.normalizer.action_low[:6] - 1e-5).all())
        self.assertTrue((action[:6] <= policy.normalizer.action_high[:6] + 1e-5).all())
        self.assertIn(float(action[6]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
