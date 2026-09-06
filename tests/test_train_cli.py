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
            "--val-fraction",
            "0",
        ]

    def test_parse_config_applies_model_overrides(self) -> None:
        config = parse_config(self.args)
        self.assertEqual(config.policy.down_dims, (8, 16))
        self.assertEqual(config.policy.diffusion_steps, 4)
        self.assertEqual(config.policy.inference_steps, 4)
        self.assertEqual(config.epochs, 2)

    def test_resume_requires_an_explicit_completed_epoch(self) -> None:
        with self.assertRaisesRegex(SystemExit, "positive --start-epoch"):
            parse_config([*self.args, "--resume", str(self.run_dir / "policy.pt")])

    def test_training_writes_checkpoints_metrics_and_normalizer(self) -> None:
        with patch(
            "giraf.learning.diffusion.nn.utils.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip_grad_norm:
            self.assertEqual(main(self.args), 0)
        self.assertTrue(clip_grad_norm.called)
        self.assertTrue(
            all(
                call.kwargs["foreach"] is False
                for call in clip_grad_norm.call_args_list
            )
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

    def test_training_resumes_existing_policy_and_optimizer(self) -> None:
        resumed_dir = Path(self._tmp.name) / "resumed-run"
        first_args = self.args.copy()
        first_args[first_args.index("--epochs") + 1] = "1"
        self.assertEqual(main(first_args), 0)

        resume_args = self.args.copy()
        resume_args[resume_args.index("--output-dir") + 1] = str(resumed_dir)
        resume_args.extend(
            [
                "--resume",
                str(self.run_dir / "policy.pt"),
                "--start-epoch",
                "1",
            ]
        )
        self.assertEqual(main(resume_args), 0)
        records = [
            json.loads(line)
            for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([record["epoch"] for record in records], [2])
        policy = DiffusionPolicy.load(resumed_dir / "policy.pt", device="cpu")
        optimizer_steps = {
            int(state["step"].item()) for state in policy.optimizer.state.values()
        }
        self.assertEqual(optimizer_steps, {6})

        same_dir_args = [
            *self.args,
            "--resume",
            str(self.run_dir / "policy.pt"),
            "--start-epoch",
            "1",
        ]
        with self.assertRaisesRegex(ValueError, "new --output-dir"):
            main(same_dir_args)

    def test_validation_split_reports_val_metrics_and_best_checkpoint(self) -> None:
        args = self.args.copy()
        args[args.index("--val-fraction") + 1] = "0.5"
        self.assertEqual(main(args), 0)

        records = [
            json.loads(line)
            for line in (self.run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all("val_loss" in record for record in records))
        self.assertTrue(all("val_action_mse" in record for record in records))
        self.assertTrue((self.run_dir / "best.pt").is_file())

        config = json.loads((self.run_dir / "config.json").read_text())
        self.assertTrue(config["train_episodes"])
        self.assertTrue(config["val_episodes"])
        self.assertTrue(
            set(config["train_episodes"]).isdisjoint(config["val_episodes"])
        )


if __name__ == "__main__":
    unittest.main()
