from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from giraf.learning.dataset import ReplayDataset, episode_windows


def write_dataset(path: Path, episode_ends=(3, 4, 10), image_size=(6, 8)) -> Path:
    n_steps = int(episode_ends[-1])
    root = zarr.open_group(str(path), mode="w")
    root.attrs["schema_version"] = "giraf-replay-v1"
    data = root.require_group("data")
    meta = root.require_group("meta")
    camera = np.zeros((n_steps, *image_size, 3), dtype=np.uint8)
    camera[:, 0, 0, 0] = np.arange(n_steps)  # frame index stamped in a pixel
    data.array("camera_rgb", camera, chunks=(2, *image_size, 3))
    action = np.tile(np.arange(n_steps, dtype=np.float32)[:, None], (1, 7))
    action[:, 6] = np.arange(n_steps) % 2
    data.array("action", action, chunks=(4, 7))
    data.array(
        "state",
        np.tile(np.arange(n_steps, dtype=np.float32)[:, None], (1, 15)),
        chunks=(4, 15),
    )
    valid = np.ones(n_steps, dtype=np.uint8)
    valid[5] = 0
    data.array("alignment_valid", valid, chunks=(4,))
    meta.array("episode_ends", np.asarray(episode_ends, dtype=np.int64), chunks=(4,))
    return path


class EpisodeWindowTests(unittest.TestCase):
    def test_padding_repeats_first_observation_and_last_action(self) -> None:
        obs_idx, act_idx = episode_windows(
            np.array([3]), None, observation_horizon=2, prediction_horizon=4
        )
        np.testing.assert_array_equal(obs_idx, [[0, 0], [0, 1], [1, 2]])
        np.testing.assert_array_equal(
            act_idx, [[0, 0, 1, 2], [0, 1, 2, 2], [1, 2, 2, 2]]
        )

    def test_windows_never_cross_episode_boundaries(self) -> None:
        obs_idx, act_idx = episode_windows(
            np.array([3, 4]), None, observation_horizon=2, prediction_horizon=3
        )
        # Episode 1 is the single step 3.
        np.testing.assert_array_equal(obs_idx[-1], [3, 3])
        np.testing.assert_array_equal(act_idx[-1], [3, 3, 3])
        self.assertTrue((act_idx[:3] <= 2).all())

    def test_invalid_anchors_are_dropped(self) -> None:
        valid = np.array([1, 0, 1])
        obs_idx, _ = episode_windows(
            np.array([3]), valid, observation_horizon=2, prediction_horizon=2
        )
        np.testing.assert_array_equal(obs_idx[:, -1], [0, 2])

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            episode_windows(
                np.array([3, 3]), None, observation_horizon=2, prediction_horizon=4
            )
        with self.assertRaises(ValueError):
            episode_windows(
                np.array([3]), None, observation_horizon=4, prediction_horizon=2
            )


class ReplayDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = write_dataset(Path(self._tmp.name) / "buffer.zarr")

    def test_batches_have_contract_shapes_and_aligned_content(self) -> None:
        dataset = ReplayDataset(
            self.path,
            batch_size=4,
            observation_horizon=2,
            prediction_horizon=4,
            shuffle=False,
        )
        self.assertEqual(dataset.n_windows, 9)  # 10 steps minus one invalid anchor
        self.assertEqual(len(dataset), 3)
        batches = list(dataset)
        first = batches[0]
        self.assertEqual(first.observations["camera_rgb"].shape, (4, 2, 6, 8, 3))
        self.assertEqual(first.observations["camera_rgb"].dtype, np.uint8)
        self.assertEqual(first.observations["state"].shape, (4, 2, 15))
        self.assertEqual(first.actions.shape, (4, 4, 7))
        self.assertEqual(first.actions.dtype, np.float32)
        # Frame index stamped in the image must match the state value for that step.
        stamped = first.observations["camera_rgb"][:, :, 0, 0, 0]
        np.testing.assert_array_equal(stamped, first.observations["state"][:, :, 0])
        self.assertEqual(sum(batch.actions.shape[0] for batch in batches), 9)

    def test_shuffle_is_seeded_and_changes_per_epoch(self) -> None:
        a = ReplayDataset(self.path, batch_size=9, prediction_horizon=4, seed=3)
        b = ReplayDataset(self.path, batch_size=9, prediction_horizon=4, seed=3)
        first_a, first_b = next(iter(a)), next(iter(b))
        np.testing.assert_array_equal(first_a.actions, first_b.actions)
        second_a = next(iter(a))
        self.assertFalse(np.array_equal(first_a.actions, second_a.actions))

        resumed = ReplayDataset(
            self.path,
            batch_size=9,
            prediction_horizon=4,
            seed=3,
            start_epoch=1,
        )
        np.testing.assert_array_equal(second_a.actions, next(iter(resumed)).actions)

    def test_preloaded_images_match_zarr_reads(self) -> None:
        streamed = ReplayDataset(
            self.path, batch_size=9, prediction_horizon=4, shuffle=False
        )
        preloaded = ReplayDataset(
            self.path,
            batch_size=9,
            prediction_horizon=4,
            shuffle=False,
            preload_images=True,
        )
        np.testing.assert_array_equal(
            next(iter(streamed)).observations["camera_rgb"],
            next(iter(preloaded)).observations["camera_rgb"],
        )

    def test_normalizer_is_fitted_on_full_dataset(self) -> None:
        dataset = ReplayDataset(self.path, batch_size=4, prediction_horizon=4)
        normalizer = dataset.fit_normalizer()
        self.assertEqual(float(normalizer.action_low[0]), 0.0)
        self.assertEqual(float(normalizer.action_high[0]), 9.0)


if __name__ == "__main__":
    unittest.main()
