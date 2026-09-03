from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from giraf.data.prune import prune_replay_buffer


def create_test_replay(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["test_marker"] = "preserved"
    data = root.require_group("data")
    meta = root.require_group("meta")

    actions = np.zeros((10, 7), dtype=np.float32)
    actions[[1, 2, 4], 0] = 0.1
    actions[7:, 6] = 1.0
    data.create_dataset("action", data=actions, chunks=(3, 7))
    data.create_dataset(
        "timestamp_ns",
        data=np.arange(100, 200, 10, dtype=np.int64),
        chunks=(3,),
    )
    data.create_dataset(
        "alignment_valid",
        data=np.asarray([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=np.uint8),
        chunks=(3,),
    )
    data.create_dataset("sample_index", data=np.arange(10), chunks=(3,))

    meta.create_dataset("episode_ends", data=np.asarray([6, 10]), chunks=(2,))
    meta.create_dataset(
        "episode_start_monotonic_ns", data=np.asarray([90, 160]), chunks=(2,)
    )
    meta.create_dataset(
        "episode_start_wall_time_ns", data=np.asarray([1000, 2000]), chunks=(2,)
    )
    meta.create_dataset("episode_valid_steps", data=np.asarray([4, 3]), chunks=(2,))
    meta.create_dataset("episode_invalid_steps", data=np.asarray([2, 1]), chunks=(2,))
    meta.create_dataset("custom_episode_value", data=np.asarray([7, 8]), chunks=(2,))


class PruneReplayBufferTests(unittest.TestCase):
    def test_preview_is_read_only_and_copy_splits_inactive_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            output = Path(directory) / "pruned.zarr"
            create_test_replay(source)

            preview = prune_replay_buffer(source)

            self.assertIsNone(preview.output)
            self.assertFalse(output.exists())
            self.assertEqual(preview.input_episodes, 2)
            self.assertEqual(preview.output_episodes, 3)
            self.assertEqual(preview.kept_steps, 6)
            self.assertEqual(preview.removed_steps, 4)

            report = prune_replay_buffer(source, output)

            self.assertEqual(report.output, str(output.resolve()))
            pruned = zarr.open_group(str(output), mode="r")
            np.testing.assert_array_equal(pruned["meta/episode_ends"][:], [2, 3, 6])
            np.testing.assert_array_equal(
                pruned["data/sample_index"][:], [1, 2, 4, 7, 8, 9]
            )
            np.testing.assert_array_equal(
                pruned["meta/custom_episode_value"][:], [7, 7, 8]
            )
            np.testing.assert_array_equal(
                pruned["meta/episode_start_monotonic_ns"][:], [110, 140, 170]
            )
            np.testing.assert_array_equal(
                pruned["meta/episode_start_wall_time_ns"][:], [1020, 1050, 2010]
            )
            np.testing.assert_array_equal(
                pruned["meta/episode_valid_steps"][:], [1, 0, 3]
            )
            np.testing.assert_array_equal(
                pruned["meta/episode_invalid_steps"][:], [1, 1, 0]
            )
            self.assertEqual(pruned.attrs["test_marker"], "preserved")
            self.assertEqual(pruned.attrs["prune"]["grasp_cooldown_s"], 2.5)

    def test_grasp_cooldown_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            create_test_replay(source)

            preview = prune_replay_buffer(source, grasp_cooldown_s=0.0)

            self.assertEqual(preview.kept_steps, 4)
            self.assertEqual(preview.removed_steps, 6)

    def test_grasp_cooldown_stops_after_two_and_a_half_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            create_test_replay(source)
            root = zarr.open_group(str(source), mode="r+")
            root["data/timestamp_ns"][:] = np.arange(10, dtype=np.int64) * 2_000_000_000

            preview = prune_replay_buffer(source)

            # The transition is at step 7. Step 8 is two seconds later and is
            # retained; step 9 is four seconds later and is pruned.
            self.assertEqual(preview.kept_steps, 5)
            self.assertEqual(preview.removed_steps, 5)

    def test_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            output = Path(directory) / "existing.zarr"
            create_test_replay(source)
            output.mkdir()

            with self.assertRaises(FileExistsError):
                prune_replay_buffer(source, output)


if __name__ == "__main__":
    unittest.main()
