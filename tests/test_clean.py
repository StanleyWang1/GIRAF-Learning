from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from giraf.clean import audit_replay_buffer, clean_replay_buffer, default_clean_path


def create_partially_damaged_replay(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({"schema_version": "giraf-replay-v1", "marker": "preserved"})
    data = root.require_group("data")
    meta = root.require_group("meta")

    timestamps = np.arange(9, dtype=np.int64) * 100_000_000 + 1_000_000_000
    actions = np.zeros((9, 7), dtype=np.float32)
    actions[4, 0] = 0.2
    actions[6:, 0] = 0.2
    values = {
        "timestamp_ns": timestamps,
        "alignment_valid": np.ones(9, dtype=np.uint8),
        "camera_rgb": np.full((9, 2, 2, 3), 127, dtype=np.uint8),
        "action": actions,
        "state": np.ones((9, 15), dtype=np.float32),
        "sample_index": np.arange(9, dtype=np.int64),
    }
    for name, source in values.items():
        array = data.create_dataset(
            name,
            shape=source.shape,
            chunks=(3,) + source.shape[1:],
            dtype=source.dtype,
        )
        # Episode 0 is deliberately left without physical chunks. Zarr reads
        # those rows as fill values; episodes 1 and 2 are fully written.
        array[3:] = source[3:]

    meta.create_dataset("episode_ends", data=np.asarray([3, 6, 9], dtype=np.int64))
    meta.create_dataset(
        "episode_valid_steps", data=np.asarray([0, 3, 3], dtype=np.int64)
    )
    meta.create_dataset(
        "episode_invalid_steps", data=np.asarray([3, 0, 0], dtype=np.int64)
    )
    meta.create_dataset(
        "episode_start_monotonic_ns",
        data=np.asarray([900_000_000, 1_300_000_000, 1_600_000_000], dtype=np.int64),
    )
    meta.create_dataset(
        "episode_start_wall_time_ns",
        data=np.asarray([10, 20, 30], dtype=np.int64),
    )
    meta.create_dataset("custom_episode_value", data=np.asarray([10, 20, 30]))


class CleanReplayBufferTests(unittest.TestCase):
    def test_dry_run_finds_only_fully_stored_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trials.zarr"
            create_partially_damaged_replay(source)

            report = clean_replay_buffer(source, dry_run=True)

            self.assertEqual(report.output, str(default_clean_path(source)))
            self.assertFalse(report.written)
            self.assertEqual(report.healthy_episodes, (1, 2))
            self.assertEqual(report.rejected_episodes, (0,))
            self.assertEqual(report.healthy_steps, 6)
            self.assertEqual(report.removed_unhealthy_steps, 3)
            self.assertFalse(default_clean_path(source).exists())
            damaged = report.audits[0]
            self.assertEqual(damaged.missing_data_rows["camera_rgb"], 3)
            self.assertEqual(damaged.missing_data_rows["action"], 3)
            self.assertEqual(damaged.missing_data_rows["state"], 3)

    def test_writes_default_clean_copy_and_preserves_episode_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trials.zarr"
            create_partially_damaged_replay(source)

            report = clean_replay_buffer(source)

            self.assertTrue(report.written)
            output = default_clean_path(source)
            cleaned = zarr.open_group(str(output), mode="r")
            np.testing.assert_array_equal(cleaned["meta/episode_ends"][:], [3, 6])
            np.testing.assert_array_equal(
                cleaned["data/sample_index"][:], [3, 4, 5, 6, 7, 8]
            )
            np.testing.assert_array_equal(
                cleaned["meta/custom_episode_value"][:], [20, 30]
            )
            np.testing.assert_array_equal(
                cleaned["meta/episode_valid_steps"][:], [3, 3]
            )
            self.assertEqual(cleaned.attrs["marker"], "preserved")
            self.assertEqual(cleaned.attrs["clean"]["healthy_source_episodes"], [1, 2])

            _root, _ends, audits = audit_replay_buffer(output)
            self.assertTrue(all(audit.healthy for audit in audits))

            with self.assertRaises(FileExistsError):
                clean_replay_buffer(source)

    def test_optional_inactive_pruning_splits_only_healthy_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trials.zarr"
            output = Path(directory) / "pruned_clean.zarr"
            create_partially_damaged_replay(source)

            report = clean_replay_buffer(
                source,
                output,
                prune_inactive=True,
                keep_grasp_transitions=False,
            )

            self.assertEqual(report.output_episodes, 2)
            self.assertEqual(report.output_steps, 4)
            self.assertEqual(report.removed_inactive_steps, 2)
            cleaned = zarr.open_group(str(output), mode="r")
            np.testing.assert_array_equal(cleaned["meta/episode_ends"][:], [1, 4])
            np.testing.assert_array_equal(cleaned["data/sample_index"][:], [4, 6, 7, 8])
            np.testing.assert_array_equal(
                cleaned["meta/episode_valid_steps"][:], [1, 3]
            )


if __name__ == "__main__":
    unittest.main()
