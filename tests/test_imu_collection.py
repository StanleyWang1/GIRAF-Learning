"""Collection/storage regression tests; no hardware is opened."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zarr

from giraf.data.config import CameraConfig, CollectorConfig, DatasetConfig, ImuConfig
from giraf.data.episode_stage import EpisodeStage
from giraf.data.imu import IMU_STREAMS, ReportDecoder, align_imu, report_example
from giraf.data.imu_storage import validate_streams
from giraf.data.prune import prune_replay_buffer
from giraf.data.replay_buffer import ReplayBufferWriter
from giraf.data.schema import aligned_example


def reports(name, times):
    rows = []
    for index, timestamp in enumerate(times):
        row = report_example(name)
        row.update(
            timestamp_ns=np.int64(timestamp),
            device_timestamp_ns=np.int64(timestamp),
            receive_timestamp_ns=np.int64(timestamp + 1_000_000),
            sequence_num=np.int64(index),
            report_valid=np.uint8(1),
        )
        row["values"][-1] = 1
        rows.append(row)
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}


class AlignmentTests(unittest.TestCase):
    def test_independent_causal_selection_and_staleness(self):
        batches = {
            "accelerometer": reports(
                "accelerometer", [10_000_000, 90_000_000, 110_000_000]
            ),
            "gyroscope": reports("gyroscope", [60_000_000]),
        }
        aligned = align_imu(batches, 100_000_000, 25_000_000)
        np.testing.assert_array_equal(
            aligned["imu_timestamp_ns"], [90_000_000, 60_000_000, -1]
        )
        np.testing.assert_array_equal(aligned["imu_available"], [1, 1, 0])
        np.testing.assert_array_equal(aligned["imu_sensor_valid"], [1, 0, 0])
        self.assertEqual(aligned["imu_valid"], 0)
        self.assertTrue(np.isnan(aligned["imu"][6:]).all())

    def test_threshold_inclusive_and_bad_report(self):
        batches = {name: reports(name, [75_000_000]) for name in IMU_STREAMS}
        self.assertEqual(align_imu(batches, 100_000_000, 25_000_000)["imu_valid"], 1)
        batches["gyroscope"]["report_valid"][0] = 0
        self.assertEqual(align_imu(batches, 100_000_000, 25_000_000)["imu_valid"], 0)

    def test_decoder_duplicate_wrap_gap_and_clock(self):
        def report(device_us, seq, host_us=None):
            return SimpleNamespace(
                getTimestamp=lambda: timedelta(microseconds=host_us or device_us),
                getTimestampDevice=lambda: timedelta(microseconds=device_us),
                getSequenceNum=lambda: seq,
                x=0,
                y=0,
                z=9.81,
                accuracy=3,
            )

        decoder = ReportDecoder()
        first = report(10000, 2**32 - 1)
        self.assertIsNotNone(decoder.decode("accelerometer", first, 10_000_000))
        self.assertIsNone(decoder.decode("accelerometer", first, 11_000_000))
        self.assertEqual(
            decoder.decode("accelerometer", report(20000, 0), 20_000_000)[
                "sequence_gap"
            ],
            0,
        )
        self.assertEqual(
            decoder.decode("accelerometer", report(50000, 3), 50_000_000)[
                "sequence_gap"
            ],
            2,
        )
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            decoder.decode("accelerometer", report(10000, 4), 50_000_000)
        with self.assertRaisesRegex(RuntimeError, "clock"):
            decoder.decode("gyroscope", report(60000, 1), 60_000_000_000)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = CollectorConfig(
            camera=CameraConfig(width=8, height=8),
            dataset=DatasetConfig(
                output_dir=Path(self.tmp.name) / "source",
                resize_dim=(8, 8),
                save_raw_video=False,
                saver_batch_size=2,
                imu_zarr_chunk_length=4,
            ),
        )
        self.writer = ReplayBufferWriter(self.config)

    def stage(self, offset=0):
        start = 2_000_000_000 + offset
        stage = EpisodeStage(
            self.config,
            start_wall_time_ns=start,
            start_monotonic_ns=start,
            imu_metadata={"enabled": True, "session_id": "a" * 32},
        )
        times = [start + i * 33_333_333 for i in range(6)]
        for name in IMU_STREAMS:
            stage.append_imu(
                name,
                reports(
                    name,
                    [
                        start - 1_100_000_000,
                        start - 500_000_000,
                        *times,
                        times[-1] + 500_000_000,
                    ],
                ),
            )
        for i, timestamp in enumerate(times):
            row = aligned_example(self.config)
            row.update(
                timestamp_ns=np.int64(timestamp),
                camera_receive_timestamp_ns=np.int64(timestamp + 1_000_000),
                camera_device_timestamp_ns=np.int64(timestamp),
                control_timestamp_ns=np.int64(timestamp),
                motor_timestamp_ns=np.int64(timestamp),
                alignment_valid=np.uint8(1),
                camera_sequence_num=np.int64(i),
            )
            row["task_twist"][0] = 1 if i in (1, 2, 4) else 0
            # Deliberately missing IMU must not filter RGB/state training.
            stage.append(row)
        stage.finish_imu(times[-1])
        return stage

    def test_two_axes_commit_and_pre_roll_trim(self):
        stage = self.stage()
        result = stage.commit(self.writer)
        root = self.writer.root
        self.assertEqual(result["num_steps"], 6)
        self.assertEqual(root["data/imu"].shape, (6, 10))
        self.assertEqual(root["meta/episode_imu_valid_steps"][0], 0)
        for name in IMU_STREAMS:
            group = root[f"imu/{name}"]
            self.assertEqual(len(group["values"]), 7)
            self.assertEqual(group["timestamp_ns"][0], 1_500_000_000)
        validate_streams(root, 1)
        self.stage(10_000_000_000).commit(self.writer)
        np.testing.assert_array_equal(root["imu/gyroscope/episode_ends"][:], [7, 14])
        self.assertEqual(root["meta/episode_imu_session_id"][0], "a" * 32)

    def test_recovery_truncates_all_uncommitted_axes(self):
        self.stage().commit(self.writer)
        root = self.writer.root
        root["data/imu"].resize(10, 10)
        group = root["imu/accelerometer"]
        for key in report_example("accelerometer"):
            array = group[key]
            array.resize((len(array) + 3,) + array.shape[1:])
        group["episode_ends"].resize(2)
        group["episode_ends"][1] = 10
        group["episode_saver_dropped"].resize(2)
        self.writer.recover_uncommitted_tail()
        self.assertEqual(root["data/imu"].shape, (6, 10))
        validate_streams(root, 1)

    def test_failed_commit_recovers_and_preserves_stage(self):
        from giraf.data.imu_storage import append_streams

        stage = self.stage()

        def append_then_fail(*args):
            append_streams(*args)
            raise OSError("disk failure after raw offsets, before image commit")

        with patch(
            "giraf.data.replay_buffer.append_streams",
            side_effect=append_then_fail,
        ):
            with self.assertRaises(OSError):
                stage.commit(self.writer)
        self.assertEqual(self.writer.n_episodes, 0)
        validate_streams(self.writer.root, 0)
        self.assertTrue(stage.directory.exists())
        rejected = stage.reject("test failure")
        self.assertTrue((rejected / "episode.zarr/imu").exists())

    def test_pruning_rebuilds_offsets_and_preserves_source(self):
        self.stage().commit(self.writer)
        self.stage(10_000_000_000).commit(self.writer)
        output = Path(self.tmp.name) / "pruned.zarr"
        report = prune_replay_buffer(
            self.config.zarr_path, output, keep_grasp_transitions=False
        )
        self.assertEqual(report.output_episodes, 4)
        target = zarr.open_group(str(output), mode="r")
        validate_streams(target, 4)
        np.testing.assert_array_equal(target["meta/episode_ends"][:], [2, 3, 5, 6])
        raw = target["imu/accelerometer"]
        self.assertLess(
            raw["timestamp_ns"][int(raw["episode_ends"][1]) - 1], 3_000_000_000
        )
        self.assertGreater(
            raw["timestamp_ns"][int(raw["episode_ends"][1])], 10_000_000_000
        )
        self.assertEqual(self.writer.n_steps, 12)

    def test_existing_training_and_viewer_accept_v2_and_v1(self):
        self.stage().commit(self.writer)
        from giraf.learning.dataset import ReplayDataset
        from giraf.viewer.dataset import GirafDataset

        for version in ("giraf-replay-v2", "giraf-replay-v1"):
            self.writer.root.attrs["schema_version"] = version
            dataset = ReplayDataset(self.config.zarr_path, batch_size=2)
            self.assertEqual(dataset.n_windows, 6)
            self.assertEqual(
                set(next(iter(dataset)).observations), {"camera_rgb", "state"}
            )
            self.assertEqual(GirafDataset(self.config.zarr_path).total_steps, 6)
            import json

            payload = GirafDataset(self.config.zarr_path).signal_payload(
                0, ["imu"], 0, 6
            )
            json.dumps(payload, allow_nan=False)
            self.assertIsNone(payload["series"]["imu"][0]["values"][0])

    def test_legacy_append_rejected_without_migration(self):
        self.writer.root.attrs["schema_version"] = "giraf-replay-v1"
        with self.assertRaisesRegex(RuntimeError, "fresh dataset"):
            ReplayBufferWriter(self.config)
        self.assertEqual(self.writer.root.attrs["schema_version"], "giraf-replay-v1")

    def test_disabled_collection_has_empty_streams(self):
        config = replace(
            self.config,
            imu=ImuConfig(enabled=False),
            dataset=replace(
                self.config.dataset, output_dir=Path(self.tmp.name) / "disabled"
            ),
        )
        writer = ReplayBufferWriter(config)
        stage = EpisodeStage(config, start_wall_time_ns=1, start_monotonic_ns=1)
        row = aligned_example(config)
        row["timestamp_ns"] = np.int64(2)
        stage.append(row)
        stage.finish_imu(3)
        stage.commit(writer)
        validate_streams(writer.root, 1)
        self.assertEqual(len(writer.root["imu/gyroscope/values"]), 0)

    def test_saver_loss_and_stopped_imu_preserve_episode(self):
        from giraf.data.saver import SaverProcess

        batch = reports(
            "gyroscope", [2_000_000_000 + i * 10_000_000 for i in range(10)]
        )
        ring = SimpleNamespace(
            count=10,
            buffer_size=4,
            get_max_k=4,
            get_range=lambda start, stop: {k: v[start:stop] for k, v in batch.items()},
            get=lambda: {k: v[-1] for k, v in batch.items()},
        )
        saver = SaverProcess(
            self.config,
            None,
            {"gyroscope": ring},
            {"gyroscope": SimpleNamespace(value=3)},
        )
        self.addCleanup(saver.parent_connection.close)
        self.addCleanup(saver.child_connection.close)
        saver._writer = self.writer
        saver._stage = EpisodeStage(
            self.config,
            start_wall_time_ns=2_000_000_000,
            start_monotonic_ns=2_000_000_000,
            imu_metadata={"enabled": True, "session_id": "a" * 32},
        )
        row = aligned_example(self.config)
        row["timestamp_ns"] = np.int64(2_000_000_000)
        row["alignment_valid"] = np.uint8(1)
        saver._stage.append(row)
        saver._imu_cursors = {"gyroscope": 0}
        saver._imu_dropped_start = {"gyroscope": 0}
        with patch("giraf.data.saver.IMU_STOP_GRACE_S", 0):
            result = saver._stop_episode(
                {"end_count": 0, "stop_monotonic_ns": 2_100_000_000}
            )
        self.assertEqual(result["num_steps"], 1)
        self.assertEqual(result["imu"]["valid_steps"], 0)
        self.assertEqual(self.writer.root["imu/gyroscope/episode_saver_dropped"][0], 6)
        self.assertEqual(
            self.writer.root["imu/gyroscope/episode_producer_dropped"][0], 3
        )
        self.assertEqual(len(self.writer.root["imu/gyroscope/values"]), 4)

    def test_clean_copy_keeps_full_rate_data(self):
        from giraf.clean import clean_replay_buffer

        self.stage().commit(self.writer)
        output = Path(self.tmp.name) / "clean.zarr"
        clean_replay_buffer(self.config.zarr_path, output)
        root = zarr.open_group(str(output), mode="r")
        validate_streams(root, 1)
        np.testing.assert_array_equal(
            root["imu/accelerometer/values"][:],
            self.writer.root["imu/accelerometer/values"][:],
        )


if __name__ == "__main__":
    unittest.main()
