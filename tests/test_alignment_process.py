"""Regressions for teleop stalls; synthetic sources never open hardware."""

import ctypes
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zarr

from giraf.data.alignment import AlignmentProcess
from giraf.data.config import CameraConfig, CollectorConfig, DatasetConfig
from giraf.data.episode_stage import EpisodeStage
from giraf.data.imu import report_example
from giraf.data.pipeline import DataCollectionPipeline
from giraf.data.producer import CameraProducer
from giraf.data.schema import camera_example
from giraf.data.shared_memory import RingBufferOverrun


class SyntheticCamera(CameraProducer):
    def run_producer(self):
        config = CollectorConfig(camera=self.config)
        self._metadata_child.send(
            {"enabled": self.imu_config.enabled, "session_id": "b" * 32}
        )
        sequence = 0
        imu_sequence = 0
        next_camera = time.monotonic()
        while not self.stop_event.wait(0.005):
            now = time.monotonic_ns()
            for name, ring in self.imu_rings.items():
                report = report_example(name)
                report.update(
                    timestamp_ns=np.int64(now),
                    device_timestamp_ns=np.int64(now),
                    receive_timestamp_ns=np.int64(now),
                    sequence_num=np.int64(imu_sequence),
                    report_valid=np.uint8(1),
                )
                report["values"][-1] = 1
                ring.put(report)
            imu_sequence += 1
            if time.monotonic() >= next_camera:
                frame = camera_example(config)
                frame.update(
                    timestamp_ns=np.int64(now),
                    device_timestamp_ns=np.int64(now),
                    receive_timestamp_ns=np.int64(now),
                    sequence_num=np.int64(sequence),
                )
                self.ring.put(frame)
                sequence += 1
                next_camera += 1 / self.config.fps
                self.ready_event.set()


class AlignmentUnitTests(unittest.TestCase):
    def make_aligner(self, camera):
        aligner = AlignmentProcess(
            CollectorConfig(),
            camera,
            None,
            None,
            {},
            SimpleNamespace(count=0),
            hardware_enabled=False,
        )
        self.addCleanup(aligner.parent_connection.close)
        self.addCleanup(aligner.child_connection.close)
        return aligner

    def test_idle_overwritten_frames_are_not_recording_errors(self):
        def unreadable(*args):
            self.fail("idle camera history should not be copied")

        camera = SimpleNamespace(count=10000, get_range=unreadable)
        aligner = self.make_aligner(camera)
        aligner._process_camera_samples()
        self.assertEqual(aligner._camera_cursor, 10000)

    def test_active_overrun_is_still_an_error(self):
        def overwritten(*args):
            raise RingBufferOverrun("sample overwritten")

        camera = SimpleNamespace(count=10000, get_max_k=64, get_range=overwritten)
        aligner = self.make_aligner(camera)
        aligner._recording = True
        with self.assertRaisesRegex(RuntimeError, "camera ring overrun"):
            aligner._process_camera_samples()

    def test_backlog_larger_than_get_max_k_is_copied_in_chunks(self):
        calls = []

        def copy(start, stop):
            self.assertLessEqual(stop - start, 4)
            calls.append((start, stop))
            return {"timestamp_ns": np.arange(start, stop, dtype=np.int64)}

        camera = SimpleNamespace(count=12, get_max_k=5, get_range=copy)
        aligner = self.make_aligner(camera)
        aligner._recording = True
        aligner._episode_start_ns = 100  # Discard pre-episode samples after copying.
        aligner._process_camera_samples()
        self.assertEqual(calls, [(0, 4), (4, 8), (8, 12)])

    def test_delayed_stop_acknowledges_already_saved_frames(self):
        aligner = self.make_aligner(SimpleNamespace(count=0))
        aligner._last_emit_ns = 200
        result = aligner._handle({"operation": "stop", "timestamp_ns": 100})
        self.assertEqual(result["stop_monotonic_ns"], 200)
        self.assertEqual(result["requested_stop_monotonic_ns"], 100)


class ProcessIsolationTests(unittest.TestCase):
    def test_parent_gil_stall_and_slow_commit_do_not_drop_camera_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CollectorConfig(
                camera=CameraConfig(width=8, height=8),
                dataset=DatasetConfig(
                    output_dir=Path(directory), resize_dim=(8, 8), save_raw_video=False
                ),
            )
            original_commit = EpisodeStage.commit

            def slow_commit(stage, writer):
                time.sleep(2.5)  # Longer than the camera's retained history.
                return original_commit(stage, writer)

            with patch("giraf.data.pipeline.CameraProducer", SyntheticCamera):
                collector = DataCollectionPipeline(config, hardware_enabled=False)
            publisher_stop = threading.Event()

            def publish():
                while not publisher_stop.is_set():
                    collector.publish_control(
                        timestamp_ns=time.monotonic_ns(),
                        task_twist=np.zeros(6),
                        joint_velocity_command=np.zeros(6),
                        joint_position_command=np.zeros(6),
                        state=np.zeros(15),
                        grasp=False,
                        clutch=False,
                        tracking=True,
                    )
                    publisher_stop.wait(0.01)

            publisher = threading.Thread(target=publish)
            try:
                # The saver inherits the injected slow commit when forked.
                with patch.object(EpisodeStage, "commit", slow_commit):
                    collector.start(lambda: {"record_toggle_count": 0})
                publisher.start()
                time.sleep(0.1)
                self.assertTrue(collector.start_episode())
                time.sleep(0.2)
                before = collector.aligned_ring.count
                # PyDLL deliberately holds this process's GIL during the C call.
                # A thread-based aligner cannot run during this three-second stall.
                libc = ctypes.PyDLL(None)
                libc.usleep.argtypes = [ctypes.c_uint]
                libc.usleep(3_000_000)
                after = collector.aligned_ring.count
                self.assertGreaterEqual(after - before, 80)
                time.sleep(0.15)
                collector.end_episode()
                self.assertFalse(collector.error)
                self.assertTrue(collector.start_episode())
                time.sleep(0.25)
                collector.end_episode()
                self.assertFalse(collector.error)
            finally:
                publisher_stop.set()
                if publisher.ident is not None:
                    publisher.join(2)
                collector.stop()
            root = zarr.open_group(str(config.zarr_path), mode="r")
            ends = root["meta/episode_ends"][:]
            self.assertEqual(len(ends), 2)
            start = 0
            for end in ends:
                sequences = root["data/camera_sequence_num"][start:end]
                np.testing.assert_array_equal(np.diff(sequences), 1)
                self.assertTrue(root["data/imu_valid"][start:end].all())
                start = end
            # Missing control updates must be labeled stale, not fabricated fresh.
            self.assertGreater(
                np.count_nonzero(root["data/alignment_valid"][:] == 0), 70
            )


if __name__ == "__main__":
    unittest.main()
