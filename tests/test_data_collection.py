from __future__ import annotations

import tempfile
import time
import unittest
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np
import zarr

from giraf.data.config import (
    CameraConfig,
    CollectorConfig,
    DatasetConfig,
    SharedMemoryConfig,
)
from giraf.data.episode_stage import EpisodeStage
from giraf.data.pipeline import DataCollectionPipeline, _sample_at_or_before
from giraf.data.replay_buffer import ReplayBufferWriter
from giraf.data.saver import SaverProcess
from giraf.data.schema import aligned_example, state_vector
from giraf.data.shared_memory import RingBufferOverrun, SharedMemoryRingBuffer


def test_config(output_dir: Path, *, raw_video: bool = False) -> CollectorConfig:
    return CollectorConfig(
        camera=CameraConfig(width=8, height=6, fps=30.0),
        dataset=DatasetConfig(
            output_dir=output_dir,
            aligned_hz=30.0,
            resize_dim=(4, 4),
            save_raw_video=raw_video,
            zarr_chunk_length=2,
            saver_batch_size=2,
        ),
        shared_memory=SharedMemoryConfig(
            get_time_budget_s=0.001,
            safety_margin=1.5,
            camera_history=4,
            control_history=8,
            motor_history=8,
            aligned_history=8,
        ),
    )


def aligned_sample(config: CollectorConfig, index: int) -> dict[str, np.ndarray]:
    sample = aligned_example(config)
    timestamp = 1_000_000_000 + index * 33_333_333
    sample.update(
        {
            "camera_rgb_source": np.full(
                (config.camera.height, config.camera.width, 3),
                index * 20,
                dtype=np.uint8,
            ),
            "timestamp_ns": np.int64(timestamp),
            "camera_device_timestamp_ns": np.int64(10_000 + index),
            "camera_receive_timestamp_ns": np.int64(timestamp + 2_000_000),
            "camera_sequence_num": np.int64(index),
            "control_timestamp_ns": np.int64(timestamp - 4_000_000),
            "motor_timestamp_ns": np.int64(timestamp - 3_000_000),
            "task_twist": np.arange(6, dtype=np.float32) + index,
            "joint_velocity_command": np.full(6, 0.1 * index, np.float32),
            "joint_position_command": np.full(6, 0.2 * index, np.float32),
            "state": np.arange(15, dtype=np.float32) + index,
            "grasp": np.uint8(index % 2),
            "clutch": np.uint8(1),
            "tracking": np.uint8(1),
            "can_position_target": np.arange(3, dtype=np.float32),
            "dynamixel_target_ticks": np.arange(4, dtype=np.int32) + 100,
            "motor_command_accepted": np.uint8(1),
            "alignment_valid": np.uint8(index != 1),
        }
    )
    return sample


class SharedMemoryRingBufferTests(unittest.TestCase):
    def test_range_alignment_and_overrun(self) -> None:
        with SharedMemoryManager() as manager:
            ring = SharedMemoryRingBuffer.create_from_examples(
                manager,
                {
                    "timestamp_ns": np.int64(0),
                    "value": np.zeros(2, dtype=np.float32),
                },
                get_max_k=3,
                get_time_budget=0.001,
                put_desired_frequency=10,
            )
            for index in range(3):
                ring.put(
                    {
                        "timestamp_ns": np.int64(100 + index * 10),
                        "value": np.full(2, index, dtype=np.float32),
                    },
                    wait=True,
                )
            batch = ring.get_range(0, 3)
            np.testing.assert_array_equal(batch["timestamp_ns"], [100, 110, 120])
            sample = _sample_at_or_before(ring, 115)
            self.assertIsNotNone(sample)
            assert sample is not None
            self.assertEqual(int(sample["timestamp_ns"]), 110)
            self.assertIsNone(_sample_at_or_before(ring, 99))

            for index in range(3, 6):
                ring.put(
                    {
                        "timestamp_ns": np.int64(100 + index * 10),
                        "value": np.full(2, index, dtype=np.float32),
                    },
                    wait=True,
                )
            with self.assertRaises(RingBufferOverrun):
                ring.get_range(0, 1)

    def test_state_vector_order(self) -> None:
        joints = np.arange(6, dtype=float)
        position = np.array([6.0, 7.0, 8.0])
        rotation = np.eye(3)
        state = state_vector(joints, position, rotation)
        np.testing.assert_array_equal(state[:9], np.arange(9, dtype=np.float32))
        np.testing.assert_array_equal(state[9:], [1, 0, 0, 0, 1, 0])


class ConductorTests(unittest.TestCase):
    def test_alignment_is_causal_and_uses_dispatched_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory))
            pipeline = DataCollectionPipeline(config, hardware_enabled=True)
            base = 1_000_000_000
            try:
                for offset, value in ((0, 1.0), (20_000_000, 2.0)):
                    pipeline.publish_control(
                        timestamp_ns=base + offset,
                        task_twist=np.full(6, value, dtype=np.float32),
                        joint_velocity_command=np.zeros(6),
                        joint_position_command=np.zeros(6),
                        state=np.zeros(15),
                        grasp=False,
                        clutch=True,
                        tracking=True,
                    )
                    time.sleep(0.002)
                pipeline.publish_control(
                    timestamp_ns=base + 40_000_000,
                    task_twist=np.full(6, 99, dtype=np.float32),
                    joint_velocity_command=np.zeros(6),
                    joint_position_command=np.zeros(6),
                    state=np.zeros(15),
                    grasp=False,
                    clutch=True,
                    tracking=True,
                )
                pipeline.publish_motor(
                    timestamp_ns=base + 15_000_000,
                    can_position_target=np.arange(3),
                    dynamixel_target_ticks=np.arange(4),
                    grasp=True,
                    command_accepted=True,
                )
                camera = {
                    "camera_rgb_source": np.zeros((6, 8, 3), dtype=np.uint8),
                    "timestamp_ns": np.int64(base + 25_000_000),
                    "device_timestamp_ns": np.int64(100),
                    "receive_timestamp_ns": np.int64(base + 27_000_000),
                    "sequence_num": np.int64(5),
                }
                aligned = pipeline._align(camera)
                self.assertIsNotNone(aligned)
                assert aligned is not None
                np.testing.assert_array_equal(aligned["task_twist"], np.full(6, 2.0))
                self.assertEqual(
                    int(aligned["control_timestamp_ns"]), base + 20_000_000
                )
                self.assertEqual(int(aligned["grasp"]), 1)
                self.assertEqual(int(aligned["alignment_valid"]), 1)
            finally:
                pipeline.manager.shutdown()


class StorageTests(unittest.TestCase):
    def test_staging_append_recovery_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory), raw_video=True)
            writer = ReplayBufferWriter(config)
            stage = EpisodeStage(
                config,
                start_wall_time_ns=20,
                start_monotonic_ns=10,
            )
            for index in range(3):
                stage.append(aligned_sample(config, index))
            metadata = stage.commit(writer)
            self.assertEqual(metadata["num_steps"], 3)
            self.assertEqual(metadata["valid_steps"], 2)
            self.assertEqual(metadata["invalid_steps"], 1)

            root = zarr.open_group(str(config.zarr_path), mode="r+")
            np.testing.assert_array_equal(root["meta/episode_ends"][:], [3])
            self.assertEqual(root["data/action"].shape, (3, 7))
            self.assertEqual(root["data/state"].shape, (3, 15))
            self.assertEqual(root["data/camera_rgb"].shape, (3, 4, 4, 3))
            np.testing.assert_array_equal(root["data/action"][:, -1], [0, 1, 0])
            video_path = config.video_dir / "0" / "camera.mp4"
            self.assertTrue(video_path.is_file())
            import av

            with av.open(str(video_path)) as container:
                self.assertEqual(sum(1 for _frame in container.decode(video=0)), 3)

            action = root["data/action"]
            action.resize((4, 7))
            action[3] = 999
            ReplayBufferWriter(config)
            recovered = zarr.open_group(str(config.zarr_path), mode="r")
            self.assertEqual(recovered["data/action"].shape, (3, 7))

    def test_saver_process_drains_shared_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = test_config(Path(directory), raw_video=False)
            with SharedMemoryManager() as manager:
                ring = SharedMemoryRingBuffer.create_from_examples(
                    manager,
                    aligned_example(config),
                    get_max_k=config.shared_memory.aligned_history,
                    get_time_budget=config.shared_memory.get_time_budget_s,
                    put_desired_frequency=config.dataset.aligned_hz,
                )
                saver = SaverProcess(config, ring)
                saver.start()
                try:
                    saver.start_wait()
                    saver.request(
                        "start",
                        start_count=ring.count,
                        start_wall_time_ns=20,
                        start_monotonic_ns=10,
                    )
                    for index in range(4):
                        ring.put(aligned_sample(config, index), wait=True)
                    result = saver.request("stop", end_count=ring.count)
                    self.assertEqual(result["num_steps"], 4)
                finally:
                    saver.shutdown()
            root = zarr.open_group(str(config.zarr_path), mode="r")
            np.testing.assert_array_equal(root["meta/episode_ends"][:], [4])
            self.assertEqual(root["data/camera_rgb"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
