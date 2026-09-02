"""Stable stream and ReplayBuffer schemas."""

from __future__ import annotations

from typing import Final

import numpy as np

from .config import CollectorConfig

SCHEMA_VERSION: Final[str] = "giraf-replay-v1"
ACTION_FIELDS: Final[tuple[str, ...]] = (
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "wx_rad_s",
    "wy_rad_s",
    "wz_rad_s",
    "grasp_command",
)
JOINT_FIELDS: Final[tuple[str, ...]] = (
    "base_roll_rad",
    "base_pitch_rad",
    "boom_extension_m",
    "wrist_1_rad",
    "wrist_2_rad",
    "wrist_3_rad",
)
STATE_FIELDS: Final[tuple[str, ...]] = JOINT_FIELDS + (
    "eef_x_m",
    "eef_y_m",
    "eef_z_m",
    "eef_rotation_col0_x",
    "eef_rotation_col0_y",
    "eef_rotation_col0_z",
    "eef_rotation_col1_x",
    "eef_rotation_col1_y",
    "eef_rotation_col1_z",
)


def camera_example(config: CollectorConfig) -> dict[str, np.ndarray]:
    camera = config.camera
    return {
        "camera_rgb_source": np.zeros((camera.height, camera.width, 3), dtype=np.uint8),
        "timestamp_ns": np.int64(0),
        "device_timestamp_ns": np.int64(0),
        "receive_timestamp_ns": np.int64(0),
        "sequence_num": np.int64(0),
    }


def control_example() -> dict[str, np.ndarray]:
    return {
        "timestamp_ns": np.int64(0),
        "task_twist": np.zeros(6, dtype=np.float32),
        "joint_velocity_command": np.zeros(6, dtype=np.float32),
        "joint_position_command": np.zeros(6, dtype=np.float32),
        "state": np.zeros(15, dtype=np.float32),
        "grasp": np.uint8(0),
        "clutch": np.uint8(0),
        "tracking": np.uint8(0),
    }


def motor_example() -> dict[str, np.ndarray]:
    return {
        "timestamp_ns": np.int64(0),
        "can_position_target": np.zeros(3, dtype=np.float32),
        "dynamixel_target_ticks": np.zeros(4, dtype=np.int32),
        "grasp": np.uint8(0),
        "command_accepted": np.uint8(0),
    }


def aligned_example(config: CollectorConfig) -> dict[str, np.ndarray]:
    camera = camera_example(config)
    control = control_example()
    motor = motor_example()
    return {
        "camera_rgb_source": camera["camera_rgb_source"],
        "timestamp_ns": np.int64(0),
        "camera_device_timestamp_ns": np.int64(0),
        "camera_receive_timestamp_ns": np.int64(0),
        "camera_sequence_num": np.int64(0),
        "control_timestamp_ns": np.int64(0),
        "motor_timestamp_ns": np.int64(0),
        "task_twist": control["task_twist"],
        "joint_velocity_command": control["joint_velocity_command"],
        "joint_position_command": control["joint_position_command"],
        "state": control["state"],
        "grasp": np.uint8(0),
        "clutch": control["clutch"],
        "tracking": control["tracking"],
        "can_position_target": motor["can_position_target"],
        "dynamixel_target_ticks": motor["dynamixel_target_ticks"],
        "motor_command_accepted": motor["command_accepted"],
        "alignment_valid": np.uint8(0),
    }


def state_vector(
    joints: np.ndarray, position: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Create the documented command-derived 15D state vector."""

    joints = np.asarray(joints, dtype=np.float32)
    position = np.asarray(position, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    if joints.shape != (6,) or position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("invalid joint or forward-kinematics shape")
    rotation_6d = np.concatenate((rotation[:, 0], rotation[:, 1]))
    return np.concatenate((joints, position, rotation_6d)).astype(np.float32)
