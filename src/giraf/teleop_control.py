"""Shared relative OptiTrack pose control, independent of device ownership."""

from __future__ import annotations

import math

import numpy as np

from giraf.kinematics import num_forward_transform

POSE_TIMEOUT = 0.15
POSITION_GAIN = ROTATION_GAIN = 5.0
LINEAR_LIMIT = np.array((0.5, 0.5, 0.5))
ANGULAR_LIMIT = np.array((1.0, 1.0, 1.0))
ROTATION_BASIS = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


def normalize_quaternion(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or norm < 1e-12:
        raise ValueError("invalid quaternion")
    return quaternion / norm


def quaternion_matrix(value) -> np.ndarray:
    x, y, z, w = normalize_quaternion(value)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def relative_quaternion(anchor, current) -> np.ndarray:
    ax, ay, az, aw = normalize_quaternion(anchor)
    bx, by, bz, bw = normalize_quaternion(current)
    return normalize_quaternion(
        (
            aw * bx - ax * bw - ay * bz + az * by,
            aw * by + ax * bz - ay * bw - az * bx,
            aw * bz - ax * by + ay * bx - az * bw,
            aw * bw + ax * bx + ay * by + az * bz,
        )
    )


def rotation_vector(rotation) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if angle < 1e-7:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        raise RuntimeError("orientation error is too close to 180 degrees")
    return angle * skew / (2.0 * math.sin(angle))


def limited_velocity(error, gain, deadband, limit) -> np.ndarray:
    magnitude = float(np.linalg.norm(error))
    if magnitude <= deadband:
        return np.zeros(3)
    return np.clip(gain * error * (magnitude - deadband) / magnitude, -limit, limit)


def model_joints(joints) -> np.ndarray:
    return np.asarray(joints) + np.array(
        (0.0, math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, 0.0)
    )


def end_effector_pose(joints) -> tuple[np.ndarray, np.ndarray]:
    transform = num_forward_transform(model_joints(joints))
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("invalid forward kinematics")
    return transform[:3, 3].copy(), transform[:3, :3].copy()


class RelativePoseController:
    """Re-anchor on activation; retain the original teleop twist calculation."""

    def anchor(self, joints, position, quaternion) -> None:
        self.position = np.asarray(position).copy()
        self.quaternion = np.asarray(quaternion).copy()
        self.rotation = quaternion_matrix(quaternion)
        self.robot_position, self.robot_rotation = end_effector_pose(joints)

    def twist(self, joints, position, quaternion) -> np.ndarray:
        relative_position = self.rotation.T @ (position - self.position)
        controller_rotation = quaternion_matrix(
            relative_quaternion(self.quaternion, quaternion)
        )
        target_position = self.robot_position + relative_position
        target_rotation = self.robot_rotation @ (
            ROTATION_BASIS.T @ controller_rotation @ ROTATION_BASIS
        )
        current_position, current_rotation = end_effector_pose(joints)
        linear = limited_velocity(
            target_position - current_position, POSITION_GAIN, 0.002, LINEAR_LIMIT
        )
        angular = limited_velocity(
            rotation_vector(target_rotation @ current_rotation.T),
            ROTATION_GAIN,
            0.01,
            ANGULAR_LIMIT,
        )
        return np.concatenate((linear, angular))
