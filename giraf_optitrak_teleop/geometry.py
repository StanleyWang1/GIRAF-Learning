"""ROS-independent pose math for relative OptiTrack teleoperation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def copy(self) -> "Pose":
        return Pose(self.position.copy(), self.rotation.copy())


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    return quaternion / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    return np.array((-qx, -qy, -qz, qw), dtype=float)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = normalize_quaternion(left)
    rx, ry, rz, rw = normalize_quaternion(right)
    return normalize_quaternion(
        np.array(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ),
            dtype=float,
        )
    )


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    return np.array(
        (
            (
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ),
            (
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ),
            (
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ),
        ),
        dtype=float,
    )


def project_to_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    left, _singular_values, right_transpose = np.linalg.svd(rotation)
    projected = left @ right_transpose
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_transpose
    return projected


def rotation_matrix_to_vector(rotation: np.ndarray) -> np.ndarray:
    rotation = project_to_rotation_matrix(rotation)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    antisymmetric = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=float,
    )
    if angle < 1e-7:
        return 0.5 * antisymmetric
    if math.pi - angle < 1e-5:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        if axis[0] > 1e-7:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif axis[1] > 1e-7:
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        norm = float(np.linalg.norm(axis))
        return np.zeros(3) if norm < 1e-12 else angle * axis / norm
    return angle * antisymmetric / (2.0 * math.sin(angle))


def pose_from_optitrack(position: np.ndarray, quaternion: np.ndarray) -> Pose:
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("OptiTrack position must contain three finite values")
    return Pose(position.copy(), quaternion_to_matrix(quaternion))


class RelativePoseMapper:
    """Map controller movement relative to a clutch pose into the robot base frame."""

    def __init__(self, position_scale: float = 1.0, orientation_scale: float = 1.0):
        if position_scale <= 0.0 or orientation_scale <= 0.0:
            raise ValueError("pose scales must be positive")
        self.position_scale = float(position_scale)
        self.orientation_scale = float(orientation_scale)
        self._controller_anchor: Optional[Pose] = None
        self._robot_anchor: Optional[Pose] = None

    @property
    def enabled(self) -> bool:
        return self._controller_anchor is not None and self._robot_anchor is not None

    def enable(self, controller_pose: Pose, robot_pose: Pose) -> None:
        self._controller_anchor = controller_pose.copy()
        self._robot_anchor = robot_pose.copy()

    def disable(self) -> None:
        self._controller_anchor = None
        self._robot_anchor = None

    def update(self, controller_pose: Pose) -> Pose:
        if not self.enabled:
            raise RuntimeError("relative pose mapper is not enabled")
        assert self._controller_anchor is not None
        assert self._robot_anchor is not None

        controller_anchor = self._controller_anchor
        robot_anchor = self._robot_anchor
        world_delta = controller_pose.position - controller_anchor.position
        relative_position = (
            self.position_scale * controller_anchor.rotation.T @ world_delta
        )
        relative_rotation = controller_anchor.rotation.T @ controller_pose.rotation
        if self.orientation_scale != 1.0:
            relative_vector = rotation_matrix_to_vector(relative_rotation)
            relative_rotation = rotation_vector_to_matrix(
                self.orientation_scale * relative_vector
            )
        return Pose(
            robot_anchor.position + relative_position,
            project_to_rotation_matrix(robot_anchor.rotation @ relative_rotation),
        )


def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    if rotation_vector.shape != (3,) or not np.all(np.isfinite(rotation_vector)):
        raise ValueError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(rotation_vector))
    x, y, z = rotation_vector if angle < 1e-12 else rotation_vector / angle
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    if angle < 1e-12:
        return project_to_rotation_matrix(np.eye(3) + skew)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def task_space_velocity(
    target_pose: Pose,
    current_pose: Pose,
    position_gain: float,
    rotation_gain: float,
    linear_limits: np.ndarray,
    angular_limits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute base-frame twist and independently apply per-axis limits."""

    linear_limits = np.asarray(linear_limits, dtype=float)
    angular_limits = np.asarray(angular_limits, dtype=float)
    if linear_limits.shape != (3,) or angular_limits.shape != (3,):
        raise ValueError("linear and angular limits must each contain three values")
    if np.any(linear_limits <= 0.0) or np.any(angular_limits <= 0.0):
        raise ValueError("velocity limits must be positive")

    position_error = target_pose.position - current_pose.position
    rotation_error = target_pose.rotation @ current_pose.rotation.T
    rotation_error_vector = rotation_matrix_to_vector(rotation_error)
    linear = np.clip(position_gain * position_error, -linear_limits, linear_limits)
    angular = np.clip(
        rotation_gain * rotation_error_vector, -angular_limits, angular_limits
    )
    return np.concatenate((linear, angular)), position_error, rotation_error_vector
