"""Pure kinematic and safety operations for guarded policy deployment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from giraf.data.schema import ACTION_DIM, GRASP_INDEX, STATE_DIM, state_vector
from giraf.drivers.dynamixel_config import (
    MOTOR21_HOME,
    MOTOR21_LIMITS,
    MOTOR22_HOME,
    MOTOR22_LIMITS,
    MOTOR23_HOME,
    MOTOR23_LIMITS,
    MOTOR24_CLOSED,
    MOTOR24_OPEN,
    TICKS_PER_REV,
)
from giraf.kinematics import num_forward_transform, num_jacobian


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Independent bounds applied after policy inference."""

    linear_velocity: tuple[float, float, float] = (0.5, 0.5, 0.5)
    angular_velocity: tuple[float, float, float] = (1.0, 1.0, 1.0)
    joint_speed: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
    )
    staging_joint_speed: tuple[float, float, float, float, float, float] = (
        0.15,
        0.15,
        0.05,
        0.25,
        0.25,
        0.25,
    )
    roll_limit: float = math.pi / 2
    pitch_min: float = 0.0
    pitch_max: float = math.pi / 2
    boom_extension_min: float = 0.31
    boom_motor_min: float = -30.0
    boom_motor_max: float = 0.0
    jacobian_rcond: float = 1e-3

    def __post_init__(self) -> None:
        positive = (
            *self.linear_velocity,
            *self.angular_velocity,
            *self.joint_speed,
            *self.staging_joint_speed,
        )
        if min(positive) <= 0:
            raise ValueError("velocity limits must be positive")
        if self.roll_limit <= 0 or self.pitch_min > self.pitch_max:
            raise ValueError("invalid revolute-joint limits")
        if self.boom_extension_min <= 0 or self.boom_motor_min > self.boom_motor_max:
            raise ValueError("invalid boom limits")
        if self.jacobian_rcond <= 0:
            raise ValueError("jacobian_rcond must be positive")


@dataclass(frozen=True, slots=True)
class JointCommand:
    """One bounded 100 Hz command derived from a policy task-space action."""

    action: np.ndarray
    joint_velocity: np.ndarray
    joint_position: np.ndarray
    can_position_target: tuple[float, float, float]
    dynamixel_target_ticks: tuple[int, int, int, int]
    grasp: bool


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def guard_policy_action(
    action,
    *,
    scale: float,
    allow_grasp: bool,
    limits: SafetyLimits | None = None,
) -> np.ndarray:
    """Validate, scale, and independently clamp a physical policy action."""

    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("action scale must be finite and in [0, 1]")
    limits = limits or SafetyLimits()
    guarded = _finite_vector(action, ACTION_DIM, "policy action").copy()
    guarded[:3] = np.clip(
        guarded[:3], -np.asarray(limits.linear_velocity), limits.linear_velocity
    ) * scale
    guarded[3:6] = np.clip(
        guarded[3:6], -np.asarray(limits.angular_velocity), limits.angular_velocity
    ) * scale
    guarded[GRASP_INDEX] = (
        float(guarded[GRASP_INDEX] >= 0.5) if allow_grasp else 0.0
    )
    return guarded.astype(np.float32)


def boom_motor_position(extension: float) -> float:
    return (
        -0.0508 * extension**3
        - 0.4122 * extension**2
        - 15.2992 * extension
        + 4.7840
    )


def boom_extension(motor_position: float) -> float:
    extension = (motor_position - 4.7840) / -15.2992
    for _ in range(20):
        error = boom_motor_position(extension) - motor_position
        slope = -0.1524 * extension**2 - 0.8244 * extension - 15.2992
        extension -= error / slope
        if abs(error) < 1e-10:
            break
    return extension


def model_joints(joints) -> np.ndarray:
    return _finite_vector(joints, 6, "joints") + np.array(
        (0.0, math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, 0.0)
    )


def end_effector_pose(joints) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(num_forward_transform(model_joints(joints)), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError("invalid forward kinematics")
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def state_from_joints(joints) -> np.ndarray:
    joints = _finite_vector(joints, 6, "joints")
    position, rotation = end_effector_pose(joints)
    state = state_vector(joints, position, rotation)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise RuntimeError("invalid deployment state")
    return state


def _radians_to_ticks(radians: float) -> int:
    return int(radians / (2 * math.pi) * TICKS_PER_REV)


def wrist_ticks(joints) -> tuple[int, int, int]:
    joints = _finite_vector(joints, 6, "joints")
    return (
        MOTOR21_HOME + _radians_to_ticks(float(joints[3])),
        MOTOR22_HOME + _radians_to_ticks(float(-joints[4])),
        MOTOR23_HOME + _radians_to_ticks(float(joints[5])),
    )


def _command_from_position(
    current: np.ndarray,
    proposed: np.ndarray,
    *,
    dt: float,
    action: np.ndarray,
    limits: SafetyLimits,
) -> JointCommand:
    proposed = proposed.copy()
    proposed[0] = np.clip(proposed[0], -limits.roll_limit, limits.roll_limit)
    proposed[1] = np.clip(proposed[1], limits.pitch_min, limits.pitch_max)
    boom = float(
        np.clip(
            boom_motor_position(max(limits.boom_extension_min, proposed[2])),
            limits.boom_motor_min,
            limits.boom_motor_max,
        )
    )
    proposed[2] = max(limits.boom_extension_min, boom_extension(boom))

    ticks = wrist_ticks(proposed)
    for index, tick, tick_limits in zip(
        (3, 4, 5), ticks, (MOTOR21_LIMITS, MOTOR22_LIMITS, MOTOR23_LIMITS)
    ):
        if not tick_limits[0] <= tick <= tick_limits[1]:
            proposed[index] = current[index]

    effective_velocity = (proposed - current) / dt
    if not np.isfinite(effective_velocity).all():
        raise RuntimeError("bounded joint velocity is non-finite")
    boom = float(
        np.clip(
            boom_motor_position(float(proposed[2])),
            limits.boom_motor_min,
            limits.boom_motor_max,
        )
    )
    grasp = bool(action[GRASP_INDEX] >= 0.5)
    gripper = MOTOR24_CLOSED if grasp else MOTOR24_OPEN
    wrist = wrist_ticks(proposed)
    return JointCommand(
        action=action.astype(np.float32),
        joint_velocity=effective_velocity.astype(np.float32),
        joint_position=proposed.astype(np.float32),
        can_position_target=(float(proposed[0]), float(proposed[1]), boom),
        dynamixel_target_ticks=(*wrist, int(gripper)),
        grasp=grasp,
    )


def plan_joint_command(
    joints,
    action,
    *,
    dt: float,
    limits: SafetyLimits | None = None,
) -> JointCommand:
    """Map a guarded task-space action to bounded position targets."""

    if not math.isfinite(dt) or not 0.0 < dt <= 0.02:
        raise ValueError("dt must be finite and in (0, 0.02]")
    limits = limits or SafetyLimits()
    current = _finite_vector(joints, 6, "joints")
    guarded = _finite_vector(action, ACTION_DIM, "guarded action")

    jacobian = np.asarray(num_jacobian(model_joints(current)), dtype=np.float64)
    if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
        raise RuntimeError("invalid Jacobian")
    velocity = np.linalg.pinv(jacobian, rcond=limits.jacobian_rcond) @ guarded[:6]
    if not np.isfinite(velocity).all():
        raise RuntimeError("RMRC produced a non-finite joint velocity")

    speed_limit = np.asarray(limits.joint_speed)
    ratio = float(np.max(np.abs(velocity) / speed_limit))
    if ratio > 1.0:
        velocity /= ratio

    return _command_from_position(
        current,
        current + dt * velocity,
        dt=dt,
        action=guarded,
        limits=limits,
    )


def validate_staging_target(
    target,
    *,
    limits: SafetyLimits | None = None,
) -> np.ndarray:
    """Validate a recorded start pose against independent hardware limits."""

    limits = limits or SafetyLimits()
    target = _finite_vector(target, 6, "staging target")
    if not -limits.roll_limit <= target[0] <= limits.roll_limit:
        raise ValueError("staging target violates the roll limit")
    if not limits.pitch_min <= target[1] <= limits.pitch_max:
        raise ValueError("staging target violates the pitch limit")
    boom = boom_motor_position(float(target[2]))
    if (
        target[2] < limits.boom_extension_min
        or not limits.boom_motor_min <= boom <= limits.boom_motor_max
    ):
        raise ValueError("staging target violates the boom limit")
    for tick, tick_limits in zip(
        wrist_ticks(target), (MOTOR21_LIMITS, MOTOR22_LIMITS, MOTOR23_LIMITS)
    ):
        if not tick_limits[0] <= tick <= tick_limits[1]:
            raise ValueError("staging target violates a wrist limit")
    return target.astype(np.float32)


def plan_staging_command(
    joints,
    target,
    *,
    dt: float,
    limits: SafetyLimits | None = None,
) -> JointCommand:
    """Move directly toward a validated start pose at conservative speeds."""

    if not math.isfinite(dt) or not 0.0 < dt <= 0.02:
        raise ValueError("dt must be finite and in (0, 0.02]")
    limits = limits or SafetyLimits()
    current = _finite_vector(joints, 6, "joints")
    target = validate_staging_target(target, limits=limits).astype(np.float64)
    max_step = np.asarray(limits.staging_joint_speed) * dt
    proposed = current + np.clip(target - current, -max_step, max_step)
    return _command_from_position(
        current,
        proposed,
        dt=dt,
        action=np.zeros(ACTION_DIM, dtype=np.float64),
        limits=limits,
    )


def state_bound_violations(
    state,
    low,
    high,
    *,
    margin_fraction: float = 0.05,
) -> tuple[int, ...]:
    """Return state dimensions outside expanded training-data bounds."""

    if not math.isfinite(margin_fraction) or margin_fraction < 0:
        raise ValueError("margin_fraction must be finite and non-negative")
    state = _finite_vector(state, STATE_DIM, "state")
    low = _finite_vector(low, STATE_DIM, "state low")
    high = _finite_vector(high, STATE_DIM, "state high")
    if np.any(high < low):
        raise ValueError("state bounds must satisfy low <= high")
    span = np.maximum(high - low, 1e-6)
    below = state < low - margin_fraction * span
    above = state > high + margin_fraction * span
    return tuple(int(index) for index in np.flatnonzero(below | above))
