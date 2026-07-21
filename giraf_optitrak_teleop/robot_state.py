"""Validation and FK reconstruction for the robot controller's JSON state."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from control.RRPRRR_kinematic_model import num_forward_transform

from .geometry import Pose, project_to_rotation_matrix


PITCH_KIN_OFFSET = np.pi / 2.0
THETA4_KIN_OFFSET = np.pi / 2.0
THETA5_KIN_OFFSET = -np.pi / 2.0
THETA6_KIN_OFFSET = 0.0

BOOM_MIN_RAD = -30.0
BOOM_MAX_RAD = 0.0
D3_MIN_M = 0.31
D3_MAX_M = 2.121
THETA4_MIN_RAD = -1000.0 * 2.0 * np.pi / 4096.0
THETA4_MAX_RAD = 1660.0 * 2.0 * np.pi / 4096.0
THETA5_MIN_RAD = -1000.0 * 2.0 * np.pi / 4096.0
THETA5_MAX_RAD = 1000.0 * 2.0 * np.pi / 4096.0
THETA6_MIN_RAD = -2000.0 * 2.0 * np.pi / 4096.0
THETA6_MAX_RAD = 2000.0 * 2.0 * np.pi / 4096.0

_BOOM_P1 = -0.0508
_BOOM_P2 = -0.4122
_BOOM_P3 = -15.2992
_BOOM_P4 = 4.7840


def get_boom_motor_rad(d3: float) -> float:
    """Convert prismatic extension in metres to spool motor radians."""

    d3 = float(d3)
    return _BOOM_P1 * d3**3 + _BOOM_P2 * d3**2 + _BOOM_P3 * d3 + _BOOM_P4


def get_boom_length_d3(
    boom_pos: float, tol: float = 1e-10, max_iter: int = 20
) -> float:
    """Invert the deployed cubic spool mapping using bounded Newton iteration."""

    boom_pos = float(boom_pos)
    if not math.isfinite(boom_pos):
        raise ValueError("boom position is not finite")
    d3 = (boom_pos - _BOOM_P4) / _BOOM_P3
    residual = math.inf
    for _ in range(max_iter):
        f = get_boom_motor_rad(d3) - boom_pos
        residual = abs(f)
        fp = 3.0 * _BOOM_P1 * d3**2 + 2.0 * _BOOM_P2 * d3 + _BOOM_P3
        if abs(fp) < 1e-12:
            raise ValueError("boom mapping derivative is singular")
        d3 -= f / fp
        if abs(f) < tol:
            break
    residual = abs(get_boom_motor_rad(d3) - boom_pos)
    if not math.isfinite(d3):
        raise ValueError("boom conversion produced a non-finite extension")
    if residual >= tol:
        raise ValueError("boom conversion did not converge")
    return d3


@dataclass(frozen=True)
class RobotState:
    stamp_sec: float
    receipt_ns: int
    active_source: str
    command_source_param: str
    stop_latched: bool
    roll: float
    pitch: float
    boom: float
    d3: float
    th4: float
    th5: float
    th6: float
    grip: float
    pose: Pose


def parse_robot_state(encoded: str, receipt_ns: Optional[int] = None) -> RobotState:
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("robot state must be a JSON object")
    arm = payload.get("arm")
    if not isinstance(arm, dict):
        raise ValueError("robot state is missing the arm object")

    def finite_float(container: dict, key: str) -> float:
        value = float(container[key])
        if not math.isfinite(value):
            raise ValueError("robot state field %s is not finite" % key)
        return value

    stamp_sec = finite_float(payload, "stamp_sec")
    roll = finite_float(arm, "roll")
    pitch = finite_float(arm, "pitch")
    boom = finite_float(arm, "boom")
    th4 = finite_float(arm, "th4")
    th5 = finite_float(arm, "th5")
    th6 = finite_float(arm, "th6")
    grip = finite_float(arm, "grip")
    if stamp_sec < 0.0:
        raise ValueError("robot state stamp is negative")
    if roll < -np.pi / 2.0 - 1e-6 or roll > np.pi / 2.0 + 1e-6:
        raise ValueError("roll is outside the controller range")
    if pitch < -1e-6 or pitch > np.pi / 2.0 + 1e-6:
        raise ValueError("pitch is outside the controller range")
    if grip < -1e-6 or grip > 1.0 + 1e-6:
        raise ValueError("gripper state is outside the normalized range")
    for name, value, lower, upper in (
        ("th4", th4, THETA4_MIN_RAD, THETA4_MAX_RAD),
        ("th5", th5, THETA5_MIN_RAD, THETA5_MAX_RAD),
        ("th6", th6, THETA6_MIN_RAD, THETA6_MAX_RAD),
    ):
        if value < lower - 1e-6 or value > upper + 1e-6:
            raise ValueError("%s is outside the controller range" % name)
    if boom < BOOM_MIN_RAD - 1e-6 or boom > BOOM_MAX_RAD + 1e-6:
        raise ValueError("boom position is outside the controller range")
    d3 = get_boom_length_d3(boom)
    if d3 < D3_MIN_M - 0.01 or d3 > D3_MAX_M + 0.01:
        raise ValueError("converted boom extension is outside the expected range")

    joint_coordinates = np.array(
        (
            roll,
            pitch + PITCH_KIN_OFFSET,
            d3,
            th4 + THETA4_KIN_OFFSET,
            th5 + THETA5_KIN_OFFSET,
            th6 + THETA6_KIN_OFFSET,
        ),
        dtype=float,
    )
    transform = np.asarray(num_forward_transform(joint_coordinates), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("forward kinematics produced an invalid transform")

    stop_latched = payload.get("stop_latched")
    if not isinstance(stop_latched, bool):
        raise ValueError("robot state stop_latched must be a Boolean")

    return RobotState(
        stamp_sec=stamp_sec,
        receipt_ns=time.monotonic_ns() if receipt_ns is None else int(receipt_ns),
        active_source=str(payload.get("active_source", "")).strip().lower(),
        command_source_param=str(payload.get("command_source_param", ""))
        .strip()
        .lower(),
        stop_latched=stop_latched,
        roll=roll,
        pitch=pitch,
        boom=boom,
        d3=d3,
        th4=th4,
        th5=th5,
        th6=th6,
        grip=grip,
        pose=Pose(
            transform[:3, 3].copy(), project_to_rotation_matrix(transform[:3, :3])
        ),
    )


class RobotStateTracker:
    """Reject a latched stale state until a second progressing message arrives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[RobotState] = None
        self._progressing_messages = 0
        self._last_error = "no robot state received"

    def update(self, encoded: str, receipt_ns: Optional[int] = None) -> bool:
        try:
            state = parse_robot_state(encoded, receipt_ns=receipt_ns)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            with self._lock:
                self._last_error = "invalid robot state: %s" % exc
                self._progressing_messages = 0
            return False
        with self._lock:
            if self._latest is not None and state.stamp_sec > self._latest.stamp_sec:
                self._progressing_messages = min(2, self._progressing_messages + 1)
            else:
                self._progressing_messages = 1
            self._latest = state
            self._last_error = ""
        return True

    def snapshot(self) -> Optional[RobotState]:
        with self._lock:
            return self._latest

    def health(self, now_ns: int, max_age_ms: float) -> Tuple[bool, str, float]:
        with self._lock:
            state = self._latest
            count = self._progressing_messages
            error = self._last_error
        if state is None:
            return False, error, math.inf
        age_ms = (now_ns - state.receipt_ns) / 1e6
        if error:
            return False, error, age_ms
        if count < 2:
            return False, "waiting for a second progressing robot state", age_ms
        if age_ms > max_age_ms:
            return False, "robot state is stale", age_ms
        if state.stop_latched:
            return False, "robot controller stop is latched", age_ms
        if state.active_source != "teleop" or state.command_source_param != "teleop":
            return False, "robot command source is not teleop", age_ms
        return True, "ready", age_ms
