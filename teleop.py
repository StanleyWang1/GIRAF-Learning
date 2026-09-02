#!/usr/bin/env python3
"""Pure-Python OptiTrack teleoperation for the GIRAF arm."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from control_table import (
    MOTOR21_HOME,
    MOTOR21_LIMITS,
    MOTOR22_HOME,
    MOTOR22_LIMITS,
    MOTOR23_HOME,
    MOTOR23_LIMITS,
    MOTOR24_CLOSED,
    MOTOR24_OPEN,
    TORQUE_ENABLE,
)
from dynamixel_driver import (
    GRIPPER,
    dynamixel_connect,
    dynamixel_disconnect,
    dynamixel_drive,
    radians_to_ticks,
)
from keyboard_driver import (
    keyboard_connect,
    keyboard_control,
    keyboard_disconnect,
    keyboard_status,
)
from kinematic_model import num_forward_transform, num_jacobian
from mab_worker import MabWorker
from optitrack_driver import DEFAULT_RIGID_BODY_ID, DEFAULT_SERVER_IP, OptiTrackDriver


CONTROL_HZ = 100.0
POSE_TIMEOUT = 0.15
POSITION_GAIN = ROTATION_GAIN = 5.0
LINEAR_LIMIT = np.array((0.5, 0.5, 0.5))
ANGULAR_LIMIT = np.array((1.0, 1.0, 1.0))
MAX_JOINT_SPEED = np.array((1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
ROLL_LIMIT, PITCH_MIN, PITCH_MAX = math.pi / 2, 0.0, math.pi / 2
D3_MIN, BOOM_MIN, BOOM_MAX = 0.31, -30.0, 0.0
ROTATION_BASIS = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


@dataclass
class SharedState:
    pose_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pose_quaternion: np.ndarray = field(default_factory=lambda: np.array((0.0, 0.0, 0.0, 1.0)))
    pose_time: float = 0.0
    pose_frame: int = 0
    joints: np.ndarray = field(default_factory=lambda: np.array((0.0, 0.0, D3_MIN, 0.0, 0.0, 0.0)))
    tracking: bool = False
    optitrack_ready: bool = False
    motors_ready: bool = False
    grasp: bool = False
    error: str = ""


STATE = SharedState()
LOCK = threading.Lock()
STOP = threading.Event()


def fail(source: str, error: Exception | str) -> None:
    with LOCK:
        if not STATE.error:
            STATE.error = f"{source}: {error}"
    STOP.set()


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
    skew = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]))
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


def boom_motor_position(extension: float) -> float:
    return -0.0508 * extension**3 - 0.4122 * extension**2 - 15.2992 * extension + 4.7840


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
    return np.asarray(joints) + np.array((0.0, math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, 0.0))


def end_effector_pose(joints) -> tuple[np.ndarray, np.ndarray]:
    transform = num_forward_transform(model_joints(joints))
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("invalid forward kinematics")
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def wrist_ticks(joints) -> list[int]:
    return [
        MOTOR21_HOME + radians_to_ticks(joints[3]),
        MOTOR22_HOME + radians_to_ticks(-joints[4]),
        MOTOR23_HOME + radians_to_ticks(joints[5]),
    ]


def optitrack_loop(server_ip: str, client_ip: str | None, rigid_id: int) -> None:
    driver = None
    try:
        print(f"[INIT][OPTI] Connecting to {server_ip}, rigid body {rigid_id}...", flush=True)
        driver = OptiTrackDriver(server_ip, rigid_id, client_ip=client_ip)
        driver.connect()
        print("[INIT][OPTI] Connected; waiting for pose frames.", flush=True)
        with LOCK:
            STATE.optitrack_ready = True
        while not STOP.is_set():
            try:
                pose = driver.get_latest_pose(timeout=0.25)
            except TimeoutError:
                continue
            with LOCK:
                STATE.pose_position = np.asarray(pose.position_m, dtype=float)
                STATE.pose_quaternion = np.asarray(pose.quaternion_xyzw, dtype=float)
                STATE.pose_time = pose.received_monotonic_ns / 1e9
                STATE.pose_frame = pose.frame_number
    except Exception as exc:
        fail("OptiTrack", exc)
    finally:
        if driver is not None:
            driver.close()


def control_loop(keyboard) -> None:
    tracking = False
    armed = False
    controller_anchor_position = controller_anchor_quaternion = controller_anchor_rotation = None
    robot_anchor_position = robot_anchor_rotation = None
    last = time.monotonic()
    period = 1.0 / CONTROL_HZ

    try:
        while not STOP.is_set():
            start = time.monotonic()
            dt = min(max(start - last, 0.0), 0.02)
            last = start
            keys = keyboard_status(keyboard)

            with LOCK:
                joints = STATE.joints.copy()
                pose_position = STATE.pose_position.copy()
                pose_quaternion = STATE.pose_quaternion.copy()
                pose_time = STATE.pose_time
                motors_ready = STATE.motors_ready

            fresh = pose_time > 0.0 and start - pose_time <= POSE_TIMEOUT
            if not keys["clutch"]:
                tracking, armed = False, True
            elif tracking and (not fresh or not motors_ready):
                tracking, armed = False, False
            elif not tracking and armed and fresh and motors_ready:
                controller_anchor_position = pose_position.copy()
                controller_anchor_quaternion = pose_quaternion.copy()
                controller_anchor_rotation = quaternion_matrix(pose_quaternion)
                robot_anchor_position, robot_anchor_rotation = end_effector_pose(joints)
                tracking, armed = True, False

            if tracking:
                relative_position = controller_anchor_rotation.T @ (pose_position - controller_anchor_position)
                controller_rotation = quaternion_matrix(relative_quaternion(controller_anchor_quaternion, pose_quaternion))
                target_position = robot_anchor_position + relative_position
                target_rotation = robot_anchor_rotation @ (ROTATION_BASIS.T @ controller_rotation @ ROTATION_BASIS)
                current_position, current_rotation = end_effector_pose(joints)
                linear = limited_velocity(target_position - current_position, POSITION_GAIN, 0.002, LINEAR_LIMIT)
                angular = limited_velocity(rotation_vector(target_rotation @ current_rotation.T), ROTATION_GAIN, 0.01, ANGULAR_LIMIT)
                jacobian = num_jacobian(model_joints(joints))
                if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
                    raise RuntimeError("invalid Jacobian")
                qdot = np.linalg.pinv(jacobian, rcond=1e-3) @ np.concatenate((linear, angular))
                if not np.all(np.isfinite(qdot)):
                    raise RuntimeError("RMRC produced a non-finite joint velocity")
                ratio = float(np.max(np.abs(qdot) / MAX_JOINT_SPEED))
                if ratio > 1.0:
                    qdot /= ratio

                proposed = joints + dt * qdot
                proposed[0] = np.clip(proposed[0], -ROLL_LIMIT, ROLL_LIMIT)
                proposed[1] = np.clip(proposed[1], PITCH_MIN, PITCH_MAX)
                boom = float(np.clip(boom_motor_position(max(D3_MIN, proposed[2])), BOOM_MIN, BOOM_MAX))
                proposed[2] = max(D3_MIN, boom_extension(boom))
                ticks = wrist_ticks(proposed)
                for index, tick, limits in zip((3, 4, 5), ticks, (MOTOR21_LIMITS, MOTOR22_LIMITS, MOTOR23_LIMITS)):
                    if not limits[0] <= tick <= limits[1]:
                        proposed[index] = joints[index]
                joints = proposed

            with LOCK:
                STATE.joints = joints
                STATE.tracking = tracking
                STATE.grasp = keys["grasp"]

            STOP.wait(max(0.0, period - (time.monotonic() - start)))
    except Exception as exc:
        fail("control", exc)


def motor_loop() -> None:
    mab = dxl = sync_write = None
    period = 1.0 / CONTROL_HZ
    try:
        print("[INIT][DXL] Connecting and configuring motors...", flush=True)
        dxl, sync_write = dynamixel_connect()
        print("[INIT][DXL] Wrist motors configured.", flush=True)
        print("[INIT][DXL] Enabling gripper torque...", flush=True)
        if not dxl.WRITE(GRIPPER, TORQUE_ENABLE, 1):
            raise RuntimeError("could not enable gripper torque")
        print("[INIT][DXL] Gripper torque enabled.", flush=True)
        print("[INIT][MAB] Starting isolated worker...", flush=True)
        mab = MabWorker()
        mab.start()
        with LOCK:
            STATE.motors_ready = True
        print("[INIT] All motors ready; control output active.", flush=True)

        while not STOP.is_set():
            start = time.monotonic()
            with LOCK:
                joints = STATE.joints.copy()
                grasp = STATE.grasp
            boom = float(np.clip(boom_motor_position(joints[2]), BOOM_MIN, BOOM_MAX))
            mab.command(joints[0], joints[1], boom)
            gripper = MOTOR24_CLOSED if grasp else MOTOR24_OPEN
            if not dynamixel_drive(dxl, sync_write, wrist_ticks(joints) + [gripper]):
                raise RuntimeError("Dynamixel command failed")
            STOP.wait(max(0.0, period - (time.monotonic() - start)))
    except Exception as exc:
        fail("motors", exc)
    finally:
        with LOCK:
            STATE.motors_ready = False
            STATE.tracking = False
        if dxl is not None:
            print("[STOP][DXL] Disabling torque...", flush=True)
            dynamixel_disconnect(dxl)
            dxl.close_port()
            print("[STOP][DXL] Torque disabled and port closed.", flush=True)
        if mab is not None:
            mab.stop()


def status_loop(keyboard, workers: list[threading.Thread]) -> None:
    while not STOP.wait(0.25):
        for thread in workers:
            if not thread.is_alive():
                fail(thread.name, "thread stopped")
                return
        keys = keyboard_status(keyboard)
        with LOCK:
            age = time.monotonic() - STATE.pose_time if STATE.pose_time else math.inf
            status = (
                STATE.tracking,
                STATE.optitrack_ready,
                STATE.motors_ready,
                STATE.grasp,
                STATE.joints.copy(),
            )
        age_text = "n/a" if not math.isfinite(age) else f"{age * 1000:.0f}ms"
        print(
            f"\rtracking={status[0]} pedal={keys['clutch']} grasp={status[3]} "
            f"opti={status[1]} motors={status[2]} pose={age_text} "
            f"q={np.round(status[4], 3)}   ",
            end="",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    parser.add_argument("--client-ip", default=None)
    parser.add_argument("--rigid-id", type=int, default=DEFAULT_RIGID_BODY_ID)
    parser.add_argument("--hardware", action="store_true", help="enable and command the physical motors")
    args = parser.parse_args()

    keyboard = None
    workers: list[threading.Thread] = []
    status = None
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())

    try:
        print("[INIT][KEYBOARD] Opening input devices...", flush=True)
        keyboard = keyboard_connect()
        print(f"[INIT][KEYBOARD] Opened {len(keyboard.devices)} input devices.", flush=True)
        workers.append(threading.Thread(target=keyboard_control, args=(keyboard, STOP), name="keyboard"))
        workers.append(threading.Thread(target=optitrack_loop, args=(args.server_ip, args.client_ip, args.rigid_id), name="optitrack"))
        workers.append(threading.Thread(target=control_loop, args=(keyboard,), name="control"))

        if args.hardware:
            print("HARDWARE MODE: stage MD80 joints at home and begin with the pedal released.")
        else:
            with LOCK:
                STATE.motors_ready = True
            print("DRY RUN: motors will not be opened.")

        print("[INIT] Starting keyboard, OptiTrack, and control workers...", flush=True)
        for thread in workers:
            thread.start()
        print("[INIT] Worker threads started.", flush=True)
        status = threading.Thread(target=status_loop, args=(keyboard, workers), name="status")
        status.start()

        if args.hardware:
            motor_loop()
        else:
            STOP.wait()
    except Exception as exc:
        fail("startup", exc)
    finally:
        STOP.set()
        for thread in workers:
            if thread.is_alive():
                thread.join(timeout=3.0)
        if status is not None and status.is_alive():
            status.join(timeout=1.0)
        if keyboard is not None:
            keyboard_disconnect(keyboard)
        with LOCK:
            error = STATE.error
        print(f"\nStopped{': ' + error if error else '.'}")

    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
