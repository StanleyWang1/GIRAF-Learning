#!/usr/bin/env python3
"""Teleoperate the arm-only GIRAF simulation from one OptiTrack rigid body.

Space acts as a clutch. While tracking is enabled, the controller pose is
mapped relative to the pose captured at enable time using the same convention
as stream_and_visualize_ONE_controller.py. Cartesian pose error is converted
to a base-frame twist, mapped through the provided kinematic Jacobian
pseudoinverse, and integrated into MuJoCo position-actuator targets.
"""

from __future__ import annotations

import argparse
import math
import queue
import signal
import socket
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = REPO_ROOT / "control"
for import_path in (REPO_ROOT, CONTROL_DIR):
    path_string = str(import_path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from RRPRRR_kinematic_model import num_forward_transform, num_jacobian
from sim_model import GirafSimulation
from sim_model.simulation import ROBOT_ACTUATORS, ROBOT_JOINTS

try:
    from natnet import NatNetClient, Version
    from natnet.packet_buffer import PacketBuffer
except ModuleNotFoundError as exc:
    NatNetClient = None
    Version = None
    PacketBuffer = None
    NATNET_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    NATNET_IMPORT_ERROR = None


DEFAULT_RIGID_ID = 33
DEFAULT_SERVER_IP = "172.24.68.77"
DEFAULT_DATA_PORT = 1511
DEFAULT_COMMAND_PORT = 1510
DEFAULT_USE_MULTICAST = False

MAX_OPTI_AGE_MS = 100.0
POSITION_SCALE = 1.0
ORIENTATION_SCALE = 1.0

POSITION_GAIN = 4.0
ROTATION_GAIN = 4.0
MAX_LINEAR_SPEED_M_S = 0.5
MAX_ANGULAR_SPEED_RAD_S = 2.0

JACOBIAN_RCOND = 1e-4
MAX_JOINT_SPEED = np.array([1.5, 1.5, 0.5, 2.0, 2.0, 2.0])
KINEMATIC_JOINT_OFFSETS = np.array(
    [0.0, np.pi / 2.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0]
)

STATUS_PERIOD_S = 0.25
KEY_DEBOUNCE_S = 0.2


@dataclass(frozen=True)
class OptiSample:
    local_ns: int
    frame: int | None
    motive_timestamp: float | None
    rigid_id: int
    seen: bool | None
    px: float | None
    py: float | None
    pz: float | None
    qx: float | None
    qy: float | None
    qz: float | None
    qw: float | None


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray


class OptiTrackReceiver:
    """Thread-safe latest-sample receiver for one rigid body."""

    def __init__(
        self,
        server_ip: str,
        client_ip: str,
        rigid_id: int,
        data_port: int,
        command_port: int,
        use_multicast: bool,
    ) -> None:
        if NATNET_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The NatNet Python package is required for OptiTrack streaming"
            ) from NATNET_IMPORT_ERROR

        self.rigid_id = rigid_id
        self.lock = threading.Lock()
        self.sample: OptiSample | None = None
        self.client = NatNetClient(
            server_ip_address=server_ip,
            local_ip_address=client_ip,
            command_port=command_port,
            data_port=data_port,
            use_multicast=use_multicast,
        )
        self.client._NatNetClient__current_protocol_version = Version(4, 3)
        self.client.on_data_frame_received_event.handlers.append(self._on_frame)

    def _on_frame(self, frame) -> None:
        local_ns = time.monotonic_ns()
        body = next(
            (
                rigid_body
                for rigid_body in frame.rigid_bodies or ()
                if rigid_body.id_num == self.rigid_id
            ),
            None,
        )
        if body is None:
            sample = OptiSample(
                local_ns=local_ns,
                frame=frame.prefix.frame_number,
                motive_timestamp=frame.suffix.timestamp,
                rigid_id=self.rigid_id,
                seen=None,
                px=None,
                py=None,
                pz=None,
                qx=None,
                qy=None,
                qz=None,
                qw=None,
            )
        else:
            sample = OptiSample(
                local_ns=local_ns,
                frame=frame.prefix.frame_number,
                motive_timestamp=frame.suffix.timestamp,
                rigid_id=body.id_num,
                seen=body.tracking_valid,
                px=body.pos[0],
                py=body.pos[1],
                pz=body.pos[2],
                qx=body.rot[0],
                qy=body.rot[1],
                qz=body.rot[2],
                qw=body.rot[3],
            )
        with self.lock:
            self.sample = sample

    def start(self) -> None:
        self.client.connect(timeout=5.0)
        print(
            "OptiTrack connected: "
            f"protocol={self.client.protocol_version} "
            f"server={self.client.server_info}"
        )
        self.client.run_async()

    def latest(self) -> OptiSample | None:
        with self.lock:
            return self.sample

    def stop(self) -> None:
        self.client.shutdown()


class RelativePoseMapper:
    """Map OptiTrack motion from a clutch pose to a robot target pose."""

    def __init__(
        self,
        position_scale: float = POSITION_SCALE,
        orientation_scale: float = ORIENTATION_SCALE,
    ) -> None:
        self.position_scale = position_scale
        self.orientation_scale = orientation_scale
        self.enabled = False
        self.controller_anchor_position: np.ndarray | None = None
        self.controller_anchor_quaternion: np.ndarray | None = None
        self.controller_anchor_rotation: np.ndarray | None = None
        self.robot_anchor_pose: Pose | None = None
        self.target_pose: Pose | None = None

    def enable(self, sample: OptiSample, robot_pose: Pose) -> None:
        position, quaternion = sample_pose(sample)
        self.controller_anchor_position = position
        self.controller_anchor_quaternion = quaternion
        self.controller_anchor_rotation = quaternion_to_matrix(quaternion)
        self.robot_anchor_pose = copy_pose(robot_pose)
        self.target_pose = copy_pose(robot_pose)
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def update(self, sample: OptiSample) -> Pose:
        if not self.enabled:
            raise RuntimeError("RelativePoseMapper is not enabled")
        assert self.controller_anchor_position is not None
        assert self.controller_anchor_quaternion is not None
        assert self.controller_anchor_rotation is not None
        assert self.robot_anchor_pose is not None

        current_position, current_quaternion = sample_pose(sample)
        world_delta = current_position - self.controller_anchor_position
        relative_position = (
            self.position_scale * self.controller_anchor_rotation.T @ world_delta
        )
        relative_quaternion = normalize_quaternion(
            quaternion_multiply(
                quaternion_conjugate(self.controller_anchor_quaternion),
                current_quaternion,
            )
        )
        scaled_relative_quaternion = scale_quaternion_rotation(
            relative_quaternion, self.orientation_scale
        )

        self.target_pose = Pose(
            position=self.robot_anchor_pose.position + relative_position,
            rotation=project_to_rotation_matrix(
                self.robot_anchor_pose.rotation
                @ quaternion_to_matrix(scaled_relative_quaternion)
            ),
        )
        return copy_pose(self.target_pose)


def patch_natnet_string_decoder() -> None:
    if PacketBuffer is None:
        return
    original = PacketBuffer.read_string
    if getattr(original, "_bota_optitrack_patched", False):
        return

    def read_string_lossy(self, max_length=None, static_length=False):
        if max_length is None:
            data_slice = self._PacketBuffer__data[self.pointer :]
        else:
            data_slice = self._PacketBuffer__data[
                self.pointer : self.pointer + max_length
            ]
        encoded, _separator, _remainder = bytes(data_slice).partition(b"\0")
        decoded = encoded.decode("utf-8", errors="replace")
        if static_length:
            assert max_length is not None
            self.pointer += max_length
        else:
            self.pointer += len(encoded) + 1
        return decoded

    read_string_lossy._bota_optitrack_patched = True
    PacketBuffer.read_string = read_string_lossy


def local_ip_for_server(server_ip: str, command_port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, command_port))
        return sock.getsockname()[0]
    finally:
        sock.close()


def copy_pose(pose: Pose) -> Pose:
    return Pose(pose.position.copy(), pose.rotation.copy())


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("zero-length quaternion")
    return quaternion / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quaternion
    return np.array((-qx, -qy, -qz, qw), dtype=float)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=float,
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


def scale_quaternion_rotation(quaternion: np.ndarray, scale: float) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    half_angle = math.acos(float(np.clip(quaternion[3], -1.0, 1.0)))
    sin_half_angle = math.sin(half_angle)
    if sin_half_angle < 1e-9:
        return np.array((0.0, 0.0, 0.0, 1.0))
    axis = quaternion[:3] / sin_half_angle
    scaled_half_angle = scale * half_angle
    return normalize_quaternion(
        np.append(axis * math.sin(scaled_half_angle), math.cos(scaled_half_angle))
    )


def rotation_matrix_to_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    antisymmetric = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
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
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return np.zeros(3)
        return angle * axis / axis_norm
    return angle * antisymmetric / (2.0 * math.sin(angle))


def project_to_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    left, _singular_values, right_transpose = np.linalg.svd(rotation)
    projected = left @ right_transpose
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_transpose
    return projected


def limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 0.0:
        return vector
    return vector * (maximum / norm)


def sample_pose(sample: OptiSample) -> tuple[np.ndarray, np.ndarray]:
    values = (
        sample.px,
        sample.py,
        sample.pz,
        sample.qx,
        sample.qy,
        sample.qz,
        sample.qw,
    )
    if any(value is None for value in values):
        raise ValueError("OptiTrack sample does not contain a full pose")
    position = np.array((sample.px, sample.py, sample.pz), dtype=float)
    quaternion = normalize_quaternion(
        np.array((sample.qx, sample.qy, sample.qz, sample.qw), dtype=float)
    )
    return position, quaternion


def sample_status(
    receiver: OptiTrackReceiver, max_age_ms: float
) -> tuple[OptiSample | None, float, str]:
    sample = receiver.latest()
    if sample is None:
        return None, math.inf, "no sample"
    age_ms = (time.monotonic_ns() - sample.local_ns) / 1e6
    if sample.px is None:
        return None, age_ms, "rigid body absent"
    if sample.seen is False:
        return None, age_ms, "tracking invalid"
    if age_ms > max_age_ms:
        return None, age_ms, "sample stale"
    return sample, age_ms, "fresh"


def kinematic_joint_coordinates(sim_joint_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(sim_joint_positions, dtype=float)
    if positions.shape != (6,):
        raise ValueError(f"expected six arm joints, got shape {positions.shape}")
    return positions + KINEMATIC_JOINT_OFFSETS


def end_effector_pose(sim_joint_positions: np.ndarray) -> Pose:
    transform = np.asarray(
        num_forward_transform(kinematic_joint_coordinates(sim_joint_positions)),
        dtype=float,
    )
    return Pose(
        position=transform[:3, 3].copy(),
        rotation=project_to_rotation_matrix(transform[:3, :3]),
    )


def task_space_velocity(
    target_pose: Pose,
    current_pose: Pose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_error = target_pose.position - current_pose.position
    rotation_error = target_pose.rotation @ current_pose.rotation.T
    rotation_error_vector = rotation_matrix_to_vector(rotation_error)
    linear_velocity = limit_norm(POSITION_GAIN * position_error, MAX_LINEAR_SPEED_M_S)
    angular_velocity = limit_norm(
        ROTATION_GAIN * rotation_error_vector, MAX_ANGULAR_SPEED_RAD_S
    )
    twist = np.concatenate((linear_velocity, angular_velocity))
    return twist, position_error, rotation_error_vector


def joint_velocity_from_twist(
    sim_joint_positions: np.ndarray, twist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = np.asarray(
        num_jacobian(kinematic_joint_coordinates(sim_joint_positions)),
        dtype=float,
    )
    joint_velocity = np.linalg.pinv(jacobian, rcond=JACOBIAN_RCOND) @ twist
    if not np.all(np.isfinite(joint_velocity)):
        raise FloatingPointError("Jacobian pseudoinverse produced non-finite qdot")

    speed_ratio = float(np.max(np.abs(joint_velocity) / MAX_JOINT_SPEED))
    if speed_ratio > 1.0:
        joint_velocity = joint_velocity / speed_ratio
    return joint_velocity, jacobian


def arm_control_limits(simulation: GirafSimulation) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(6)
    upper = np.empty(6)
    for index, actuator_name in enumerate(ROBOT_ACTUATORS[:6]):
        actuator_id = simulation.actuator_id(actuator_name)
        if not simulation.model.actuator_ctrllimited[actuator_id]:
            lower[index] = -math.inf
            upper[index] = math.inf
        else:
            lower[index], upper[index] = simulation.model.actuator_ctrlrange[
                actuator_id
            ]
    return lower, upper


def apply_arm_targets(simulation: GirafSimulation, joint_targets: np.ndarray) -> None:
    for actuator_name, target in zip(ROBOT_ACTUATORS[:6], joint_targets):
        simulation.set_actuator_target(actuator_name, float(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid-id", type=int, default=DEFAULT_RIGID_ID)
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    parser.add_argument("--client-ip")
    parser.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    parser.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument(
        "--multicast",
        action="store_true",
        default=DEFAULT_USE_MULTICAST,
        help="receive NatNet data using multicast",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="stop after this many wall-clock seconds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the MuJoCo viewer (Space clutch is unavailable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be positive")

    patch_natnet_string_decoder()
    client_ip = args.client_ip or local_ip_for_server(args.server_ip, args.command_port)
    receiver = OptiTrackReceiver(
        server_ip=args.server_ip,
        client_ip=client_ip,
        rigid_id=args.rigid_id,
        data_port=args.data_port,
        command_port=args.command_port,
        use_multicast=args.multicast,
    )
    mapper = RelativePoseMapper()
    stop_event = threading.Event()
    actions: queue.SimpleQueue[str] = queue.SimpleQueue()
    last_key_time: dict[str, float] = {}

    def request_stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    def on_key(keycode: int) -> None:
        action = None
        if keycode == ord(" "):
            action = "toggle"
        elif keycode in (ord("R"), ord("r")):
            action = "reset"
        if action is None:
            return
        now = time.monotonic()
        if now - last_key_time.get(action, -math.inf) >= KEY_DEBOUNCE_S:
            last_key_time[action] = now
            actions.put(action)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"Streaming OptiTrack rigid body ID {args.rigid_id}")
    print(f"OptiTrack server/client: {args.server_ip} / {client_ip}")
    print("Loading arm-only GIRAF simulation.")
    print("Controls: Space toggles tracking, R resets the simulation.")

    viewer_context = nullcontext(None)
    try:
        with GirafSimulation("arm") as simulation:
            joint_targets = simulation.joint_positions(ROBOT_JOINTS)
            control_lower, control_upper = arm_control_limits(simulation)
            apply_arm_targets(simulation, joint_targets)
            receiver.start()

            if not args.headless:
                viewer_context = mujoco.viewer.launch_passive(
                    simulation.model,
                    simulation.data,
                    key_callback=on_key,
                )

            started_at = time.monotonic()
            last_status_time = -math.inf
            with viewer_context as viewer:
                while not stop_event.is_set() and (
                    viewer is None or viewer.is_running()
                ):
                    loop_started = time.perf_counter()
                    actual_joints = simulation.joint_positions(ROBOT_JOINTS)
                    current_pose = end_effector_pose(actual_joints)

                    while True:
                        try:
                            action = actions.get_nowait()
                        except queue.Empty:
                            break
                        if action == "toggle":
                            if mapper.enabled:
                                mapper.disable()
                                joint_targets = actual_joints.copy()
                                print(
                                    "Tracking disabled: task velocity forced to zero."
                                )
                            else:
                                sample, age_ms, stream_state = sample_status(
                                    receiver, MAX_OPTI_AGE_MS
                                )
                                if sample is None:
                                    print(
                                        "Tracking not enabled: "
                                        f"OptiTrack stream is {stream_state}."
                                    )
                                else:
                                    mapper.enable(sample, current_pose)
                                    joint_targets = actual_joints.copy()
                                    print(
                                        "Tracking enabled: "
                                        f"frame={sample.frame} age={age_ms:.1f} ms."
                                    )
                        elif action == "reset":
                            simulation.reset()
                            actual_joints = simulation.joint_positions(ROBOT_JOINTS)
                            current_pose = end_effector_pose(actual_joints)
                            joint_targets = actual_joints.copy()
                            mapper.disable()
                            print("Simulation reset; tracking is OFF.")

                    sample, age_ms, stream_state = sample_status(
                        receiver, MAX_OPTI_AGE_MS
                    )
                    command_enabled = mapper.enabled and sample is not None
                    twist = np.zeros(6)
                    position_error = np.zeros(3)
                    rotation_error = np.zeros(3)
                    joint_velocity = np.zeros(6)

                    if command_enabled:
                        assert sample is not None
                        target_pose = mapper.update(sample)
                        twist, position_error, rotation_error = task_space_velocity(
                            target_pose, current_pose
                        )
                        joint_velocity, _jacobian = joint_velocity_from_twist(
                            actual_joints, twist
                        )
                        joint_targets = joint_targets + (
                            simulation.step_dt * joint_velocity
                        )
                        joint_targets = np.clip(
                            joint_targets, control_lower, control_upper
                        )

                    apply_arm_targets(simulation, joint_targets)
                    simulation.step()
                    if viewer is not None:
                        viewer.sync()

                    now = time.monotonic()
                    if now - last_status_time >= STATUS_PERIOD_S:
                        last_status_time = now
                        age_text = (
                            "n/a" if not math.isfinite(age_ms) else f"{age_ms:.1f}"
                        )
                        print(
                            f"tracking={'ON' if mapper.enabled else 'OFF'} "
                            f"command={'ON' if command_enabled else 'ZERO'} "
                            f"stream={stream_state} age={age_text}ms "
                            f"|ep|={np.linalg.norm(position_error):.4f}m "
                            f"|er|={np.linalg.norm(rotation_error):.4f}rad "
                            f"|twist|={np.linalg.norm(twist):.4f} "
                            f"|qdot|={np.linalg.norm(joint_velocity):.4f}",
                            flush=True,
                        )

                    if args.duration is not None and now - started_at >= args.duration:
                        break
                    remaining = simulation.step_dt - (
                        time.perf_counter() - loop_started
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
        return 0
    except Exception as exc:
        print(f"FATAL: {exc!r}")
        traceback.print_exc()
        return 1
    finally:
        stop_event.set()
        try:
            receiver.stop()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
