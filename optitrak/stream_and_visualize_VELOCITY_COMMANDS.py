#!/usr/bin/env python3
"""Stream one OptiTrack rigid body and visualize derived velocity commands.

The translucent prism is the desired pose accumulated from incremental
OptiTrack motion. The solid prism moves only by integrating the generated
base-frame linear and angular velocity commands.
"""

from __future__ import annotations

import importlib.util
import math
import os
import signal
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig")
)

import matplotlib
import numpy as np
from natnet import NatNetClient, Version
from natnet.packet_buffer import PacketBuffer


def configure_matplotlib_backend() -> None:
    backend_override = os.environ.get("MPLBACKEND")
    if backend_override:
        return
    if importlib.util.find_spec("tkinter"):
        matplotlib.use("TkAgg")
        return
    if importlib.util.find_spec("PyQt6") or importlib.util.find_spec("PySide6"):
        matplotlib.use("QtAgg")
        return
    if importlib.util.find_spec("PyQt5"):
        matplotlib.use("Qt5Agg")
        return
    if importlib.util.find_spec("gi"):
        matplotlib.use("GTK3Agg")


configure_matplotlib_backend()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


RIGID_ID = 33
SERVER_IP = "172.24.68.77"
CLIENT_IP = None
DATA_PORT = 1511
COMMAND_PORT = 1510
USE_MULTICAST = False

MAX_OPTI_AGE_MS = 100.0
PLOT_UPDATE_HZ = 60.0
MAX_INTEGRATION_DT_S = 0.05

POSITION_GAIN = 4.0
ROTATION_GAIN = 4.0
MAX_LINEAR_SPEED_M_S = 0.5
MAX_ANGULAR_SPEED_RAD_S = 2.0

POSITION_SCALE_STEP = 0.1
MIN_POSITION_SCALE = 0.1
ORIENTATION_SCALE_STEP = 0.1
MIN_ORIENTATION_SCALE = 0.1

AXIS_LIMIT_M = 0.35
PRISM_SIZE_M = (0.16, 0.05, 0.03)
VIEW_ELEV_DEG = 22.0
VIEW_AZIM_DEG = -55.0
VELOCITY_ARROW_TIME_S = 0.25
ANGULAR_ARROW_SCALE_M_PER_RAD_S = 0.06

# Maps OptiTrack displacement coordinates into robot-base command coordinates.
# Replace this identity matrix after the controller-to-robot axes are calibrated.
COMMAND_FROM_OPTITRACK = np.eye(3, dtype=float)

STOP = False


def patch_natnet_string_decoder() -> None:
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
        str_enc, _separator, _remainder = bytes(data_slice).partition(b"\0")
        str_dec = str_enc.decode("utf-8", errors="replace")
        if static_length:
            assert max_length is not None
            self.pointer += max_length
        else:
            self.pointer += len(str_enc) + 1
        return str_dec

    read_string_lossy._bota_optitrack_patched = True
    PacketBuffer.read_string = read_string_lossy


patch_natnet_string_decoder()


def on_signal(_signum, _frame) -> None:
    global STOP
    STOP = True
    plt.close("all")


def local_ip_for_server(server_ip: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, COMMAND_PORT))
        return sock.getsockname()[0]
    finally:
        sock.close()


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


class OptiTrackReceiver:
    """Thread-safe receiver for the latest pose of one OptiTrack rigid body."""

    def __init__(
        self,
        server_ip: str,
        client_ip: str,
        rigid_id: int,
        data_port: int,
        command_port: int,
        use_multicast: bool,
    ) -> None:
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
                local_ns,
                frame.prefix.frame_number,
                frame.suffix.timestamp,
                self.rigid_id,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
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


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class TrackingState:
    enabled: bool = False
    needs_resync: bool = True
    previous_sample_ns: int | None = None
    previous_position: np.ndarray | None = None
    previous_quaternion: np.ndarray | None = None


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


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector = quaternion[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[3]))
    return vector * (angle / vector_norm)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3) + skew(rotation_vector)
    axis = rotation_vector / angle
    axis_skew = skew(axis)
    return (
        np.eye(3)
        + math.sin(angle) * axis_skew
        + (1.0 - math.cos(angle)) * (axis_skew @ axis_skew)
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


def limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm <= 0.0:
        return vector
    return vector * (maximum / norm)


def project_to_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Remove accumulated floating-point drift while preserving orientation."""

    left, _singular_values, right_transpose = np.linalg.svd(rotation)
    projected = left @ right_transpose
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_transpose
    return projected


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


def fresh_sample(
    receiver: OptiTrackReceiver, max_age_ms: float
) -> tuple[OptiSample, float] | None:
    now_ns = time.monotonic_ns()
    sample = receiver.latest()
    if sample is None or sample.px is None:
        return None
    age_ms = (now_ns - sample.local_ns) / 1e6
    if age_ms > max_age_ms:
        return None
    return sample, age_ms


def prism_vertices(size: tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = (dimension / 2.0 for dimension in size)
    return np.array(
        (
            (-sx, -sy, -sz),
            (sx, -sy, -sz),
            (sx, sy, -sz),
            (-sx, sy, -sz),
            (-sx, -sy, sz),
            (sx, -sy, sz),
            (sx, sy, sz),
            (-sx, sy, sz),
        )
    )


def transformed_faces(
    pose: Pose, body_vertices: np.ndarray
) -> list[list[np.ndarray]]:
    transformed = (pose.rotation @ body_vertices.T).T + pose.position
    face_indices = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    )
    return [[transformed[index] for index in face] for face in face_indices]


def main() -> int:
    global STOP

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "pdf", "pgf", "ps", "svg", "template"}:
        print(
            "FATAL: Matplotlib is using a non-interactive backend "
            f"({matplotlib.get_backend()!r}), so no window can be shown.\n"
            "Install a GUI backend, then rerun. Typical fixes:\n"
            "  1. apt install python3-tk\n"
            "  2. or python -m pip install PyQt6"
        )
        return 1

    client_ip = CLIENT_IP or local_ip_for_server(SERVER_IP)
    print(f"Streaming rigid body ID {RIGID_ID}")
    print(f"OptiTrack server/client: {SERVER_IP} / {client_ip}")
    print(
        "Controls: space toggles tracking, r resets both poses, "
        "Up/Down adjust translation scale, Left/Right adjust rotation scale."
    )

    receiver = OptiTrackReceiver(
        SERVER_IP,
        client_ip,
        RIGID_ID,
        DATA_PORT,
        COMMAND_PORT,
        USE_MULTICAST,
    )
    tracking = TrackingState()
    target_pose = Pose(np.zeros(3), np.eye(3))
    command_pose = Pose(np.zeros(3), np.eye(3))
    velocity_linear = np.zeros(3)
    velocity_angular = np.zeros(3)
    position_scale = 1.0
    orientation_scale = 1.0
    last_update_time = time.monotonic()
    last_status_print = 0.0
    body_vertices = prism_vertices(PRISM_SIZE_M)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("OptiTrack Velocity Command Validation")
    ax.set_title(f"Rigid Body {RIGID_ID}: Target vs. Integrated Velocity Command")
    ax.set_xlabel("Command X (m)")
    ax.set_ylabel("Command Y (m)")
    ax.set_zlabel("Command Z (m)")
    ax.set_xlim(-AXIS_LIMIT_M, AXIS_LIMIT_M)
    ax.set_ylim(-AXIS_LIMIT_M, AXIS_LIMIT_M)
    ax.set_zlim(-AXIS_LIMIT_M, AXIS_LIMIT_M)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=VIEW_ELEV_DEG, azim=VIEW_AZIM_DEG)

    ax.plot(
        [-AXIS_LIMIT_M, AXIS_LIMIT_M],
        [0.0, 0.0],
        [0.0, 0.0],
        color="tab:red",
        alpha=0.3,
    )
    ax.plot(
        [0.0, 0.0],
        [-AXIS_LIMIT_M, AXIS_LIMIT_M],
        [0.0, 0.0],
        color="tab:green",
        alpha=0.3,
    )
    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [-AXIS_LIMIT_M, AXIS_LIMIT_M],
        color="tab:blue",
        alpha=0.3,
    )

    target_prism = Poly3DCollection(
        transformed_faces(target_pose, body_vertices),
        facecolors="gold",
        edgecolors="darkorange",
        linewidths=1.0,
        alpha=0.2,
        label="OptiTrack target",
    )
    command_prism = Poly3DCollection(
        transformed_faces(command_pose, body_vertices),
        facecolors="cornflowerblue",
        edgecolors="black",
        linewidths=1.0,
        alpha=0.8,
        label="Velocity-integrated pose",
    )
    ax.add_collection3d(target_prism)
    ax.add_collection3d(command_prism)

    target_marker, = ax.plot(
        [0.0], [0.0], [0.0], marker="o", color="darkorange", markersize=5
    )
    command_marker, = ax.plot(
        [0.0], [0.0], [0.0], marker="o", color="black", markersize=4
    )
    linear_velocity_line, = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        color="magenta",
        linewidth=2.0,
        label="Linear velocity",
    )
    angular_velocity_line, = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        color="cyan",
        linewidth=2.0,
        label="Angular velocity axis",
    )
    command_x_axis, = ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        color="red",
        linewidth=2.0,
    )

    status_text = fig.text(
        0.02,
        0.03,
        "Tracking OFF. Press space to enable.",
        family="monospace",
    )
    fig.subplots_adjust(bottom=0.24)
    ax.legend(loc="upper left")

    def set_previous_controller_pose(sample: OptiSample) -> None:
        position, quaternion = sample_pose(sample)
        tracking.previous_sample_ns = sample.local_ns
        tracking.previous_position = position
        tracking.previous_quaternion = quaternion
        tracking.needs_resync = False

    def toggle_tracking() -> None:
        nonlocal velocity_linear, velocity_angular

        if tracking.enabled:
            tracking.enabled = False
            tracking.needs_resync = True
            velocity_linear = np.zeros(3)
            velocity_angular = np.zeros(3)
            print("Tracking disabled: velocity command forced to zero.")
            return

        fresh = fresh_sample(receiver, MAX_OPTI_AGE_MS)
        if fresh is None:
            print("Tracking not enabled: no fresh OptiTrack sample available.")
            return
        sample, age_ms = fresh
        if sample.seen is False:
            print("Tracking not enabled: OptiTrack tracking is invalid.")
            return

        set_previous_controller_pose(sample)
        target_pose.position = command_pose.position.copy()
        target_pose.rotation = command_pose.rotation.copy()
        tracking.enabled = True
        print(
            f"Tracking enabled at frame={sample.frame}, age={age_ms:.1f} ms. "
            "Target initialized from the integrated command pose."
        )

    def reset_poses() -> None:
        nonlocal velocity_linear, velocity_angular

        target_pose.position = np.zeros(3)
        target_pose.rotation = np.eye(3)
        command_pose.position = np.zeros(3)
        command_pose.rotation = np.eye(3)
        velocity_linear = np.zeros(3)
        velocity_angular = np.zeros(3)
        fresh = fresh_sample(receiver, MAX_OPTI_AGE_MS)
        if tracking.enabled and fresh is not None and fresh[0].seen is not False:
            set_previous_controller_pose(fresh[0])
        else:
            tracking.needs_resync = True
        print("Target and velocity-integrated poses reset to the origin.")

    def on_key_press(event) -> None:
        nonlocal position_scale, orientation_scale

        if event.key == " ":
            toggle_tracking()
        elif event.key == "r":
            reset_poses()
        elif event.key == "up":
            position_scale = round(
                max(MIN_POSITION_SCALE, position_scale + POSITION_SCALE_STEP), 10
            )
            print(f"Translation scale set to {position_scale:.1f}x.")
        elif event.key == "down":
            position_scale = round(
                max(MIN_POSITION_SCALE, position_scale - POSITION_SCALE_STEP), 10
            )
            print(f"Translation scale set to {position_scale:.1f}x.")
        elif event.key == "right":
            orientation_scale = round(
                max(
                    MIN_ORIENTATION_SCALE,
                    orientation_scale + ORIENTATION_SCALE_STEP,
                ),
                10,
            )
            print(f"Orientation scale set to {orientation_scale:.1f}x.")
        elif event.key == "left":
            orientation_scale = round(
                max(
                    MIN_ORIENTATION_SCALE,
                    orientation_scale - ORIENTATION_SCALE_STEP,
                ),
                10,
            )
            print(f"Orientation scale set to {orientation_scale:.1f}x.")

    def on_close(_event) -> None:
        global STOP
        STOP = True

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("close_event", on_close)

    def update_target_from_new_sample(sample: OptiSample) -> None:
        current_position, current_quaternion = sample_pose(sample)
        if tracking.needs_resync:
            set_previous_controller_pose(sample)
            return
        if sample.local_ns == tracking.previous_sample_ns:
            return

        assert tracking.previous_position is not None
        assert tracking.previous_quaternion is not None

        if np.dot(current_quaternion, tracking.previous_quaternion) < 0.0:
            current_quaternion = -current_quaternion

        controller_delta_position = (
            current_position - tracking.previous_position
        )
        mapped_delta_position = (
            position_scale
            * COMMAND_FROM_OPTITRACK
            @ controller_delta_position
        )

        controller_delta_quaternion = normalize_quaternion(
            quaternion_multiply(
                quaternion_conjugate(tracking.previous_quaternion),
                current_quaternion,
            )
        )
        controller_delta_rotation_vector = quaternion_to_rotation_vector(
            controller_delta_quaternion
        )
        scaled_controller_delta_rotation = rotation_vector_to_matrix(
            orientation_scale * controller_delta_rotation_vector
        )
        mapped_delta_rotation = (
            COMMAND_FROM_OPTITRACK
            @ scaled_controller_delta_rotation
            @ COMMAND_FROM_OPTITRACK.T
        )

        target_pose.position = target_pose.position + mapped_delta_position
        target_pose.rotation = project_to_rotation_matrix(
            target_pose.rotation @ mapped_delta_rotation
        )

        tracking.previous_sample_ns = sample.local_ns
        tracking.previous_position = current_position
        tracking.previous_quaternion = current_quaternion

    def update(_frame_index):
        nonlocal last_update_time, last_status_print
        nonlocal velocity_linear, velocity_angular

        now = time.monotonic()
        integration_dt = min(now - last_update_time, MAX_INTEGRATION_DT_S)
        last_update_time = now
        sample = receiver.latest()
        sample_age_ms = math.inf
        stream_status = "no sample"
        command_enabled = False

        if sample is not None:
            sample_age_ms = (time.monotonic_ns() - sample.local_ns) / 1e6
            if sample.px is None:
                stream_status = "rigid body absent"
            elif sample.seen is False:
                stream_status = "tracking invalid"
            elif sample_age_ms > MAX_OPTI_AGE_MS:
                stream_status = "sample stale"
            else:
                stream_status = "fresh"
                if tracking.enabled:
                    update_target_from_new_sample(sample)
                    command_enabled = True

        if tracking.enabled and not command_enabled:
            tracking.needs_resync = True

        if command_enabled:
            position_error = target_pose.position - command_pose.position
            rotation_error = target_pose.rotation @ command_pose.rotation.T
            rotation_error_vector = rotation_matrix_to_vector(rotation_error)
            velocity_linear = limit_norm(
                POSITION_GAIN * position_error, MAX_LINEAR_SPEED_M_S
            )
            velocity_angular = limit_norm(
                ROTATION_GAIN * rotation_error_vector,
                MAX_ANGULAR_SPEED_RAD_S,
            )

            command_pose.position = (
                command_pose.position + velocity_linear * integration_dt
            )
            command_pose.rotation = project_to_rotation_matrix(
                rotation_vector_to_matrix(velocity_angular * integration_dt)
                @ command_pose.rotation
            )
        else:
            position_error = target_pose.position - command_pose.position
            rotation_error = target_pose.rotation @ command_pose.rotation.T
            rotation_error_vector = rotation_matrix_to_vector(rotation_error)
            velocity_linear = np.zeros(3)
            velocity_angular = np.zeros(3)

        target_prism.set_verts(transformed_faces(target_pose, body_vertices))
        command_prism.set_verts(transformed_faces(command_pose, body_vertices))
        target_marker.set_data_3d(
            [target_pose.position[0]],
            [target_pose.position[1]],
            [target_pose.position[2]],
        )
        command_marker.set_data_3d(
            [command_pose.position[0]],
            [command_pose.position[1]],
            [command_pose.position[2]],
        )

        linear_endpoint = (
            command_pose.position + VELOCITY_ARROW_TIME_S * velocity_linear
        )
        linear_velocity_line.set_data_3d(
            [command_pose.position[0], linear_endpoint[0]],
            [command_pose.position[1], linear_endpoint[1]],
            [command_pose.position[2], linear_endpoint[2]],
        )
        angular_endpoint = (
            command_pose.position
            + ANGULAR_ARROW_SCALE_M_PER_RAD_S * velocity_angular
        )
        angular_velocity_line.set_data_3d(
            [command_pose.position[0], angular_endpoint[0]],
            [command_pose.position[1], angular_endpoint[1]],
            [command_pose.position[2], angular_endpoint[2]],
        )
        command_x_endpoint = (
            command_pose.position + 0.1 * command_pose.rotation[:, 0]
        )
        command_x_axis.set_data_3d(
            [command_pose.position[0], command_x_endpoint[0]],
            [command_pose.position[1], command_x_endpoint[1]],
            [command_pose.position[2], command_x_endpoint[2]],
        )

        frame_text = "none" if sample is None else str(sample.frame)
        age_text = "n/a" if not math.isfinite(sample_age_ms) else f"{sample_age_ms:.1f}"
        lines = [
            f"tracking={'ON' if tracking.enabled else 'OFF'} "
            f"command={'ON' if command_enabled else 'ZERO'}",
            f"stream={stream_status} frame={frame_text} age={age_text} ms "
            f"dt={integration_dt * 1000.0:.1f} ms",
            f"scale_pos={position_scale:.1f}x "
            f"scale_rot={orientation_scale:.1f}x "
            f"Kp={POSITION_GAIN:.2f} Kr={ROTATION_GAIN:.2f}",
            "target p=" + np.array2string(target_pose.position, precision=3, sign="+"),
            "command p=" + np.array2string(command_pose.position, precision=3, sign="+"),
            "ep="
            + np.array2string(position_error, precision=3, sign="+")
            + f" |ep|={np.linalg.norm(position_error):.4f} m",
            "er="
            + np.array2string(rotation_error_vector, precision=3, sign="+")
            + " |er|="
            + f"{math.degrees(np.linalg.norm(rotation_error_vector)):.2f} deg",
            "v ="
            + np.array2string(velocity_linear, precision=3, sign="+")
            + f" m/s  |v|={np.linalg.norm(velocity_linear):.3f}",
            "w ="
            + np.array2string(velocity_angular, precision=3, sign="+")
            + f" rad/s |w|={np.linalg.norm(velocity_angular):.3f}",
        ]
        status_text.set_text("\n".join(lines))

        if now - last_status_print >= 0.5:
            last_status_print = now
            print(" | ".join(lines), flush=True)

        if STOP:
            plt.close(fig)
        return (
            target_prism,
            command_prism,
            target_marker,
            command_marker,
            linear_velocity_line,
            angular_velocity_line,
            command_x_axis,
            status_text,
        )

    try:
        receiver.start()
        animation = FuncAnimation(
            fig,
            update,
            interval=1000.0 / PLOT_UPDATE_HZ,
            blit=False,
            cache_frame_data=False,
        )
        fig._animation = animation
        plt.show()
        return 0
    except Exception as exc:
        print(f"FATAL: {exc!r}")
        traceback.print_exc()
        return 1
    finally:
        STOP = True
        try:
            receiver.stop()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
