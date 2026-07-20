#!/usr/bin/env python3
"""Stream one OptiTrack rigid body and visualize its pose with a space-bar tracking toggle."""

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

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
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
PLOT_UPDATE_HZ = 30.0
AXIS_LIMIT_M = 0.35
PRISM_SIZE_M = (0.16, 0.05, 0.03)
VIEW_ELEV_DEG = 22.0
VIEW_AZIM_DEG = -55.0
POSITION_SCALE_STEP = 0.1
MIN_POSITION_SCALE = 0.1
ORIENTATION_SCALE_STEP = 0.1
MIN_ORIENTATION_SCALE = 0.1
DOUBLE_R_WINDOW_S = 0.4

STOP = False


def patch_natnet_string_decoder() -> None:
    original = PacketBuffer.read_string
    if getattr(original, "_bota_optitrack_patched", False):
        return

    def read_string_lossy(self, max_length=None, static_length=False):
        if max_length is None:
            data_slice = self._PacketBuffer__data[self.pointer :]
        else:
            data_slice = self._PacketBuffer__data[self.pointer : self.pointer + max_length]
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


def on_signal(_signum, _frame):
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


@dataclass
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
    def __init__(self, server_ip: str, client_ip: str, rigid_id: int, data_port: int, command_port: int, use_multicast: bool):
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

    def _on_frame(self, frame):
        local_ns = time.monotonic_ns()
        body = None
        for rigid_body in frame.rigid_bodies or ():
            if rigid_body.id_num == self.rigid_id:
                body = rigid_body
                break
        if body is None:
            sample = OptiSample(local_ns, frame.prefix.frame_number, frame.suffix.timestamp, self.rigid_id, None, None, None, None, None, None, None, None)
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

    def start(self):
        self.client.connect(timeout=5.0)
        print(f"OptiTrack connected: protocol={self.client.protocol_version} server={self.client.server_info}")
        self.client.run_async()

    def latest(self) -> OptiSample | None:
        with self.lock:
            return self.sample

    def stop(self):
        self.client.shutdown()


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 0.0:
        raise ValueError("zero-length quaternion")
    return tuple(v / norm for v in q)


def quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = q
    return (-qx, -qy, -qz, qw)


def quat_scale_rotation(q: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = quat_normalize(q)
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    qw = max(-1.0, min(1.0, qw))
    half_angle = math.acos(qw)
    sin_half = math.sin(half_angle)
    if sin_half < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    axis = (qx / sin_half, qy / sin_half, qz / sin_half)
    scaled_half_angle = scale * half_angle
    scaled_sin = math.sin(scaled_half_angle)
    return quat_normalize((
        axis[0] * scaled_sin,
        axis[1] * scaled_sin,
        axis[2] * scaled_sin,
        math.cos(scaled_half_angle),
    ))


def quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_to_matrix(q: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    qx, qy, qz, qw = quat_normalize(q)
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def mat_transpose_vec_mul(matrix, vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vec[0] + matrix[1][0] * vec[1] + matrix[2][0] * vec[2],
        matrix[0][1] * vec[0] + matrix[1][1] * vec[1] + matrix[2][1] * vec[2],
        matrix[0][2] * vec[0] + matrix[1][2] * vec[1] + matrix[2][2] * vec[2],
    )


def mat_vec_mul(matrix, vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2],
        matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2],
    )


def sample_pose(sample: OptiSample) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if None in (sample.px, sample.py, sample.pz, sample.qx, sample.qy, sample.qz, sample.qw):
        raise ValueError("OptiTrack sample does not contain a full pose")
    return (sample.px, sample.py, sample.pz), quat_normalize((sample.qx, sample.qy, sample.qz, sample.qw))


def fresh_sample(receiver: OptiTrackReceiver, max_age_ms: float) -> tuple[OptiSample, float] | None:
    now_ns = time.monotonic_ns()
    sample = receiver.latest()
    if sample is None or sample.px is None:
        return None
    age_ms = (now_ns - sample.local_ns) / 1e6
    if age_ms > max_age_ms:
        return None
    return sample, age_ms


def prism_vertices(size: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    sx, sy, sz = (dim / 2.0 for dim in size)
    return [
        (-sx, -sy, -sz),
        (sx, -sy, -sz),
        (sx, sy, -sz),
        (-sx, sy, -sz),
        (-sx, -sy, sz),
        (sx, -sy, sz),
        (sx, sy, sz),
        (-sx, sy, sz),
    ]


def transformed_faces(center: tuple[float, float, float], quat: tuple[float, float, float, float], body_vertices: list[tuple[float, float, float]]) -> list[list[tuple[float, float, float]]]:
    rotation = quat_to_matrix(quat)
    transformed = []
    for vertex in body_vertices:
        rx, ry, rz = mat_vec_mul(rotation, vertex)
        transformed.append((rx + center[0], ry + center[1], rz + center[2]))
    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]
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
            "  2. or .venv/bin/pip install PyQt6"
        )
        return 1

    client_ip = CLIENT_IP or local_ip_for_server(SERVER_IP)
    print(f"Streaming rigid body ID {RIGID_ID}")
    print(f"OptiTrack server/client: {SERVER_IP} / {client_ip}")
    print("Controls: space toggles tracking, o/c set gripper open/closed while tracking is ON, r resets prism position and orientation to the origin while tracking is ON, double-tap r quickly to reset translation and orientation scales to 1.0x, Up/Down adjust translation scale by 0.1, Left/Right adjust orientation scale by 0.1.")

    receiver = OptiTrackReceiver(SERVER_IP, client_ip, RIGID_ID, DATA_PORT, COMMAND_PORT, USE_MULTICAST)
    tracking_state = {
        "enabled": False,
        "controller_anchor_pos": (0.0, 0.0, 0.0),
        "controller_anchor_quat": (0.0, 0.0, 0.0, 1.0),
        "controller_anchor_rot": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "controller_anchor_inv_quat": (0.0, 0.0, 0.0, 1.0),
        "display_anchor_pos": (0.0, 0.0, 0.0),
        "display_anchor_quat": (0.0, 0.0, 0.0, 1.0),
    }
    display_pose = {
        "pos": (0.0, 0.0, 0.0),
        "quat": (0.0, 0.0, 0.0, 1.0),
    }
    gripper_state = {"status": "open"}
    position_scale = {"value": 1.0}
    orientation_scale = {"value": 1.0}
    r_key_state = {"last_press_time": None}
    last_status_print = 0.0
    body_vertices = prism_vertices(PRISM_SIZE_M)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title(f"OptiTrack Rigid Body {RIGID_ID}")
    ax.set_title(f"Rigid Body {RIGID_ID} With Toggleable Relative Tracking")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-AXIS_LIMIT_M, AXIS_LIMIT_M)
    ax.set_ylim(-AXIS_LIMIT_M, AXIS_LIMIT_M)
    ax.set_zlim(-AXIS_LIMIT_M, AXIS_LIMIT_M)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=VIEW_ELEV_DEG, azim=VIEW_AZIM_DEG)

    ax.plot([-AXIS_LIMIT_M, AXIS_LIMIT_M], [0.0, 0.0], [0.0, 0.0], color="tab:red", alpha=0.35)
    ax.plot([0.0, 0.0], [-AXIS_LIMIT_M, AXIS_LIMIT_M], [0.0, 0.0], color="tab:green", alpha=0.35)
    ax.plot([0.0, 0.0], [0.0, 0.0], [-AXIS_LIMIT_M, AXIS_LIMIT_M], color="tab:blue", alpha=0.35)

    faces = transformed_faces((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), body_vertices)
    prism = Poly3DCollection(faces, facecolors="cornflowerblue", edgecolors="black", linewidths=1.0, alpha=0.75)
    ax.add_collection3d(prism)

    marker, = ax.plot([0.0], [0.0], [0.0], marker="o", color="black", markersize=4)
    status_text = fig.text(0.02, 0.04, "Tracking OFF. Press space to enable.", family="monospace")
    gripper_text = fig.text(0.02, 0.96, "Gripper status: open", family="monospace", va="top")
    fig.subplots_adjust(bottom=0.12)

    def toggle_tracking_from_latest():
        fresh = fresh_sample(receiver, MAX_OPTI_AGE_MS)
        if tracking_state["enabled"]:
            tracking_state["enabled"] = False
            print(
                "Tracking disabled: prism frozen at "
                f"pos=({display_pose['pos'][0]:.4f},{display_pose['pos'][1]:.4f},{display_pose['pos'][2]:.4f}) "
                f"quat=({display_pose['quat'][0]:.4f},{display_pose['quat'][1]:.4f},{display_pose['quat'][2]:.4f},{display_pose['quat'][3]:.4f})"
            )
            return

        if fresh is None:
            print("Tracking not enabled: no fresh OptiTrack sample available.")
            return
        sample, age_ms = fresh
        if sample.seen is False:
            print("Tracking not enabled: rigid body is present but tracking is invalid.")
            return

        pos, quat = sample_pose(sample)
        tracking_state["enabled"] = True
        tracking_state["controller_anchor_pos"] = pos
        tracking_state["controller_anchor_quat"] = quat
        tracking_state["controller_anchor_rot"] = quat_to_matrix(quat)
        tracking_state["controller_anchor_inv_quat"] = quat_conjugate(quat)
        tracking_state["display_anchor_pos"] = display_pose["pos"]
        tracking_state["display_anchor_quat"] = display_pose["quat"]
        print(
            f"Tracking enabled: frame={sample.frame} age={age_ms:.1f}ms "
            f"pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) "
            f"quat=({quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f})"
        )

    def set_gripper_status(new_status: str):
        if not tracking_state["enabled"]:
            print(f"Gripper command ignored while tracking is OFF: requested {new_status}.")
            return
        gripper_state["status"] = new_status
        print(f"Gripper status set to {new_status}.")

    def reset_pose_from_latest():
        if not tracking_state["enabled"]:
            print("Reset ignored while tracking is OFF.")
            return

        fresh = fresh_sample(receiver, MAX_OPTI_AGE_MS)
        if fresh is None:
            print("Reset ignored: no fresh OptiTrack sample available.")
            return
        sample, age_ms = fresh
        if sample.seen is False:
            print("Reset ignored: rigid body is present but tracking is invalid.")
            return

        pos, quat = sample_pose(sample)
        tracking_state["controller_anchor_pos"] = pos
        tracking_state["controller_anchor_quat"] = quat
        tracking_state["controller_anchor_rot"] = quat_to_matrix(quat)
        tracking_state["controller_anchor_inv_quat"] = quat_conjugate(quat)
        display_pose["pos"] = (0.0, 0.0, 0.0)
        display_pose["quat"] = (0.0, 0.0, 0.0, 1.0)
        tracking_state["display_anchor_pos"] = display_pose["pos"]
        tracking_state["display_anchor_quat"] = display_pose["quat"]
        print(
            f"Pose reset: frame={sample.frame} age={age_ms:.1f}ms "
            f"controller_pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) "
            f"controller_quat=({quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f}) "
            "prism_pos=(0.0000,0.0000,0.0000) prism_quat=(0.0000,0.0000,0.0000,1.0000)"
        )

    def adjust_position_scale(delta: float):
        next_scale = max(MIN_POSITION_SCALE, position_scale["value"] + delta)
        position_scale["value"] = round(next_scale, 10)
        print(f"Position scale set to {position_scale['value']:.1f}x.")

    def adjust_orientation_scale(delta: float):
        next_scale = max(MIN_ORIENTATION_SCALE, orientation_scale["value"] + delta)
        orientation_scale["value"] = round(next_scale, 10)
        print(f"Orientation scale set to {orientation_scale['value']:.1f}x.")

    def reset_scales_to_default():
        position_scale["value"] = 1.0
        orientation_scale["value"] = 1.0
        print("Translation and orientation scales reset to 1.0x via double-tap r.")

    def on_key_press(event):
        if event.key == " ":
            toggle_tracking_from_latest()
        elif event.key == "o":
            set_gripper_status("open")
        elif event.key == "c":
            set_gripper_status("closed")
        elif event.key == "r":
            now = time.monotonic()
            is_double_tap = (
                r_key_state["last_press_time"] is not None
                and (now - r_key_state["last_press_time"]) <= DOUBLE_R_WINDOW_S
            )
            r_key_state["last_press_time"] = None if is_double_tap else now
            reset_pose_from_latest()
            if is_double_tap:
                reset_scales_to_default()
        elif event.key == "up":
            adjust_position_scale(POSITION_SCALE_STEP)
        elif event.key == "down":
            adjust_position_scale(-POSITION_SCALE_STEP)
        elif event.key == "right":
            adjust_orientation_scale(ORIENTATION_SCALE_STEP)
        elif event.key == "left":
            adjust_orientation_scale(-ORIENTATION_SCALE_STEP)

    def on_close(_event):
        global STOP
        STOP = True

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("close_event", on_close)

    def update(_frame_index):
        nonlocal last_status_print
        now = time.monotonic()
        sample = receiver.latest()
        lines = []

        lines.append(f"tracking={'ON' if tracking_state['enabled'] else 'OFF'}")
        lines.append(f"gripper={gripper_state['status']}")
        lines.append(f"scale={position_scale['value']:.1f}x")
        lines.append(f"rot_scale={orientation_scale['value']:.1f}x")

        if sample is None:
            lines.append("No OptiTrack sample yet.")
        elif sample.px is None:
            lines.append("Rigid body not present in the current frame.")
        else:
            age_ms = (time.monotonic_ns() - sample.local_ns) / 1e6
            lines.append(f"frame={sample.frame} age={age_ms:.1f}ms opti_seen={sample.seen}")
            pos, quat = sample_pose(sample)
            lines.append(f"abs pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")

            if not tracking_state["enabled"]:
                lines.append("Tracking OFF: prism held.")
            elif sample.seen is False:
                lines.append("OptiTrack invalid: prism held.")
            elif age_ms > MAX_OPTI_AGE_MS:
                lines.append("Sample stale: prism held.")
            else:
                world_delta = (
                    pos[0] - tracking_state["controller_anchor_pos"][0],
                    pos[1] - tracking_state["controller_anchor_pos"][1],
                    pos[2] - tracking_state["controller_anchor_pos"][2],
                )
                unscaled_rel_pos = mat_transpose_vec_mul(tracking_state["controller_anchor_rot"], world_delta)
                rel_pos = (
                    position_scale["value"] * unscaled_rel_pos[0],
                    position_scale["value"] * unscaled_rel_pos[1],
                    position_scale["value"] * unscaled_rel_pos[2],
                )
                rel_quat = quat_normalize(quat_multiply(tracking_state["controller_anchor_inv_quat"], quat))
                scaled_rel_quat = quat_scale_rotation(rel_quat, orientation_scale["value"])
                display_pose["pos"] = (
                    tracking_state["display_anchor_pos"][0] + rel_pos[0],
                    tracking_state["display_anchor_pos"][1] + rel_pos[1],
                    tracking_state["display_anchor_pos"][2] + rel_pos[2],
                )
                display_pose["quat"] = quat_normalize(quat_multiply(tracking_state["display_anchor_quat"], scaled_rel_quat))
                angle_deg = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(scaled_rel_quat[3])))))
                lines.append(f"rel pos=({rel_pos[0]:+.3f}, {rel_pos[1]:+.3f}, {rel_pos[2]:+.3f}) m")
                lines.append(f"rel angle={angle_deg:.1f} deg")
        lines.append(
            "prism pos="
            f"({display_pose['pos'][0]:+.3f}, {display_pose['pos'][1]:+.3f}, {display_pose['pos'][2]:+.3f}) m"
        )

        prism.set_verts(transformed_faces(display_pose["pos"], display_pose["quat"], body_vertices))
        marker.set_data_3d([display_pose["pos"][0]], [display_pose["pos"][1]], [display_pose["pos"][2]])
        status_text.set_text("\n".join(lines))
        gripper_text.set_text(f"Gripper status: {gripper_state['status']}")

        if now - last_status_print >= 0.5:
            last_status_print = now
            print(" | ".join(lines), flush=True)

        if STOP:
            plt.close(fig)
        return prism, marker, status_text, gripper_text

    try:
        receiver.start()
        animation = FuncAnimation(fig, update, interval=1000.0 / PLOT_UPDATE_HZ, blit=False, cache_frame_data=False)
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
