#!/usr/bin/env python3
"""Small, ROS-free OptiTrack driver for one rigid body.

The driver returns the raw pose reported by Motive:

* position is in metres;
* quaternion order is ``(x, y, z, w)``;
* no GIRAF-specific axis mapping or relative-pose transform is applied.

For repeated reads, create one :class:`OptiTrackDriver` and keep it connected.
The module-level :func:`get_latest_pose` helper is intended for one-shot tests.
"""

from __future__ import annotations

import argparse
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

# COMMON BUGS
    # Streaming NOT enabled in Motive
    # Rigid body NOT enabled/checked in Motive
    # Set to Multicast (BAD) instead of Unicast mode in Motive!

DEFAULT_SERVER_IP = "172.24.68.77"
DEFAULT_RIGID_BODY_ID = 40  # Tiffany_controller_1 in Motive
DEFAULT_COMMAND_PORT = 1510
DEFAULT_DATA_PORT = 1511
DEFAULT_USE_MULTICAST = False 
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_POSE_TIMEOUT = 5.0

try:
    from natnet import NatNetClient, Version
    from natnet.packet_buffer import PacketBuffer
except ModuleNotFoundError as exc:
    NatNetClient = None
    Version = None
    PacketBuffer = None
    _NATNET_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _NATNET_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class OptiTrackPose:
    """One raw rigid-body pose from an OptiTrack frame."""

    rigid_body_id: int
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    tracking_valid: bool
    frame_number: int
    motive_timestamp_s: float
    received_monotonic_ns: int

    @property
    def age_s(self) -> float:
        """Age of this sample according to the local monotonic clock."""

        return (time.monotonic_ns() - self.received_monotonic_ns) / 1e9


def local_ip_for_server(
    server_ip: str,
    command_port: int = DEFAULT_COMMAND_PORT,
) -> str:
    """Return the local interface address routed toward the Motive computer."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect selects a route; it does not send a packet here.
        sock.connect((server_ip, command_port))
        return str(sock.getsockname()[0])
    finally:
        sock.close()


def _patch_natnet_string_decoder() -> None:
    """Tolerate non-UTF8 padding seen in this Motive/NatNet deployment."""

    if PacketBuffer is None:
        return

    original = PacketBuffer.read_string
    if getattr(original, "_giraf_lossy_utf8_patch", False):
        return

    def read_string_lossy(
        self: Any,
        max_length: int | None = None,
        static_length: bool = False,
    ) -> str:
        if max_length is None:
            data_slice = self._PacketBuffer__data[self.pointer :]
        else:
            data_slice = self._PacketBuffer__data[
                self.pointer : self.pointer + max_length
            ]

        encoded, _separator, _remainder = bytes(data_slice).partition(b"\0")
        decoded = encoded.decode("utf-8", errors="replace")
        if static_length:
            if max_length is None:
                raise ValueError("static NatNet string requires max_length")
            self.pointer += max_length
        else:
            self.pointer += len(encoded) + 1
        return decoded

    read_string_lossy._giraf_lossy_utf8_patch = True  # type: ignore[attr-defined]
    PacketBuffer.read_string = read_string_lossy


class OptiTrackDriver:
    """Thread-safe receiver for a single OptiTrack rigid body."""

    def __init__(
        self,
        server_ip: str = DEFAULT_SERVER_IP,
        rigid_body_id: int = DEFAULT_RIGID_BODY_ID,
        *,
        client_ip: str | None = None,
        command_port: int = DEFAULT_COMMAND_PORT,
        data_port: int = DEFAULT_DATA_PORT,
        use_multicast: bool = DEFAULT_USE_MULTICAST,
    ) -> None:
        if _NATNET_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The OptiTrack driver requires natnet==0.2.0. "
                "Install it in the Python environment running this file."
            ) from _NATNET_IMPORT_ERROR
        if not server_ip:
            raise ValueError("server_ip cannot be empty")
        if rigid_body_id < 0:
            raise ValueError("rigid_body_id cannot be negative")
        if not 0 < command_port <= 65535 or not 0 < data_port <= 65535:
            raise ValueError("NatNet ports must be between 1 and 65535")

        _patch_natnet_string_decoder()

        self.server_ip = server_ip
        self.rigid_body_id = int(rigid_body_id)
        self.client_ip = client_ip or local_ip_for_server(server_ip, command_port)
        self.command_port = int(command_port)
        self.data_port = int(data_port)
        self.use_multicast = bool(use_multicast)

        self._condition = threading.Condition()
        self._latest_pose: OptiTrackPose | None = None
        self._last_status = "no NatNet frame received"
        self._started = False

        self._client = NatNetClient(
            server_ip_address=self.server_ip,
            local_ip_address=self.client_ip,
            command_port=self.command_port,
            data_port=self.data_port,
            use_multicast=self.use_multicast,
        )
        # This installation uses Motive/NatNet 4.3. natnet 0.2.0 has no public
        # pre-connection setter, so this mirrors the previously working code.
        self._client._NatNetClient__current_protocol_version = Version(4, 3)
        self._client.on_data_frame_received_event.handlers.append(self._on_frame)

    @property
    def connected(self) -> bool:
        return self._started and bool(self._client.connected)

    def _on_frame(self, frame: Any) -> None:
        received_ns = time.monotonic_ns()
        body = next(
            (
                candidate
                for candidate in frame.rigid_bodies or ()
                if candidate.id_num == self.rigid_body_id
            ),
            None,
        )

        with self._condition:
            if body is None:
                self._latest_pose = None
                self._last_status = (
                    f"rigid body {self.rigid_body_id} is absent from "
                    f"frame {frame.prefix.frame_number}"
                )
            else:
                position = tuple(float(value) for value in body.pos)
                quaternion = tuple(float(value) for value in body.rot)
                values = position + quaternion

                if (
                    len(position) != 3
                    or len(quaternion) != 4
                    or not all(math.isfinite(value) for value in values)
                ):
                    self._latest_pose = None
                    self._last_status = "received a malformed or non-finite pose"
                else:
                    self._latest_pose = OptiTrackPose(
                        rigid_body_id=int(body.id_num),
                        position_m=position,
                        quaternion_xyzw=quaternion,
                        tracking_valid=body.tracking_valid is True,
                        frame_number=int(frame.prefix.frame_number),
                        motive_timestamp_s=float(frame.suffix.timestamp),
                        received_monotonic_ns=received_ns,
                    )
                    self._last_status = (
                        "tracking is valid"
                        if body.tracking_valid is True
                        else "Motive marked tracking invalid"
                    )
            self._condition.notify_all()

    def connect(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        """Connect to Motive and start receiving frames in the background."""

        if timeout <= 0.0:
            raise ValueError("connection timeout must be positive")
        if self._started:
            return

        with self._condition:
            self._latest_pose = None
            self._last_status = "connected; waiting for a NatNet frame"

        try:
            self._client.connect(timeout=timeout)
            self._client.run_async()
        except TimeoutError as exc:
            self._client.shutdown()
            raise TimeoutError(
                f"no response from OptiTrack server "
                f"{self.server_ip}:{self.command_port} within {timeout:.1f} s"
            ) from exc
        except Exception:
            self._client.shutdown()
            raise

        self._started = True

    def get_latest_pose(
        self,
        timeout: float = DEFAULT_POSE_TIMEOUT,
        *,
        require_tracking: bool = True,
    ) -> OptiTrackPose:
        """Return the latest pose, waiting briefly if none is available yet.

        By default, frames Motive marks as invalid are ignored while waiting.
        Set ``require_tracking=False`` when invalid poses are useful for
        diagnostics.
        """

        if not self.connected:
            raise RuntimeError("OptiTrackDriver.connect() must be called first")
        if timeout < 0.0:
            raise ValueError("pose timeout cannot be negative")

        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                pose = self._latest_pose
                if pose is not None and (
                    pose.tracking_valid or not require_tracking
                ):
                    return pose

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"no usable pose for rigid body {self.rigid_body_id} "
                        f"within {timeout:.1f} s ({self._last_status})"
                    )
                self._condition.wait(remaining)

    def close(self) -> None:
        """Stop the receiver threads and close the NatNet sockets."""

        self._client.shutdown()
        self._started = False

    def __enter__(self) -> OptiTrackDriver:
        self.connect()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def get_latest_pose(
    server_ip: str = DEFAULT_SERVER_IP,
    rigid_body_id: int = DEFAULT_RIGID_BODY_ID,
    *,
    client_ip: str | None = None,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    pose_timeout: float = DEFAULT_POSE_TIMEOUT,
) -> OptiTrackPose:
    """Connect, obtain one valid pose, and disconnect.

    For a loop, reuse :class:`OptiTrackDriver` instead of reconnecting for
    every sample.
    """

    driver = OptiTrackDriver(
        server_ip=server_ip,
        rigid_body_id=rigid_body_id,
        client_ip=client_ip,
    )
    try:
        driver.connect(timeout=connect_timeout)
        return driver.get_latest_pose(timeout=pose_timeout)
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read one OptiTrack pose.")
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    parser.add_argument("--client-ip", default=None)
    parser.add_argument("--rigid-id", type=int, default=DEFAULT_RIGID_BODY_ID)
    parser.add_argument("--timeout", type=float, default=DEFAULT_POSE_TIMEOUT)
    args = parser.parse_args()

    try:
        pose = get_latest_pose(
            server_ip=args.server_ip,
            rigid_body_id=args.rigid_id,
            client_ip=args.client_ip,
            pose_timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"OptiTrack test failed: {exc}\n")

    print(f"Rigid body {pose.rigid_body_id}, frame {pose.frame_number}")
    print(f"position_m:       {pose.position_m}")
    print(f"quaternion_xyzw:  {pose.quaternion_xyzw}")
    print(f"tracking_valid:   {pose.tracking_valid}")
    print(f"motive_timestamp: {pose.motive_timestamp_s:.6f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
