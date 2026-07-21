"""Headless, thread-safe NatNet receiver for one OptiTrack rigid body."""

from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import Pose, pose_from_optitrack

try:
    from natnet import NatNetClient, Version
    from natnet.packet_buffer import PacketBuffer
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in the container
    NatNetClient = None
    Version = None
    PacketBuffer = None
    NATNET_IMPORT_ERROR = exc
else:
    NATNET_IMPORT_ERROR = None


@dataclass(frozen=True)
class OptiSample:
    local_ns: int
    frame: Optional[int]
    motive_timestamp: Optional[float]
    rigid_id: int
    tracking_valid: Optional[bool]
    position: Optional[np.ndarray]
    quaternion: Optional[np.ndarray]

    def pose(self) -> Pose:
        if self.position is None or self.quaternion is None:
            raise ValueError("OptiTrack sample has no complete pose")
        return pose_from_optitrack(self.position, self.quaternion)


def patch_natnet_string_decoder() -> None:
    """Decode malformed/non-UTF8 server strings without dropping the connection."""

    if PacketBuffer is None:
        return
    original = PacketBuffer.read_string
    if getattr(original, "_giraf_lossy_utf8_patch", False):
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
            if max_length is None:
                raise ValueError("static NatNet string requires max_length")
            self.pointer += max_length
        else:
            self.pointer += len(encoded) + 1
        return decoded

    read_string_lossy._giraf_lossy_utf8_patch = True
    PacketBuffer.read_string = read_string_lossy


def local_ip_for_server(server_ip: str, command_port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((server_ip, command_port))
        return str(sock.getsockname()[0])
    finally:
        sock.close()


class OptiTrackReceiver:
    def __init__(
        self,
        server_ip: str,
        client_ip: str,
        rigid_id: int,
        data_port: int = 1511,
        command_port: int = 1510,
        use_multicast: bool = False,
    ) -> None:
        if NATNET_IMPORT_ERROR is not None:
            raise RuntimeError("natnet==0.2.0 is required") from NATNET_IMPORT_ERROR
        patch_natnet_string_decoder()
        self.rigid_id = int(rigid_id)
        self._lock = threading.Lock()
        self._sample: Optional[OptiSample] = None
        self._client = NatNetClient(
            server_ip_address=server_ip,
            local_ip_address=client_ip,
            command_port=int(command_port),
            data_port=int(data_port),
            use_multicast=bool(use_multicast),
        )
        # Motive in this deployment streams NatNet 4.3. The upstream package
        # currently exposes no public pre-connect protocol setter.
        self._client._NatNetClient__current_protocol_version = Version(4, 3)
        self._client.on_data_frame_received_event.handlers.append(self._on_frame)

    def _on_frame(self, frame) -> None:
        local_ns = time.monotonic_ns()
        body = next(
            (
                candidate
                for candidate in frame.rigid_bodies or ()
                if candidate.id_num == self.rigid_id
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
            )
        else:
            sample = OptiSample(
                local_ns=local_ns,
                frame=frame.prefix.frame_number,
                motive_timestamp=frame.suffix.timestamp,
                rigid_id=body.id_num,
                tracking_valid=body.tracking_valid,
                position=np.asarray(body.pos, dtype=float),
                quaternion=np.asarray(body.rot, dtype=float),
            )
        with self._lock:
            self._sample = sample

    def start(self) -> str:
        self._client.connect(timeout=5.0)
        description = "protocol=%s server=%s" % (
            self._client.protocol_version,
            self._client.server_info,
        )
        self._client.run_async()
        return description

    def stop(self) -> None:
        self._client.shutdown()

    def latest(self) -> Optional[OptiSample]:
        with self._lock:
            return self._sample

    def health(
        self, now_ns: int, max_age_ms: float
    ) -> Tuple[bool, str, float, Optional[OptiSample]]:
        sample = self.latest()
        if sample is None:
            return False, "no OptiTrack sample received", math.inf, None
        age_ms = (now_ns - sample.local_ns) / 1e6
        if age_ms > max_age_ms:
            return False, "OptiTrack sample is stale", age_ms, sample
        if sample.position is None or sample.quaternion is None:
            return False, "OptiTrack rigid body is absent", age_ms, sample
        if sample.tracking_valid is not True:
            return False, "OptiTrack tracking is invalid", age_ms, sample
        if not np.all(np.isfinite(sample.position)) or not np.all(
            np.isfinite(sample.quaternion)
        ):
            return False, "OptiTrack pose is non-finite", age_ms, sample
        try:
            sample.pose()
        except ValueError as exc:
            return False, "invalid OptiTrack pose: %s" % exc, age_ms, sample
        return True, "ready", age_ms, sample
