"""Camera alignment isolated from teleoperation threads and their Python GIL."""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from .imu import align_imu
from .producer import ProducerProcess
from .schema import motor_example
from .shared_memory import RingBufferOverrun, SharedMemoryRingBuffer


def _sample_at_or_before(
    ring: SharedMemoryRingBuffer,
    timestamp_ns: int,
) -> dict[str, np.ndarray] | None:
    count = ring.count
    if count == 0:
        return None
    k = min(count, ring.get_max_k)
    batch = ring.get_last_k(k)
    timestamps = batch["timestamp_ns"]
    indices = np.flatnonzero(timestamps <= timestamp_ns)
    if indices.size == 0:
        return None
    index = int(indices[-1])
    return {key: value[index] for key, value in batch.items()}


class AlignmentProcess(ProducerProcess):
    """Single consumer of camera frames; publishes aligned data without disk I/O."""

    def __init__(
        self,
        config,
        camera_ring,
        control_ring,
        motor_ring,
        imu_rings,
        aligned_ring,
        *,
        hardware_enabled,
    ):
        super().__init__(name="camera-aligner")
        self.config = config
        self.camera_ring = camera_ring
        self.control_ring = control_ring
        self.motor_ring = motor_ring
        self.imu_rings = imu_rings
        self.aligned_ring = aligned_ring
        self.hardware_enabled = hardware_enabled
        self.parent_connection, self.child_connection = mp.Pipe()
        self._recording = False
        self._camera_cursor = 0
        self._episode_start_ns = 0
        self._next_emit_ns = None
        self._last_emit_ns = 0

    def run_producer(self):
        self._camera_cursor = self.camera_ring.count
        self.ready_event.set()
        while not self.stop_event.wait(0.002):
            if self.child_connection.poll():
                command = self.child_connection.recv()
                try:
                    result = self._handle(command)
                except BaseException as exc:
                    self.child_connection.send({"ok": False, "error": str(exc)})
                    raise
                self.child_connection.send({"ok": True, "result": result})
            self._process_camera_samples()

    def _handle(self, command):
        operation = command["operation"]
        if operation == "start":
            if self._recording:
                raise RuntimeError("alignment episode is already active")
            self._episode_start_ns = int(command["timestamp_ns"])
            self._next_emit_ns = self._episode_start_ns
            self._last_emit_ns = 0
            count = self.camera_ring.count
            self._camera_cursor = max(0, count - self.camera_ring.get_max_k)
            if count:
                first = self.camera_ring.get_range(
                    self._camera_cursor, self._camera_cursor + 1
                )
                if (
                    int(first["timestamp_ns"][0]) - self._episode_start_ns
                    > 1e9 / self.config.camera.fps
                ):
                    raise RuntimeError("recording start is older than camera history")
            self._recording = True
            return {"start_count": self.aligned_ring.count}
        if operation == "stop":
            requested = int(command["timestamp_ns"])
            # A delayed UI request cannot retract frames already saved to video.
            # Acknowledge an actual boundary covering everything already emitted.
            stop_ns = max(requested, self._last_emit_ns)
            self._process_camera_samples(max_timestamp_ns=stop_ns)
            self._recording = False
            return {
                "end_count": self.aligned_ring.count,
                "stop_monotonic_ns": stop_ns,
                "requested_stop_monotonic_ns": requested,
            }
        if operation == "abort":
            self._recording = False
            return {}
        raise ValueError(f"unknown alignment operation: {operation}")

    def request(self, operation, *, timeout=15.0, **payload):
        if self.pid is None or self.exitcode is not None:
            raise RuntimeError("camera aligner is not running")
        self.parent_connection.send({"operation": operation, **payload})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.parent_connection.poll(0.05):
                response = self.parent_connection.recv()
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                return response["result"]
            if self.exitcode is not None:
                raise RuntimeError("camera aligner exited during request")
        raise TimeoutError(f"camera aligner {operation} timed out")

    def _process_camera_samples(self, *, max_timestamp_ns=None):
        end = self.camera_ring.count
        if not self._recording:
            # Frames outside episodes may be discarded even after a long idle or
            # disk commit. Their overwritten history is not a recording failure.
            self._camera_cursor = end
            return
        while self._camera_cursor < end:
            # Bound RGB copies and catch-up work; get_max_k is a history limit,
            # not a requirement to copy the entire history in one operation.
            stop = min(end, self._camera_cursor + min(4, self.camera_ring.get_max_k))
            try:
                batch = self.camera_ring.get_range(self._camera_cursor, stop)
            except RingBufferOverrun as exc:
                raise RuntimeError(f"camera ring overrun: {exc}") from exc
            for index in range(stop - self._camera_cursor):
                timestamp_ns = int(batch["timestamp_ns"][index])
                if timestamp_ns < self._episode_start_ns:
                    continue
                if max_timestamp_ns is not None and timestamp_ns > max_timestamp_ns:
                    continue
                if not self._should_emit(timestamp_ns):
                    continue
                camera = {key: values[index] for key, values in batch.items()}
                aligned = self._align(camera)
                if aligned is not None:
                    self.aligned_ring.put(aligned, wait=True)
                    self._last_emit_ns = timestamp_ns
            self._camera_cursor = stop

    def _should_emit(self, timestamp_ns: int) -> bool:
        if self.config.dataset.aligned_hz >= self.config.camera.fps - 1e-6:
            return True
        if self._next_emit_ns is None:
            self._next_emit_ns = timestamp_ns
        if timestamp_ns < self._next_emit_ns:
            return False
        period_ns = int(round(1_000_000_000 / self.config.dataset.aligned_hz))
        self._next_emit_ns = timestamp_ns + period_ns
        return True

    def _align(self, camera: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        timestamp_ns = int(camera["timestamp_ns"])
        control = _sample_at_or_before(self.control_ring, timestamp_ns)
        if control is None:
            return None
        motor = _sample_at_or_before(self.motor_ring, timestamp_ns)
        if self.hardware_enabled and motor is None:
            return None
        if motor is None:
            # Dry run: no motor stream, so mirror the control sample's timing.
            motor = motor_example()
            motor["timestamp_ns"] = control["timestamp_ns"]
            motor["grasp"] = control["grasp"]

        control_age = timestamp_ns - int(control["timestamp_ns"])
        motor_age = timestamp_ns - int(motor["timestamp_ns"])
        valid = (
            0 <= control_age <= self.config.alignment.max_control_age_ms * 1_000_000
            and (
                not self.hardware_enabled
                or (
                    0 <= motor_age <= self.config.alignment.max_motor_age_ms * 1_000_000
                    and bool(motor["command_accepted"])
                )
            )
        )
        grasp = motor["grasp"] if self.hardware_enabled else control["grasp"]
        batches = {}
        for name, ring in self.imu_rings.items():
            count = ring.count
            if count:
                try:
                    batches[name] = ring.get_last_k(min(count, ring.get_max_k))
                except (RingBufferOverrun, TimeoutError):
                    pass  # Missing IMU invalidates only IMU, never RGB/control.
        imu = align_imu(
            batches,
            timestamp_ns,
            round(self.config.alignment.max_imu_age_ms * 1_000_000),
        )
        return {
            **imu,
            "camera_rgb_source": camera["camera_rgb_source"],
            "timestamp_ns": camera["timestamp_ns"],
            "camera_device_timestamp_ns": camera["device_timestamp_ns"],
            "camera_receive_timestamp_ns": camera["receive_timestamp_ns"],
            "camera_sequence_num": camera["sequence_num"],
            "control_timestamp_ns": control["timestamp_ns"],
            "motor_timestamp_ns": motor["timestamp_ns"],
            "task_twist": control["task_twist"],
            "joint_velocity_command": control["joint_velocity_command"],
            "joint_position_command": control["joint_position_command"],
            "state": control["state"],
            "grasp": np.uint8(grasp),
            "clutch": control["clutch"],
            "tracking": control["tracking"],
            "can_position_target": motor["can_position_target"],
            "dynamixel_target_ticks": motor["dynamixel_target_ticks"],
            "motor_command_accepted": motor["command_accepted"],
            "alignment_valid": np.uint8(valid),
        }
