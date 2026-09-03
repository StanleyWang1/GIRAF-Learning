"""Live conductor and teleoperation-facing collection API."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np

from giraf.settings import CONTROL_HZ

from .config import CollectorConfig, load_config
from .producer import CameraProducer
from .saver import SaverProcess
from .schema import aligned_example, camera_example, control_example, motor_example
from .shared_memory import RingBufferOverrun, SharedMemoryRingBuffer

KeyboardStatus = Callable[[], dict[str, bool | int]]


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


class DataCollectionPipeline:
    """Coordinate source rings, timestamp alignment, and episode saving."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        hardware_enabled: bool,
    ) -> None:
        config.validate()
        self.config = config
        self.hardware_enabled = bool(hardware_enabled)
        self.manager = SharedMemoryManager()
        self.manager.start()
        shm = config.shared_memory
        common = {
            "manager": self.manager,
            "get_time_budget": shm.get_time_budget_s,
            "safety_margin": shm.safety_margin,
        }
        self.camera_ring = SharedMemoryRingBuffer.create_from_examples(
            examples=camera_example(config),
            get_max_k=shm.camera_history,
            put_desired_frequency=config.camera.fps,
            **common,
        )
        self.control_ring = SharedMemoryRingBuffer.create_from_examples(
            examples=control_example(),
            get_max_k=shm.control_history,
            put_desired_frequency=CONTROL_HZ,
            **common,
        )
        self.motor_ring = SharedMemoryRingBuffer.create_from_examples(
            examples=motor_example(),
            get_max_k=shm.motor_history,
            put_desired_frequency=CONTROL_HZ,
            **common,
        )
        self.aligned_ring = SharedMemoryRingBuffer.create_from_examples(
            examples=aligned_example(config),
            get_max_k=shm.aligned_history,
            put_desired_frequency=config.dataset.aligned_hz,
            **common,
        )
        self.camera = CameraProducer(config.camera, self.camera_ring)
        self.saver = SaverProcess(config, self.aligned_ring)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.keyboard_status: KeyboardStatus | None = None
        self._episode_lock = threading.Lock()
        self._recording = False
        self._episode_start_ns = 0
        self._camera_cursor = 0
        self._last_toggle_count = 0
        self._next_emit_ns: int | None = None
        self._hard_error = threading.Event()
        self._error_lock = threading.Lock()
        self._error_reason = ""
        self._started = False

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        hardware_enabled: bool,
    ) -> DataCollectionPipeline:
        return cls(load_config(path), hardware_enabled=hardware_enabled)

    @property
    def is_recording(self) -> bool:
        with self._episode_lock:
            return self._recording

    @property
    def error(self) -> str:
        with self._error_lock:
            return self._error_reason

    def start(self, keyboard_status: KeyboardStatus, timeout: float = 20.0) -> None:
        if self._started:
            raise RuntimeError("data collection pipeline is already started")
        self.keyboard_status = keyboard_status
        try:
            self.saver.start()
            self.saver.start_wait(timeout=timeout)
            self.camera.start()
            self.camera.start_wait(timeout=timeout)
            initial = keyboard_status()
            self._last_toggle_count = int(initial.get("record_toggle_count", 0))
            self._camera_cursor = self.camera_ring.count
            self.thread = threading.Thread(
                target=self._conductor_loop,
                name="data-conductor",
                daemon=True,
            )
            self.thread.start()
            self._started = True
        except BaseException:
            self.camera.stop()
            self.saver.shutdown()
            self.manager.shutdown()
            raise

    def publish_control(
        self,
        *,
        timestamp_ns: int,
        task_twist,
        joint_velocity_command,
        joint_position_command,
        state,
        grasp: bool,
        clutch: bool,
        tracking: bool,
    ) -> bool:
        try:
            self.control_ring.put(
                {
                    "timestamp_ns": np.int64(timestamp_ns),
                    "task_twist": np.asarray(task_twist, dtype=np.float32),
                    "joint_velocity_command": np.asarray(
                        joint_velocity_command, dtype=np.float32
                    ),
                    "joint_position_command": np.asarray(
                        joint_position_command, dtype=np.float32
                    ),
                    "state": np.asarray(state, dtype=np.float32),
                    "grasp": np.uint8(grasp),
                    "clutch": np.uint8(clutch),
                    "tracking": np.uint8(tracking),
                },
                wait=False,
            )
            return True
        except Exception as exc:
            self._set_hard_error(f"control stream: {type(exc).__name__}: {exc}")
            return False

    def publish_motor(
        self,
        *,
        timestamp_ns: int,
        can_position_target,
        dynamixel_target_ticks,
        grasp: bool,
        command_accepted: bool,
    ) -> bool:
        try:
            self.motor_ring.put(
                {
                    "timestamp_ns": np.int64(timestamp_ns),
                    "can_position_target": np.asarray(
                        can_position_target, dtype=np.float32
                    ),
                    "dynamixel_target_ticks": np.asarray(
                        dynamixel_target_ticks, dtype=np.int32
                    ),
                    "grasp": np.uint8(grasp),
                    "command_accepted": np.uint8(command_accepted),
                },
                wait=False,
            )
            if not command_accepted:
                self._set_hard_error("motor command was not accepted for dispatch")
            return True
        except Exception as exc:
            self._set_hard_error(f"motor stream: {type(exc).__name__}: {exc}")
            return False

    def _set_hard_error(self, reason: str) -> None:
        with self._error_lock:
            if not self._error_reason:
                self._error_reason = reason
        self._hard_error.set()

    def _sources_ready(self) -> bool:
        return (
            self.camera_ring.count > 0
            and self.control_ring.count > 0
            and (not self.hardware_enabled or self.motor_ring.count > 0)
        )

    def _conductor_loop(self) -> None:
        try:
            while not self.stop_event.wait(0.002):
                camera_error = self.camera.poll_error()
                if camera_error:
                    self._set_hard_error(f"camera producer: {camera_error}")
                if self.saver.exitcode is not None:
                    self._set_hard_error(
                        f"dataset saver exited with code {self.saver.exitcode}"
                    )
                if self._hard_error.is_set():
                    if self.is_recording:
                        self.abort_episode(self.error)
                    continue
                self._poll_record_toggle()
                self._process_camera_samples()
        except BaseException as exc:
            self._set_hard_error(f"conductor: {type(exc).__name__}: {exc}")
            if self.is_recording:
                try:
                    self.abort_episode(self.error)
                except BaseException:
                    pass

    def _poll_record_toggle(self) -> None:
        assert self.keyboard_status is not None
        status = self.keyboard_status()
        toggle_count = int(status.get("record_toggle_count", 0))
        if toggle_count == self._last_toggle_count:
            return
        event_ns = int(status.get("record_event_timestamp_ns", time.monotonic_ns()))
        n_events = toggle_count - self._last_toggle_count
        self._last_toggle_count = toggle_count
        for _ in range(max(1, n_events)):
            if self.is_recording:
                self.end_episode(event_ns)
            else:
                self.start_episode(event_ns)

    def start_episode(self, start_monotonic_ns: int | None = None) -> bool:
        with self._episode_lock:
            if self._recording:
                return False
            if not self._sources_ready():
                print(
                    "\n[DATA] Record request ignored: sources are not ready.",
                    flush=True,
                )
                return False
            start_ns = int(start_monotonic_ns or time.monotonic_ns())
            now_monotonic_ns = time.monotonic_ns()
            start_wall_time_ns = time.time_ns() - (now_monotonic_ns - start_ns)
            self.saver.request(
                "start",
                start_count=self.aligned_ring.count,
                start_wall_time_ns=start_wall_time_ns,
                start_monotonic_ns=start_ns,
            )
            self._recording = True
            self._episode_start_ns = start_ns
            self._next_emit_ns = start_ns
            print("\n[DATA] Episode recording started.", flush=True)
            return True

    def end_episode(self, stop_monotonic_ns: int | None = None):
        with self._episode_lock:
            if not self._recording:
                return None
            stop_ns = int(stop_monotonic_ns or time.monotonic_ns())
            self._process_camera_samples(max_timestamp_ns=stop_ns, lock_held=True)
            result = self.saver.request(
                "stop",
                end_count=self.aligned_ring.count,
                timeout=600.0,
            )
            self._recording = False
            self._next_emit_ns = None
            if result.get("rejected"):
                print("\n[DATA] Empty episode rejected.", flush=True)
                return result
            print(
                f"\n[DATA] Episode {result['episode_index']} saved "
                f"({result['num_steps']} steps, {result['invalid_steps']} invalid).",
                flush=True,
            )
            return result

    def abort_episode(self, reason: str):
        with self._episode_lock:
            if not self._recording:
                return None
            result = self.saver.request("abort", reason=reason, timeout=30.0)
            self._recording = False
            self._next_emit_ns = None
            print(f"\n[DATA] Episode rejected: {reason}", flush=True)
            return result

    def _process_camera_samples(
        self,
        *,
        max_timestamp_ns: int | None = None,
        lock_held: bool = False,
    ) -> None:
        end = self.camera_ring.count
        if end == self._camera_cursor:
            return
        try:
            batch = self.camera_ring.get_range(self._camera_cursor, end)
        except RingBufferOverrun as exc:
            self._camera_cursor = end
            self._set_hard_error(f"camera ring overrun: {exc}")
            return
        start_count = self._camera_cursor
        self._camera_cursor = end
        for index in range(end - start_count):
            timestamp_ns = int(batch["timestamp_ns"][index])
            if max_timestamp_ns is not None and timestamp_ns > max_timestamp_ns:
                continue
            recording = self._recording if lock_held else self.is_recording
            if not recording or timestamp_ns < self._episode_start_ns:
                continue
            if not self._should_emit(timestamp_ns):
                continue
            camera = {key: value[index] for key, value in batch.items()}
            aligned = self._align(camera)
            if aligned is None:
                continue
            try:
                # Camera frames can accumulate while this thread is descheduled,
                # particularly around an episode stop. Pace catch-up writes so a
                # saver copy retains its configured shared-memory safety window.
                self.aligned_ring.put(aligned, wait=True)
            except BaseException as exc:
                self._set_hard_error(f"aligned stream: {type(exc).__name__}: {exc}")
                return

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
        return {
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

    def stop(self) -> None:
        if not self._started:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.is_recording:
            try:
                self.end_episode(time.monotonic_ns())
            except BaseException as exc:
                try:
                    self.abort_episode(f"shutdown commit failed: {exc}")
                except BaseException:
                    pass
        self.camera.stop()
        self.saver.shutdown()
        for ring in (
            self.camera_ring,
            self.control_ring,
            self.motor_ring,
            self.aligned_ring,
        ):
            ring.close()
        self.manager.shutdown()
        self._started = False
