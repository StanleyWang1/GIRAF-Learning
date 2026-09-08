"""Live conductor and teleoperation-facing collection API."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np

from giraf.settings import CONTROL_HZ

from .alignment import AlignmentProcess
from .config import CollectorConfig, load_config
from .imu import IMU_STREAMS, report_example
from .producer import CameraProducer
from .saver import SaverProcess
from .schema import aligned_example, camera_example, control_example, motor_example
from .shared_memory import RingBufferOverrun, SharedMemoryRingBuffer

KeyboardStatus = Callable[[], dict[str, bool | int]]


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
        self.imu_rings = (
            {
                name: SharedMemoryRingBuffer.create_from_examples(
                    examples=report_example(name),
                    get_max_k=shm.imu_history,
                    # Allow report-rate rounding and USB batches without blocking
                    # the producer on the shared-memory protection window.
                    put_desired_frequency=500,
                    **common,
                )
                for name in IMU_STREAMS
            }
            if config.imu.enabled
            else {}
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
        self.camera = CameraProducer(
            config.camera, self.camera_ring, config.imu, self.imu_rings
        )
        self.saver = SaverProcess(
            config, self.aligned_ring, self.imu_rings, self.camera.imu_dropped
        )
        self.aligner = AlignmentProcess(
            config,
            self.camera_ring,
            self.control_ring,
            self.motor_ring,
            self.imu_rings,
            self.aligned_ring,
            hardware_enabled=self.hardware_enabled,
        )
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.keyboard_status: KeyboardStatus | None = None
        self._episode_lock = threading.Lock()
        self._recording = False
        self._last_toggle_count = 0
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
            if not self.camera.metadata_parent.poll(timeout):
                raise RuntimeError("camera producer did not report IMU metadata")
            self.saver.request(
                "configure_imu", metadata=self.camera.metadata_parent.recv()
            )
            self.aligner.start()
            self.aligner.start_wait(timeout=timeout)
            initial = keyboard_status()
            self._last_toggle_count = int(initial.get("record_toggle_count", 0))
            self.thread = threading.Thread(
                target=self._conductor_loop,
                name="data-conductor",
                daemon=True,
            )
            self.thread.start()
            self._started = True
        except BaseException:
            self.aligner.stop()
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
        now_ns = time.monotonic_ns()
        for ring in self.imu_rings.values():
            if not ring.count:
                return False
            try:
                sample = ring.get()
            except (RingBufferOverrun, TimeoutError):
                return False
            if not sample["report_valid"] or not (
                0 <= now_ns - int(sample["timestamp_ns"]) <= 250_000_000
            ):
                return False
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
                alignment_error = self.aligner.poll_error()
                if alignment_error:
                    self._set_hard_error(f"camera alignment: {alignment_error}")
                if (
                    self.aligner.exitcode is not None
                    or self.camera.exitcode is not None
                ):
                    self._set_hard_error("camera acquisition/alignment process exited")
                if self.saver.exitcode is not None:
                    self._set_hard_error(
                        f"dataset saver exited with code {self.saver.exitcode}"
                    )
                if self._hard_error.is_set():
                    if self.is_recording:
                        self.abort_episode(self.error)
                    continue
                self._poll_record_toggle()
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
            boundary = self.aligner.request("start", timestamp_ns=start_ns)
            try:
                self.saver.request(
                    "start",
                    start_count=boundary["start_count"],
                    start_wall_time_ns=start_wall_time_ns,
                    start_monotonic_ns=start_ns,
                )
            except BaseException:
                self.aligner.request("abort")
                raise
            self._recording = True
            print("\n[DATA] Episode recording started.", flush=True)
            return True

    def end_episode(self, stop_monotonic_ns: int | None = None):
        with self._episode_lock:
            if not self._recording:
                return None
            stop_ns = int(stop_monotonic_ns or time.monotonic_ns())
            boundary = self.aligner.request("stop", timestamp_ns=stop_ns)
            result = self.saver.request(
                "stop",
                **boundary,
                timeout=600.0,
            )
            self._recording = False
            if result.get("rejected"):
                print("\n[DATA] Empty episode rejected.", flush=True)
                return result
            print(
                f"\n[DATA] Episode {result['episode_index']} saved "
                f"({result['num_steps']} steps, {result['invalid_steps']} invalid).",
                flush=True,
            )
            if self.config.imu.enabled:
                print(
                    f"[DATA] IMU valid: {result['imu']['valid_steps']}/{result['num_steps']} steps; "
                    f"reports/gaps: {result['imu']['streams']}",
                    flush=True,
                )
            return result

    def abort_episode(self, reason: str):
        with self._episode_lock:
            if not self._recording:
                return None
            if self.aligner.is_alive():
                self.aligner.request("abort")
            result = self.saver.request("abort", reason=reason, timeout=30.0)
            self._recording = False
            print(f"\n[DATA] Episode rejected: {reason}", flush=True)
            return result

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
        self.aligner.stop()
        self.camera.stop()
        self.saver.shutdown()
        for ring in (
            self.camera_ring,
            self.control_ring,
            self.motor_ring,
            self.aligned_ring,
            *self.imu_rings.values(),
        ):
            ring.close()
        self.manager.shutdown()
        self._started = False
