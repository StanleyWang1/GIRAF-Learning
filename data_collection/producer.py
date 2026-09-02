"""Generic process producer lifecycle and the DepthAI camera producer."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from abc import ABC, abstractmethod

import numpy as np

from .config import CameraConfig
from .shared_memory import SharedMemoryRingBuffer


class ProducerProcess(mp.Process, ABC):
    """A process with explicit ready, stop, and error signaling."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name)
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self._error_parent, self._error_child = mp.Pipe(duplex=False)

    def run(self) -> None:
        try:
            self.run_producer()
        except BaseException:
            try:
                self._error_child.send(traceback.format_exc())
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            self.ready_event.clear()

    @abstractmethod
    def run_producer(self) -> None:
        """Run until ``stop_event`` is set."""

    def start_wait(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            error = self.poll_error()
            if error:
                raise RuntimeError(f"{self.name} failed during startup:\n{error}")
            if self.ready_event.wait(timeout=0.05):
                return
            if self.exitcode is not None:
                raise RuntimeError(
                    f"{self.name} exited with code {self.exitcode} during startup"
                )
        raise TimeoutError(f"{self.name} did not become ready within {timeout:.1f}s")

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.pid is None:
            return
        self.join(timeout=timeout)
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)

    def poll_error(self) -> str | None:
        if self._error_parent.poll():
            try:
                return str(self._error_parent.recv())
            except EOFError:
                return f"{self.name} error channel closed"
        return None


class CameraProducer(ProducerProcess):
    """Own the DepthAI device and publish source-resolution RGB frames."""

    def __init__(
        self,
        config: CameraConfig,
        ring: SharedMemoryRingBuffer,
    ) -> None:
        super().__init__(name="camera-producer")
        self.config = config
        self.ring = ring

    @staticmethod
    def _timestamp_ns(value) -> int:
        return int(round(value.total_seconds() * 1_000_000_000))

    def run_producer(self) -> None:
        import cv2

        from camera_driver import camera_connect, camera_disconnect, camera_read_message

        pipeline = None
        try:
            pipeline, frame_queue = camera_connect(
                frame_size=(self.config.width, self.config.height),
                frame_rate=self.config.fps,
            )
            first_frame = True
            while not self.stop_event.is_set() and pipeline.isRunning():
                message = camera_read_message(frame_queue)
                receive_ns = time.monotonic_ns()
                capture_ns = self._timestamp_ns(message.getTimestamp())
                device_ns = self._timestamp_ns(message.getTimestampDevice())
                if first_frame:
                    # Both values should use the host monotonic clock. Refuse to
                    # record against an unverified clock mapping.
                    if abs(receive_ns - capture_ns) > 10_000_000_000:
                        raise RuntimeError(
                            "DepthAI host timestamp is not aligned with "
                            "time.monotonic_ns()"
                        )
                    first_frame = False
                bgr = message.getCvFrame()
                if bgr.shape != (self.config.height, self.config.width, 3):
                    raise RuntimeError(f"unexpected camera frame shape {bgr.shape}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.ring.put(
                    {
                        "camera_rgb_source": rgb,
                        "timestamp_ns": np.int64(capture_ns),
                        "device_timestamp_ns": np.int64(device_ns),
                        "receive_timestamp_ns": np.int64(receive_ns),
                        "sequence_num": np.int64(message.getSequenceNum()),
                    },
                    wait=False,
                )
                self.ready_event.set()
        finally:
            if pipeline is not None:
                camera_disconnect(pipeline)
