"""Dedicated saver process: drains aligned shared memory and owns disk writes."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from typing import Any

from .config import CollectorConfig
from .episode_stage import EpisodeStage
from .replay_buffer import ReplayBufferWriter
from .shared_memory import SharedMemoryRingBuffer


class SaverProcess(mp.Process):
    """Drain aligned shared memory and own every on-disk dataset write.

    The parent talks to the child through ``request()``; the child serves one
    operation at a time (start, stop, abort, shutdown) between drain passes.
    """

    def __init__(
        self,
        config: CollectorConfig,
        aligned_ring: SharedMemoryRingBuffer,
    ) -> None:
        super().__init__(name="dataset-saver")
        self.config = config
        self.aligned_ring = aligned_ring
        self.ready_event = mp.Event()
        self.parent_connection, self.child_connection = mp.Pipe(duplex=True)
        self._request_id = 0
        # Child-process state, populated in run().
        self._writer: ReplayBufferWriter | None = None
        self._stage: EpisodeStage | None = None
        self._cursor = 0
        self._running = False

    # -- child side ---------------------------------------------------------

    def run(self) -> None:
        self._cursor = self.aligned_ring.count
        try:
            self._writer = ReplayBufferWriter(self.config)
            self.config.video_dir.mkdir(parents=True, exist_ok=True)
            self.ready_event.set()
            self._running = True
            while self._running:
                if self.child_connection.poll(0.005):
                    self._serve(self.child_connection.recv())
                if self._stage is not None:
                    self._drain(self.aligned_ring.count)
        except BaseException:
            if self._stage is not None:
                try:
                    self._stage.reject("saver process failure")
                except BaseException:
                    pass
            try:
                self.child_connection.send(
                    {"request_id": -1, "ok": False, "error": traceback.format_exc()}
                )
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            self.ready_event.clear()

    def _serve(self, command: dict[str, Any]) -> None:
        request_id = command["request_id"]
        operation = command["operation"]
        try:
            result = self._handle(operation, command)
        except BaseException as exc:
            if self._stage is not None and operation in {"start", "stop"}:
                try:
                    self._stage.reject(f"{type(exc).__name__}: {exc}")
                finally:
                    self._stage = None
            self.child_connection.send(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        self.child_connection.send(
            {"request_id": request_id, "ok": True, "result": result}
        )

    def _handle(self, operation: str, command: dict[str, Any]) -> dict[str, Any]:
        if operation == "start":
            return self._start_episode(command)
        if operation == "stop":
            return self._stop_episode(command)
        if operation == "abort":
            return self._abort_episode(str(command["reason"]))
        if operation == "shutdown":
            if self._stage is not None:
                self._stage.reject("saver shutdown with active episode")
                self._stage = None
            self._running = False
            return {"shutdown": True}
        raise ValueError(f"unknown saver operation {operation!r}")

    def _start_episode(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._stage is not None:
            raise RuntimeError("an episode is already active")
        self._cursor = int(command["start_count"])
        self._stage = EpisodeStage(
            self.config,
            start_wall_time_ns=int(command["start_wall_time_ns"]),
            start_monotonic_ns=int(command["start_monotonic_ns"]),
        )
        return {"started": True}

    def _stop_episode(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._stage is None:
            raise RuntimeError("no episode is active")
        assert self._writer is not None
        self._drain(int(command["end_count"]))
        stage = self._stage
        if stage.length == 0:
            path = stage.reject("empty episode")
            self._stage = None
            return {
                "rejected": True,
                "reason": "empty episode",
                "rejected_path": str(path),
            }
        # _stage stays set until commit returns so _serve() can reject it on failure.
        result = stage.commit(self._writer)
        self._stage = None
        return result

    def _abort_episode(self, reason: str) -> dict[str, Any]:
        if self._stage is None:
            return {"rejected_path": None}
        stage, self._stage = self._stage, None
        return {"rejected_path": str(stage.reject(reason))}

    def _drain(self, target: int) -> None:
        assert self._stage is not None
        while self._cursor < target:
            stop = min(target, self._cursor + self.aligned_ring.get_max_k)
            batch = self.aligned_ring.get_range(self._cursor, stop)
            for index in range(stop - self._cursor):
                self._stage.append({key: value[index] for key, value in batch.items()})
            self._cursor = stop

    # -- parent side --------------------------------------------------------

    def start_wait(self, timeout: float = 15.0) -> None:
        if not self.ready_event.wait(timeout):
            if self.parent_connection.poll():
                message = self.parent_connection.recv()
                raise RuntimeError(message.get("error", "saver startup failed"))
            raise TimeoutError("dataset saver did not become ready")

    def request(self, operation: str, timeout: float = 30.0, **payload):
        if self.pid is None or self.exitcode is not None:
            raise RuntimeError("dataset saver is not running")
        self._request_id += 1
        request_id = self._request_id
        self.parent_connection.send(
            {"request_id": request_id, "operation": operation, **payload}
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.parent_connection.poll(0.05):
                response = self.parent_connection.recv()
                if response.get("request_id") not in (request_id, -1):
                    continue
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "saver request failed"))
                return response.get("result")
            if self.exitcode is not None:
                raise RuntimeError(f"dataset saver exited with code {self.exitcode}")
        raise TimeoutError(f"saver operation {operation!r} timed out")

    def shutdown(self) -> None:
        if self.pid is None:
            return
        if self.is_alive():
            try:
                self.request("shutdown", timeout=10.0)
            except (RuntimeError, TimeoutError, BrokenPipeError, EOFError, OSError):
                pass
            self.join(timeout=5.0)
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)
