"""Clutch-gated session state shared by deployment workers.

The control worker owns inputs, joint integration and motor commands. Policy
results can only enter the session while their activation generation is current.
Call state-changing methods under ``lock`` (an RLock permits nested calls).
"""

from __future__ import annotations

import threading
from enum import Enum

import numpy as np

from giraf.data.schema import ACTION_DIM, GRASP_INDEX

INITIAL_JOINTS = np.array((0.0, 0.0, 0.31, 0.0, 0.0, 0.0), dtype=np.float32)


class ActionSource(str, Enum):
    TELEOP = "teleop"
    POLICY = "policy"


class Session:
    def __init__(self, log) -> None:
        self.log = log
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.joints = INITIAL_JOINTS.copy()
        self.grasp = False
        self.source = ActionSource.TELEOP
        self.active = False
        self.armed = False
        self.clutch = True  # unknown until the initial keyboard snapshot
        self.generation = 0
        self.reason = "release SPACE, then press to enable teleop"
        self.action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.action_time = 0.0
        self.started_at = 0.0
        self.pose = None
        self.pose_error = "waiting for OptiTrack"
        self.stop_reason = ""
        self.error = ""

    def initialize_keys(self, clutch: bool) -> None:
        self.clutch = clutch
        self.armed = not clutch

    def pause(self, reason: str, *, generation: int | None = None) -> None:
        if self.stop.is_set() or (
            generation is not None and generation != self.generation
        ):
            return
        ended_generation = self.generation
        self.active = False
        self.armed = not self.clutch
        self.generation += 1
        self.action[:] = 0
        self.action[GRASP_INDEX] = self.grasp
        self.action_time = 0.0
        self.reason = reason
        self.log.write(
            "paused",
            source=self.source.value,
            reason=reason,
            generation=self.generation,
            ended_generation=ended_generation,
            joints=self.joints.tolist(),
            grasp=self.grasp,
        )

    def input(self, key: str, value: bool, now: float) -> None:
        if self.stop.is_set():
            return
        if key == "quit":
            self.finish("quit_key")
        elif key == "mode":
            self.source = (
                ActionSource.POLICY
                if self.source is ActionSource.TELEOP
                else ActionSource.TELEOP
            )
            self.pause("mode switched; fresh SPACE press required")
            self.log.write("mode_changed", source=self.source.value)
        elif key == "clutch":
            was_down = self.clutch
            self.clutch = value
            if not value:
                self.pause("clutch released")
            elif not was_down and self.armed:
                self.active = True
                self.armed = False
                self.generation += 1
                self.started_at = now
                self.action_time = 0.0
                self.reason = "active"
                self.log.write(
                    "activated",
                    source=self.source.value,
                    generation=self.generation,
                    joints=self.joints.tolist(),
                    grasp=self.grasp,
                )
        elif key == "grasp" and self.source is ActionSource.TELEOP:
            self.grasp = not self.grasp

    def accepts(self, generation: int) -> bool:
        return (
            not self.stop.is_set()
            and self.active
            and self.source is ActionSource.POLICY
            and self.generation == generation
        )

    def hold_for_replan(self, generation: int, now: float) -> bool:
        if not self.accepts(generation):
            return False
        self.action[:] = 0
        self.action[GRASP_INDEX] = self.grasp
        self.action_time = now
        return True

    def publish(
        self, generation: int, action, now: float, *, allow_grasp: bool
    ) -> bool:
        if not self.accepts(generation):
            return False
        self.action = np.asarray(action, dtype=np.float32).copy()
        if not allow_grasp:
            self.action[GRASP_INDEX] = self.grasp
        self.action_time = now
        return True

    def finish(self, reason: str) -> None:
        with self.lock:
            if self.stop.is_set():
                return
            ended_generation = self.generation
            self.active = False
            self.generation += 1
            self.stop_reason = reason
            self.stop.set()
            self.log.write(
                "stop_requested",
                reason=reason,
                generation=self.generation,
                ended_generation=ended_generation,
                joints=self.joints.tolist(),
                grasp=self.grasp,
            )

    def fail(self, source: str, error: BaseException | str) -> None:
        with self.lock:
            self.error = self.error or f"{source}: {error}"
            self.log.write(
                "error",
                source=source,
                detail=str(error),
                generation=self.generation,
            )
            self.finish("error")

    def action_expired(self, now: float, timeout: float) -> bool:
        return now - (self.action_time or self.started_at) > timeout

    def status(self) -> str:
        with self.lock:
            return (
                f"{self.source.value} {'ACTIVE' if self.active else 'PAUSED'} "
                f"grasp={self.grasp} q={np.round(self.joints, 3)} | {self.reason}"
            )
