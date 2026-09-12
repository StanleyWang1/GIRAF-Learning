"""Camera-paced policy inference and rollout video capture."""

from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import torch

from giraf.data.config import CollectorConfig

from .configuration import DeploymentConfig
from .safety import guard_policy_action, state_bound_violations, state_from_joints
from .session import ActionSource, Session


class PolicyLoop:
    """Run synchronous policy inference on the camera-owning main thread."""

    def __init__(
        self,
        runtime: Session,
        config: DeploymentConfig,
        collector_config: CollectorConfig,
        policy,
        frame_queue,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.collector_config = collector_config
        self.policy = policy
        self.frame_queue = frame_queue
        self.generation = -1
        self.actions_remaining = 0
        self._last_violations: tuple[int, ...] | None = None

    def infer(self, image: np.ndarray, sequence: int, frame_age: float) -> None:
        runtime, policy, config = self.runtime, self.policy, self.config
        with runtime.lock:
            generation = runtime.generation
            if not runtime.accepts(generation):
                return
            if frame_age > config.max_frame_age_s:
                runtime.pause(f"camera frame stale by {frame_age:.3f}s")
                return
            state = state_from_joints(runtime.joints)
        try:
            if generation != self.generation:
                policy.reset()
                torch.manual_seed(config.seed)
                self.actions_remaining = 0
                self.generation = generation
                self._last_violations = None
            violations = state_bound_violations(
                state,
                policy.normalizer.state_low,
                policy.normalizer.state_high,
                margin_fraction=config.state_margin_fraction,
            )
            if violations != self._last_violations:
                runtime.log.write(
                    "state_distribution",
                    generation=generation,
                    outside=bool(violations),
                    dimensions=list(violations),
                )
                if violations:
                    print(
                        "[DEPLOY] Warning: state outside training bounds in "
                        f"dimensions {violations}",
                        flush=True,
                    )
                self._last_violations = violations
            if violations and config.enforce_training_bounds:
                raise ValueError(f"state outside training bounds: {violations}")
            started = time.monotonic()
            replanning = self.actions_remaining == 0
            if replanning:
                with runtime.lock:
                    if not runtime.hold_for_replan(generation, started):
                        return
                runtime.log.write(
                    "inference_started", generation=generation, sequence=sequence
                )
            inference_error = None
            try:
                raw_action = policy.act({"camera_rgb": image, "state": state})
            except Exception as exc:
                inference_error = str(exc)
                raise
            finally:
                latency = time.monotonic() - started
                # Retain slow/failed inference records even after a concurrent pause.
                if replanning or inference_error is not None:
                    with runtime.lock:
                        activation_current = runtime.accepts(generation)
                    runtime.log.write(
                        "inference_finished",
                        generation=generation,
                        sequence=sequence,
                        inference_latency_s=latency,
                        activation_current=activation_current,
                        error=inference_error,
                    )
            if runtime.stop.is_set():
                return
            if replanning:
                self.actions_remaining = policy.config.action_horizon
            self.actions_remaining -= 1
            safe_action = guard_policy_action(
                raw_action, scale=config.action_scale, allow_grasp=config.allow_grasp
            )
            with runtime.lock:
                if not runtime.accepts(generation):
                    return
                now = time.monotonic()
                if runtime.action_expired(now, config.action_timeout_s):
                    runtime.pause("policy action stale", generation=generation)
                    return
                runtime.publish(
                    generation, safe_action, now, allow_grasp=config.allow_grasp
                )
                applied = runtime.action.copy()
            runtime.log.write(
                "policy",
                generation=generation,
                sequence=sequence,
                frame_age_s=frame_age,
                inference_latency_s=latency,
                replanning=replanning,
                state=state.tolist(),
                raw_action=np.asarray(raw_action).tolist(),
                safe_action=applied.tolist(),
            )
        except Exception as exc:
            with runtime.lock:
                runtime.pause(f"policy: {exc}", generation=generation)

    def run(self) -> None:
        runtime = self.runtime
        while not runtime.stop.is_set():
            try:
                if runtime.stop.is_set():
                    return
                rgb, image, sequence, frame_age = read_frame(
                    self.frame_queue, self.collector_config
                )
                if runtime.stop.is_set():
                    return
                with runtime.lock:
                    generation = runtime.generation
                    record_frame = runtime.accepts(generation)
                if record_frame:
                    runtime.log.write_frame(rgb, generation=generation)
            except Exception as exc:
                with runtime.lock:
                    if runtime.active and runtime.source is ActionSource.POLICY:
                        runtime.pause(f"camera: {exc}")
                runtime.stop.wait(0.05)
            else:
                self.infer(image, sequence, frame_age)
            finally:
                runtime.log.finish_pending()


def _timestamp_ns(value) -> int:
    return int(round(value.total_seconds() * 1_000_000_000))


def read_frame(
    frame_queue, config: CollectorConfig
) -> tuple[np.ndarray, np.ndarray, int, float]:
    import cv2

    message = frame_queue.get(timedelta(seconds=0.25))
    if message is None:
        raise RuntimeError("camera did not produce a frame within 0.25s")
    while True:
        newer = frame_queue.tryGet()
        if newer is None:
            break
        message = newer
    received = time.monotonic()
    captured_ns = _timestamp_ns(message.getTimestamp())
    frame_age_s = received - captured_ns / 1_000_000_000.0
    if frame_age_s < -0.05 or frame_age_s > 10.0:
        raise RuntimeError(
            "DepthAI capture timestamp is not on the host monotonic clock"
        )
    bgr = message.getCvFrame()
    camera = config.camera
    if bgr.shape != (camera.height, camera.width, 3):
        raise RuntimeError(f"unexpected camera frame shape {bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = config.dataset.resize_dim
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return (
        rgb,
        resized.astype(np.uint8, copy=False),
        int(message.getSequenceNum()),
        frame_age_s,
    )
