"""Guarded live rollout of a trained diffusion policy."""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from giraf.data.config import CollectorConfig, load_config
from giraf.learning import DiffusionPolicy
from giraf.settings import CONTROL_HZ

from .reference import load_reference_start
from .safety import (
    SafetyLimits,
    guard_policy_action,
    plan_joint_command,
    plan_staging_command,
    state_bound_violations,
    state_from_joints,
    validate_staging_target,
)

INITIAL_JOINTS = np.array((0.0, 0.0, 0.31, 0.0, 0.0, 0.0), dtype=np.float32)


class DeploymentMode(str, Enum):
    SHADOW = "shadow"
    DRY_RUN = "dry-run"
    HARDWARE = "hardware"


class DeploymentPhase(str, Enum):
    WAIT_HOME_RELEASE = "wait_home_release"
    WAIT_STAGE_PRESS = "wait_stage_press"
    STAGING = "staging"
    WAIT_STAGE_RELEASE = "wait_stage_release"
    PREVIEW = "preview"
    WAIT_ROLLOUT_RELEASE = "wait_rollout_release"
    WAIT_ROLLOUT_PRESS = "wait_rollout_press"
    ROLLOUT = "rollout"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    checkpoint: Path
    reference_dataset: Path
    reference_episode: int = 0
    collector_config: Path = Path("config/tape_grasping.yaml")
    mode: DeploymentMode = DeploymentMode.SHADOW
    device: str = "cuda"
    action_scale: float = 0.2
    inference_steps: int | None = None
    duration_s: float = 5.0
    action_timeout_s: float = 0.5
    max_frame_age_s: float = 0.15
    state_margin_fraction: float = 0.05
    allow_grasp: bool = False
    seed: int = 0
    log_dir: Path | None = None
    save_video: bool = True
    hardware_confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint}")
        if not self.collector_config.is_file():
            raise FileNotFoundError(
                f"collector config does not exist: {self.collector_config}"
            )
        if not self.reference_dataset.is_dir():
            raise FileNotFoundError(
                f"reference dataset does not exist: {self.reference_dataset}"
            )
        if self.reference_episode < 0:
            raise ValueError("reference_episode must be non-negative")
        if self.inference_steps is not None and self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive when provided")
        if self.mode is DeploymentMode.HARDWARE and not self.hardware_confirmed:
            raise ValueError(
                "hardware mode requires confirmation that the robot is physically "
                "at the teleop home pose"
            )
        positive = (
            self.duration_s,
            self.action_timeout_s,
            self.max_frame_age_s,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("duration and timeout values must be finite and positive")
        if not math.isfinite(self.action_scale) or not 0 <= self.action_scale <= 1:
            raise ValueError("action_scale must be finite and in [0, 1]")
        if (
            not math.isfinite(self.state_margin_fraction)
            or self.state_margin_fraction < 0
        ):
            raise ValueError("state_margin_fraction must be finite and non-negative")


class _EventLog:
    def __init__(self, directory: Path, config: DeploymentConfig) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self._lock = threading.Lock()
        self._file = (directory / "events.jsonl").open("x")
        payload = asdict(config)
        payload["checkpoint"] = str(config.checkpoint)
        payload["reference_dataset"] = str(config.reference_dataset)
        payload["collector_config"] = str(config.collector_config)
        payload["mode"] = config.mode.value
        payload["log_dir"] = str(directory)
        (directory / "config.json").write_text(json.dumps(payload, indent=2) + "\n")

    def write(self, event: str, **values) -> None:
        record = {
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }
        with self._lock:
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


class _VideoLog:
    def __init__(self, path: Path, *, width: int, height: int, fps: float) -> None:
        import av

        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=int(round(fps)))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": "21"}

    def write(self, rgb: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


class _Runtime:
    def __init__(
        self,
        log: _EventLog,
        *,
        initial_joints: np.ndarray,
        staging_target: np.ndarray,
        hardware: bool,
    ) -> None:
        self.log = log
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.joints = np.asarray(initial_joints, dtype=np.float32).copy()
        self.staging_target = np.asarray(staging_target, dtype=np.float32).copy()
        self.action = np.zeros(7, dtype=np.float32)
        self.action_time = 0.0
        self.started_at = 0.0
        self.phase = (
            DeploymentPhase.WAIT_HOME_RELEASE
            if hardware
            else DeploymentPhase.PREVIEW
        )
        self.stop_reason = ""
        self.error = ""

    def set_phase(self, phase: DeploymentPhase, event: str) -> None:
        with self.lock:
            self.phase = phase
        self.log.write(event, phase=phase.value)

    def start_trial(self) -> None:
        with self.lock:
            self.phase = DeploymentPhase.ROLLOUT
            self.started_at = time.monotonic()
            self.action_time = 0.0
        self.log.write("trial_started", phase=DeploymentPhase.ROLLOUT.value)

    def set_action(self, action: np.ndarray) -> None:
        with self.lock:
            self.action = np.asarray(action, dtype=np.float32).copy()
            self.action_time = time.monotonic()

    def finish(self, reason: str) -> None:
        with self.lock:
            if not self.stop_reason:
                self.stop_reason = reason
            self.phase = DeploymentPhase.STOPPED
        self.log.write("stop_requested", reason=reason)
        self.stop.set()

    def fail(self, source: str, error: BaseException | str) -> None:
        detail = f"{source}: {error}"
        with self.lock:
            if not self.error:
                self.error = detail
                self.stop_reason = "error"
            self.phase = DeploymentPhase.STOPPED
        self.log.write("error", source=source, detail=str(error))
        self.stop.set()


class _ControlWorker(threading.Thread):
    def __init__(
        self,
        runtime: _Runtime,
        keyboard,
        config: DeploymentConfig,
        *,
        limits: SafetyLimits,
    ) -> None:
        super().__init__(name="deployment-control")
        self.runtime = runtime
        self.keyboard = keyboard
        self.config = config
        self.limits = limits
        self.ready = threading.Event()

    def run(self) -> None:
        mab = dxl = sync_write = None
        try:
            if self.config.mode is DeploymentMode.HARDWARE:
                from giraf.drivers.dynamixel import (
                    GRIPPER,
                    dynamixel_connect,
                )
                from giraf.drivers.dynamixel_config import TORQUE_ENABLE
                from giraf.drivers.mab_worker import MabWorker

                print("[DEPLOY] Connecting Dynamixel motors...", flush=True)
                dxl, sync_write = dynamixel_connect()
                if not dxl.WRITE(GRIPPER, TORQUE_ENABLE, 1):
                    raise RuntimeError("could not enable gripper torque")
                print("[DEPLOY] Starting isolated MAB worker...", flush=True)
                mab = MabWorker()
                mab.start()
                print("[DEPLOY] Motors ready and holding the home target.", flush=True)

            self.ready.set()
            period = 1.0 / CONTROL_HZ
            last = time.monotonic()
            while not self.runtime.stop.is_set():
                loop_start = time.monotonic()
                dt = min(max(loop_start - last, 1e-6), 0.02)
                last = loop_start

                from giraf.drivers.keyboard import keyboard_status

                clutch = bool(keyboard_status(self.keyboard)["clutch"])
                with self.runtime.lock:
                    phase = self.runtime.phase
                    started_at = self.runtime.started_at
                    action_time = self.runtime.action_time
                    action = self.runtime.action.copy()
                    joints = self.runtime.joints.copy()
                    staging_target = self.runtime.staging_target.copy()

                if phase is DeploymentPhase.WAIT_HOME_RELEASE and not clutch:
                    self.runtime.set_phase(
                        DeploymentPhase.WAIT_STAGE_PRESS, "home_deadman_armed"
                    )
                    phase = DeploymentPhase.WAIT_STAGE_PRESS
                elif phase is DeploymentPhase.WAIT_STAGE_PRESS and clutch:
                    self.runtime.set_phase(DeploymentPhase.STAGING, "staging_started")
                    phase = DeploymentPhase.STAGING
                elif phase is DeploymentPhase.STAGING and not clutch:
                    self.runtime.finish("staging_deadman_released")
                    break
                elif phase is DeploymentPhase.WAIT_STAGE_RELEASE and not clutch:
                    self.runtime.set_phase(DeploymentPhase.PREVIEW, "staging_accepted")
                    phase = DeploymentPhase.PREVIEW
                elif phase is DeploymentPhase.WAIT_ROLLOUT_RELEASE and not clutch:
                    self.runtime.set_phase(
                        DeploymentPhase.WAIT_ROLLOUT_PRESS, "rollout_deadman_armed"
                    )
                    phase = DeploymentPhase.WAIT_ROLLOUT_PRESS

                if phase is DeploymentPhase.ROLLOUT and not clutch:
                    self.runtime.finish("deadman_released")
                    break
                if (
                    phase is DeploymentPhase.ROLLOUT
                    and loop_start - started_at >= self.config.duration_s
                ):
                    self.runtime.finish("duration_complete")
                    break
                if phase is DeploymentPhase.ROLLOUT:
                    freshness_origin = action_time or started_at
                    if loop_start - freshness_origin > self.config.action_timeout_s:
                        self.runtime.fail(
                            "control",
                            f"policy action stale for {loop_start - freshness_origin:.3f}s",
                        )
                        break

                staging = phase is DeploymentPhase.STAGING and clutch
                active = (
                    phase is DeploymentPhase.ROLLOUT
                    and clutch
                    and action_time > 0.0
                )
                if staging:
                    command = plan_staging_command(
                        joints, staging_target, dt=dt, limits=self.limits
                    )
                else:
                    command_action = (
                        action if active else np.zeros(7, dtype=np.float32)
                    )
                    command = plan_joint_command(
                        joints, command_action, dt=dt, limits=self.limits
                    )
                if (staging or active) and self.config.mode is not DeploymentMode.SHADOW:
                    with self.runtime.lock:
                        self.runtime.joints = command.joint_position.copy()

                if staging and np.allclose(
                    command.joint_position, staging_target, rtol=0.0, atol=1e-6
                ):
                    self.runtime.set_phase(
                        DeploymentPhase.WAIT_STAGE_RELEASE, "staging_complete"
                    )

                command_accepted = None
                if self.config.mode is DeploymentMode.HARDWARE:
                    assert mab is not None and dxl is not None and sync_write is not None
                    from giraf.drivers.dynamixel import dynamixel_drive

                    mab.command(*command.can_position_target)
                    command_accepted = dynamixel_drive(
                        dxl, sync_write, list(command.dynamixel_target_ticks)
                    )
                    if not command_accepted:
                        raise RuntimeError("Dynamixel command failed")

                self.runtime.log.write(
                    "control",
                    phase=phase.value,
                    staging=staging,
                    active=active,
                    clutch=clutch,
                    action_age_s=None
                    if action_time == 0.0
                    else loop_start - action_time,
                    action=command.action.tolist(),
                    joint_velocity=command.joint_velocity.tolist(),
                    joint_position=command.joint_position.tolist(),
                    can_position_target=list(command.can_position_target),
                    dynamixel_target_ticks=list(command.dynamixel_target_ticks),
                    command_accepted=command_accepted,
                )
                self.runtime.stop.wait(
                    max(0.0, period - (time.monotonic() - loop_start))
                )
        except BaseException as exc:  # hardware failures must stop the rollout
            self.runtime.fail("control", exc)
        finally:
            self.ready.set()
            if dxl is not None:
                try:
                    from giraf.drivers.dynamixel import (
                        dynamixel_disconnect,
                    )

                    dynamixel_disconnect(dxl)
                    dxl.close_port()
                except BaseException as exc:
                    self.runtime.fail("Dynamixel shutdown", exc)
            if mab is not None:
                try:
                    mab.stop()
                except BaseException as exc:
                    self.runtime.fail("MAB shutdown", exc)


def _timestamp_ns(value) -> int:
    return int(round(value.total_seconds() * 1_000_000_000))


def _read_frame(frame_queue, config: CollectorConfig) -> tuple[np.ndarray, np.ndarray, int, float]:
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
        raise RuntimeError("DepthAI capture timestamp is not on the host monotonic clock")
    bgr = message.getCvFrame()
    camera = config.camera
    if bgr.shape != (camera.height, camera.width, 3):
        raise RuntimeError(f"unexpected camera frame shape {bgr.shape}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = config.dataset.resize_dim
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb, resized.astype(np.uint8, copy=False), int(message.getSequenceNum()), frame_age_s


def _default_log_dir() -> Path:
    return Path("deployment_runs") / time.strftime("%Y%m%d-%H%M%S")


def run(config: DeploymentConfig) -> Path:
    """Run one guarded live rollout and return its log directory."""

    collector_config = load_config(config.collector_config)
    reference = load_reference_start(
        config.reference_dataset, config.reference_episode
    )
    resize_width, resize_height = collector_config.dataset.resize_dim
    expected_reference_shape = (resize_height, resize_width, 3)
    if reference.camera_rgb.shape != expected_reference_shape:
        raise ValueError(
            f"reference RGB shape is {reference.camera_rgb.shape}, expected "
            f"{expected_reference_shape} from {config.collector_config}"
        )
    limits = SafetyLimits()
    validate_staging_target(reference.joints, limits=limits)
    log_dir = config.log_dir or _default_log_dir()
    log = _EventLog(log_dir, config)
    reference_path = log_dir / "reference_start.png"
    hardware = config.mode is DeploymentMode.HARDWARE
    runtime = _Runtime(
        log,
        initial_joints=INITIAL_JOINTS if hardware else reference.joints,
        staging_target=reference.joints,
        hardware=hardware,
    )
    keyboard = camera_pipeline = video = control = None
    keyboard_thread = None
    old_handlers = {}

    def request_stop(signum, _frame) -> None:
        # Keep the signal handler lock-free: it can interrupt a log write on the
        # main thread. The final event records the reason during normal cleanup.
        runtime.stop_reason = f"signal_{signum}"
        runtime.stop.set()

    try:
        log.write("startup")
        import cv2

        if not cv2.imwrite(
            str(reference_path),
            cv2.cvtColor(reference.camera_rgb, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError(f"could not save reference image: {reference_path}")
        print(f"[DEPLOY] Loading {config.checkpoint} on {config.device}...", flush=True)
        torch.manual_seed(config.seed)
        policy = DiffusionPolicy.load(config.checkpoint, device=config.device)
        if policy.config.action_space != "twist":
            raise SystemExit(
                "deployment executes twist actions; checkpoint action_space is "
                f"{policy.config.action_space!r}"
            )
        if policy.normalizer is None:
            raise RuntimeError("checkpoint does not contain a training normalizer")
        if config.inference_steps is not None:
            if config.inference_steps > policy.config.diffusion_steps:
                raise ValueError(
                    "inference_steps cannot exceed the checkpoint diffusion_steps"
                )
            policy.config = replace(
                policy.config, inference_steps=config.inference_steps
            )
        reference_violations = state_bound_violations(
            reference.state,
            policy.normalizer.state_low,
            policy.normalizer.state_high,
            margin_fraction=config.state_margin_fraction,
        )
        if reference_violations:
            raise RuntimeError(
                "reference start is outside checkpoint training bounds in dimensions "
                f"{reference_violations}"
            )
        log.write(
            "reference_loaded",
            dataset=str(reference.dataset),
            episode=reference.episode,
            step=reference.step,
            joints=reference.joints.tolist(),
            state=reference.state.tolist(),
        )

        from giraf.drivers.camera import camera_connect, camera_disconnect
        from giraf.drivers.keyboard import keyboard_connect, keyboard_control

        print("[DEPLOY] Opening keyboard and camera...", flush=True)
        keyboard = keyboard_connect()
        keyboard_thread = threading.Thread(
            target=keyboard_control,
            args=(keyboard, runtime.stop),
            name="deployment-keyboard",
        )
        keyboard_thread.start()
        camera_pipeline, frame_queue = camera_connect(
            frame_size=(collector_config.camera.width, collector_config.camera.height),
            frame_rate=collector_config.camera.fps,
        )
        frame_queue.setMaxSize(2)
        frame_queue.setBlocking(False)
        if config.save_video:
            video = _VideoLog(
                log_dir / "camera.mp4",
                width=collector_config.camera.width,
                height=collector_config.camera.height,
                fps=collector_config.camera.fps,
            )

        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            old_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, request_stop)

        control = _ControlWorker(runtime, keyboard, config, limits=limits)
        if hardware:
            print(
                "[HARDWARE] Using the current physical pose as MAB encoder zero. "
                "It must be the same teleop home pose used for data collection.",
                flush=True,
            )
        control.start()
        if not control.ready.wait(timeout=20.0):
            raise TimeoutError("control worker did not become ready within 20s")
        if runtime.error:
            raise RuntimeError(runtime.error)

        print(
            f"[DEPLOY] mode={config.mode.value} scale={config.action_scale:.2f} "
            f"duration={config.duration_s:.1f}s grasp={config.allow_grasp} "
            f"inference_steps={policy.config.inference_steps}",
            flush=True,
        )
        print(
            f"[DEPLOY] reference={reference.dataset} episode={reference.episode} "
            f"q0={np.round(reference.joints, 4)}",
            flush=True,
        )
        print(f"[DEPLOY] reference image: {reference_path}", flush=True)

        last_status = 0.0
        last_phase = None
        actions_remaining = 0
        while not runtime.stop.is_set():
            rgb, image, sequence, frame_age = _read_frame(frame_queue, collector_config)
            if video is not None:
                video.write(rgb)
            if frame_age > config.max_frame_age_s:
                raise RuntimeError(f"camera frame is stale by {frame_age:.3f}s")

            from giraf.drivers.keyboard import keyboard_status

            clutch = bool(keyboard_status(keyboard)["clutch"])
            with runtime.lock:
                phase = runtime.phase
                joints = runtime.joints.copy()

            if phase is not last_phase:
                print()
                if phase is DeploymentPhase.WAIT_HOME_RELEASE:
                    print(
                        "[STAGE] Robot must be physically at teleop home; "
                        "release SPACE to arm staging.",
                        flush=True,
                    )
                elif phase is DeploymentPhase.WAIT_STAGE_PRESS:
                    print(
                        "[STAGE] Hold SPACE continuously to move slowly to the "
                        "recorded start pose; release early to stop.",
                        flush=True,
                    )
                elif phase is DeploymentPhase.WAIT_STAGE_RELEASE:
                    print(
                        "[STAGE] Recorded start pose reached. Release SPACE to "
                        "accept it and run the preview.",
                        flush=True,
                    )
                elif phase is DeploymentPhase.PREVIEW:
                    print("[PREVIEW] Running one no-motion policy inference...", flush=True)
                elif phase is DeploymentPhase.WAIT_ROLLOUT_RELEASE:
                    print("[RUN] Release SPACE to arm the rollout.", flush=True)
                elif phase is DeploymentPhase.WAIT_ROLLOUT_PRESS:
                    print(
                        "[RUN] Hold SPACE to begin; release it to stop. "
                        "Ctrl-C also stops.",
                        flush=True,
                    )
                last_phase = phase

            if phase is DeploymentPhase.STAGING:
                now = time.monotonic()
                if now - last_status >= 0.25:
                    remaining = reference.joints - joints
                    print(
                        f"\r[STAGE] q={np.round(joints, 3)} "
                        f"remaining={np.round(remaining, 3)}   ",
                        end="",
                        flush=True,
                    )
                    last_status = now
                continue

            if phase is DeploymentPhase.PREVIEW:
                state = state_from_joints(joints)
                if not np.allclose(joints, reference.joints, rtol=0.0, atol=1e-5):
                    raise RuntimeError("preview pose does not match the reference start")
                preview_started = time.monotonic()
                raw_preview = policy.act({"camera_rgb": image, "state": state})
                preview_latency = time.monotonic() - preview_started
                safe_preview = guard_policy_action(
                    raw_preview,
                    scale=config.action_scale,
                    allow_grasp=config.allow_grasp,
                    limits=limits,
                )
                preview_command = plan_joint_command(
                    joints, safe_preview, dt=1.0 / CONTROL_HZ, limits=limits
                )
                log.write(
                    "preview",
                    sequence=sequence,
                    frame_age_s=frame_age,
                    inference_latency_s=preview_latency,
                    state=state.tolist(),
                    raw_action=np.asarray(raw_preview).tolist(),
                    safe_action=safe_preview.tolist(),
                    joint_velocity=preview_command.joint_velocity.tolist(),
                    joint_position=preview_command.joint_position.tolist(),
                )
                print(f"[PREVIEW] raw={np.round(raw_preview, 4)}", flush=True)
                print(f"[PREVIEW] safe={np.round(safe_preview, 4)}", flush=True)
                print(
                    f"[PREVIEW] qdot={np.round(preview_command.joint_velocity, 4)} "
                    f"latency={preview_latency * 1000:.1f}ms",
                    flush=True,
                )
                policy.reset()
                actions_remaining = 0
                torch.manual_seed(config.seed)
                runtime.set_phase(
                    DeploymentPhase.WAIT_ROLLOUT_RELEASE, "preview_complete"
                )
                continue

            if phase is DeploymentPhase.WAIT_ROLLOUT_PRESS and clutch:
                runtime.start_trial()
                policy.reset()
                actions_remaining = 0
                torch.manual_seed(config.seed)
                phase = DeploymentPhase.ROLLOUT

            if phase is not DeploymentPhase.ROLLOUT:
                continue

            state = state_from_joints(joints)
            violations = state_bound_violations(
                state,
                policy.normalizer.state_low,
                policy.normalizer.state_high,
                margin_fraction=config.state_margin_fraction,
            )
            if violations:
                raise RuntimeError(
                    f"state left training bounds in dimensions {violations}"
                )

            inference_started = time.monotonic()
            replanning = actions_remaining == 0
            if replanning:
                # A synchronous diffusion sample can take longer than one 30 Hz
                # action period. Hold zero while sampling rather than extending
                # the previous velocity command for an unintended duration.
                runtime.set_action(np.zeros(7, dtype=np.float32))
            raw_action = policy.act({"camera_rgb": image, "state": state})
            inference_latency = time.monotonic() - inference_started
            if replanning:
                actions_remaining = policy.config.action_horizon
            actions_remaining -= 1
            safe_action = guard_policy_action(
                raw_action,
                scale=config.action_scale,
                allow_grasp=config.allow_grasp,
                limits=limits,
            )
            runtime.set_action(safe_action)
            log.write(
                "policy",
                sequence=sequence,
                frame_age_s=frame_age,
                inference_latency_s=inference_latency,
                replanning=replanning,
                state=state.tolist(),
                raw_action=np.asarray(raw_action).tolist(),
                safe_action=safe_action.tolist(),
            )
            now = time.monotonic()
            if now - last_status >= 0.25:
                print(
                    f"\r[RUN] t={now - runtime.started_at:5.2f}s "
                    f"frame={sequence} infer={inference_latency * 1000:6.1f}ms "
                    f"action={np.round(safe_action, 3)}   ",
                    end="",
                    flush=True,
                )
                last_status = now
    except KeyboardInterrupt:
        runtime.finish("keyboard_interrupt")
    except BaseException as exc:
        runtime.fail("runner", exc)
    finally:
        runtime.stop.set()
        if control is not None and control.is_alive():
            control.join(timeout=8.0)
            if control.is_alive():
                runtime.fail("shutdown", "control worker did not stop")
        if keyboard_thread is not None and keyboard_thread.is_alive():
            keyboard_thread.join(timeout=2.0)
        if camera_pipeline is not None:
            try:
                camera_disconnect(camera_pipeline)
            except BaseException as exc:
                runtime.fail("camera shutdown", exc)
        if keyboard is not None:
            try:
                from giraf.drivers.keyboard import keyboard_disconnect

                keyboard_disconnect(keyboard)
            except BaseException as exc:
                runtime.fail("keyboard shutdown", exc)
        if video is not None:
            try:
                video.close()
            except BaseException as exc:
                runtime.fail("video shutdown", exc)
        for selected_signal, old_handler in old_handlers.items():
            signal.signal(selected_signal, old_handler)
        log.write(
            "finished",
            reason=runtime.stop_reason or "runner_finished",
            error=runtime.error or None,
        )
        log.close()
        print()
        if runtime.error:
            print(f"[DEPLOY][ERROR] {runtime.error}", flush=True)
        else:
            print(
                f"[DEPLOY] Finished: {runtime.stop_reason or 'runner_finished'}",
                flush=True,
            )
        print(f"[DEPLOY] Logs: {log_dir}", flush=True)
    if runtime.error:
        raise RuntimeError(runtime.error)
    return log_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--reference-dataset",
        required=True,
        type=Path,
        help="replay dataset containing the rollout's recorded initial pose",
    )
    parser.add_argument(
        "--reference-episode",
        type=int,
        default=0,
        help="episode whose first pose is used for staging (default: 0)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/tape_grasping.yaml")
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in DeploymentMode],
        default=DeploymentMode.SHADOW.value,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="override checkpoint diffusion inference steps",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="seconds")
    parser.add_argument("--action-timeout", type=float, default=0.5, help="seconds")
    parser.add_argument("--max-frame-age", type=float, default=0.15, help="seconds")
    parser.add_argument("--state-margin", type=float, default=0.05)
    parser.add_argument("--allow-grasp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help=(
            "confirm hardware control and that the robot is physically at the "
            "teleop home pose"
        ),
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> DeploymentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = DeploymentMode(args.mode)
    if mode is DeploymentMode.HARDWARE and not args.confirm_hardware:
        parser.error("--mode hardware requires --confirm-hardware")
    return DeploymentConfig(
        checkpoint=args.checkpoint,
        reference_dataset=args.reference_dataset,
        reference_episode=args.reference_episode,
        collector_config=args.config,
        mode=mode,
        device=args.device,
        action_scale=args.action_scale,
        inference_steps=args.inference_steps,
        duration_s=args.duration,
        action_timeout_s=args.action_timeout,
        max_frame_age_s=args.max_frame_age,
        state_margin_fraction=args.state_margin,
        allow_grasp=args.allow_grasp,
        seed=args.seed,
        log_dir=args.log_dir,
        save_video=not args.no_video,
        hardware_confirmed=args.confirm_hardware,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_config(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[DEPLOY][FATAL] {exc}", flush=True)
        return 1
    return 0
