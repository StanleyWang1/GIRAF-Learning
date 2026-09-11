"""Persistent teleop / diffusion-policy deployment session."""

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
from giraf.data.schema import GRASP_INDEX
from giraf.drivers.optitrack import DEFAULT_RIGID_BODY_ID, DEFAULT_SERVER_IP
from giraf.learning import DiffusionPolicy
from giraf.settings import CONTROL_HZ
from giraf.teleop_control import POSE_TIMEOUT, RelativePoseController

from .safety import (
    SafetyLimits,
    guard_policy_action,
    plan_joint_command,
    state_bound_violations,
    state_from_joints,
)
from .session import ActionSource, Session


class DeploymentMode(str, Enum):
    SHADOW = "shadow"
    DRY_RUN = "dry-run"
    HARDWARE = "hardware"


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    checkpoint: Path
    collector_config: Path = Path("config/tape_grasping.yaml")
    mode: DeploymentMode = DeploymentMode.SHADOW
    device: str = "cuda"
    action_scale: float = 0.2
    inference_steps: int | None = None
    server_ip: str = DEFAULT_SERVER_IP
    client_ip: str | None = None
    rigid_id: int = DEFAULT_RIGID_BODY_ID
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
        if self.inference_steps is not None and self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive when provided")
        if self.mode is DeploymentMode.HARDWARE and not self.hardware_confirmed:
            raise ValueError(
                "hardware mode requires confirmation that the robot is physically "
                "at the teleop home pose"
            )
        positive = (
            self.action_timeout_s,
            self.max_frame_age_s,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("timeout values must be finite and positive")
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
        self._closed = False
        self._file = (directory / "events.jsonl").open("x")
        payload = asdict(config)
        payload["checkpoint"] = str(config.checkpoint)
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
            if self._closed:
                return
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._closed = True
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


class _ControlWorker(threading.Thread):
    def __init__(self, runtime, keyboard, config, *, limits, state_low, state_high):
        super().__init__(name="deployment-control")
        self.runtime = runtime
        self.keyboard = keyboard
        self.config = config
        self.limits = limits
        self.state_low = state_low
        self.state_high = state_high
        self.ready = threading.Event()
        self.controller = RelativePoseController()
        self.teleop_generation = -1

    def step(self, events, *, now, dt):
        """Resolve inputs and produce one command; never run policy inference here."""
        runtime = self.runtime
        with runtime.lock:
            for key, value in events:
                runtime.input(key, value, now)
            if runtime.stop.is_set():
                return None
            # Invalid shared state is fatal, including while paused.
            state = state_from_joints(runtime.joints)
            action = np.zeros(7, dtype=np.float32)
            action[GRASP_INDEX] = runtime.grasp
            if runtime.active and runtime.source is ActionSource.TELEOP:
                pose = runtime.pose
                if (
                    pose is None
                    or not pose.tracking_valid
                    or now - pose.received_monotonic_ns / 1e9 > POSE_TIMEOUT
                ):
                    runtime.pause(runtime.pose_error or "OptiTrack pose stale")
                else:
                    try:
                        position = np.asarray(pose.position_m, dtype=float)
                        quaternion = np.asarray(pose.quaternion_xyzw, dtype=float)
                        if not np.isfinite(position).all() or position.shape != (3,):
                            raise ValueError("invalid controller position")
                        if self.teleop_generation != runtime.generation:
                            self.controller.anchor(runtime.joints, position, quaternion)
                            self.teleop_generation = runtime.generation
                        action[:6] = self.controller.twist(
                            runtime.joints, position, quaternion
                        )
                    except (ValueError, RuntimeError) as exc:
                        runtime.pause(f"teleop: {exc}")
                        action[:6] = 0
            elif runtime.active:
                violations = state_bound_violations(
                    state,
                    self.state_low,
                    self.state_high,
                    margin_fraction=self.config.state_margin_fraction,
                )
                if violations:
                    runtime.pause(f"state outside training bounds: {violations}")
                elif runtime.action_expired(now, self.config.action_timeout_s):
                    runtime.pause("policy action stale")
                elif runtime.action_time > 0:
                    action = runtime.action.copy()

            command = plan_joint_command(
                runtime.joints, action, dt=dt, limits=self.limits
            )
            if runtime.active and self.config.mode is not DeploymentMode.SHADOW:
                runtime.joints = command.joint_position.copy()
            runtime.grasp = command.grasp
            return command

    def run(self):
        from giraf.drivers.keyboard import keyboard_session_events

        runtime = self.runtime
        mab = dxl = sync_write = None
        try:
            if self.config.mode is DeploymentMode.HARDWARE:
                from giraf.drivers.dynamixel import GRIPPER, dynamixel_connect
                from giraf.drivers.dynamixel_config import TORQUE_ENABLE
                from giraf.drivers.mab_worker import MabWorker

                print("[DEPLOY] Connecting motors at physical home...", flush=True)
                dxl, sync_write = dynamixel_connect()
                if not dxl.WRITE(GRIPPER, TORQUE_ENABLE, 1):
                    raise RuntimeError("could not enable gripper torque")
                if not runtime.stop.is_set():
                    mab = MabWorker()
                    mab.start()

            # Discard motion inputs during startup; never discard a quit request.
            _, keys = keyboard_session_events(self.keyboard)
            with runtime.lock:
                runtime.initialize_keys(bool(keys["clutch"]))
                if keys["quit_requested"]:
                    runtime.finish("quit_key")
            self.ready.set()
            if not runtime.stop.is_set():
                print(
                    "[DEPLOY] Controls ready: teleop paused; fresh SPACE press to enable.",
                    flush=True,
                )
            period = 1.0 / CONTROL_HZ
            last = time.monotonic()
            while not runtime.stop.is_set():
                now = time.monotonic()
                dt = min(max(now - last, 1e-6), 0.02)
                last = now
                events, keys = keyboard_session_events(self.keyboard)
                command = self.step(events, now=now, dt=dt)
                if command is None or runtime.stop.is_set():
                    break
                command_accepted = None
                if self.config.mode is DeploymentMode.HARDWARE:
                    from giraf.drivers.dynamixel import dynamixel_drive

                    mab.command(*command.can_position_target)
                    command_accepted = dynamixel_drive(
                        dxl, sync_write, list(command.dynamixel_target_ticks)
                    )
                    if not command_accepted:
                        raise RuntimeError("Dynamixel command failed")
                with runtime.lock:
                    source, active = runtime.source.value, runtime.active
                runtime.log.write(
                    "control",
                    source=source,
                    active=active,
                    clutch=bool(keys["clutch"]),
                    action=command.action.tolist(),
                    joint_velocity=command.joint_velocity.tolist(),
                    joint_position=command.joint_position.tolist(),
                    can_position_target=list(command.can_position_target),
                    dynamixel_target_ticks=list(command.dynamixel_target_ticks),
                    command_accepted=command_accepted,
                )
                runtime.stop.wait(max(0.0, period - (time.monotonic() - now)))
        except BaseException as exc:
            runtime.fail("control", exc)
        finally:
            self.ready.set()
            if dxl is not None:
                try:
                    from giraf.drivers.dynamixel import dynamixel_disconnect

                    try:
                        dynamixel_disconnect(dxl)
                    finally:
                        dxl.close_port()
                except BaseException as exc:
                    runtime.fail("Dynamixel shutdown", exc)
            if mab is not None:
                try:
                    mab.stop()
                except BaseException as exc:
                    runtime.fail("MAB shutdown", exc)


def _optitrack_loop(runtime, config):
    from giraf.drivers.optitrack import OptiTrackDriver

    while not runtime.stop.is_set():
        driver = None
        try:
            driver = OptiTrackDriver(
                config.server_ip, config.rigid_id, client_ip=config.client_ip
            )
            driver.connect()
            while not runtime.stop.is_set():
                with runtime.lock:
                    teleop_selected = runtime.source is ActionSource.TELEOP
                if not teleop_selected:
                    # Keep NatNet connected for handoff, but do not consume poses
                    # or update robot-session state while policy is selected.
                    runtime.stop.wait(1.0 / CONTROL_HZ)
                    continue
                poll_started = time.monotonic()
                try:
                    pose = driver.get_latest_pose(timeout=0.25)
                except TimeoutError:
                    continue
                with runtime.lock:
                    runtime.pose = pose
                    runtime.pose_error = ""
                # get_latest_pose returns its cached sample immediately. Without
                # a wait this spins on Python and runtime.lock alongside inference,
                # even in policy mode. Match consumption to the control frequency;
                # NatNet continues receiving poses independently in the driver.
                runtime.stop.wait(
                    max(0.0, 1.0 / CONTROL_HZ - (time.monotonic() - poll_started))
                )
        except Exception as exc:
            with runtime.lock:
                runtime.pose = None
                runtime.pose_error = f"OptiTrack unavailable: {exc}"
                if runtime.active and runtime.source is ActionSource.TELEOP:
                    runtime.pause(runtime.pose_error)
            runtime.log.write("optitrack_error", detail=str(exc))
        finally:
            if driver is not None:
                driver.close()
        runtime.stop.wait(1.0)


class _PolicyLoop:
    """Run camera-paced, synchronous inference on the main thread, as before handoff."""

    def __init__(self, runtime, config, collector_config, policy, frame_queue, video):
        self.runtime = runtime
        self.config = config
        self.collector_config = collector_config
        self.policy = policy
        self.frame_queue = frame_queue
        self.video = video
        self.generation = -1
        self.actions_remaining = 0

    def infer(self, image, sequence, frame_age):
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
            violations = state_bound_violations(
                state,
                policy.normalizer.state_low,
                policy.normalizer.state_high,
                margin_fraction=config.state_margin_fraction,
            )
            if violations:
                raise ValueError(f"state outside training bounds: {violations}")
            started = time.monotonic()
            replanning = self.actions_remaining == 0
            if replanning:
                with runtime.lock:
                    if not runtime.hold_for_replan(generation, started):
                        return
            if replanning:
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
                # Record slow/failed samples even after the watchdog paused the
                # activation. The policy event below only records accepted actions.
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
                    return  # a late prediction must not undo a pause or handoff
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

    def run(self):
        runtime = self.runtime
        while not runtime.stop.is_set():
            try:
                if runtime.stop.is_set():
                    return
                rgb, image, sequence, frame_age = _read_frame(
                    self.frame_queue, self.collector_config
                )
                if runtime.stop.is_set():
                    return
                if self.video is not None:
                    try:
                        self.video.write(rgb)
                    except Exception as exc:
                        runtime.log.write("video_error", detail=str(exc))
                        self.video = None
            except Exception as exc:
                with runtime.lock:
                    if runtime.active and runtime.source is ActionSource.POLICY:
                        runtime.pause(f"camera: {exc}")
                runtime.stop.wait(0.05)
                continue
            self.infer(image, sequence, frame_age)


def _timestamp_ns(value) -> int:
    return int(round(value.total_seconds() * 1_000_000_000))


def _read_frame(
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


def _default_log_dir() -> Path:
    return Path("deployment_runs") / time.strftime("%Y%m%d-%H%M%S")


def run(config: DeploymentConfig) -> Path:
    """Run a persistent session; connect motors once and retain targets on pause."""
    from giraf.drivers.camera import camera_connect, camera_disconnect
    from giraf.drivers.keyboard import (
        keyboard_connect,
        keyboard_control,
        keyboard_disconnect,
        keyboard_status,
    )

    collector_config = load_config(config.collector_config)
    log_dir = config.log_dir or _default_log_dir()
    log = _EventLog(log_dir, config)
    runtime = Session(log)
    keyboard = camera_pipeline = video = control = policy_loop = None
    workers = []
    old_handlers = {}

    def request_stop(signum, _frame):
        # Signals can interrupt a log write: do not acquire locks here.
        runtime.stop_reason = f"signal_{signum}"
        runtime.stop.set()

    def guarded_worker(source, function, *args):
        try:
            function(*args)
        except BaseException as exc:
            runtime.fail(source, exc)

    try:
        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            old_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, request_stop)
        log.write("startup")
        print(f"[DEPLOY] Loading {config.checkpoint} on {config.device}...", flush=True)
        torch.manual_seed(config.seed)
        policy = DiffusionPolicy.load(config.checkpoint, device=config.device)
        if policy.config.action_space != "twist":
            raise ValueError("deployment requires a twist-action checkpoint")
        if policy.normalizer is None:
            raise ValueError("checkpoint does not contain a training normalizer")
        if config.inference_steps is not None:
            if config.inference_steps > policy.config.diffusion_steps:
                raise ValueError(
                    "inference_steps cannot exceed checkpoint diffusion_steps"
                )
            policy.config = replace(
                policy.config, inference_steps=config.inference_steps
            )
        if runtime.stop.is_set():
            if runtime.error:
                raise RuntimeError(runtime.error)
            return log_dir

        keyboard = keyboard_connect(session_events=True)
        keyboard_thread = threading.Thread(
            target=guarded_worker,
            args=("keyboard", keyboard_control, keyboard, runtime.stop),
            name="deployment-keyboard",
        )
        keyboard_thread.start()
        workers.append(keyboard_thread)
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
        if keyboard_status(keyboard)["quit_requested"]:
            runtime.finish("quit_key")
        if runtime.stop.is_set():
            if runtime.error:
                raise RuntimeError(runtime.error)
            return log_dir

        optitrack = threading.Thread(
            target=guarded_worker,
            args=("OptiTrack worker", _optitrack_loop, runtime, config),
            name="deployment-optitrack",
            daemon=True,
        )
        optitrack.start()
        workers.append(optitrack)
        policy_loop = _PolicyLoop(
            runtime, config, collector_config, policy, frame_queue, video
        )
        control = _ControlWorker(
            runtime,
            keyboard,
            config,
            limits=SafetyLimits(),
            state_low=policy.normalizer.state_low,
            state_high=policy.normalizer.state_high,
        )
        if config.mode is DeploymentMode.HARDWARE:
            print(
                "[HARDWARE] Physical robot must be at teleop home for encoder zero.",
                flush=True,
            )
        control.start()
        workers.append(control)
        startup_deadline = time.monotonic() + 20
        print(
            f"[DEPLOY] backend={config.mode.value} scale={config.action_scale:.2f} "
            f"policy_grasp={config.allow_grasp} inference_steps={policy.config.inference_steps}\n"
            "[DEPLOY] D: teleop/policy | SPACE: hold to run, release to pause | "
            "B: teleop grasp | Q / Ctrl+C: shutdown\n"
            "[DEPLOY] Each mode switch requires a fresh SPACE press. No rollout duration limit.",
            flush=True,
        )

        def supervise():
            last_status = ""
            last_status_time = 0.0
            while not runtime.stop.wait(0.02):
                if keyboard_status(keyboard)["quit_requested"]:
                    with runtime.lock:
                        runtime.finish("quit_key")
                    break
                if not control.ready.is_set() and time.monotonic() > startup_deadline:
                    raise TimeoutError("control worker did not become ready within 20s")
                for worker in workers:
                    if not worker.is_alive() and not runtime.stop.is_set():
                        raise RuntimeError(f"{worker.name} stopped unexpectedly")
                now = time.monotonic()
                if now - last_status_time >= 0.25:
                    status = runtime.status()
                    if status != last_status:
                        print(f"[DEPLOY] {status}", flush=True)
                        last_status = status
                    last_status_time = now

        supervisor = threading.Thread(
            target=guarded_worker,
            args=("supervisor", supervise),
            name="deployment-supervisor",
        )
        supervisor.start()
        workers.append(supervisor)
        # Loading and inference share the main thread, matching the original
        # deployment path. Keyboard/control/supervision still run during sampling.
        policy_loop.run()
    except KeyboardInterrupt:
        runtime.finish("keyboard_interrupt")
    except BaseException as exc:
        runtime.fail("runner", exc)
    finally:
        runtime.stop.set()
        # Motors stop independently of camera/model progress.
        if control is not None and control.is_alive():
            control.join(timeout=20.0)
            if control.is_alive():
                runtime.fail("shutdown", "control worker did not stop")
        for worker in workers:
            if worker is not control and worker.is_alive():
                worker.join(timeout=2.0)
                if worker.is_alive():
                    log.write("worker_shutdown_pending", worker=worker.name)

        def cleanup(source, function, *args):
            try:
                function(*args)
            except BaseException as exc:
                runtime.fail(source, exc)

        def close_camera_logs():
            if video is not None:
                cleanup("video shutdown", video.close)
            if camera_pipeline is not None:
                cleanup("camera shutdown", camera_disconnect, camera_pipeline)

        # Camera/inference and cleanup share this thread. Motor shutdown has
        # already run independently if Q or a guard ended control during sampling.
        close_camera_logs()
        if keyboard is not None:
            cleanup("keyboard shutdown", keyboard_disconnect, keyboard)
        for selected_signal, old_handler in old_handlers.items():
            signal.signal(selected_signal, old_handler)
        log.write(
            "finished",
            reason=runtime.stop_reason or "session_finished",
            error=runtime.error or None,
        )
        log.close()
        print(
            f"[DEPLOY] Finished: {runtime.error or runtime.stop_reason}\n"
            f"[DEPLOY] Logs: {log_dir}",
            flush=True,
        )
    if runtime.error:
        raise RuntimeError(runtime.error)
    return log_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
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
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    parser.add_argument("--client-ip", default=None)
    parser.add_argument("--rigid-id", type=int, default=DEFAULT_RIGID_BODY_ID)
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
        collector_config=args.config,
        mode=mode,
        device=args.device,
        action_scale=args.action_scale,
        inference_steps=args.inference_steps,
        server_ip=args.server_ip,
        client_ip=args.client_ip,
        rigid_id=args.rigid_id,
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
