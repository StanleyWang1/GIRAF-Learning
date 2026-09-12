"""Persistent teleop / diffusion-policy deployment session."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from giraf.data.config import load_config
from giraf.learning import DiffusionPolicy

from .configuration import DeploymentConfig, DeploymentMode, parse_config
from .control import ControlWorker, run_optitrack
from .policy_loop import PolicyLoop
from .recording import DeploymentRecorder, VideoSettings
from .safety import SafetyLimits
from .session import Session


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
    payload = asdict(config)
    payload["checkpoint"] = str(config.checkpoint)
    payload["collector_config"] = str(config.collector_config)
    payload["mode"] = config.mode.value
    payload["log_dir"] = str(log_dir)
    log = DeploymentRecorder(
        log_dir,
        config=payload,
        video=VideoSettings(
            enabled=config.save_video,
            width=collector_config.camera.width,
            height=collector_config.camera.height,
            fps=collector_config.camera.fps,
        ),
    )
    runtime = Session(log)
    keyboard = camera_pipeline = control = policy_loop = None
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
        if policy.config.imu_input != "none":
            raise ValueError("live deployment does not yet supply IMU observations")
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
        if keyboard_status(keyboard)["quit_requested"]:
            runtime.finish("quit_key")
        if runtime.stop.is_set():
            if runtime.error:
                raise RuntimeError(runtime.error)
            return log_dir

        optitrack = threading.Thread(
            target=guarded_worker,
            args=("OptiTrack worker", run_optitrack, runtime, config),
            name="deployment-optitrack",
            daemon=True,
        )
        optitrack.start()
        workers.append(optitrack)
        policy_loop = PolicyLoop(runtime, config, collector_config, policy, frame_queue)
        control = ControlWorker(
            runtime,
            keyboard,
            config,
            limits=SafetyLimits(),
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

        # Camera/inference and cleanup share this thread. Motor shutdown has
        # already run independently if Q or a guard ended control during sampling.
        if camera_pipeline is not None:
            cleanup("camera shutdown", camera_disconnect, camera_pipeline)
        if keyboard is not None:
            cleanup("keyboard shutdown", keyboard_disconnect, keyboard)
        for selected_signal, old_handler in old_handlers.items():
            signal.signal(selected_signal, old_handler)
        with runtime.lock:
            log.write(
                "finished",
                reason=runtime.stop_reason or "session_finished",
                error=runtime.error or None,
                joints=runtime.joints.tolist(),
                grasp=runtime.grasp,
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


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_config(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[DEPLOY][FATAL] {exc}", flush=True)
        return 1
    return 0
