"""The 100 Hz motor-control worker and OptiTrack receiver loop."""

from __future__ import annotations

import threading
import time

import numpy as np

from giraf.data.schema import ACTION_DIM, GRASP_INDEX
from giraf.settings import CONTROL_HZ
from giraf.teleop_control import POSE_TIMEOUT, RelativePoseController

from .configuration import DeploymentConfig, DeploymentMode
from .safety import SafetyLimits, plan_joint_command, state_from_joints
from .session import ActionSource, Session


class ControlWorker(threading.Thread):
    def __init__(
        self,
        runtime: Session,
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
        self.controller = RelativePoseController()
        self.teleop_generation = -1

    def step(self, events, *, now: float, dt: float):
        """Resolve inputs and produce one command; never run inference here."""

        runtime = self.runtime
        with runtime.lock:
            for key, value in events:
                runtime.input(key, value, now)
            if runtime.stop.is_set():
                return None
            # Invalid shared state is fatal, including while paused.
            state_from_joints(runtime.joints)
            action = np.zeros(ACTION_DIM, dtype=np.float32)
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
                if runtime.action_expired(now, self.config.action_timeout_s):
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

    def run(self) -> None:
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
                    source = runtime.source.value
                    active = runtime.active
                    generation = runtime.generation
                runtime.log.write(
                    "control",
                    generation=generation,
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


def run_optitrack(runtime: Session, config: DeploymentConfig) -> None:
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
                # NatNet receives independently; cap cached-pose polling at 100 Hz.
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
