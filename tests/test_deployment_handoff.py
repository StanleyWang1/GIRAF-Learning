"""Handoff and motor lifecycle regression tests; never open physical devices."""

import json
import signal
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from evdev import ecodes

from giraf.deployment.runner import (
    DeploymentConfig,
    DeploymentMode,
    _ControlWorker,
    _optitrack_loop,
    _PolicyLoop,
    build_parser,
    run,
)
from giraf.deployment.safety import SafetyLimits, state_from_joints
from giraf.deployment.session import ActionSource, Session
from giraf.drivers.keyboard import KeyboardState, keyboard_read, keyboard_session_events
from giraf.teleop_control import RelativePoseController


class FakeDevice:
    def __init__(self, path="keyboard", active=()):
        self.path = path
        self.active = active
        self.events = []

    def active_keys(self):
        return self.active

    def read(self):
        events, self.events = self.events, []
        return events

    def key(self, code, value, timestamp):
        self.events.append(
            SimpleNamespace(
                type=ecodes.EV_KEY, code=code, value=value, timestamp=lambda: timestamp
            )
        )


def config(**kwargs):
    return DeploymentConfig(checkpoint=Path(__file__), **kwargs)


def pose(now, position=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        position_m=position,
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        received_monotonic_ns=int(now * 1e9),
        tracking_valid=True,
    )


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Session(Mock())
        self.runtime.initialize_keys(False)
        self.config = config(mode=DeploymentMode.DRY_RUN)
        self.control = _ControlWorker(
            self.runtime,
            None,
            self.config,
            limits=SafetyLimits(),
            state_low=np.full(15, -100.0),
            state_high=np.full(15, 100.0),
        )
        self.now = time.monotonic()
        self.runtime.pose = pose(self.now)

    def step(self, events=(), elapsed=0):
        return self.control.step(events, now=self.now + elapsed, dt=0.01)

    def policy_on(self):
        self.step([("mode", True), ("clutch", True)])
        self.assertTrue(self.runtime.active)

    def fake_policy(self, act=None):
        return SimpleNamespace(
            reset=Mock(),
            act=act or Mock(return_value=np.array([0.1, 0, 0, 0, 0, 0, 1])),
            config=SimpleNamespace(action_horizon=2),
            normalizer=SimpleNamespace(
                state_low=np.full(15, -100.0), state_high=np.full(15, 100.0)
            ),
        )

    def infer_worker(self, policy):
        return _PolicyLoop(self.runtime, self.config, None, policy, None, None)

    def test_startup_held_and_both_switch_directions_need_fresh_clutch(self):
        self.runtime.initialize_keys(True)
        self.step([("clutch", True)])
        self.assertFalse(self.runtime.active)
        self.step([("clutch", False), ("clutch", True)])
        self.assertTrue(self.runtime.active)
        for source in (ActionSource.POLICY, ActionSource.TELEOP):
            self.step([("mode", True)])
            self.assertEqual(self.runtime.source, source)
            self.assertFalse(self.runtime.active)
            self.step([("clutch", True)])
            self.assertFalse(self.runtime.active)
            self.step([("clutch", False), ("clutch", True)])
            self.assertTrue(self.runtime.active)

    def test_teleop_moves_state_and_reanchors_after_handoff(self):
        self.step([("clutch", True)])
        before = self.runtime.joints.copy()
        self.runtime.pose = pose(self.now, (0.05, 0, 0))
        self.step()
        self.assertGreater(np.linalg.norm(self.runtime.joints - before), 0)
        self.step([("mode", True)])
        np.testing.assert_allclose(
            state_from_joints(self.runtime.joints)[:6], self.runtime.joints
        )
        self.step([("mode", True), ("clutch", False)])
        self.runtime.pose = pose(self.now, (1, 2, 3))
        command = self.step([("clutch", True)])
        np.testing.assert_allclose(command.action[:6], 0, atol=1e-7)

    def test_pause_preserves_pose_and_grasp_and_b_is_source_specific(self):
        self.step([("grasp", True), ("clutch", True)])
        self.assertTrue(self.runtime.grasp)
        before = self.runtime.joints.copy()
        command = self.step([("clutch", False), ("mode", True), ("grasp", True)])
        self.assertTrue(command.grasp)
        self.assertFalse(self.runtime.stop.is_set())
        np.testing.assert_allclose(self.runtime.joints, before)
        np.testing.assert_allclose(command.action[:6], 0)
        self.step([("mode", True), ("grasp", True)])
        self.assertFalse(self.runtime.grasp)

    def test_bounds_pause_and_teleop_recovery(self):
        self.control.state_low = np.full(15, 100.0)
        self.control.state_high = np.full(15, 101.0)
        self.step([("mode", True), ("clutch", True)])
        self.assertFalse(self.runtime.active)
        self.assertIn("training bounds", self.runtime.reason)
        self.assertFalse(self.runtime.stop.is_set())
        self.step([("clutch", True)])
        self.assertFalse(self.runtime.active)
        self.step([("mode", True), ("clutch", False), ("clutch", True)])
        self.assertTrue(self.runtime.active)

    def test_stale_pose_requires_release_after_recovery(self):
        self.step([("clutch", True)])
        self.step(elapsed=1)
        self.assertFalse(self.runtime.active)
        self.runtime.pose = pose(self.now + 1)
        self.step(elapsed=1)
        self.assertFalse(self.runtime.active)
        self.step([("clutch", False), ("clutch", True)], elapsed=1)
        self.assertTrue(self.runtime.active)

    def test_stale_action_pauses_and_late_publish_is_discarded(self):
        self.policy_on()
        generation = self.runtime.generation
        self.step(elapsed=1)
        self.assertFalse(self.runtime.active)
        self.assertFalse(self.runtime.stop.is_set())
        self.assertFalse(
            self.runtime.publish(generation, np.ones(7), self.now + 1, allow_grasp=True)
        )

    def test_no_duration_limit_with_fresh_actions(self):
        self.policy_on()
        self.runtime.hold_for_replan(self.runtime.generation, self.now + 3600)
        self.step(elapsed=3600)
        self.assertTrue(self.runtime.active)
        self.assertFalse(self.runtime.stop.is_set())

    def test_replan_holds_arm_and_keeps_grasp(self):
        self.step([("grasp", True)])
        self.policy_on()

        def act(observation):
            np.testing.assert_array_equal(self.runtime.action[:6], 0)
            self.assertEqual(self.runtime.action[6], 1)
            np.testing.assert_allclose(
                observation["state"], state_from_joints(self.runtime.joints)
            )
            return np.zeros(7)

        worker = self.infer_worker(self.fake_policy(act))
        worker.infer(np.zeros((8, 8, 3), dtype=np.uint8), 1, 0.01)
        self.assertEqual(self.runtime.action[6], 1)  # --allow-grasp off

    def test_allow_grasp_policy_result_applied_by_control(self):
        self.policy_on()
        worker = self.infer_worker(self.fake_policy())
        worker.config = replace(self.config, allow_grasp=True)
        worker.infer(np.zeros((8, 8, 3), dtype=np.uint8), 1, 0.01)
        self.step()
        self.assertTrue(self.runtime.grasp)
        self.step([("mode", True)])
        self.assertTrue(self.runtime.grasp)

    def test_policy_resume_resets_once_per_activation(self):
        self.policy_on()
        policy = self.fake_policy()
        worker = self.infer_worker(policy)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        worker.infer(image, 1, 0.01)
        worker.infer(image, 2, 0.01)
        self.assertEqual(policy.reset.call_count, 1)
        self.step([("clutch", False), ("clutch", True)])
        worker.infer(image, 3, 0.01)
        self.assertEqual(policy.reset.call_count, 2)

    def test_invalid_action_and_stale_camera_pause_without_shutdown(self):
        self.policy_on()
        worker = self.infer_worker(
            self.fake_policy(Mock(return_value=np.full(7, np.nan)))
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        worker.infer(image, 1, 0.01)
        self.assertFalse(self.runtime.active)
        self.assertFalse(self.runtime.stop.is_set())
        self.step([("clutch", False), ("clutch", True)])
        worker.infer(image, 2, 1.0)
        self.assertFalse(self.runtime.active)
        self.assertIn("camera", self.runtime.reason)

    def test_blocked_inference_cannot_undo_handoff_or_shutdown(self):
        self.policy_on()
        entered, release = threading.Event(), threading.Event()

        def act(_observation):
            entered.set()
            release.wait(2)
            return np.ones(7)

        worker = self.infer_worker(self.fake_policy(act))
        thread = threading.Thread(
            target=worker.infer, args=(np.zeros((8, 8, 3), dtype=np.uint8), 1, 0.01)
        )
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            command = self.step([("mode", True)])
            np.testing.assert_array_equal(command.action[:6], 0)
            self.assertFalse(self.runtime.active)
            self.assertIsNone(self.step([("quit", True)]))
            self.assertTrue(self.runtime.stop.is_set())
        finally:
            release.set()
            thread.join(2)
        np.testing.assert_array_equal(self.runtime.action[:6], 0)

    def test_failed_inference_from_old_generation_cannot_pause_new_activation(self):
        self.policy_on()

        def act(_):
            self.step([("mode", True), ("clutch", False), ("clutch", True)])
            raise ValueError("late error")

        self.infer_worker(self.fake_policy(act)).infer(np.zeros((8, 8, 3)), 1, 0.01)
        self.assertTrue(self.runtime.active)
        self.assertEqual(self.runtime.source, ActionSource.TELEOP)

    def test_timed_out_prediction_retains_inference_timing(self):
        self.policy_on()

        def act(_):
            self.step(elapsed=1)  # watchdog pauses before the result arrives
            return np.zeros(7)

        self.infer_worker(self.fake_policy(act)).infer(np.zeros((8, 8, 3)), 1, 0.01)
        records = self.runtime.log.write.call_args_list
        finished = [
            call.kwargs for call in records if call.args[0] == "inference_finished"
        ]
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0]["activation_current"])
        self.assertGreaterEqual(finished[0]["inference_latency_s"], 0)
        self.assertFalse(any(call.args[0] == "policy" for call in records))

    def test_nonfinite_shared_state_is_fatal(self):
        self.runtime.joints[0] = np.nan
        with self.assertRaises(ValueError):
            self.step()

    def test_cli_no_longer_requires_reference_or_duration(self):
        args = build_parser().parse_args(["--checkpoint", "test.pt"])
        self.assertFalse(hasattr(args, "reference_dataset"))
        self.assertFalse(hasattr(args, "duration"))
        self.assertEqual(args.rigid_id, 40)


class OptiTrackPollingTests(unittest.TestCase):
    def test_policy_skips_pose_polling_and_teleop_resumes_on_same_connection(self):
        runtime = Session(Mock())
        runtime.source = ActionSource.POLICY
        driver = Mock()
        sample = pose(time.monotonic())
        driver.get_latest_pose.return_value = sample
        waits = []

        def wait(timeout):
            waits.append(timeout)
            if len(waits) == 1:
                driver.get_latest_pose.assert_not_called()
                runtime.source = ActionSource.TELEOP
            else:
                runtime.stop.set()
            return runtime.stop.is_set()

        with (
            patch("giraf.drivers.optitrack.OptiTrackDriver", return_value=driver),
            patch.object(runtime.stop, "wait", side_effect=wait),
        ):
            _optitrack_loop(runtime, config())
        driver.connect.assert_called_once()
        driver.get_latest_pose.assert_called_once()
        self.assertIs(runtime.pose, sample)
        driver.close.assert_called_once()

    def test_cached_pose_yields_instead_of_spinning(self):
        runtime = Session(Mock())
        driver = Mock()
        sample = pose(time.monotonic())
        waits = []

        def cached_pose(**_kwargs):
            # Bound the test even if polling regresses to a tight loop.
            if driver.get_latest_pose.call_count >= 10:
                runtime.stop.set()
            return sample

        def wait(timeout):
            waits.append(timeout)
            runtime.stop.set()
            return True

        driver.get_latest_pose.side_effect = cached_pose
        with (
            patch("giraf.drivers.optitrack.OptiTrackDriver", return_value=driver),
            patch.object(runtime.stop, "wait", side_effect=wait),
        ):
            _optitrack_loop(runtime, config())
        driver.get_latest_pose.assert_called_once()
        self.assertIs(runtime.pose, sample)
        self.assertGreater(waits[0], 0)
        self.assertLessEqual(waits[0], 0.01)
        driver.close.assert_called_once()


class KeyboardTests(unittest.TestCase):
    def test_ordered_quick_switch_release_press_and_no_repeat(self):
        device = FakeDevice(active=(ecodes.KEY_SPACE,))
        keys = KeyboardState([device], session_events=True)
        self.assertTrue(keys.clutch)
        for code, value, stamp in [
            (ecodes.KEY_D, 1, 1),
            (ecodes.KEY_D, 2, 2),
            (ecodes.KEY_SPACE, 0, 3),
            (ecodes.KEY_SPACE, 1, 4),
        ]:
            device.key(code, value, stamp)
        keyboard_read(keys)
        events, status = keyboard_session_events(keys)
        self.assertEqual(events, [("mode", True), ("clutch", False), ("clutch", True)])
        self.assertTrue(status["clutch"])
        self.assertEqual(keyboard_session_events(keys)[0], [])

    def test_multiple_devices_keep_aggregate_clutch_held(self):
        first, second = FakeDevice("a", (ecodes.KEY_SPACE,)), FakeDevice("b")
        keys = KeyboardState([first, second], session_events=True)
        second.key(ecodes.KEY_SPACE, 1, 1)
        first.key(ecodes.KEY_SPACE, 0, 2)
        keyboard_read(keys)
        self.assertEqual(keyboard_session_events(keys)[0], [])
        self.assertTrue(keys.clutch)
        second.key(ecodes.KEY_SPACE, 0, 3)
        keyboard_read(keys)
        self.assertEqual(keyboard_session_events(keys)[0], [("clutch", False)])

    def test_b_r_compatibility_and_q_latches(self):
        device = FakeDevice()
        keys = KeyboardState([device])
        for stamp, code in enumerate((ecodes.KEY_B, ecodes.KEY_R, ecodes.KEY_Q)):
            device.key(code, 1, stamp)
            device.key(code, 2, stamp + 0.1)
        status = keyboard_read(keys)
        self.assertTrue(status["grasp"])
        self.assertEqual(status["record_toggle_count"], 1)
        self.assertTrue(status["quit_requested"])
        self.assertIsNone(keys._events)


class MotorLifecycleTests(unittest.TestCase):
    def test_pause_keeps_sending_targets_then_q_disconnects_once(self):
        runtime = Session(Mock())
        now = time.monotonic()
        runtime.pose = pose(now)
        cfg = config(mode=DeploymentMode.HARDWARE, hardware_confirmed=True)
        worker = _ControlWorker(
            runtime,
            None,
            cfg,
            limits=SafetyLimits(),
            state_low=np.full(15, -100),
            state_high=np.full(15, 100),
        )
        snapshots = iter(
            [
                ([], {"clutch": False, "quit_requested": False}),
                ([("grasp", True), ("clutch", True)], {"clutch": True}),
                ([("clutch", False)], {"clutch": False}),
                ([("mode", True)], {"clutch": False}),
                ([("quit", True)], {"clutch": False}),
            ]
        )
        dxl, sync, mab = Mock(), Mock(), Mock()
        dxl.WRITE.return_value = True
        with (
            patch(
                "giraf.drivers.keyboard.keyboard_session_events",
                side_effect=lambda _: next(snapshots),
            ),
            patch(
                "giraf.drivers.dynamixel.dynamixel_connect", return_value=(dxl, sync)
            ),
            patch(
                "giraf.drivers.dynamixel.dynamixel_drive", return_value=True
            ) as drive,
            patch("giraf.drivers.dynamixel.dynamixel_disconnect") as disconnect,
            patch("giraf.drivers.mab_worker.MabWorker", return_value=mab),
        ):
            worker.run()
        self.assertEqual(runtime.stop_reason, "quit_key")
        self.assertEqual(runtime.error, "")
        self.assertEqual(drive.call_count, 3)
        self.assertEqual(mab.command.call_count, 3)
        self.assertTrue(runtime.grasp)
        mab.start.assert_called_once()
        mab.stop.assert_called_once()
        disconnect.assert_called_once_with(dxl)
        dxl.close_port.assert_called_once()


class SessionIntegrationTests(unittest.TestCase):
    def test_q_and_sigint_stop_motors_while_inference_is_blocked(self):
        for quit_method in ("q", "sigint"):
            with (
                self.subTest(quit_method=quit_method),
                tempfile.TemporaryDirectory() as tmp,
            ):
                started, predicting, disconnected, release = (
                    threading.Event() for _ in range(4)
                )
                actor_errors = []
                device = FakeDevice()
                keyboard = KeyboardState([device], session_events=True)
                dxl, sync, mab, pipeline, queue = (Mock() for _ in range(5))
                dxl.WRITE.return_value = True
                mab.command.side_effect = lambda *_: started.set()

                def act(_observation):
                    self.assertIs(threading.current_thread(), threading.main_thread())
                    predicting.set()
                    if not release.wait(5):
                        raise RuntimeError("test timed out waiting for shutdown")
                    return np.ones(7)

                policy = SimpleNamespace(
                    reset=Mock(),
                    act=act,
                    config=SimpleNamespace(
                        action_space="twist", action_horizon=8, inference_steps=16
                    ),
                    normalizer=SimpleNamespace(
                        state_low=np.full(15, -100.0), state_high=np.full(15, 100.0)
                    ),
                )

                def keyboard_actor(_keyboard, stop):
                    try:
                        self.assertTrue(started.wait(3))
                        device.key(ecodes.KEY_D, 1, 1)
                        device.key(ecodes.KEY_SPACE, 1, 2)
                        keyboard_read(keyboard)
                        self.assertTrue(predicting.wait(3))
                        if quit_method == "q":
                            device.key(ecodes.KEY_Q, 1, 3)
                            keyboard_read(keyboard)
                        else:
                            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                        self.assertTrue(disconnected.wait(1))
                        self.assertFalse(release.is_set())
                    except BaseException as exc:
                        actor_errors.append(exc)
                        stop.set()
                    finally:
                        release.set()
                    stop.wait(1)

                def frame(*_args):
                    time.sleep(1 / 30)
                    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
                    return rgb, rgb, 1, 0.001

                def optitrack(runtime, _config):
                    runtime.stop.wait()

                camera_module = SimpleNamespace(
                    camera_connect=Mock(return_value=(pipeline, queue)),
                    camera_disconnect=Mock(),
                )
                cfg = config(
                    mode=DeploymentMode.HARDWARE,
                    hardware_confirmed=True,
                    save_video=False,
                    log_dir=Path(tmp) / "session",
                )
                old_sigint = signal.getsignal(signal.SIGINT)
                with (
                    patch.dict(sys.modules, {"giraf.drivers.camera": camera_module}),
                    patch(
                        "giraf.deployment.runner.DiffusionPolicy.load",
                        return_value=policy,
                    ),
                    patch("giraf.deployment.runner._read_frame", side_effect=frame),
                    patch(
                        "giraf.deployment.runner._optitrack_loop", side_effect=optitrack
                    ),
                    patch(
                        "giraf.drivers.keyboard.keyboard_connect", return_value=keyboard
                    ),
                    patch(
                        "giraf.drivers.keyboard.keyboard_control",
                        side_effect=keyboard_actor,
                    ),
                    patch(
                        "giraf.drivers.keyboard.keyboard_disconnect"
                    ) as close_keyboard,
                    patch(
                        "giraf.drivers.dynamixel.dynamixel_connect",
                        return_value=(dxl, sync),
                    ),
                    patch("giraf.drivers.dynamixel.dynamixel_drive", return_value=True),
                    patch(
                        "giraf.drivers.dynamixel.dynamixel_disconnect",
                        side_effect=lambda _: disconnected.set(),
                    ),
                    patch("giraf.drivers.mab_worker.MabWorker", return_value=mab),
                ):
                    output = run(cfg)
                self.assertEqual(actor_errors, [])
                self.assertIs(signal.getsignal(signal.SIGINT), old_sigint)
                mab.start.assert_called_once()
                mab.stop.assert_called_once()
                dxl.close_port.assert_called_once()
                close_keyboard.assert_called_once_with(keyboard)
                camera_module.camera_disconnect.assert_called_once_with(pipeline)
                records = [
                    json.loads(line)
                    for line in (output / "events.jsonl").read_text().splitlines()
                ]
                self.assertEqual(records[-1]["event"], "finished")
                self.assertIsNone(records[-1]["error"])
                self.assertFalse(any(row["event"] == "policy" for row in records))

    def test_motor_failure_is_fatal_and_cleans_up_both_drivers(self):
        runtime = Session(Mock())
        cfg = config(mode=DeploymentMode.HARDWARE, hardware_confirmed=True)
        worker = _ControlWorker(
            runtime,
            None,
            cfg,
            limits=SafetyLimits(),
            state_low=np.full(15, -100.0),
            state_high=np.full(15, 100.0),
        )
        dxl, sync, mab = Mock(), Mock(), Mock()
        dxl.WRITE.return_value = True
        with (
            patch(
                "giraf.drivers.keyboard.keyboard_session_events",
                return_value=([], {"clutch": False, "quit_requested": False}),
            ),
            patch(
                "giraf.drivers.dynamixel.dynamixel_connect", return_value=(dxl, sync)
            ),
            patch("giraf.drivers.dynamixel.dynamixel_drive", return_value=False),
            patch("giraf.drivers.dynamixel.dynamixel_disconnect") as disconnect,
            patch("giraf.drivers.mab_worker.MabWorker", return_value=mab),
        ):
            worker.run()
        self.assertTrue(runtime.stop.is_set())
        self.assertIn("Dynamixel command failed", runtime.error)
        disconnect.assert_called_once_with(dxl)
        mab.stop.assert_called_once()


class TeleopMathTests(unittest.TestCase):
    def test_original_translation_gain_and_deadband(self):
        controller = RelativePoseController()
        joints = np.array([0.1, 0.2, 0.5, 0.1, 0.1, 0.1])
        origin, quaternion = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        controller.anchor(joints, origin, quaternion)
        np.testing.assert_allclose(
            controller.twist(joints, np.array([0.001, 0, 0]), quaternion), 0, atol=1e-12
        )
        twist = controller.twist(joints, np.array([0.05, 0, 0]), quaternion)
        np.testing.assert_allclose(twist, [0.24, 0, 0, 0, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
