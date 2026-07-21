from __future__ import annotations

import json
import math
import unittest

import numpy as np

from control.RRPRRR_kinematic_model import num_forward_transform
from giraf_optitrak_teleop.geometry import (
    Pose,
    RelativePoseMapper,
    quaternion_to_matrix,
    task_space_velocity,
)
from giraf_optitrak_teleop.health import (
    JointStateHealthTracker,
    competing_publishers,
)
from giraf_optitrak_teleop.interlock import HoldToRunGate
from giraf_optitrak_teleop.robot_state import (
    get_boom_length_d3,
    get_boom_motor_rad,
    parse_robot_state,
    RobotStateTracker,
)


def state_json(stamp=1.0, boom=None, stop=False, source="teleop"):
    if boom is None:
        # The deployed controller clamps the small positive polynomial residue
        # at D3_MIN to BOOM_MAX == 0 before publishing state.
        boom = 0.0
    return json.dumps(
        {
            "stamp_sec": stamp,
            "active_source": source,
            "command_source_param": source,
            "stop_latched": stop,
            "arm": {
                "roll": 0.1,
                "pitch": 0.2,
                "boom": boom,
                "th4": 0.3,
                "th5": -0.4,
                "th6": 0.5,
                "grip": 1.0,
            },
        }
    )


class BoomMappingTest(unittest.TestCase):
    def test_round_trip_across_controller_range(self):
        upper = get_boom_length_d3(-30.0)
        for d3 in np.linspace(0.31, upper, 20):
            recovered = get_boom_length_d3(get_boom_motor_rad(float(d3)))
            self.assertAlmostEqual(float(d3), recovered, places=9)


class RobotStateTest(unittest.TestCase):
    def test_state_reconstructs_authoritative_full_transform(self):
        state = parse_robot_state(state_json(), receipt_ns=100)
        expected = np.asarray(
            num_forward_transform(
                [
                    state.roll,
                    state.pitch + np.pi / 2.0,
                    state.d3,
                    state.th4 + np.pi / 2.0,
                    state.th5 - np.pi / 2.0,
                    state.th6,
                ]
            ),
            dtype=float,
        )
        np.testing.assert_allclose(state.pose.position, expected[:3, 3])
        np.testing.assert_allclose(state.pose.rotation, expected[:3, :3], atol=1e-12)

    def test_state_tracker_rejects_latched_state_until_stamp_progresses(self):
        tracker = RobotStateTracker()
        tracker.update(state_json(stamp=1.0), receipt_ns=1_000_000_000)
        healthy, reason, _age = tracker.health(1_010_000_000, 300.0)
        self.assertFalse(healthy)
        self.assertIn("second", reason)

        tracker.update(state_json(stamp=1.1), receipt_ns=1_020_000_000)
        healthy, reason, _age = tracker.health(1_030_000_000, 300.0)
        self.assertTrue(healthy, reason)

    def test_state_tracker_rejects_stop_and_wrong_source(self):
        for encoded, expected in (
            (state_json(stamp=2.0, stop=True), "stop"),
            (state_json(stamp=2.0, source="auto"), "teleop"),
        ):
            tracker = RobotStateTracker()
            tracker.update(encoded, receipt_ns=1_000_000_000)
            tracker.update(
                encoded.replace('"stamp_sec": 2.0', '"stamp_sec": 2.1'),
                receipt_ns=1_010_000_000,
            )
            healthy, reason, _age = tracker.health(1_020_000_000, 300.0)
            self.assertFalse(healthy)
            self.assertIn(expected, reason)

    def test_invalid_or_stale_state_is_never_healthy(self):
        tracker = RobotStateTracker()
        tracker.update(state_json(stamp=3.0), receipt_ns=1_000_000_000)
        tracker.update(state_json(stamp=3.1), receipt_ns=1_010_000_000)
        self.assertTrue(tracker.health(1_020_000_000, 300.0)[0])
        self.assertFalse(tracker.update("not-json", receipt_ns=1_030_000_000))
        healthy, reason, _age = tracker.health(1_040_000_000, 300.0)
        self.assertFalse(healthy)
        self.assertIn("invalid", reason)

        tracker.update(state_json(stamp=3.2), receipt_ns=2_000_000_000)
        tracker.update(state_json(stamp=3.3), receipt_ns=2_010_000_000)
        healthy, reason, _age = tracker.health(2_400_000_000, 300.0)
        self.assertFalse(healthy)
        self.assertIn("stale", reason)

    def test_non_finite_and_out_of_range_state_is_rejected(self):
        payload = json.loads(state_json())
        for field, value in (
            ("roll", float("nan")),
            ("pitch", -0.2),
            ("th4", 3.0),
            ("th5", -2.0),
            ("th6", 3.5),
        ):
            invalid = json.loads(json.dumps(payload))
            invalid["arm"][field] = value
            with self.assertRaises(ValueError):
                parse_robot_state(json.dumps(invalid))


class RelativePoseControlTest(unittest.TestCase):
    def test_press_anchor_has_zero_error_and_no_extra_control_transform(self):
        robot = Pose(np.array([0.2, -0.1, 0.4]), np.eye(3))
        controller = Pose(np.array([1.0, 2.0, 3.0]), np.eye(3))
        mapper = RelativePoseMapper()
        mapper.enable(controller, robot)
        target = mapper.update(controller)
        np.testing.assert_allclose(target.position, robot.position)
        np.testing.assert_allclose(target.rotation, robot.rotation)

    def test_relative_motion_and_velocity_caps(self):
        mapper = RelativePoseMapper()
        robot = Pose(np.zeros(3), np.eye(3))
        controller = Pose(np.zeros(3), np.eye(3))
        mapper.enable(controller, robot)
        angle = 0.5
        moved = Pose(
            np.array([0.2, -0.1, 0.05]),
            quaternion_to_matrix(
                np.array([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])
            ),
        )
        target = mapper.update(moved)
        velocity, _position_error, _rotation_error = task_space_velocity(
            target,
            robot,
            position_gain=1.0,
            rotation_gain=1.0,
            linear_limits=np.array([0.05, 0.05, 0.025]),
            angular_limits=np.array([0.125, 0.125, 0.125]),
        )
        np.testing.assert_allclose(velocity[:3], [0.05, -0.05, 0.025])
        np.testing.assert_allclose(velocity[3:], [0.0, 0.0, 0.125], atol=1e-12)


class SafetyGateTest(unittest.TestCase):
    def test_hold_to_run_requires_release_and_fresh_press(self):
        gate = HoldToRunGate()
        self.assertFalse(gate.step(True, True).engaged)
        self.assertFalse(gate.step(False, True).engaged)
        self.assertTrue(gate.step(True, True).newly_engaged)
        self.assertTrue(gate.engaged)

        self.assertTrue(gate.step(True, False).newly_disengaged)
        self.assertFalse(gate.step(True, True).engaged)
        self.assertFalse(gate.step(False, True).engaged)
        self.assertTrue(gate.step(True, True).newly_engaged)

    def test_md80_feedback_maps_by_name_and_requires_two_messages(self):
        tracker = JointStateHealthTracker(["Joint 11", "Joint 12", "Joint 13"])
        names = ["Joint 13", "Joint 11", "Joint 12"]
        positions = [0.3, 0.1, 0.2]
        tracker.update(names, positions, receipt_ns=1_000_000_000)
        self.assertFalse(tracker.health(1_010_000_000, 300.0)[0])
        tracker.update(names, positions, receipt_ns=1_020_000_000)
        self.assertTrue(tracker.health(1_030_000_000, 300.0)[0])
        self.assertFalse(tracker.health(1_500_000_000, 300.0)[0])

        self.assertFalse(
            tracker.update(
                ["Joint 11", "Joint 12"],
                [0.1, 0.2],
                receipt_ns=1_510_000_000,
            )
        )
        self.assertFalse(tracker.health(1_520_000_000, 300.0)[0])

    def test_competing_publishers_excludes_only_this_node(self):
        system_publishers = [
            ["/unrelated", ["/some_node"]],
            [
                "/giraf_arm/teleop_task_velocity_cmd",
                ["/teleop_console", "/rogue", "/another_rogue"],
            ],
        ]
        self.assertEqual(
            competing_publishers(
                system_publishers,
                "/giraf_arm/teleop_task_velocity_cmd",
                "/teleop_console",
            ),
            ("/another_rogue", "/rogue"),
        )


if __name__ == "__main__":
    unittest.main()
