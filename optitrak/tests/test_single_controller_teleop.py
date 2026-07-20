from __future__ import annotations

import time
import unittest

import numpy as np

from optitrak import single_controller_teleop as teleop


class SingleControllerTeleopTest(unittest.TestCase):
    def setUp(self) -> None:
        now = time.monotonic_ns()
        self.anchor_sample = teleop.OptiSample(
            now,
            1,
            0.0,
            33,
            True,
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.initial_joints = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
        self.initial_pose = teleop.end_effector_pose(self.initial_joints)

    def test_mapper_uses_one_controller_relative_pose_convention(self) -> None:
        angle = 0.3
        current_sample = teleop.OptiSample(
            time.monotonic_ns(),
            2,
            0.01,
            33,
            True,
            1.1,
            1.8,
            3.05,
            0.0,
            0.0,
            np.sin(angle / 2.0),
            np.cos(angle / 2.0),
        )
        mapper = teleop.RelativePoseMapper()
        mapper.enable(self.anchor_sample, self.initial_pose)

        target = mapper.update(current_sample)

        np.testing.assert_allclose(
            target.position - self.initial_pose.position,
            [0.1, -0.2, 0.05],
        )
        relative_rotation = teleop.quaternion_to_matrix(
            np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
        )
        np.testing.assert_allclose(
            target.rotation,
            self.initial_pose.rotation @ relative_rotation,
            atol=1e-12,
        )

    def test_anchor_pose_produces_zero_task_velocity(self) -> None:
        mapper = teleop.RelativePoseMapper()
        mapper.enable(self.anchor_sample, self.initial_pose)

        target = mapper.update(self.anchor_sample)
        twist, position_error, rotation_error = teleop.task_space_velocity(
            target, self.initial_pose
        )

        np.testing.assert_allclose(twist, np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(position_error, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(rotation_error, np.zeros(3), atol=1e-12)

    def test_kinematic_model_applies_required_joint_offsets(self) -> None:
        joint_positions = np.array([0.1, -0.2, 0.35, 0.4, -0.5, 0.6])

        model_coordinates = teleop.kinematic_joint_coordinates(joint_positions)

        expected = joint_positions.copy()
        expected[1] += np.pi / 2.0
        expected[3] += np.pi / 2.0
        expected[4] -= np.pi / 2.0
        np.testing.assert_allclose(model_coordinates, expected)

    def test_pseudoinverse_reconstructs_reachable_twist(self) -> None:
        jacobian = np.asarray(
            teleop.num_jacobian(
                teleop.kinematic_joint_coordinates(self.initial_joints)
            ),
            dtype=float,
        )
        twist = jacobian @ np.array([0.02, -0.03, 0.01, 0.05, -0.04, 0.03])

        joint_velocity, computed_jacobian = teleop.joint_velocity_from_twist(
            self.initial_joints, twist
        )

        np.testing.assert_allclose(computed_jacobian, jacobian)
        np.testing.assert_allclose(
            computed_jacobian @ joint_velocity, twist, atol=1e-10
        )

    def test_orientation_only_twist_preserves_zero_linear_velocity(self) -> None:
        twist = np.array([0.0, 0.0, 0.0, 0.12, -0.08, 0.05])

        joint_velocity, jacobian = teleop.joint_velocity_from_twist(
            self.initial_joints, twist
        )

        achieved_twist = jacobian @ joint_velocity
        np.testing.assert_allclose(achieved_twist[:3], np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(achieved_twist[3:], twist[3:], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
