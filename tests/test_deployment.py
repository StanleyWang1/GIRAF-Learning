from __future__ import annotations

import unittest

import numpy as np

from giraf.deployment.safety import (
    SafetyLimits,
    guard_policy_action,
    plan_joint_command,
    plan_staging_command,
    state_bound_violations,
    state_from_joints,
    validate_staging_target,
)


class DeploymentSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.joints = np.array(
            (-0.01, 0.68, 0.91, -1.16, -0.09, 0.0), dtype=np.float32
        )

    def test_guard_scales_twist_and_disables_grasp(self) -> None:
        action = np.array((0.2, -0.4, 0.6, 0.5, -1.5, 2.0, 1.0))
        guarded = guard_policy_action(action, scale=0.2, allow_grasp=False)
        np.testing.assert_allclose(
            guarded, (0.04, -0.08, 0.1, 0.1, -0.2, 0.2, 0.0), atol=1e-7
        )

    def test_guard_independently_clamps_and_binarizes(self) -> None:
        action = np.array((5.0, -5.0, 5.0, 5.0, -5.0, 5.0, 0.7))
        guarded = guard_policy_action(action, scale=1.0, allow_grasp=True)
        np.testing.assert_allclose(
            guarded, (0.5, -0.5, 0.5, 1.0, -1.0, 1.0, 1.0)
        )

    def test_scale_also_reduces_the_hard_velocity_ceiling(self) -> None:
        action = np.array((5.0, -5.0, 5.0, 5.0, -5.0, 5.0, 0.0))
        guarded = guard_policy_action(action, scale=0.2, allow_grasp=False)
        np.testing.assert_allclose(
            guarded, (0.1, -0.1, 0.1, 0.2, -0.2, 0.2, 0.0)
        )

    def test_guard_rejects_nonfinite_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            guard_policy_action(
                np.array((0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0)),
                scale=0.2,
                allow_grasp=False,
            )

    def test_zero_action_holds_joint_target(self) -> None:
        command = plan_joint_command(
            self.joints, np.zeros(7), dt=0.01, limits=SafetyLimits()
        )
        np.testing.assert_allclose(command.joint_position, self.joints, atol=1e-6)
        np.testing.assert_allclose(command.joint_velocity, 0.0, atol=1e-4)
        self.assertFalse(command.grasp)

    def test_joint_velocity_is_limited(self) -> None:
        limits = SafetyLimits()
        guarded = guard_policy_action(
            np.array((10.0, -10.0, 10.0, 10.0, -10.0, 10.0, 0.0)),
            scale=1.0,
            allow_grasp=False,
            limits=limits,
        )
        command = plan_joint_command(self.joints, guarded, dt=0.01, limits=limits)
        self.assertTrue(
            np.all(np.abs(command.joint_velocity) <= np.asarray(limits.joint_speed) + 1e-4)
        )

    def test_staging_reaches_target_at_conservative_speed(self) -> None:
        limits = SafetyLimits()
        current = np.array((0.0, 0.0, 0.31, 0.0, 0.0, 0.0), dtype=np.float32)
        target = validate_staging_target(self.joints, limits=limits)
        for _ in range(2_000):
            command = plan_staging_command(
                current, target, dt=0.01, limits=limits
            )
            self.assertTrue(
                np.all(
                    np.abs(command.joint_velocity)
                    <= np.asarray(limits.staging_joint_speed) + 1e-4
                )
            )
            current = command.joint_position
            if np.allclose(current, target, rtol=0.0, atol=1e-6):
                break
        else:
            self.fail("staging did not reach its target")

    def test_staging_rejects_hardware_limit_violation(self) -> None:
        target = self.joints.copy()
        target[1] = -0.1
        with self.assertRaisesRegex(ValueError, "pitch"):
            validate_staging_target(target)

    def test_state_shape_and_training_bound_violations(self) -> None:
        state = state_from_joints(self.joints)
        self.assertEqual(state.shape, (15,))
        self.assertTrue(np.isfinite(state).all())
        low, high = state - 1.0, state + 1.0
        self.assertEqual(state_bound_violations(state, low, high), ())
        outside = state.copy()
        outside[4] = high[4] + 1.0
        self.assertEqual(state_bound_violations(outside, low, high), (4,))


if __name__ == "__main__":
    unittest.main()
