from __future__ import annotations

import unittest

import numpy as np

from sim_model import GirafSimulation, available_scenes


class GirafSimulationTest(unittest.TestCase):
    def test_default_scene_is_arm_only_giraf_model(self) -> None:
        with GirafSimulation() as sim:
            self.assertEqual(sim.scene.name, "arm")
            self.assertEqual(sim.scene.path.name, "GIRAF.xml")
            self.assertEqual(sim.model.nq, 8)

    def test_base_model_uses_updated_wrist_axes_and_offsets(self) -> None:
        with GirafSimulation() as sim:
            joint_ids = [
                sim.model.joint(name).id for name in ("R4", "R5", "R6")
            ]

            np.testing.assert_allclose(
                sim.model.jnt_axis[joint_ids],
                [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            )
            np.testing.assert_allclose(
                sim.model.jnt_pos[joint_ids],
                [[0.0, 0.0, 0.0], [0.0597, 0.0, 0.0], [0.0597, 0.0, 0.0]],
            )

    def test_all_builtin_scenes_load_and_step(self) -> None:
        for scene in available_scenes():
            with self.subTest(scene=scene.name), GirafSimulation(scene) as sim:
                initial_time = sim.data.time
                state = sim.step(n_steps=2)

                self.assertEqual(sim.model.nu, 8)
                self.assertEqual(sim.camera_names, ("wrist_cam",))
                self.assertAlmostEqual(
                    state.time, initial_time + 2 * sim.physics_dt
                )

    def test_reset_initializes_robot_and_actuator_targets(self) -> None:
        with GirafSimulation("arm") as sim:
            self.assertAlmostEqual(sim.joint_position("P3"), 0.25)
            np.testing.assert_allclose(
                sim.joint_positions(),
                [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            np.testing.assert_allclose(
                sim.state().control,
                [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            )

    def test_state_snapshot_does_not_alias_live_data(self) -> None:
        with GirafSimulation("arm") as sim:
            snapshot = sim.state()
            sim.set_joint_position("R1", 0.5)

            self.assertEqual(snapshot.qpos[0], 0.0)
            self.assertEqual(sim.joint_position("R1"), 0.5)

    def test_frame_skip_controls_step_duration(self) -> None:
        with GirafSimulation("arm", frame_skip=5) as sim:
            state = sim.step()

            self.assertAlmostEqual(sim.step_dt, 0.005)
            self.assertAlmostEqual(state.time, 0.005)

    def test_named_accessors_report_unknown_names(self) -> None:
        with GirafSimulation("arm") as sim:
            with self.assertRaisesRegex(KeyError, "Unknown body"):
                sim.body_pose("missing")
            with self.assertRaisesRegex(KeyError, "Unknown joint"):
                sim.joint_position("missing")
            with self.assertRaisesRegex(KeyError, "Unknown actuator"):
                sim.set_actuator_target("missing", 0.0)

    def test_control_shape_is_validated(self) -> None:
        with GirafSimulation("arm") as sim:
            with self.assertRaisesRegex(ValueError, "control must have shape"):
                sim.step(np.zeros(3))


if __name__ == "__main__":
    unittest.main()
