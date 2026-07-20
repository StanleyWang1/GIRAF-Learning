"""Stateful MuJoCo simulation backend for GIRAF scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scenes import SceneInput, resolve_scene


ROBOT_JOINTS = ("R1", "R2", "P3", "R4", "R5", "R6")
GRIPPER_JOINTS = ("left_grip_joint", "right_grip_joint")
ROBOT_ACTUATORS = (
    "actuator_R1",
    "actuator_R2",
    "actuator_P3",
    "actuator_R4",
    "actuator_R5",
    "actuator_R6",
    "actuator_left_grip",
    "actuator_right_grip",
)

DEFAULT_JOINT_POSITIONS = {
    "R1": 0.0,
    "R2": 0.0,
    "P3": 0.25,
    "R4": 0.0,
    "R5": 0.0,
    "R6": 0.0,
    "left_grip_joint": 0.0,
    "right_grip_joint": 0.0,
}

_ACTUATOR_TO_JOINT = dict(zip(ROBOT_ACTUATORS, DEFAULT_JOINT_POSITIONS))


@dataclass(frozen=True)
class SimulationState:
    """Copy of the dynamic MuJoCo state at one instant."""

    time: float
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    control: NDArray[np.float64]


@dataclass(frozen=True)
class BodyPose:
    """World-frame pose using a MuJoCo-order ``wxyz`` quaternion."""

    position: NDArray[np.float64]
    quaternion: NDArray[np.float64]


class GirafSimulation:
    """Own one independent MuJoCo model/data pair.

    The class contains no controller, policy, viewer, or wall-clock timing. Each
    instance can therefore be stepped headlessly or observed by optional tools.
    """

    def __init__(
        self,
        scene: SceneInput = "arm",
        *,
        frame_skip: int = 1,
    ) -> None:
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        self.scene = resolve_scene(scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene.path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self._renderer: mujoco.Renderer | None = None
        self._render_size: tuple[int, int] | None = None
        self.reset()

    @property
    def physics_dt(self) -> float:
        """Duration of one MuJoCo physics step in seconds."""

        return float(self.model.opt.timestep)

    @property
    def step_dt(self) -> float:
        """Simulated time advanced by one :meth:`step` call."""

        return self.physics_dt * self.frame_skip

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Names of all fixed or tracking cameras in the loaded scene."""

        return tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(self.model.ncam)
        )

    def reset(
        self,
        *,
        joint_positions: Mapping[str, float] | None = None,
        control: ArrayLike | None = None,
    ) -> SimulationState:
        """Reset physics and return a copy of the initialized state."""

        mujoco.mj_resetData(self.model, self.data)
        positions = dict(DEFAULT_JOINT_POSITIONS)
        if joint_positions is not None:
            positions.update(joint_positions)

        for name, value in positions.items():
            self.set_joint_position(name, value)

        if control is None:
            for actuator_name, joint_name in _ACTUATOR_TO_JOINT.items():
                actuator_id = self.actuator_id(actuator_name)
                self.data.ctrl[actuator_id] = positions[joint_name]
        else:
            self.set_control(control)

        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def step(
        self,
        control: ArrayLike | None = None,
        *,
        n_steps: int | None = None,
    ) -> SimulationState:
        """Apply optional actuator controls and advance simulation physics."""

        if control is not None:
            self.set_control(control)
        steps = self.frame_skip if n_steps is None else n_steps
        if steps < 1:
            raise ValueError("n_steps must be at least 1")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def state(self) -> SimulationState:
        """Return an immutable snapshot whose arrays do not alias MuJoCo data."""

        return SimulationState(
            time=float(self.data.time),
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            control=self.data.ctrl.copy(),
        )

    def set_control(self, control: ArrayLike) -> None:
        """Set the complete MuJoCo actuator control vector."""

        values = np.asarray(control, dtype=np.float64)
        if values.shape != (self.model.nu,):
            raise ValueError(
                f"control must have shape ({self.model.nu},), got {values.shape}"
            )
        self.data.ctrl[:] = values

    def set_actuator_target(self, name: str, value: float) -> None:
        """Set one named actuator target without advancing physics."""

        self.data.ctrl[self.actuator_id(name)] = value

    def actuator_id(self, name: str) -> int:
        """Return a named actuator ID or raise a descriptive error."""

        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if actuator_id < 0:
            raise KeyError(f"Unknown actuator: {name}")
        return actuator_id

    def joint_position(self, name: str) -> float:
        """Read a named one-degree-of-freedom joint position."""

        return float(self.data.qpos[self._scalar_joint_qpos_address(name)])

    def joint_positions(
        self, names: Sequence[str] = ROBOT_JOINTS + GRIPPER_JOINTS
    ) -> NDArray[np.float64]:
        """Read named scalar joint positions in the requested order."""

        return np.asarray([self.joint_position(name) for name in names])

    def set_joint_position(self, name: str, value: float) -> None:
        """Set a named one-degree-of-freedom joint position directly."""

        self.data.qpos[self._scalar_joint_qpos_address(name)] = value

    def body_pose(self, name: str) -> BodyPose:
        """Return a named body's current world-frame pose."""

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Unknown body: {name}")
        return BodyPose(
            position=self.data.xpos[body_id].copy(),
            quaternion=self.data.xquat[body_id].copy(),
        )

    def render(
        self,
        camera: str | int = "wrist_cam",
        *,
        width: int = 640,
        height: int = 480,
        depth: bool = False,
    ) -> NDArray[np.uint8] | NDArray[np.float32]:
        """Render RGB or metric depth from a named MuJoCo camera."""

        if width < 1 or height < 1:
            raise ValueError("render width and height must be positive")
        render_size = (width, height)
        if self._renderer is None or self._render_size != render_size:
            self._renderer = mujoco.Renderer(
                self.model, height=height, width=width
            )
            self._render_size = render_size

        if depth:
            self._renderer.enable_depth_rendering()
        else:
            self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        """Release optional rendering resources held by this instance."""

        close_renderer = getattr(self._renderer, "close", None)
        if close_renderer is not None:
            close_renderer()
        self._renderer = None
        self._render_size = None

    def __enter__(self) -> GirafSimulation:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _scalar_joint_qpos_address(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise KeyError(f"Unknown joint: {name}")
        joint_type = self.model.jnt_type[joint_id]
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Joint {name!r} is not one degree of freedom")
        return int(self.model.jnt_qposadr[joint_id])
