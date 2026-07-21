"""ROS1 node connecting OptiTrack and a physical deadman to GIRAF task velocity."""

from __future__ import annotations

import json
import math
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import rosgraph
import rospy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .deadman import SpaceDeadmanReader
from .geometry import RelativePoseMapper, task_space_velocity
from .health import JointStateHealthTracker, PublisherMonitor
from .interlock import HoldToRunGate
from .natnet_receiver import OptiTrackReceiver, local_ip_for_server
from .robot_state import RobotStateTracker


DEFAULT_COMMAND_TOPIC = "/giraf_arm/teleop_task_velocity_cmd"
DEFAULT_STATE_TOPIC = "/giraf_arm/state"
DEFAULT_MD80_TOPIC = "/md80/joint_states"
DEFAULT_STATUS_TOPIC = "/giraf_optitrak_teleop/status"


def _three_positive_param(name: str, default: List[float]) -> np.ndarray:
    value = np.asarray(rospy.get_param(name, default), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("%s must contain three finite positive values" % name)
    return value


class GirafOptitrakTeleopNode:
    def __init__(self) -> None:
        if bool(rospy.get_param("/use_sim_time", False)):
            raise ValueError("/use_sim_time must be false for physical teleoperation")
        self.publish_hz = float(rospy.get_param("~publish_hz", 100.0))
        self.status_hz = float(rospy.get_param("~status_hz", 5.0))
        self.position_gain = float(rospy.get_param("~position_gain", 1.0))
        self.rotation_gain = float(rospy.get_param("~rotation_gain", 1.0))
        self.linear_limits = _three_positive_param(
            "~linear_velocity_limits", [0.05, 0.05, 0.025]
        )
        self.angular_limits = _three_positive_param(
            "~angular_velocity_limits", [0.125, 0.125, 0.125]
        )
        self.position_scale = float(rospy.get_param("~position_scale", 1.0))
        self.orientation_scale = float(rospy.get_param("~orientation_scale", 1.0))
        self.opti_timeout_ms = float(rospy.get_param("~opti_timeout_ms", 100.0))
        self.state_timeout_ms = float(rospy.get_param("~state_timeout_ms", 300.0))
        self.md80_timeout_ms = float(rospy.get_param("~md80_timeout_ms", 300.0))
        self.shutdown_zero_duration_sec = float(
            rospy.get_param("~shutdown_zero_duration_sec", 0.25)
        )
        scalar_parameters = (
            self.publish_hz,
            self.status_hz,
            self.position_gain,
            self.rotation_gain,
            self.position_scale,
            self.orientation_scale,
            self.opti_timeout_ms,
            self.state_timeout_ms,
            self.md80_timeout_ms,
            self.shutdown_zero_duration_sec,
        )
        if not all(math.isfinite(value) for value in scalar_parameters):
            raise ValueError("teleop scalar parameters must be finite")
        if self.publish_hz <= 0.0 or self.status_hz <= 0.0:
            raise ValueError("publish and status rates must be positive")
        if self.position_gain <= 0.0 or self.rotation_gain <= 0.0:
            raise ValueError("pose-error gains must be positive")
        if self.position_scale <= 0.0 or self.orientation_scale <= 0.0:
            raise ValueError("pose scales must be positive")
        if (
            min(self.opti_timeout_ms, self.state_timeout_ms, self.md80_timeout_ms)
            <= 0.0
        ):
            raise ValueError("input timeouts must be positive")
        if self.shutdown_zero_duration_sec < 0.0:
            raise ValueError("shutdown zero duration cannot be negative")

        self.command_topic = str(
            rospy.get_param("~command_topic", DEFAULT_COMMAND_TOPIC)
        )
        self.state_topic = str(rospy.get_param("~state_topic", DEFAULT_STATE_TOPIC))
        self.md80_topic = str(rospy.get_param("~md80_topic", DEFAULT_MD80_TOPIC))
        self.status_topic = str(rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC))
        expected_md80_names = rospy.get_param(
            "~expected_md80_joint_names", ["Joint 11", "Joint 12", "Joint 13"]
        )
        if not isinstance(expected_md80_names, list) or not expected_md80_names:
            raise ValueError("expected_md80_joint_names must be a non-empty list")

        server_ip = str(rospy.get_param("~optitrack_server_ip", "172.24.68.77"))
        command_port = int(rospy.get_param("~optitrack_command_port", 1510))
        configured_client_ip = str(rospy.get_param("~optitrack_client_ip", "")).strip()
        client_ip = configured_client_ip or local_ip_for_server(server_ip, command_port)
        rigid_id = int(rospy.get_param("~optitrack_rigid_id", 33))
        data_port = int(rospy.get_param("~optitrack_data_port", 1511))
        multicast = bool(rospy.get_param("~optitrack_multicast", False))
        keyboard_path = str(
            rospy.get_param("~keyboard_device", "/dev/input/deadman_keyboard")
        )

        self.robot_states = RobotStateTracker()
        self.md80_states = JointStateHealthTracker(expected_md80_names)
        self.mapper = RelativePoseMapper(self.position_scale, self.orientation_scale)
        self.gate = HoldToRunGate()
        self.deadman = SpaceDeadmanReader(keyboard_path)
        self.optitrack = OptiTrackReceiver(
            server_ip=server_ip,
            client_ip=client_ip,
            rigid_id=rigid_id,
            data_port=data_port,
            command_port=command_port,
            use_multicast=multicast,
        )

        self.command_pub = rospy.Publisher(
            self.command_topic, TwistStamped, queue_size=1, tcp_nodelay=True
        )
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=1, latch=True
        )
        self.state_sub = rospy.Subscriber(
            self.state_topic,
            String,
            self._state_callback,
            queue_size=2,
            tcp_nodelay=True,
        )
        self.md80_sub = rospy.Subscriber(
            self.md80_topic,
            JointState,
            self._md80_callback,
            queue_size=2,
            tcp_nodelay=True,
        )
        master = rosgraph.Master(rospy.get_name())
        self.publisher_monitor = PublisherMonitor(
            master,
            rospy.resolve_name(self.command_topic),
            rospy.get_name(),
        )
        self._last_status_sec = -math.inf
        self._stopped = False
        self._last_interlocks: Tuple[str, ...] = ("starting",)
        self._last_ages: Dict[str, Optional[float]] = {}

        rospy.loginfo(
            "Configured OptiTrack rigid body %d at %s via local IP %s",
            rigid_id,
            server_ip,
            client_ip,
        )

    def _state_callback(self, message: String) -> None:
        if not self.robot_states.update(message.data):
            rospy.logwarn_throttle(2.0, "Rejected malformed /giraf_arm/state")

    def _md80_callback(self, message: JointState) -> None:
        if not self.md80_states.update(message.name, message.position):
            rospy.logwarn_throttle(2.0, "Rejected malformed /md80/joint_states")

    @staticmethod
    def _zero_message() -> TwistStamped:
        message = TwistStamped()
        # Deliberately leave stamp at zero: the robot controller substitutes
        # receipt time, avoiding cross-machine clock skew in its 150 ms watchdog.
        return message

    @staticmethod
    def _twist_message(velocity: np.ndarray) -> TwistStamped:
        message = TwistStamped()
        message.twist.linear.x = float(velocity[0])
        message.twist.linear.y = float(velocity[1])
        message.twist.linear.z = float(velocity[2])
        message.twist.angular.x = float(velocity[3])
        message.twist.angular.y = float(velocity[4])
        message.twist.angular.z = float(velocity[5])
        return message

    def _health_snapshot(self, now_ns: int):
        interlocks: List[str] = []
        ages: Dict[str, Optional[float]] = {}

        keyboard = self.deadman.snapshot()
        if not keyboard.device_ok:
            interlocks.append(keyboard.error or "keyboard device is unavailable")

        opti_ok, opti_reason, opti_age, opti_sample = self.optitrack.health(
            now_ns, self.opti_timeout_ms
        )
        ages["optitrack_ms"] = (
            None if not math.isfinite(opti_age) else round(opti_age, 3)
        )
        if not opti_ok:
            interlocks.append(opti_reason)

        state_ok, state_reason, state_age = self.robot_states.health(
            now_ns, self.state_timeout_ms
        )
        ages["robot_state_ms"] = (
            None if not math.isfinite(state_age) else round(state_age, 3)
        )
        if not state_ok:
            interlocks.append(state_reason)

        md80_ok, md80_reason, md80_age = self.md80_states.health(
            now_ns, self.md80_timeout_ms
        )
        ages["md80_state_ms"] = (
            None if not math.isfinite(md80_age) else round(md80_age, 3)
        )
        if not md80_ok:
            interlocks.append(md80_reason)

        publisher_health = self.publisher_monitor.health(now_ns)
        ages["publisher_check_ms"] = (
            None
            if not math.isfinite(publisher_health.age_ms)
            else round(publisher_health.age_ms, 3)
        )
        if not publisher_health.healthy:
            detail = publisher_health.error
            if publisher_health.competitors:
                detail += ": " + ", ".join(publisher_health.competitors)
            interlocks.append(detail)

        return keyboard, opti_sample, tuple(interlocks), ages, publisher_health

    def _publish_status(
        self,
        now_sec: float,
        keyboard,
        interlocks: Tuple[str, ...],
        ages: Dict[str, Optional[float]],
        publisher_health,
        velocity: np.ndarray,
    ) -> None:
        if now_sec - self._last_status_sec < 1.0 / self.status_hz:
            return
        if self.gate.engaged:
            state = "engaged"
        elif interlocks:
            state = "interlocked"
        elif keyboard.held:
            state = "release_then_press_required"
        else:
            state = "space_released"
        payload = {
            "stamp_sec": round(now_sec, 6),
            "state": state,
            "engaged": self.gate.engaged,
            "space_held": keyboard.held,
            "interlocks": list(interlocks),
            "ages": ages,
            "competing_publishers": list(publisher_health.competitors),
            "command": {
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "wx": float(velocity[3]),
                "wy": float(velocity[4]),
                "wz": float(velocity[5]),
            },
        }
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        self._last_status_sec = now_sec

    def run(self) -> int:
        keyboard_description = self.deadman.start()
        rospy.loginfo("Deadman keyboard: %s", keyboard_description)
        optitrack_description = self.optitrack.start()
        rospy.loginfo("OptiTrack connected: %s", optitrack_description)
        self.publisher_monitor.start()

        rate = rospy.Rate(self.publish_hz)
        try:
            while not rospy.is_shutdown():
                now_ns = time.monotonic_ns()
                now_sec = rospy.get_time()
                (
                    keyboard,
                    opti_sample,
                    interlocks,
                    ages,
                    publisher_health,
                ) = self._health_snapshot(now_ns)
                prerequisites_ok = not interlocks
                transition = self.gate.step(keyboard.held, prerequisites_ok)
                if transition.newly_disengaged:
                    self.mapper.disable()
                    rospy.logwarn("Teleoperation disengaged; publishing zero twist")

                if transition.newly_engaged:
                    robot_state = self.robot_states.snapshot()
                    try:
                        if robot_state is None or opti_sample is None:
                            raise RuntimeError("anchor inputs disappeared")
                        self.mapper.enable(opti_sample.pose(), robot_state.pose)
                        rospy.loginfo(
                            "Teleoperation engaged from fresh relative anchors"
                        )
                    except Exception as exc:
                        self.mapper.disable()
                        self.gate.invalidate(keyboard.held)
                        interlocks = interlocks + (
                            "failed to capture anchors: %s" % exc,
                        )

                velocity = np.zeros(6, dtype=float)
                if self.gate.engaged:
                    robot_state = self.robot_states.snapshot()
                    try:
                        if robot_state is None or opti_sample is None:
                            raise RuntimeError("control inputs disappeared")
                        target_pose = self.mapper.update(opti_sample.pose())
                        (
                            velocity,
                            _position_error,
                            _rotation_error,
                        ) = task_space_velocity(
                            target_pose,
                            robot_state.pose,
                            self.position_gain,
                            self.rotation_gain,
                            self.linear_limits,
                            self.angular_limits,
                        )
                    except Exception as exc:
                        rospy.logerr("Control calculation failed: %s", exc)
                        self.mapper.disable()
                        self.gate.invalidate(keyboard.held)
                        interlocks = interlocks + ("control calculation failed",)
                        velocity.fill(0.0)

                self.command_pub.publish(self._twist_message(velocity))
                self._last_interlocks = interlocks
                self._last_ages = ages
                self._publish_status(
                    now_sec,
                    keyboard,
                    interlocks,
                    ages,
                    publisher_health,
                    velocity,
                )
                rate.sleep()
            return 0
        except rospy.ROSInterruptException:
            return 0
        except Exception as exc:
            rospy.logerr(
                "Fatal GIRAF teleop error: %s\n%s", exc, traceback.format_exc()
            )
            return 1
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.mapper.disable()
        self.gate.invalidate(False)
        deadline = time.monotonic() + max(0.0, self.shutdown_zero_duration_sec)
        period = 1.0 / self.publish_hz
        while time.monotonic() < deadline:
            try:
                self.command_pub.publish(self._zero_message())
            except Exception:
                break
            time.sleep(period)
        for stop_name, stop_action in (
            ("publisher monitor", self.publisher_monitor.stop),
            ("OptiTrack", self.optitrack.stop),
            ("deadman keyboard", self.deadman.stop),
        ):
            try:
                stop_action()
            except Exception as exc:
                rospy.logwarn("Failed to stop %s cleanly: %s", stop_name, exc)


def main() -> int:
    rospy.init_node("giraf_optitrak_teleop", anonymous=False, disable_signals=False)
    node: Optional[GirafOptitrakTeleopNode] = None
    try:
        node = GirafOptitrakTeleopNode()
        return node.run()
    except Exception as exc:
        rospy.logfatal(
            "Unable to start GIRAF OptiTrack teleop: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        if node is not None:
            node.stop()
        return 1
