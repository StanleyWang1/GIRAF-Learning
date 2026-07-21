# GIRAF ROS1 Real-Arm OptiTrack Teleoperation Runbook

## TL;DR

This is the operator-side console for controlling the GIRAF arm through the
robot's existing ROS1 Noetic controller. It receives a tracked controller pose
from OptiTrack, reconstructs the commanded end-effector pose from
`/giraf_arm/state`, and publishes a bounded Cartesian `TwistStamped` command.

The essential operating rules are:

1. Start with the robot's **motor-disabled dry-run backend**, not live motors.
2. Keep Space released while starting the console.
3. Verify `/giraf_optitrak_teleop/status` has no interlocks.
4. **Hold Space to move. Release Space to stop commanding motion.**
5. After any tracking, feedback, keyboard, ROS, or controller fault, release
   Space completely and press it again to re-arm.
6. Do not run another publisher on
   `/giraf_arm/teleop_task_velocity_cmd`.
7. Confirm every Cartesian and rotational direction at low speed with motors
   disabled before enabling hardware.

This console is intentionally not a hardware driver. It never opens CAN or
Dynamixel devices, never publishes direct MD80 commands, never commands the
gripper, and never calls the stop or motor services.

## How to use it

### 1. Prepare the robot side

Start the robot's ROS master and motor-disabled version of
`/giraf_arm_controller`. The dry-run backend should expose the same ROS contract
as the hardware controller while preventing actuator output.

The console requires:

| Interface | Type | Requirement |
| --- | --- | --- |
| `/giraf_arm/state` | `std_msgs/String` | Progressing JSON state, nominally 10 Hz |
| `/md80/joint_states` | `sensor_msgs/JointState` | Joints 11, 12, and 13, nominally 10 Hz |
| `/giraf_arm/teleop_task_velocity_cmd` | `geometry_msgs/TwistStamped` | Robot controller subscribes |
| `/use_sim_time` | ROS parameter | Must be `false` |

The state JSON must report both `active_source` and `command_source_param` as
`teleop`, and `stop_latched` must be false. The console deliberately does not
change these values.

On a machine with the ROS environment sourced, perform read-only checks:

```bash
rosnode list
rostopic info /giraf_arm/teleop_task_velocity_cmd
rostopic echo -n 1 /giraf_arm/state
rostopic hz /giraf_arm/state
rostopic echo -n 1 /md80/joint_states
rostopic hz /md80/joint_states
rosparam get /use_sim_time
```

Before starting this console, the command topic should have no unexpected
publishers. Do not call motor services as part of these checks.

### 2. Prepare the Linux operator computer

The operator computer needs:

- Docker Engine with Docker Compose;
- a robot-facing network address reachable from the robot NUC;
- a route to the Motive/OptiTrack server;
- access to the selected physical keyboard's Linux evdev device; and
- firewall rules that allow ROS1's master and dynamically allocated peer ports
  on the robot network, plus NatNet UDP traffic on the OptiTrack network.

Find a stable keyboard device path:

```bash
ls -l /dev/input/by-id/*event-kbd
```

Prefer a `/dev/input/by-id/...-event-kbd` path rather than `/dev/input/eventN`,
because event numbers can change after reboot or reconnection.

Create the local configuration:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
ROS_MASTER_URI=http://ROBOT_NUC_ADDRESS:11311
ROS_IP=OPERATOR_PC_ROBOT_FACING_IP
DEADMAN_KEYBOARD_DEVICE=/dev/input/by-id/YOUR_KEYBOARD-event-kbd

OPTITRACK_SERVER_IP=172.24.68.77
OPTITRACK_CLIENT_IP=
OPTITRACK_RIGID_ID=33
```

`ROS_IP` must be an operator-computer address to which the robot NUC can open a
connection. Do not use loopback, a Docker bridge address, or an unrelated Wi-Fi
address. Leaving `OPTITRACK_CLIENT_IP` empty selects the local address from the
route to Motive; set it explicitly if automatic route selection is ambiguous.

The console uses host networking. This is intended for native Linux Docker,
not Docker Desktop on macOS or Windows.

### 3. Build the console

```bash
docker compose build
```

The image contains ROS Noetic on Ubuntu Focal, NumPy, SymPy, evdev, and the
pinned `natnet==0.2.0` client. It does not contain the simulation, camera,
Dynamixel, CVXPY, or GUI dependencies because the operator console does not use
them.

The build also compiles the catkin package and runs the ROS-independent safety
and kinematics tests. Rebuild after changing the Dockerfile, dependencies,
package metadata, or CMake configuration.

### 4. Start against the motor-disabled backend

Keep Space released, then run:

```bash
docker compose up
```

The process should connect to OptiTrack, open the selected keyboard, connect to
the external ROS master, and begin publishing zero velocity at 100 Hz. It does
not start a ROS master.

In another terminal, inspect the status:

```bash
docker compose exec giraf-teleop bash -lc \
  'source /catkin_ws/devel/setup.bash && rostopic echo /giraf_optitrak_teleop/status'
```

Also inspect the command stream during dry-run:

```bash
docker compose exec giraf-teleop bash -lc \
  'source /catkin_ws/devel/setup.bash && rostopic hz /giraf_arm/teleop_task_velocity_cmd'
```

With Space released, all six command components must remain zero.

### 5. Exercise the clutch in dry-run

When status reports `space_released` and `interlocks: []`:

1. Hold the tracked controller still.
2. Press and continue holding Space. This captures fresh controller and robot
   pose anchors; it does not jump to an absolute OptiTrack pose.
3. Move or rotate the controller slowly along only one axis.
4. Verify the dry-run robot receives the expected bounded Cartesian command.
5. Release Space and verify all six velocity components return to zero.

Repeat for positive and negative X, Y, Z, roll, pitch, and yaw. Initial command
caps are deliberately conservative:

| Component | Limit |
| --- | ---: |
| X velocity | 0.05 m/s |
| Y velocity | 0.05 m/s |
| Z velocity | 0.025 m/s |
| Roll rate | 0.125 rad/s |
| Pitch rate | 0.125 rad/s |
| Yaw rate | 0.125 rad/s |

Test each interlock in dry-run: release Space, obscure the rigid body, stop the
fake feedback stream, disconnect the keyboard, and start a competing test
publisher one case at a time. Each fault must force zero. Motion must remain
disarmed after recovery until Space is fully released and pressed again.

### 6. Shut down

Release Space first, then stop Compose with Ctrl-C or:

```bash
docker compose down
```

On a clean shutdown the console publishes zero commands for approximately
250 ms before disconnecting. This complements—but does not replace—the
robot-side 150 ms command watchdog.

### 7. Hardware commissioning

Do not enable motors merely because the software dry-run passed. For the first
hardware session:

- use the physical estop and have a second person ready to actuate it;
- place the arm in the controller's required startup configuration;
- verify the live MD80 calibration, wrist homes, limits, and metal-wrist sign
  convention on the physical robot;
- make the workspace safe for a possible direction or frame mismatch;
- begin with one axis at a time and very small controller motion;
- release Space after every individual check; and
- stop if commanded state and physical motion disagree.

The existing robot controller initializes the wrist state from constants and
does not publish measured wrist feedback. Consequently, this implementation is
not a trustworthy measured-feedback six-joint servo. Treat successful remote
teleoperation as supervised operation of the existing controller, not as proof
of full hardware-state observability.

## What the status means

The node publishes latched JSON on `/giraf_optitrak_teleop/status` at 5 Hz.

| `state` | Meaning | Operator action |
| --- | --- | --- |
| `space_released` | Inputs are healthy and Space is released | Safe to press Space when ready |
| `engaged` | Space is held and command generation is active | Move slowly; release to disengage |
| `interlocked` | One or more required inputs are unhealthy | Release Space and correct every listed fault |
| `release_then_press_required` | Inputs recovered while Space remained held | Fully release Space, then press again |

The `interlocks` array contains human-readable reasons. `ages` reports local
monotonic receipt age for OptiTrack, robot state, MD80 state, and the ROS-master
publisher check. `competing_publishers` names other nodes found on the command
topic. `command` reports the most recently published six-axis velocity.

## Architecture and ROS contract

```text
Motive / NatNet 4.3 ──> tracked pose ─┐
                                     ├─> health + hold-to-run gate
Linux evdev Space key ────────────────┤             │
/giraf_arm/state ──> GIRAF FK ───────┤             v
/md80/joint_states ──────────────────┘     bounded TwistStamped at 100 Hz
                                                        │
                                                        v
                                  /giraf_arm/teleop_task_velocity_cmd
                                                        │
                                                        v
                                       robot-side /giraf_arm_controller
```

Published interfaces:

| Topic | Type | Behavior |
| --- | --- | --- |
| `/giraf_arm/teleop_task_velocity_cmd` | `geometry_msgs/TwistStamped` | 100 Hz bounded command or explicit zero |
| `/giraf_optitrak_teleop/status` | `std_msgs/String` | 5 Hz latched diagnostic JSON |

Subscribed interfaces:

| Topic | Type | Purpose |
| --- | --- | --- |
| `/giraf_arm/state` | `std_msgs/String` | Commanded joint state, source, and stop latch |
| `/md80/joint_states` | `sensor_msgs/JointState` | Fresh measured base-joint feedback gate |

No gripper topic, direct MD80 topic, stop topic, command-source topic, or motor
service is published or called.

## Kinematics and command mapping

The console imports the repository's existing
`control/RRPRRR_kinematic_model.py`; it does not contain a second kinematic
model. State coordinates are reconstructed in this order:

```text
[roll, pitch + pi/2, d3, th4 + pi/2, th5 - pi/2, th6]
```

The software-published boom motor angle is converted back to extension with the
same cubic spool relationship used by the robot controller. Incoming state is
rejected if it is malformed, non-finite, outside the controller's known joint
ranges, stopped, on the wrong command source, stale, or not progressing.

When Space is freshly pressed, the current tracked-controller pose and current
commanded robot pose become anchors. Subsequent controller motion is interpreted
relative to that controller anchor and applied to the robot anchor. The node
then computes pose error and publishes a proportional, per-axis-clamped
Cartesian velocity. It does not apply the MuJoCo-only endpoint/control-frame
offset used by some simulation scripts.

The `TwistStamped.header.stamp` is intentionally left at zero. The deployed
robot controller replaces a zero stamp with its own receipt time, avoiding
cross-machine clock skew in the 150 ms command timeout.

## Safety behavior

Motion is allowed only when all conditions remain true:

- the evdev keyboard is connected and Space is physically held;
- OptiTrack has a fresh, finite, tracking-valid sample for the configured rigid
  body;
- two progressing robot-state messages have been observed;
- robot state is fresh, unlatched, and in teleop source mode;
- two correctly named MD80 feedback messages have been observed and feedback
  remains fresh;
- the ROS-master publisher check is fresh; and
- no other publisher owns the teleop command topic.

Any failed condition publishes zero and invalidates the clutch anchor. Merely
recovering the failed condition cannot resume motion: the operator must release
and press Space again.

This console's zero-command behavior is a controlled hold request to the
robot-side controller. It is not torque-off and is not a physical estop. The
robot controller and hardware still require their own watchdog, limit, fault,
and emergency-stop protections.

## Configuration reference

Operational defaults live in `launch/giraf_optitrak_teleop.launch`.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `optitrack_server_ip` | `172.24.68.77` | Motive/NatNet server |
| `optitrack_client_ip` | empty | Auto-select local routed address |
| `optitrack_rigid_id` | `33` | Tracked controller rigid-body ID |
| `optitrack_command_port` | `1510` | NatNet command port |
| `optitrack_data_port` | `1511` | NatNet data port |
| `optitrack_multicast` | `false` | Use unicast by default |
| `keyboard_device` | `/dev/input/deadman_keyboard` | Container evdev mapping |
| `publish_hz` | `100.0` | Command publication rate |
| `status_hz` | `5.0` | Diagnostic publication rate |
| `position_gain` | `1.0` | Translational pose-error gain |
| `rotation_gain` | `1.0` | Rotational pose-error gain |
| `position_scale` | `1.0` | Controller translation scale |
| `orientation_scale` | `1.0` | Controller rotation scale |
| `opti_timeout_ms` | `100.0` | Maximum OptiTrack receipt age |
| `state_timeout_ms` | `300.0` | Maximum controller-state receipt age |
| `md80_timeout_ms` | `300.0` | Maximum MD80-feedback receipt age |

Velocity arrays and expected MD80 names are also defined in the launch file.
Change conservative limits only after dry-run and low-speed physical validation.

## Troubleshooting

### The container exits immediately

Inspect logs:

```bash
docker compose logs --no-color giraf-teleop
```

Startup intentionally fails if the keyboard cannot open, NatNet cannot connect,
required parameters are invalid, or `/use_sim_time` is true.

### Keyboard permission or device errors

Confirm that `DEADMAN_KEYBOARD_DEVICE` exists on the host and points to the
actual keyboard event device. Replugging a device may change `/dev/input/eventN`
but should not change a correct `/dev/input/by-id` symlink. The container is not
privileged; only the configured device is passed through.

### No OptiTrack data

Confirm Motive is streaming rigid body 33, the rigid body is tracking-valid,
the server address and NatNet ports are correct, and the operator machine uses
the intended interface to reach Motive. Set `OPTITRACK_CLIENT_IP` explicitly if
the machine has ambiguous routes.

### ROS master is reachable but topics do not connect

ROS1 nodes establish peer-to-peer connections after querying the master. Both
machines must be able to initiate connections to each other's advertised
addresses. Verify `ROS_IP` is reachable from the robot NUC and that the firewall
does not allow port 11311 while blocking the dynamic XMLRPC/TCPROS ports.

### `robot command source is not teleop`

Select teleop on the robot using the established robot-side operational
procedure. This console refuses to change source because silently taking
command ownership would be unsafe.

### `competing teleop publisher detected`

Stop the named publisher and release Space. The console excludes itself from
the ROS-master result but rejects every other publisher on the command topic.

### State or MD80 feedback is stale

Verify that the robot-side dry-run or hardware controller is actively
publishing rather than merely leaving a latched state behind. The console
requires two progressing state messages and two correctly formed MD80 messages
before it becomes ready.

## Development validation

Run the console-specific suite without ROS:

```bash
python3 -m unittest tests.test_teleop_core -v
```

The Docker build runs the same suite using the Noetic image's Python 3.8. The
tests cover spool conversion, authoritative FK reconstruction, controller-state
validation, freshness, MD80 name mapping, velocity limits, competing publisher
detection, and release/re-press clutch behavior.

Source and launch files are mounted read-only into the development container,
so Python or launch edits take effect after restarting Compose. Rebuild when
dependencies or package/build metadata change.

## Known limitations

- `/giraf_arm/state` contains software-integrated commanded joints, not measured
  full-arm state.
- Only MD80 joints 11–13 have measured ROS feedback in the current robot stack.
- The wrist and gripper are owned directly by the robot-side Dynamixel code and
  are not measured over ROS by this console.
- There is no unified robot ready/fault/estop message in the inspected stack.
- ROS1 provides no authentication or exclusive publisher lease.
- A competing publisher cannot be made safe by this console; it must be stopped.
- Clean-shutdown zeros cannot protect against every host, network, or process
  failure. Robot-side watchdog behavior remains essential.

A more robust final robot architecture would have one robot-side process own
both actuator buses, publish measured state and health for every joint, enforce
limits and receipt-time watchdogs, and accept only bounded desired motion plus
an explicit heartbeat/deadman signal from the operator console.
