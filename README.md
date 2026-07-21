# GIRAF-Learning
Teleop, Data Collection, Model Training, and Deployment of Learned Policies for the GIRAF Manipulator Arm

## ROS1 real-arm OptiTrack teleoperation

This repository is also a ROS Noetic catkin package containing a headless,
hold-to-run teleoperation node. It connects to the existing robot-side
`/giraf_arm_controller`; it does **not** own motor devices or start a ROS master.

Data flow:

```text
Motive/NatNet ----> relative pose + pose error ----> bounded TwistStamped
                         ^                                |
/giraf_arm/state ------- FK                               v
Space via evdev -------- hold-to-run gate     /giraf_arm/teleop_task_velocity_cmd
/md80/joint_states ----- health gate
```

Space is a momentary clutch, never a toggle. A fresh press captures the current
controller and commanded robot poses. Releasing Space publishes zero task
velocity immediately. Any stale/malformed input, keyboard loss, robot stop,
wrong command source, ROS-master ownership failure, or competing publisher also
forces zero and requires a complete release followed by a fresh press.

### Safety boundary

- The node publishes only `geometry_msgs/TwistStamped` at 100 Hz.
- It never selects command source, calls `/giraf_arm/stop`, commands the
  gripper, or publishes `/md80/motion_command`.
- Current pose comes from the robot controller's software-integrated state; the
  wrist is not measured feedback in the current robot stack.
- Initial caps are 25% of joystick teleop: X/Y 0.05 m/s, Z 0.025 m/s, and
  rotation 0.125 rad/s.
- The command header stamp is intentionally zero so the robot controller uses
  receipt time for its 150 ms watchdog.
- A physical estop and supervised dry-run commissioning remain required.

### Configuration and startup

Run this only from a Linux control PC with routes to both the robot and Motive.
Docker Desktop networking is not the deployment target.

```bash
cp .env.example .env
ls -l /dev/input/by-id/*event-kbd
# Edit .env with the live master, control-PC ROS address, and keyboard path.
docker compose build
docker compose up
```

The service uses host networking because ROS1 peers must connect directly back
to the control PC and NatNet uses UDP. Only the selected keyboard device is
passed through; the container is not privileged.

Before a dry-run or hardware session, verify the live graph read-only:

```bash
rosnode list
rostopic info /giraf_arm/teleop_task_velocity_cmd
rostopic hz /giraf_arm/state
rostopic hz /md80/joint_states
rostopic echo /giraf_optitrak_teleop/status
```

The robot-side controller must already be running, publishing progressing state
with `active_source` and `command_source_param` equal to `teleop`, and have no
other publisher on the teleop velocity topic. Start with its motor-disabled
dry-run backend. Confirm axis directions and all interlocks before enabling
motors. `/use_sim_time` must be false; the node refuses to start otherwise.

### Development and tests

Compose read-only mounts the current repository over the copy baked into the
image. Existing Python and launch-file edits therefore take effect on restart;
dependency, package metadata, CMake, or generated-code changes require a rebuild.

Run the ROS-independent safety and kinematics suite with:

```bash
python3 -m unittest tests.test_teleop_core -v
```

The same suite runs during the Noetic image build under Python 3.8.
