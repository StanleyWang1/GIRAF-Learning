# Teleop / policy deployment

See the repository [deployment guide](../../../README.md#deployment) for setup,
connection options, controls, inference timing, pause recovery, and logs.

Run `python -m giraf.deployment`. The console starts at logical home in teleop;
D switches action sources, a fresh SPACE press enables motion, releasing SPACE
pauses with motor targets retained, and Q/Ctrl+C shuts down. Recorded start
poses and rollout durations are no longer required or accepted.

- `runner.py`: device ownership, control workers and main-thread inference, CLI, and logs.
- `session.py`: shared state, clutch gating, pauses, and inference generations.
- `safety.py`: twist guards, kinematics, and command/state limits.
- `../teleop_control.py`: relative OptiTrack control shared with standalone teleop.

`reference.py` and the staging helpers remain available for offline reference
inspection; they are not used by the deployment console. Training and collection
entry points retain their existing behavior.

OptiTrack's cached pose is polled at the 100 Hz control rate in teleop only.
Pose polling is suspended in policy mode; the NatNet receiver remains connected
for handoff. Camera acquisition and inference run on the main thread, with
keyboard, motor control, and supervision in separate threads.
`inference_started` / `inference_finished` events record replanning latency even
when the activation times out or is paused before the prediction returns.
