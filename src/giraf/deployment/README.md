# Teleop / policy deployment

See the repository [deployment guide](../../../README.md#deployment) for setup,
connection options, controls, inference timing, pause recovery, and logs.

Run `python -m giraf.deployment`. The console starts at logical home in teleop;
D switches action sources, a fresh SPACE press enables motion, releasing SPACE
pauses with motor targets retained, and Q/Ctrl+C shuts down. Recorded start
poses and rollout durations are no longer required or accepted.

- `configuration.py`: validated deployment settings and command-line parsing.
- `runner.py`: device ownership, worker supervision, and shutdown orchestration.
- `control.py`: the 100 Hz command/motor worker and OptiTrack receiver.
- `policy_loop.py`: camera acquisition, inference, and action publication.
- `recording.py`: session logs and finalized per-rollout video/data directories.
- `session.py`: shared state, clutch gating, pauses, and inference generations.
- `safety.py`: twist guards, kinematics, and command/state limits.
- `../teleop_control.py`: relative OptiTrack control shared with standalone teleop.

OptiTrack's cached pose is polled at the 100 Hz control rate in teleop only.
Pose polling is suspended in policy mode; the NatNet receiver remains connected
for handoff. Camera acquisition and inference run on the main thread, with
keyboard, motor control, and supervision in separate threads.
`inference_started` / `inference_finished` events record replanning latency even
when the activation times out or is paused before the prediction returns.
Training-bound excursions are observable but non-blocking by default; pass
`--enforce-training-bounds` to turn them into policy pauses. Physical limits and
finite-value checks are always enforced.
