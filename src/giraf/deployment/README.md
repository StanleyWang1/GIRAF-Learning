# Guarded policy deployment

The current usage guide is in the repository
[README's Deployment section](../../../README.md#deployment), immediately after
Training. It covers environment setup, encoder/action compatibility, ResNet
trial commands, all three modes, the physical-home and SPACE staging sequence,
inference timing, the gripper replan limitation, logs, and teleop handoff status.

Run this package as `python -m giraf.deployment`; no project entry point was
added. The implementation is split into:

- `runner.py`: command-line options, camera/inference loop, motor worker,
  rollout phases, and logging;
- `reference.py`: recorded start-pose loading and consistency checks;
- `safety.py`: twist guards, kinematics, staging, and command/state limits.

This package does not alter the training, teleoperation, or data collection
paths. See the main guide for the current behavior before a hardware trial.
