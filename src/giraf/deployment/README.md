# Guarded policy deployment

This package is additive: it does not alter the training, teleoperation, or data
collection paths. Run it as `python -m giraf.deployment`; no project entry point
was added.

## What it does

- Loads one explicit episode start from a GIRAF replay dataset and checks its
  state against the checkpoint normalizer and independent hardware limits.
- Uses live RGB with the same RGB conversion and resize configuration as data
  collection.
- Scales and clamps policy twists, caps joint rates, checks command-derived
  state bounds, disables grasp unless explicitly enabled, and stops on stale
  actions or stale camera frames.
- Uses SPACE as a deadman and caps the rollout duration (five seconds by
  default).
- Writes `config.json`, `events.jsonl`, `camera.mp4`, and
  `reference_start.png` under `deployment_runs/`.

`shadow` and `dry-run` never open the motors. Shadow keeps the logical arm pose
fixed while reporting policy outputs. Dry-run integrates the bounded commands
into a simulated command-derived pose. Hardware mode first stages from the
teleop home pose to the selected recorded pose at conservative joint speeds.

## Before any hardware run

1. Stop teleop and any other process using the motors, camera, or input devices.
2. Put the physical robot at the exact home pose used when starting teleop for
   data collection. The MAB driver defines the current pose as encoder zero
   when it connects.
3. Place the camera, tape, and surrounding scene like the chosen demonstration.
   The run directory contains `reference_start.png` for comparison.
4. Keep the physical kill switch reachable and begin with SPACE released.
5. Do the first run with grasp disabled, `--action-scale 0.2`, and a five-second
   duration.

Episode 16 is a reasonable first Stanley reference because its initial joint
pose is near the center of the 50 Stanley episode starts. Scene matching still
matters more than that statistical convenience.

At `--action-scale 0.2`, both ordinary predictions and the independent hard
twist ceilings are reduced to 20%. Scaling does not affect the binary grasp
channel.

## Recommended sequence

Check command generation without opening motors:

```bash
uv run --frozen python -m giraf.deployment \
  --checkpoint checkpoints/tape_grasping/policy.pt \
  --reference-dataset data/tape_grasping/stanley_trials_cleaned.zarr \
  --reference-episode 16 \
  --mode dry-run \
  --action-scale 0.2 \
  --inference-steps 10 \
  --duration 5
```

For the first physical run, remain attached to the terminal (a tmux session is
fine, but do not detach during motion):

```bash
tmux new -s giraf-deploy

uv run --frozen python -m giraf.deployment \
  --checkpoint checkpoints/tape_grasping/policy.pt \
  --reference-dataset data/tape_grasping/stanley_trials_cleaned.zarr \
  --reference-episode 16 \
  --mode hardware \
  --action-scale 0.2 \
  --inference-steps 10 \
  --duration 5 \
  --confirm-hardware
```

Hardware interaction is deliberately two-stage:

1. With the arm physically at teleop home, release SPACE to arm staging.
2. Hold SPACE continuously while the robot moves slowly to the recorded start.
   Releasing early stops the program and shuts down the motor interfaces.
3. Once staging reports complete, release SPACE. The runner performs one
   no-motion inference and prints the raw action, guarded action, joint rate,
   and inference latency.
4. Hold SPACE again to begin the five-second rollout. Release it at any time to
   stop. `Ctrl-C` also stops.

Do not add `--allow-grasp` until arm-only motion has been reviewed. When it is
enabled, the final policy channel is thresholded into an open/closed command;
it is not scaled like the six twist channels.

The checkpoint stores 100 diffusion inference steps. On this RTX 4090, an
offline warm replan measured about 0.20 seconds at 100 steps versus about 0.022
seconds at 10 steps. Ten steps is therefore explicit in the first-run commands,
but it changes the diffusion sampling procedure and may change policy quality.
Deployment commands zero the task velocity while a synchronous replan is in
progress instead of holding the preceding velocity longer than intended.

## Monitoring and limitations

In another terminal:

```bash
nvidia-smi -l 1
```

After the run, inspect the newest directory:

```bash
ls -lt deployment_runs
tail -n 30 deployment_runs/YYYYMMDD-HHMMSS/events.jsonl
```

The current robot state is command-derived, matching the demonstrations; it is
not encoder feedback. Staging completion therefore means that the commanded
trajectory finished, not that measured joints were verified. This package also
does not add collision detection, contact sensing, or task-success evaluation.
The physical kill switch and supervision remain necessary.
