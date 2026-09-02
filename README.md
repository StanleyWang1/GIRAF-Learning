# GIRAF teleoperation and data collection

The existing teleoperation stack can optionally record camera observations,
task-space actions, commanded state, grasp commands, and final actuator targets
into a Diffusion Policy-compatible Zarr ReplayBuffer.

## Setup

Create the Python 3.12 environment described in `instructions.md`, then install
the dependencies:

```bash
python -m pip install -r requirements.txt
```

Normal teleoperation is unchanged:

```bash
python teleop.py --hardware
```

Enable data collection with the example configuration:

```bash
python teleop.py --hardware --collect-config config/data_collection.yaml
```

Controls:

- Hold `SPACE` to clutch teleoperation.
- Press `B` to toggle the gripper command.
- Press `R` to start an episode, then press `R` again to stop and commit it.
- Press `Ctrl+C` to stop. An active healthy episode is committed during shutdown.

Recording starts only after camera, control, and motor streams have produced a
sample. A ring overrun, video encoder failure, or rejected motor dispatch aborts
the current episode without blocking the robot-control loop. Stale but available
control samples are retained with `alignment_valid=0` and source-age metadata.

## Output

The configured output directory contains:

```text
giraf_demos/
├── replay_buffer.zarr/
│   ├── data/
│   │   ├── camera_rgb          # uint8 [T, 224, 224, 3]
│   │   ├── action              # float32 [T, 7]
│   │   ├── state               # float32 [T, 15]
│   │   ├── grasp_label         # uint8 [T]
│   │   └── timing and actuator audit arrays
│   └── meta/episode_ends       # cumulative exclusive boundaries
├── videos/<episode>/camera.mp4
├── videos/<episode>/episode.json
└── rejected/                   # recoverable partial episodes
```

`action` is `[vx, vy, vz, wx, wy, wz, grasp]`. The Cartesian twist is captured
after deadband and safety saturation but before RMRC. `state` contains the six
commanded joints, command-derived FK position, and a 6D rotation representation.
It is not measured motor feedback. The grasp value is an operator command, not
contact sensing.

All camera, control, and motor producers use the host monotonic clock. The
camera frame is the 30 Hz reference; control and motor samples use latest-at-or-
before alignment so a future command is never attached to an earlier image.

The ReplayBuffer stores only aligned 30 Hz samples. Source-resolution video is
retained, but the intermediate 100 Hz control stream is not, so action alignment
cannot be reconstructed later.

## Replay and training

Inspect or extract an episode:

```bash
python -m data_collection.replay \
  --dataset data/giraf_demos/replay_buffer.zarr \
  --episode 0 --show

python -m data_collection.replay \
  --dataset data/giraf_demos/replay_buffer.zarr \
  --episode 0 --extract-dir /tmp/giraf_episode_0
```

`config/diffusion_policy_shape_meta.yaml` contains the RGB, state, and 7D action
shapes for a Diffusion Policy task configuration. Images stay RGB `uint8` on
disk; convert to grayscale or normalize in the dataset loader so collection is
not irreversible.

Depth is reserved in the configuration but intentionally rejected in v1 until
a calibrated depth source and schema are defined.
