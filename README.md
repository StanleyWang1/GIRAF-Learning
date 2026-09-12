# GIRAF Learning

GIRAF Learning contains the robot drivers, teleoperation and demonstration
collection stack, and a compact PyTorch diffusion-policy foundation. Hardware,
data, and learning code are separate packages with explicit boundaries.

See [IMU collection](notes/imu/imu_collection.md) for the v2 recording schema,
signal semantics, compatibility, hardware checks, and known device limitations.

See the [notes index](notes/README.md) for training setup, IMU testing, and
SRC-03 hardware troubleshooting.

## Repository layout

```text
.
├── config/data_collection.yaml   # collection settings
├── src/giraf/
│   ├── data/                     # capture, alignment, Zarr storage, replay
│   ├── deployment/               # guarded live policy trials and rollout logs
│   ├── drivers/                  # camera, input, OptiTrack, and motor adapters
│   ├── learning/                 # diffusion policy, datasets, and training loops
│   ├── viewer/                   # read-only local dataset web viewer
│   ├── kinematics.py             # GIRAF kinematic model
│   ├── settings.py               # shared runtime constants
│   └── teleop.py                 # teleoperation application
├── pyproject.toml                # package metadata and dependencies
└── uv.lock                       # exact reproducible dependency lock
```

Runtime datasets are written below `data/` and are ignored by Git.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the project environment
and install the core package:

```bash
uv sync
```

`uv sync` creates `.venv` automatically. Commands can be run through `uv run`
without activating it. To activate it for an interactive shell:

```bash
source .venv/bin/activate
```

Linux and Windows install PyTorch from the CUDA 13.0 index (about 3 GB of
wheels). Learning and tooling also run on CPU. Add `--extra train` for
Weights & Biases logging.

Install the device libraries on a machine connected to the robot:

```bash
uv sync --extra hardware
```

Use `uv add <package>` and `uv remove <package>` for dependency changes so
`pyproject.toml` and `uv.lock` stay synchronized. Do not maintain a separate
requirements file.

## Teleoperation

A dry run opens the keyboard and OptiTrack interfaces but does not open or
command the physical motors:

```bash
uv run --extra hardware giraf-teleop
```

Enable the existing motor output path explicitly:

```bash
uv run --extra hardware giraf-teleop --hardware
```

The relevant options are:

```text
--server-ip       Motive/NatNet server address
--client-ip       local interface address; route-derived when omitted
--rigid-id        controller rigid-body ID
--hardware        enable MD80 and DYNAMIXEL outputs
--collect-config  enable recording with the supplied YAML configuration
```

Controls:

- Hold `SPACE` to clutch teleoperation.
- Press `B` to toggle the gripper command.
- Press `R` to start or finish an episode while collection is enabled.
- Press `Ctrl+C` to stop. A healthy active episode is committed on shutdown.

Stage the robot safely before using `--hardware`.

## Data collection

Start teleoperation and collection together:

```bash
uv run --extra hardware giraf-teleop --hardware \
  --collect-config config/data_collection.yaml
```

Collection begins only after camera, control, and motor streams have produced
a sample. The camera is the 30 Hz reference. Control and motor samples are
matched latest-at-or-before, so a future command is never attached to an older
image. Ring overruns, video-encoder failures, and rejected motor dispatches
reject the active episode instead of silently producing a trusted sample.

The default output is:

```text
data/demos/
├── replay_buffer.zarr/
│   ├── data/
│   │   ├── camera_rgb       # uint8 [T, H, W, 3]
│   │   ├── action           # float32 [T, 7]
│   │   ├── state            # float32 [T, 15]
│   │   └── timing and actuator audit arrays
│   └── meta/episode_ends    # cumulative exclusive episode boundaries
├── videos/<episode>/
│   ├── camera.mp4
│   └── episode.json
└── rejected/                # inspectable incomplete/rejected episodes
```

`action` is `[vx, vy, vz, wx, wy, wz, grasp]`. The twist is captured after
deadband and safety saturation but before RMRC. `state` contains six commanded
joints, command-derived forward-kinematics position, and the first two rotation
matrix columns. It is not measured motor feedback. Grasp is an operator command,
not contact sensing.

Source-resolution video is retained when enabled. Zarr stores only aligned,
resized observations; the intermediate 100 Hz control stream is not retained.
Images remain RGB `uint8` on disk so normalization stays a training concern.

Shared runtime settings live in `config/data_collection.yaml`. Task-specific
files use `extends: data_collection.yaml` and override only their output paths.
Fixed action and state fields live in `giraf.data.schema`; diffusion shape
metadata is derived from those fields and the resolved collector configuration:

```python
from giraf.data import diffusion_shape_meta, load_config

config = load_config("config/data_collection.yaml")
shape_meta = diffusion_shape_meta(config)
```

## Dataset cleaning

Diagnose one episode without loading video:

```bash
uv run giraf-diagnose --dataset data/demos/replay_buffer.zarr --episode 0
```

This reports physical Zarr chunk coverage, alignment-rule reconstruction,
timing, metadata consistency, and stream health. Add `--json` for structured
output.

Audit every episode and preview the healthy subset:

```bash
uv run giraf-clean --dataset data/demos/replay_buffer.zarr --dry-run
```

Remove `--dry-run` to write `data/demos/replay_buffer_cleaned.zarr`. The cleaner
rejects incomplete or inconsistent episodes, copies intact episodes without
renumbering errors, and verifies the new Zarr before publishing it. It never
changes the source or overwrites an existing output; use `--output` to choose a
different destination.

Inactive-action pruning is separate and opt-in because zero-motion holds may be
intentional. Preview it with:

```bash
uv run giraf-prune --dataset data/demos/replay_buffer.zarr
```

Pass `--output PATH` to `giraf-prune` to write its result, or add
`--prune-inactive` to `giraf-clean` to combine both operations. The shared
controls are `--action-epsilon`, `--padding-steps`, `--min-segment-steps`,
`--grasp-cooldown-s`, and `--ignore-grasp-transitions`. Active runs and the
configured post-grasp window become separate output episodes.

## Dataset viewer

Open a GIRAF Zarr in the local read-only web viewer:

```bash
uv run giraf-view \
  --dataset data/demos/replay_buffer_cleaned.zarr \
  --episode 0
```

Then visit the printed URL (normally `http://127.0.0.1:8080`), or add `--open`
to open it automatically. `--episode` selects only the initial episode; use the
episode dropdown or arrow buttons in the browser to move through the entire
dataset. The viewer provides frame playback and scrubbing, action/state plots
with schema field names, validity and collection metrics, and GIRAF-specific
event markers. It never modifies the Zarr.

## Replay

Inspect, display, or extract an episode:

```bash
uv run giraf-replay --dataset data/demos/replay_buffer.zarr --episode 0
uv run giraf-replay --dataset data/demos/replay_buffer.zarr --episode 0 --show
uv run giraf-replay --dataset data/demos/replay_buffer.zarr --episode 0 \
  --extract-dir /tmp/giraf-episode-0
```

## Learning

The initial policy is a conditional DDPM implemented with PyTorch:

- a pluggable RGB encoder conditions a temporal 1D U-Net together with robot state;
- training predicts noise added by a cosine-beta DDPM scheduler and keeps an
  exponential moving average of the weights (`ema_decay`, default `0.999`);
- inference uses a 16-step DDIM schedule (`inference_steps`, default `16`) on
  the EMA weights when available, then executes an 8-step action chunk;
- checkpoints contain model, optimizer, and EMA state, configuration, and
  format version, and are written atomically.

Batches carry raw physical units straight from the ReplayBuffer:

```text
observations["camera_rgb"]  uint8 [B, 2, H, W, 3] or float32 [B, 2, 3, H, W]
observations["state"]       float32 [B, 2, 15]
observations["imu"]         float32 [B, 2, 10]  (only with IMU input enabled)
actions                     float32 [B, 16, 7]  (m/s, rad/s, grasp in {0, 1})
```

A `Normalizer` fitted on the dataset maps actions and states to `[-1, 1]`
inside the policy and is stored in the checkpoint, so `act()` returns
denormalized actions `[vx, vy, vz, wx, wy, wz, grasp]`. RGB scaling also
happens inside the policy. `act()` keeps its own two-frame observation
history; `DiffusionPolicy.reset()` clears it at an episode boundary and
deployment calls it at every policy activation. The required policy contract
remains `act`, `train_step`, and `save`.

### Encoders

`DiffusionPolicyConfig.encoder` (also `--encoder` on `giraf-train`) selects the
image encoder: `conv` is the small baseline convolutional net, trained end to
end; `resnet18` is a ResNet-18 (GroupNorm instead of BatchNorm) with a
spatial-softmax keypoint head, also trained end to end, for tasks that need
more precise spatial alignment; `dinov2` wraps a frozen, pretrained DINOv2
backbone for more robust features on limited data, and downloads its weights
through `torch.hub` on first use.

### Action spaces and temporal ensembling

`DiffusionPolicyConfig.action_space` (also `--action-space` on `giraf-train`)
selects what `act()` returns: `twist` is `[vx, vy, vz, wx, wy, wz, grasp]` in
m/s and rad/s; `joint_position` is absolute joint targets in the teleop joint
order (`base_roll`, `base_pitch`, `boom_extension`, `wrist_1`, `wrist_2`,
`wrist_3`) plus grasp. `--prediction-horizon` (default `16`) and
`--action-horizon` (default `8`) control the length of each predicted action
chunk and how many steps of it are executed before replanning. With
temporal ensembling (default on, `--no-temporal-ensemble` to disable) each
`act()` call averages every still-open plan's prediction for the current
step before thresholding grasp, smoothing the transition between replans
instead of jumping straight to a new chunk. A deployment executor must read
`action_space` from the checkpoint config to know how to interpret the
actions `act()` returns.

### Training

`--state-input full` is the default and uses all 15 state values. For a fresh
run using only the five angular joints (base roll/pitch and three wrist angles),
add `--state-input joint_angles`; this excludes boom extension and all FK
position/orientation inputs while retaining RGB images. The selection is saved
in the checkpoint and reused during inference. Existing checkpoints default to
`full`; `--resume` keeps the checkpoint's selection, so changing modes requires
a fresh run. Recorded data and action outputs are unchanged.

`--imu-input` independently selects optional camera-aligned IMU features:

| Mode | Features |
| --- | --- |
| `none` (default) | No IMU; preserves existing training behavior and older datasets/checkpoints |
| `accel_gyro` | 6 values: acceleration XYZ (includes gravity) and gyro XYZ |
| `full` | 10 values: acceleration, gyro, and game rotation quaternion XYZW |

For example, add `--state-input joint_angles --imu-input full` to a fresh
`giraf-train` command. IMU remains a separate observation field and is
concatenated with selected state features inside the policy; no extra encoder
is used. Accel/gyro scaling is fitted only on retained training observations;
quaternion components keep their native scale and sign. Selection and scaling
are saved in checkpoints; `--resume` uses the checkpoint's IMU mode.

IMU modes require `data/imu` and `data/imu_sensor_valid`. Windows are skipped
if any required IMU observation is invalid, without joining across gaps;
`accel_gyro` does not require a valid quaternion. Existing robot-alignment
filtering is unchanged. No full-rate IMU resampling is performed.

For `act()`, supply the same 10D `imu` layout alongside image/state; the caller
must ensure selected sensors are fresh and camera-aligned. The live deployment
runner does not yet supply IMU and rejects IMU-enabled checkpoints at startup.
Game rotation heading is arbitrary and can drift; evaluate across device sessions.

Install the optional W&B client and authenticate once:

```bash
uv sync --extra train
uv run wandb login
```

Use `tmux` for a long run so it survives a terminal disconnect:

```bash
cd ~/Documents/GIRAF-Learning
tmux new -s giraf-train
```

Inside `tmux`, choose the dataset and a unique run directory. The timestamp
keeps a new run from appending metrics to, or overwriting checkpoints in, an
older run:

```bash
TRAIN_DATASET=data/tape_grasping/sept03_trials.zarr
TRAIN_RUN_NAME=sept03_trials_200ep_$(date +%Y%m%d-%H%M%S)
TRAIN_RUN_DIR="checkpoints/tape_grasping/$TRAIN_RUN_NAME"

if test -e "$TRAIN_RUN_DIR"; then
  echo "ERROR: refusing to reuse existing run directory: $TRAIN_RUN_DIR"
  exit 1
fi

mkdir -p "$TRAIN_RUN_DIR"

export WANDB_MODE=online
export WANDB_NAME="$TRAIN_RUN_NAME"
export WANDB_CONSOLE=off
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -c unlimited
set -o pipefail

uv run giraf-train \
  --dataset "$TRAIN_DATASET" \
  --output-dir "$TRAIN_RUN_DIR" \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --checkpoint-every 20 \
  --device cuda \
  --seed 0 \
  --preload-images \
  --wandb \
  --wandb-project giraf-tape-grasping \
  2>&1 | tee "$TRAIN_RUN_DIR/train.log"

training_status=${PIPESTATUS[0]}
printf '%s\n' "$training_status" | tee "$TRAIN_RUN_DIR/exit_status.txt"
```

`WANDB_MODE=online` enables live monitoring. `WANDB_CONSOLE=off` leaves metric
logging enabled but prevents W&B from intercepting the terminal stream, so
`train.log` retains native crash diagnostics. The exit status is `0` after a
successful run; values above `128` identify termination by a Unix signal.

Detach from `tmux` without stopping training by pressing `Ctrl-b`, releasing
both keys, and then pressing `d`. Reattach with `tmux attach -t giraf-train`.

The run directory receives `config.json`, `normalizer.json`, `metrics.jsonl`
(one line per epoch), `policy.pt` after every epoch, and
`policy_epoch_NNNN.pt` every `--checkpoint-every` epochs. Windows are anchored
at every step whose `alignment_valid` flag is set; the first observation and
the last action repeat at episode boundaries, matching what `act()` does with
its history. Use `--preload-images` only when the uncompressed images fit in
RAM. W&B runs offline by default when `WANDB_MODE` is not set.

The learning rate follows a linear warmup to the peak value over
`--warmup-steps` optimizer steps (default `500`), then decays by a cosine
schedule to `--min-lr-ratio` of the peak (default `0.1`) by the end of
training; the current value is logged each epoch as `lr`.

`--val-fraction` (default `0.1`) holds out that fraction of episodes, at
least one whenever there are two or more, for validation. The split is
episode-level and written to `config.json` as `train_episodes` and
`val_episodes`, and the normalizer is fitted on the training episodes only.
Each epoch also logs `val_loss`, a deterministic denoising loss computed with
timesteps spread evenly over the schedule instead of random ones, and
`val_action_mse`, the error between a fully sampled action chunk and the
ground-truth actions. `val_action_mse` is the metric to watch, since
denoising loss correlates only weakly with rollout quality. `best.pt` is
saved whenever `val_action_mse` improves.

Camera images are cropped to `crop_fraction` (default `0.9`) of their height
and width and, during training only, given per-sample brightness and contrast
jitter of up to `color_jitter` (default `0.1`). Evaluation and `act()` use a
center crop with no jitter. Checkpoints saved before this feature existed
load with `crop_fraction=1.0` and `color_jitter=0.0` so they keep seeing the
uncropped, unjittered images they were trained on.

To warm-start a recovery run from an existing checkpoint, use one whose
completed epoch is known. A numbered checkpoint is preferable after an abrupt
shutdown because its filename makes that boundary explicit. Use a new output
directory, set `--start-epoch` to that boundary, and set `--epochs` to the total
target:

```bash
RECOVERY_RUN_NAME=recovered_from_0260_$(date +%Y%m%d-%H%M%S)
RECOVERY_RUN_DIR="checkpoints/tape_grasping/$RECOVERY_RUN_NAME"
export WANDB_NAME="$RECOVERY_RUN_NAME"

uv run giraf-train \
  --dataset "$TRAIN_DATASET" \
  --output-dir "$RECOVERY_RUN_DIR" \
  --epochs 500 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --checkpoint-every 20 \
  --device cuda \
  --seed 0 \
  --preload-images \
  --wandb \
  --wandb-project giraf-tape-grasping \
  --resume "$TRAIN_RUN_DIR/policy_epoch_0260.pt" \
  --start-epoch 260
```

The checkpoint restores the model, optimizer, normalizer, and policy
configuration. `--start-epoch 260` advances logging and deterministic dataset
shuffling so the next epoch is 261. Older checkpoints do not contain the exact
PyTorch or CUDA random-number state, so this is a valid stochastic continuation
rather than a bit-for-bit replay of an uninterrupted run. Use the same dataset,
batch size, and seed as the original command. The recovery directory and W&B
tracking are new; their configuration records the source checkpoint, while the
original run remains untouched.

If the source checkpoint was trained without a validation split, pass
`--val-fraction 0` on the recovery run too, or accept that reported
`val_loss`/`val_action_mse` may include episodes the checkpoint already
trained on. The episode split is reproducible across runs whenever
`n_episodes`, `--val-fraction`, and `--seed` all match, so passing the
original values recreates the same held-out episodes.

```python
from giraf.learning import DiffusionPolicy, ReplayDataset, train

dataset = ReplayDataset("data/demos/replay_buffer.zarr", batch_size=64)
policy = DiffusionPolicy(normalizer=dataset.fit_normalizer())
history = train(policy, dataset, epochs=1)
policy = DiffusionPolicy.load("checkpoints/tape_grasping/policy.pt", device="cpu")
```

### Deployment

Run the deployment console with `python -m giraf.deployment`. It keeps one
motor connection and shared commanded robot state while switching between
OptiTrack teleop and a diffusion policy. No recorded episode or start-pose
staging is required.

**Compatibility:** `conv`, `resnet18`, and `dinov2` checkpoints use the shared
policy loader. Saved encoder settings, center crop, normalization, EMA weights
when present, action horizon, and temporal ensembling remain in effect.
Training-time color jitter is disabled. DINOv2's backbone code and pretrained
weights must be cached or downloadable. Checkpoints must contain a normalizer
and use `action_space="twist"`; incompatible checkpoints are rejected before
motors are opened.

#### Start a session

From the repository root:

```bash
uv sync --frozen --extra hardware
uv run --frozen --extra hardware python -m giraf.deployment --help
uv run --frozen --extra hardware python -m giraf.deployment \
  --checkpoint checkpoints/tape_grasping/nautilus-policy-v2-resnet/best.pt \
  --config config/tape_grasping.yaml \
  --device cuda \
  --mode dry-run \
  --action-scale 0.2
```

Stop standalone teleop and other processes using the camera or motors before
launching. The console uses live camera RGB and Linux keyboard input in every
backend, plus OptiTrack for teleop. Its connection options match teleop:
`--server-ip` (default `172.24.68.77`), `--client-ip` (default automatic), and
`--rigid-id` (default `40`). Missing/stale OptiTrack poses prevent teleop motion;
policy can run without OptiTrack once the robot is positioned. Pose polling is
suspended while policy is selected; the NatNet receiver stays connected for
teleop handoff.

| Backend (`--mode`) | Behavior |
| --- | --- |
| `shadow` (default) | Reports commands using a fixed logical home pose; motors are not opened. |
| `dry-run` | Integrates commands into a local joint pose; motors are not opened. |
| `hardware` | Commands the physical motors from the same shared joint/gripper state. |

Dry-run still uses live images; it does not simulate the visual consequences of
motion or evaluate task success. `--config` supplies camera settings and RGB
resize dimensions, which should match training (224 x 224 for this example).

For hardware, first put the robot physically at the same teleop home used for
collection, then replace `--mode dry-run` with `--mode hardware` and add
`--confirm-hardware`. The MAB driver zeros encoders on connection. The console
initializes commanded joints to `[0, 0, 0.31, 0, 0, 0]`; this is not an automatic
homing move. Keep the physical kill switch accessible.

#### Teleop and policy handoff

The session starts paused in teleop. Wait for the console's **Controls ready**
message before operating it; motion inputs entered during motor startup are
discarded. Motor targets and calibration persist across every handoff.

| Key | Behavior |
| --- | --- |
| SPACE held | Enables the selected mode after a fresh press. |
| SPACE released | Pauses arm motion, holding joint targets and the last gripper command with motors enabled. |
| D | Switches teleop/policy and pauses. If SPACE is held, release it and press again to enable the new mode. |
| B | Toggles grasp in teleop, including while the arm is paused. Ignored in policy mode. |
| Q or Ctrl+C | Ends the session and runs motor shutdown. |

1. Begin with SPACE released. Hold it to teleoperate using the existing
   OptiTrack controls and gains. Each activation re-anchors the controller to
   the current commanded robot pose.
2. Position the robot and scene, then press D to select policy. Release SPACE
   if held, then hold it again to start the policy from live RGB and current
   command-derived state.
3. Release SPACE to pause. Press it again to make a fresh plan: observation
   history and queued policy actions are cleared at every activation.
4. Press D to return to teleop. A fresh clutch press re-anchors teleop at the
   current pose, without reconnecting or zeroing the motors.

Policy rollouts have no duration limit. There is no mandatory stationary
preview. Old `--reference-dataset`, `--reference-episode`, and `--duration`
arguments have been removed. This console does not record training episodes;
use standalone teleop/collection for that workflow.

#### Inference and grasp

Active policy execution retains the existing camera-paced action consumption
(30 Hz with the example configuration), checkpoint inference schedule, and
100 Hz motor command loop. At each replan it holds zero arm velocity while
sampling synchronously, preserving the current gripper command. Prediction
and chunk execution do not overlap. Camera acquisition and inference run on the
main thread, matching the original deployment path. Keyboard handling, motor
control, and supervision run separately, so pauses and motor shutdown do not
wait for an inference result. Process cleanup completes once the current model
call returns. Results from a previous activation are discarded after a pause
or switch.

The default `--action-scale 0.2` applies to policy twist channels and their
hard velocity ceilings, not teleop or grasp. CUDA remains the default device.
`--inference-steps N` overrides the checkpoint schedule and must not exceed its
diffusion training steps; changing it can affect policy quality.

Without `--allow-grasp`, the policy preserves the current gripper command,
including a grasp set during teleop. With it, the policy selects open/closed
while active. Pauses, mode switches, and replanning preserve grasp in either
case; returning to teleop does not restore an old keyboard grasp state.

#### Pauses, faults, and logs

Joint-speed and physical joint limits apply to both action sources. Policy
state is compared with expanded training bounds (`--state-margin 0.05`) and
distribution excursions are logged and printed, but do not block execution by
default. Add `--enforce-training-bounds` when an experiment should pause on
those excursions. This distinction lets deployment operate outside the
recorded distribution without weakening finite-value checks or physical limits.

Stale actions (`--action-timeout 0.5` seconds), stale camera frames
(`--max-frame-age 0.15` seconds), and invalid/failed policy predictions pause
policy and retain motor targets. OptiTrack poses older than 0.15 seconds pause
teleop. Guard recovery never restarts motion automatically: release/press SPACE
to retry, or use D and a fresh SPACE press for teleop recovery.

Q, Ctrl+C, and SIGTERM shut down. Motor communication errors, keyboard/control
worker failures, and invalid shared robot state also trigger shutdown because
reliable holding is no longer assured. State remains command-derived, without
measured arrival verification, collision detection, or task-success detection.

Each session creates `deployment_runs/YYYYMMDD-HHMMSS/` (or an unused directory
specified by `--log-dir`) containing:

- `config.json`: checkpoint, connection settings, and deployment options;
- `events.jsonl`: session-level mode changes, pauses, faults, and shutdown;
- `episodes/episode_NNNN/episode.json`: finalized rollout metadata;
- `episodes/episode_NNNN/data.jsonl`: activation, inference, policy, and 100 Hz
  control records for exactly one policy activation;
- `episodes/episode_NNNN/camera.mp4`: frames from that rollout, unless
  `--no-video`.

Holding SPACE in policy mode starts a rollout; releasing it, switching mode, a
guard pause, a fault, or shutdown finalizes that rollout. Videos are opened on
the first rollout frame and closed at its boundary. A video write failure pauses
the rollout rather than continuing with incomplete evidence. Synchronous
replanning can skip camera frames, so use the timestamps in `data.jsonl` when
assessing motion timing. `episode.json` records start/end state, duration, frame
count, termination reason, and `video_complete` status.

The provisional policy-contract decision remains machine-readable in
`giraf.learning.status`:

```text
POLICY_CONTRACT_FINAL = False
```

## Verification

This is a research repo and carries no test suite. Check a change with a
smoke run: lint, then train a tiny policy for one epoch on any collected
dataset and reload the checkpoint.

```bash
uvx ruff check src scripts
uv run giraf-train --dataset data/demos/replay_buffer.zarr \
  --output-dir /tmp/giraf-smoke --epochs 1 --batch-size 8 \
  --down-dims 8 16 --diffusion-steps 4 --device cpu
uv run python -c "from giraf.learning import DiffusionPolicy; \
  DiffusionPolicy.load('/tmp/giraf-smoke/policy.pt', device='cpu')"
uv run --extra hardware giraf-teleop --help
```

Everything hardware-facing needs the robot.
