# GIRAF Learning

GIRAF Learning contains the robot drivers, teleoperation and demonstration
collection stack, and a compact PyTorch diffusion-policy foundation. Hardware,
data, and learning code are separate packages with explicit boundaries.

See [IMU collection](notes/imu_collection.md) for the v2 recording schema,
signal semantics, compatibility, hardware checks, and known device limitations.

## Repository layout

```text
.
├── config/data_collection.yaml   # collection settings
├── src/giraf/
│   ├── data/                     # capture, alignment, Zarr storage, replay
│   ├── deployment/               # guarded live policy trials and rollout logs
│   ├── drivers/                  # camera, input, OptiTrack, and motor adapters
│   ├── learning/                 # policy/environment contracts and loops
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
wheels). Tests and all tooling also run on CPU. Add `--extra train` for
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

All data settings live in `config/data_collection.yaml`. Fixed action and state
fields live in `giraf.data.schema`; diffusion shape metadata is derived from
those fields and the resolved collector configuration:

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
actions                     float32 [B, 16, 7]  (m/s, rad/s, grasp in {0, 1})
```

A `Normalizer` fitted on the dataset maps actions and states to `[-1, 1]`
inside the policy and is stored in the checkpoint, so `act()` returns
denormalized actions `[vx, vy, vz, wx, wy, wz, grasp]`. RGB scaling also
happens inside the policy. `act()` keeps its own two-frame observation
history; `DiffusionPolicy.reset()` clears it at an episode boundary and
`rollout()` calls it when a policy provides it. The required policy contract
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

Run the deployment prototype with `python -m giraf.deployment`; there is no
`giraf-deploy` entry point. It loads one checkpoint and runs a bounded trial
using live camera RGB and command-derived robot state.

**Compatibility:** `conv`, `resnet18`, and `dinov2` checkpoints use the shared
policy loader and inference path. The saved encoder, center crop, image/state
normalization, EMA weights when present, action horizon, and temporal ensembling
are applied automatically. Training-time color jitter is disabled. DINOv2
initialization uses `torch.hub`, so its backbone code and pretrained weights
must be cached or downloadable on the deployment machine. Checkpoints must
contain a normalizer and use `action_space="twist"`; `joint_position` policies
are explicitly rejected before motors are opened.

#### Prepare and check command generation

Run from the repository root. Sync the environment after pulling learning-code
changes; the encoders require `torchvision`, including when loading an older
conv checkpoint:

```bash
uv sync --frozen --extra hardware
uv run --frozen --extra hardware python -m giraf.deployment --help
```

All modes use the live camera and Linux keyboard input. Stop teleop and any
other process using those devices before launching deployment. OptiTrack is
not used by the standalone deployment runner.

| Mode | Behavior |
| --- | --- |
| `shadow` (default) | Reports policy outputs with a fixed logical joint pose; motors are not opened. |
| `dry-run` | Integrates bounded commands into a simulated joint pose; motors are not opened. |
| `hardware` | Connects motors, stages to the recorded start, and executes bounded policy commands. |

Dry-run uses live images even as the simulated joint pose changes. It checks
command generation and limits; it does not simulate the visual consequences of
motion or evaluate task success.

For the September ResNet checkpoint, begin with:

```bash
uv run --frozen --extra hardware python -m giraf.deployment \
  --checkpoint checkpoints/tape_grasping/nautilus-policy-v2-resnet/best.pt \
  --reference-dataset data/tape_grasping/sept03_trials.zarr \
  --reference-episode 16 \
  --config config/tape_grasping.yaml \
  --device cuda \
  --mode dry-run \
  --action-scale 0.2 \
  --duration 5
```

The runner prints a no-motion inference preview. Release SPACE to arm the
trial, then hold SPACE to run it. Releasing SPACE during the trial stops the
program; `Ctrl-C` also stops it. Use `--mode shadow` for a fixed logical pose.

Replace `--checkpoint` to try another model. `best.pt` is selected by validation
action MSE; `policy.pt` is the latest saved model, and `policy_epoch_NNNN.pt`
selects a particular epoch. Choose a zero-based `--reference-episode` from a
dataset for the same task and a scene you can reproduce. The runner uses that
episode's first recorded joint pose and saves its image for comparison; policy
inference uses the live image. `--config` supplies camera settings and the
dataset RGB resize, which should match training (224 x 224 here). The runner
checks the reference image shape against this configuration.

These commands retain the checkpoint's inference schedule: this ResNet model
uses 16 inference steps. Its 100 diffusion training steps are a separate
setting. `--inference-steps N` overrides inference only and must not exceed the
checkpoint's diffusion steps. Inspect preview and rollout latency before
changing it; an override can change policy quality. The older conv trial used
an explicit 10-step override, which is not a requirement for newer models.

#### Run on the robot

Before launching hardware mode, put the physical robot at the same teleop home
pose used for collection. The MAB driver zeros its encoders on connection, and
the runner initializes its commanded joints to `[0, 0, 0.31, 0, 0, 0]`. Starting
at another physical pose would give the commands the wrong coordinate frame.
Position the camera, object, and surroundings like the selected demonstration.
Keep the physical kill switch reachable and remain at the terminal during motion.

```bash
uv run --frozen --extra hardware python -m giraf.deployment \
  --checkpoint checkpoints/tape_grasping/nautilus-policy-v2-resnet/best.pt \
  --reference-dataset data/tape_grasping/sept03_trials.zarr \
  --reference-episode 16 \
  --config config/tape_grasping.yaml \
  --device cuda \
  --mode hardware \
  --action-scale 0.2 \
  --duration 5 \
  --confirm-hardware
```

1. Begin with SPACE released. Motor connections initialize the home targets.
2. Hold SPACE continuously to stage slowly to the recorded start. Releasing
   before staging completes stops the program and closes the motor interfaces.
3. When staging completes, release SPACE. Review the printed raw action,
   guarded action, joint velocity, and inference latency from the preview.
4. Hold SPACE again to start the rollout. The five-second duration starts here,
   after staging and preview. Release SPACE or press `Ctrl-C` to stop.

The default `--action-scale 0.2` scales the six twist channels and their hard
velocity ceilings to 20%; it does not scale staging speeds or grasp. Increase
the scale or `--duration` deliberately after reviewing a short trial. CUDA is
the default device; `--device cpu` is available, but inference must still meet
the configured freshness limits.

Grasp defaults to an **open command**, not an unpowered gripper. Add
`--allow-grasp` to let the policy select open/closed after reviewing arm motion.
There is a current limitation: each synchronous replan temporarily sets all
seven action channels to zero. This holds the commanded arm pose during
inference and also commands the gripper open, even with `--allow-grasp`.
Brief holds can coexist with visually smooth motion. The command interruption
does not establish how much the gripper physically moves or explain a failed
manipulation on its own; inspect it when evaluating grasp behavior.

#### Logs, limits, and handoff

Each invocation creates `deployment_runs/YYYYMMDD-HHMMSS/` containing:

- `config.json`: checkpoint path and deployment options;
- `events.jsonl`: staging, preview, policy outputs, inference latency, control
  commands, and the final stop reason;
- `camera.mp4`: live camera video, unless `--no-video` is set;
- `reference_start.png`: the selected episode's starting image.

Use `--log-dir PATH` for a custom, unused run directory. To review a run:

```bash
ls -lt deployment_runs
tail -n 30 deployment_runs/YYYYMMDD-HHMMSS/events.jsonl
```

The executor clamps twists, caps joint speeds, enforces joint limits, and
checks rollout state against checkpoint training bounds (`--state-margin 0.05`
by default). It stops on stale actions (`--action-timeout 0.5` seconds), stale
frames (`--max-frame-age 0.15` seconds), deadman release, or the duration limit.
State and staging completion are command-derived; measured joint arrival is
not verified. There is no collision detection, contact sensing, or automatic
task-success evaluation. Retain the run directory when reporting trial results.

Matching a recorded start is a requirement of this prototype's initialization,
not of diffusion policies in general. It supplies a known commanded pose and a
familiar starting scene. Direct teleop/policy handoff is not implemented:
releasing SPACE ends this runner rather than returning control to teleop.
A future shared session could preserve motor calibration and joint/gripper
state while switching action sources, resetting policy history, and
re-anchoring teleop. Until then, each hardware launch requires physical home
and staging again.

### MuJoCo drop-in

Automated simulation evaluation still requires a MuJoCo backend.

Pass the existing Gym/Gymnasium-style MuJoCo environment directly to the
adapter:

```python
from giraf.learning import SimEnvironment, rollout

environment = SimEnvironment(mujoco_environment)
summary = rollout(policy, environment, max_steps=500, seed=0)
environment.close()
```

The backend must expose `reset(seed=...)`, `step(action)`, and `close()`, and
must emit observation mappings containing `camera_rgb` and `state`. Both the
five-value Gymnasium step result and legacy four-value Gym result are accepted.

## Verification

This is a research repo and carries no test suite. Check a change with a
smoke run: lint, then train a tiny policy for one epoch on any collected
dataset and reload the checkpoint.

```bash
uvx ruff check src
uv run giraf-train --dataset data/demos/replay_buffer.zarr \
  --output-dir /tmp/giraf-smoke --epochs 1 --batch-size 8 \
  --down-dims 8 16 --diffusion-steps 4 --device cpu
uv run python -c "from giraf.learning import DiffusionPolicy; \
  DiffusionPolicy.load('/tmp/giraf-smoke/best.pt', device='cpu')"
uv run --extra hardware giraf-teleop --help
```

Everything hardware-facing needs the robot.
