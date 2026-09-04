# GIRAF Learning

GIRAF Learning contains the robot drivers, teleoperation and demonstration
collection stack, and a compact PyTorch diffusion-policy foundation. Hardware,
data, and learning code are separate packages with explicit boundaries.

## Repository layout

```text
.
├── config/data_collection.yaml   # collection settings
├── src/giraf/
│   ├── data/                     # capture, alignment, Zarr storage, replay
│   ├── drivers/                  # camera, input, OptiTrack, and motor adapters
│   ├── learning/                 # policy/environment contracts and loops
│   ├── viewer/                   # read-only local dataset web viewer
│   ├── kinematics.py             # GIRAF kinematic model
│   ├── settings.py               # shared runtime constants
│   └── teleop.py                 # teleoperation application
├── tests/                        # non-hardware unit tests (ring buffer, saver)
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

- a small RGB encoder conditions a temporal 1D U-Net together with robot state;
- training predicts noise added by a cosine-beta DDPM scheduler;
- inference denoises a 16-step sequence and executes an 8-step action chunk;
- checkpoints contain model state, optimizer state, configuration, and format
  version and are written atomically.

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

```python
from giraf.learning import DiffusionPolicy, ReplayDataset, train

dataset = ReplayDataset("data/demos/replay_buffer.zarr", batch_size=64)
policy = DiffusionPolicy(normalizer=dataset.fit_normalizer())
history = train(policy, dataset, epochs=1)
policy = DiffusionPolicy.load("checkpoints/tape_grasping/policy.pt", device="cpu")
```

Still missing: evaluation rollouts, which need the MuJoCo simulator backend.

### MuJoCo drop-in

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

The unit tests cover the non-hardware parts that are easy to get wrong: ring
buffer overrun detection, causal alignment, staging, and the saver process.
Everything else needs the robot.

```bash
uv run python -m unittest discover tests
uvx ruff check src tests
uv run --extra hardware giraf-teleop --help
```
