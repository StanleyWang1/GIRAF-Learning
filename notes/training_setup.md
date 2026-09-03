# GIRAF training setup

This guide sets up NVIDIA/CUDA support, Weights & Biases (W&B), and `tmux`,
then starts and monitors a GIRAF training run. Run commands from the repository
root:

```bash
cd ~/Documents/GIRAF-Learning
```

Useful references:

- [uv documentation](https://docs.astral.sh/uv/)
- [Ubuntu NVIDIA driver installation](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [W&B quickstart](https://docs.wandb.ai/models/quickstart)
- [W&B environment variables](https://docs.wandb.ai/models/track/environment-variables)
- [tmux getting started](https://github.com/tmux/tmux/wiki/Getting-Started)

## 1. Install and verify the NVIDIA driver

Install Ubuntu's recommended NVIDIA driver and `tmux`:

```bash
sudo apt update
sudo ubuntu-drivers install
sudo apt install tmux
```

If the NVIDIA installation fails because files such as `module.lds` or
`tools/objtool/objtool` are missing for kernel `7.0.0-31`, install the matching
headers and finish configuring the packages:

```bash
sudo apt install \
  linux-headers-7.0.0-31-generic \
  linux-headers-generic-hwe-24.04

sudo dpkg --configure -a
sudo apt --fix-broken install
```

Verify that package configuration is clean and the NVIDIA module exists:

```bash
sudo dpkg --audit
modinfo -k 7.0.0-31-generic nvidia | head
```

`dpkg --audit` should print nothing. Reboot only after the installation has
finished successfully:

```bash
sudo reboot
```

After rebooting, check the active kernel, driver, GPU, and PyTorch CUDA access:

```bash
uname -r
nvidia-smi
uv run --frozen python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

On the current training machine, the expected results are:

```text
kernel:          7.0.0-31-generic
NVIDIA driver:  595.84
GPU:             NVIDIA GeForce RTX 4090
PyTorch:         2.14.0+cu130
CUDA available:  True
```

The CUDA version displayed by `nvidia-smi` is the newest CUDA version supported
by the installed driver. It can be newer than the CUDA runtime bundled with
PyTorch; driver 595 is compatible with the project's CUDA 13.0 PyTorch wheel.

Do not remove older kernels with `apt autoremove` until the new kernel and
NVIDIA driver have booted successfully.

## 2. Install the project and W&B

Sync the locked project environment with both training and robot-hardware
extras. Including both extras prevents a sync from dropping hardware-only
packages from the environment:

```bash
uv sync --frozen --extra train --extra hardware
```

Confirm the W&B client is installed:

```bash
uv run --frozen python -c "import wandb; print(wandb.__version__)"
```

## 3. Create and authenticate a W&B account

Create or sign into an account at [wandb.ai](https://wandb.ai/), then run:

```bash
uv run --frozen wandb login
```

Follow the URL shown by the command and paste the API key into the terminal
prompt. Do not place the API key in this repository, a training command, shell
history, or chat message. Confirm the saved configuration:

```bash
uv run --frozen wandb status
```

The GIRAF trainer intentionally uses W&B's offline mode unless
`WANDB_MODE=online` is set. The launch command below sets online mode so metrics
appear on the W&B website during training.

## 4. Run preflight checks

Check storage, dataset size, and current GPU use:

```bash
df -h .
du -sh data/tape_grasping/stanley_trials_cleaned.zarr
nvidia-smi
```

For this dataset, `--preload-images` holds approximately 3.1 GiB of decoded
images in system RAM. The numbered checkpoints plus the rolling `policy.pt`
should use approximately 1.4 GiB with the settings below.

Check whether the intended run directory already exists:

```bash
if test -e checkpoints/tape_grasping; then
  echo "Run directory already exists; choose a new --output-dir"
else
  echo "Run directory is available"
fi
```

The trainer does not currently resume an interrupted run. Reusing an existing
directory restarts at epoch 1, appends to `metrics.jsonl`, and can mix files
from separate attempts. Prefer a new output directory for each restart.

## 5. Start training in tmux

Create a named `tmux` session:

```bash
tmux new -s giraf-train
```

Inside the new session, run:

```bash
WANDB_MODE=online \
WANDB_NAME=tape-grasping-200ep \
uv run --frozen giraf-train \
  --dataset data/tape_grasping/stanley_trials_cleaned.zarr \
  --output-dir checkpoints/tape_grasping \
  --epochs 200 \
  --batch-size 64 \
  --checkpoint-every 10 \
  --device cuda \
  --preload-images \
  --wandb \
  --wandb-project giraf-tape-grasping
```

For a W&B team project, set the team account inside the `tmux` session before
running the launch command:

```bash
export WANDB_ENTITY=your-team-name
```

A healthy startup prints a line resembling:

```text
[TRAIN] 21962 windows, 344 batches/epoch, device=cuda
```

Using `--device cuda` is deliberate: it makes the command fail clearly if CUDA
is unavailable instead of silently training on the CPU.

## 6. Detach from and reconnect to tmux

To detach while leaving training running, press `Ctrl-B`, release the keys,
then press `D`.

List sessions and reconnect later:

```bash
tmux ls
tmux attach -t giraf-train
```

`tmux` protects the process from a closed terminal or dropped SSH connection.
It does not survive a reboot and does not prevent system suspension. Disable
automatic sleep while training.

## 7. Monitor training

From a second terminal, monitor GPU utilization, temperature, and VRAM:

```bash
watch -n 2 nvidia-smi
```

Follow the local per-epoch metrics:

```bash
tail -f checkpoints/tape_grasping/metrics.jsonl
```

Inspect recent terminal output without attaching interactively:

```bash
tmux capture-pane -p -S -50 -t giraf-train
```

The W&B run URL is printed near startup. The current trainer sends these values
once per epoch:

- `loss`
- `gradient_norm`
- `epoch_seconds`

It also records the complete training configuration. Model checkpoints are not
uploaded as W&B artifacts; they stay in the output directory:

```text
checkpoints/tape_grasping/
├── config.json
├── normalizer.json
├── metrics.jsonl
├── policy.pt
└── policy_epoch_NNNN.pt
```

If GPU utilization remains near zero after training starts, first confirm that
the launch line said `device=cuda`. Short dips can occur while image batches are
prepared. If CUDA reports an out-of-memory error, start a new run directory and
reduce `--batch-size` from `64` to `32`.

## 8. Stop training safely

Reconnect to the session and interrupt the trainer:

```bash
tmux attach -t giraf-train
```

Press `Ctrl-C` once and wait for the shell prompt. `policy.pt` contains the last
fully completed epoch because checkpoints are written atomically after each
epoch. The current partial epoch is not recoverable.

## 9. Optional offline W&B workflow

To train without uploading live, omit `WANDB_MODE=online` but retain `--wandb`.
The trainer stores a local offline W&B run beneath the output directory. Upload
it later with:

```bash
uv run --frozen wandb sync checkpoints/tape_grasping/wandb/offline-run-*
```

See W&B's guide to [offline runs and later syncing](https://docs.wandb.ai/support/models/articles/what-happens-if-internet-connection-is-l).
