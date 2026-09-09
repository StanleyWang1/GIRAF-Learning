# SRC-03 training crash investigation

Status: investigation in progress; no faulty CPU core identified and no cores
disabled. A passing short test does not establish hardware health.

## Active production run

The user-requested run `src-boom-1m-and-2m-x100` was started on September 9 at
12:34 Pacific time in detached tmux session `giraf-src-boom-monitor`.
It uses the requested 500 epochs, LR floor ratio 0.5, checkpoint interval 20,
and online W&B, with all logical CPUs available. Other requested training
arguments are recorded in the run's `config.json` and `monitor_status.json`.

`scripts/monitor_training.py` forwards the trainer arguments unchanged and runs
the trainer under `scripts/train_diagnostics.gdb`. The run directory is
`checkpoints/src-boom-1m-and-2m-x100/`:

- `train.log`: trainer output, GDB native stacks, signal, faulting thread's last
  logical CPU, raw physical core ID, SMT siblings, registers, instructions,
  shared libraries, and a temperature snapshot on a stopped signal.
- `cpu_temperatures.jsonl`: CPU temperature readings every two seconds.
- `monitor_status.json`: command, process IDs, affinity, start/end time, status.
- `exit_status.txt`: written when training ends; SIGSEGV maps to 139.

W&B console interception is disabled while online metric logging remains enabled.
GDB fault reporting was verified with a deliberately signaled test process;
normal exit 0 and error exit 7 were also preserved. The complete monitor passed
an integration check using the trainer's `--help` command. These are launcher
checks, not evidence that training or the CPU is stable.

```bash
tmux attach -t giraf-src-boom-monitor
tail -f checkpoints/src-boom-1m-and-2m-x100/train.log
```

## Observations

- Intel Core i9-14900KS; BIOS reports 2202; runtime microcode is 0x133.
- NVIDIA RTX 4090; driver 595.84; PyTorch 2.14.0+cu130; cuDNN 92400.
- `checkpoints/r1-src-boom-1m-2m-x100/train.log` records SIGSEGV during backward
  before any epoch completed. The directory contains no training checkpoint.
- An earlier run, `checkpoints/src-03-boom-1m-2m-x100`, completed 37 epochs before
  a similar crash. Therefore a three-epoch pass is only a screen.
- An isolated 221-batch training pass succeeded at batch size 256 with image
  preloading; peak PyTorch allocated GPU memory was about 9.6 GiB.
- Full training under GDB with default cuDNN and offline W&B crashed early in
  epoch 2. The fault was in `libcuda.so.1`, reached through `cuLaunchKernel`
  from cuDNN convolution backward.
- `TORCH_CUDNN_V8_API_DISABLED=1` also crashed after epoch 1, in `libcuda.so.1`
  while launching a PyTorch tensor addition. This is not a verified workaround.
- E-core-only affinity (`taskset -c 16-31`) completed three epochs normally,
  including validation and checkpoint saving, at about 53.6 seconds per epoch.
- P-core-only affinity (`taskset -c 0-15`) completed three epochs normally at
  about 48.8 seconds per epoch.
- Both short GDB sessions printed `No stack` after the inferior exited normally;
  that debugger command error was not a training crash.

## Longer test

An unrestricted diagnostic targeting 60 epochs ran with the same dataset,
ResNet-18 encoder, batch size 256, seed 0, validation split, and preloading.
Its scheduler uses a 60-epoch target instead of the production 500-epoch target;
these are diagnostic checkpoints, not an exact continuation of production.

It completed 14 epochs and crashed in epoch 15, in `libcuda.so.1` while launching
a GroupNorm backward kernel. GDB reported logical CPU 8 for the stopped thread,
raw physical core ID 16, with SMT siblings 8–9. This is a candidate for targeted
retesting, not proof of a damaged core. The temperature monitor collected 214
samples, with a maximum sampled temperature of 78 C across CPU sensors.

- GDB log: `/tmp/giraf-train-allcores-long-gdb.log`
- Metrics and checkpoints: `/tmp/giraf-train-allcores-long-20260909/`
- CPU temperature samples every two seconds (started after epoch 5):
  `checkpoints/training_debug/cpu-affinity-20260909/allcores-temperatures.jsonl`
- GDB records the selected faulting thread's last logical CPU, its allowed CPU
  list, and a native backtrace if it stops on a signal. The last CPU is a clue,
  not proof that corruption originated on that core.

Earlier native backtraces are saved under
`checkpoints/training_debug/r1-native-crash-20260908/` and
`checkpoints/training_debug/cpu-affinity-20260909/`.

## CPU numbering

Use sysfs topology to map kernel CPU IDs and temperature labels. `lscpu`'s
compact core numbering differs from the raw core IDs on this machine.

| Logical CPUs | Raw core ID / coretemp label | Type |
| --- | --- | --- |
| 0–1 | 0 | P |
| 2–3 | 4 | P |
| 4–5 | 8 | P |
| 6–7 | 12 | P |
| 8–9 | 16 | P |
| 10–11 | 20 | P |
| 12–13 | 24 | P |
| 14–15 | 28 | P |
| 16–31 | 32–47, respectively | E |

Both SMT siblings must be excluded from a process's affinity mask when testing
avoidance of a physical P-core. Process affinity does not offline cores systemwide.

## Remaining evidence

Kernel logs require sudo, and `sudo -n` reports that a password is required.
The user has been asked to run:

```bash
sudo journalctl -k --since '2026-09-03' --no-pager -o short-iso \
  --grep='segfault|trap|[Hh]ardware [Ee]rror|[Mm]achine [Cc]heck|mce:|EDAC|NVRM|Xid|thermal|throttl'
```

Also confirm Intel Default Settings and XMP disabled after the BIOS update.
Firmware and microcode versions alone do not establish those settings.

A CPU-only correctness sweep in
`checkpoints/training_debug/cpu-affinity-20260909/giraf_cpu_sweep.py` completed
ten seconds on each of 24 physical cores with no detected correctness failures,
including CPU 8. It pins one worker at a time to one logical CPU per physical core,
checks matrix products against exact integer arithmetic and compression/hash
round trips, and samples temperatures. It is a short custom screen, not a
replacement for extended CPU and memory testing. Keep it separate from training
comparisons to avoid changing load and thermal conditions.

## Interpretation and sources

CPU degradation is plausible, but neither a segfault location nor temperature
alone identifies a damaged core. Repeatable affinity-specific failures and
decoded machine-check records would be stronger evidence. RAM, BIOS settings,
and software remain possible causes.

Intel describes reliability aging of a core clock-tree circuit under elevated
voltage and temperature in its
[root-cause statement](https://community.intel.com/t5/Blogs/Tech-Innovation/Client/Intel-Core-13th-and-14th-Gen-Desktop-Instability-Root-Cause/post/1638315).
Its [current support guidance](https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html)
recommends Intel Default Settings and BIOS with microcode 0x12F or later and
describes extended warranty coverage for eligible processors.
