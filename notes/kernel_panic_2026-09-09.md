# Training host panic, September 9, 2026

Kernel and firmware-persisted panic logs have been collected. Logical CPUs 8–9
are now the strongest candidate for controlled isolation, but physical damage
has not been established.

## Findings from the collected kernel logs

The collection command's line break shortened its destination to
`checkpoints/training_debug/kernel-panic-/`. The journal and pstore archive were
successfully saved before the last command failed because the user's terminal
did not have `rg`. The collector now uses standard `grep -E` instead. Only the
memory-settings export was missing; rerunning the whole collection is unnecessary
for the panic analysis.

Decoded copies of archived `dmesg.txt` files are in that directory's
`pstore-decoded/` subdirectory; original journals and `pstore.tar` are preserved.

- At 12:49:39 Pacific, the kernel took an NX instruction-fetch fault on CPU 9
  in `kworker/9:2`, with workqueue `events psi_avgs_work`. The instruction pointer
  was a non-executable address near its stack. See `previous-kernel.log:1645`.
- About 20 minutes 43 seconds later, CPU 8 faulted in `swapper/8` with an
  instruction pointer of `0x3`; its trace includes the CPU-idle path.
  See `pstore-decoded/1788984622-002.log:67`.
- The subsequent panic was `Attempted to kill the idle task!`.
  See `pstore-decoded/1788984624-003.log:34`.
- The CPU 9 fault happened while training still appeared to progress normally.
  W&B/training progress did not establish kernel health.
- Deduplicating repeated excerpts across all saved pstore `dmesg.txt` records
  yields six distinct CPU-tagged Oops reports: four on CPU 8 and two on CPU 9,
  across records from September 4, 8, and 9. Some are cascading faults within
  the same boot, not six independent panics. The records include BIOS 1801
  and BIOS 2202 and several kernel execution paths.
- Current sysfs topology confirms CPUs 8 and 9 are SMT siblings, socket 0,
  raw physical core ID 16. Earlier GDB also caught a userspace fault on CPU 8.
- No matching MCE/hardware-error or NVIDIA Xid entries were found in the
  exported previous/current kernel journals; no hardware-error record positively
  attributes the cause to the CPU.

This clustering is substantially stronger evidence than the temperature readings
alone. It is consistent with core-specific instability, while memory corruption
from RAM, unstable settings, or kernel/driver software remains possible. A
stock-settings baseline with XMP off should precede conclusions about permanent
damage. A later core-isolation test must account for kernel workers and idle
tasks as well as training; restricting only the training process does not keep
those kernel tasks off the candidate core.

## Preserved training evidence

Run: `checkpoints/src-boom-1m-and-2m-x100/`, started at 12:34:43 Pacific.

- 42 complete epoch records survive in `metrics.jsonl`.
- The rolling `policy.pt` was last modified at 13:09:59.680624 Pacific, after
  epoch 42. The latest numbered checkpoint is `policy_epoch_0040.pt`.
- `best.pt`, `policy.pt`, and numbered checkpoints 20 and 40 passed ZIP member
  CRC checks. This checks archive integrity, not training quality or exact
  reproducibility of a resumed run.
- Last complete temperature sample: 13:10:14.669289 Pacific. Package: 69 C;
  physical core shared by logical CPUs 8–9: 69 C; CPUs 10–11: 42 C.
- Highest sampled package temperature: 92 C at 13:06:34.333290, also on the
  physical core shared by logical CPUs 10–11.
- `train.log`, `metrics.jsonl`, and `cpu_temperatures.jsonl` have NUL-filled
  tails following the last complete records. Originals are left untouched.
  Those tails do not by themselves establish faulty RAM.
- There is no new GDB signal/backtrace report and no `exit_status.txt`.
  The monitor's recorded `state: running` is stale after the reboot; it cannot
  finalize its status when the operating system stops.

GDB monitors userspace signals. It cannot guarantee a trace or final log flush
when the whole kernel panics. The archived pstore records are the next evidence
source, rather than extrapolating from an earlier userspace crash on CPU 8.

## Kernel evidence

Previous boot ID: `8fdd8d1d08df455ebbdffa6f4b0fd073`.
Current boot ID: `32b61ba5-a106-4c94-8dfd-589404cdae97`.

New archived records were found under `/var/lib/systemd/pstore/` in directories
`1788983379`, `1788984622`, `1788984624`, and `1788984625`.
Their contents and the full kernel journal required root access. The user
collected them through sudo; the saved copies above are now readable.

Collect them from a normal terminal in the repository:

```bash
bash scripts/collect_kernel_panic.sh checkpoints/training_debug/kernel-panic-20260909
```

The script prompts through sudo, reads the previous/current kernel journals,
archives saved pstore records, and records memory speeds/voltages reported by
firmware. It does not alter BIOS, drivers, or core availability.

## BIOS changes reported after the panic

The user changed Extreme to Performance and enabled XMP I. The exact previous
CPU profile remains unverified. ASUS lists both Performance and Extreme under
Intel Default Settings separately from ASUS Advanced OC Profile, so the label
Extreme alone does not establish manual CPU overclocking.

XMP I enables a memory overclock. For an initial stock-settings comparison,
use Intel Default Settings with Performance and disable XMP (Ai Overclock Tuner
Auto), keeping CPU ratios/voltages at their defaults. This separates memory
overclock instability from the suspected CPU problem. No new training or stress
test has been started since the panic.

Sources:

- [ASUS profile definitions](https://zentalk.asus.com/t5/faq/motherboard-intel-13th-and-14th-gen-k-series-processor-stability/ta-p/432878)
- [Intel XMP](https://www.intel.com/content/www/us/en/gaming/extreme-memory-profile-xmp.html)
- [Intel instability guidance](https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html)
