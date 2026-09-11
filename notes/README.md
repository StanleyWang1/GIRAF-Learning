# Notes

Setup guides, testing reports, and hardware investigations, grouped by topic.
Commands and paths inside the notes assume the repository root unless stated
otherwise. Dated reports describe findings at the time they were written.

## Training

- [Training setup](training/training_setup.md): NVIDIA/CUDA, W&B, tmux, and
  starting and monitoring training runs.

## IMU collection and testing

- [IMU collection](imu/imu_collection.md): recording schema, sensor signals,
  compatibility, and hardware checks.
- [Health audit — 2026-09-08](imu/imu_test_health_2026-09-08.md): recording
  integrity, timing, and follow-ups.
- Audit data: [detailed results](imu/imu_test_health_2026-09-08.json) and
  [core structural and timing results](imu/imu_test_core_health_2026-09-08.json).

## SRC-03 hardware troubleshooting

- [CPU instability investigation — 2026-09-09](hardware/src-03/cpu_instability_investigation_2026-09-09.md):
  training crashes, diagnostics, and observations.
- [Kernel panic investigation — 2026-09-09](hardware/src-03/kernel_panic_2026-09-09.md):
  kernel logs, CPU topology, and preserved training evidence.
- [BIOS update — 2026-09-07](hardware/src-03/bios-update/README.md): update
  procedure and provenance, with the original firmware archive, extracted
  firmware, and USB preparation script kept together in the same folder.

Keep future training guides in `training/`, IMU guides and reports in `imu/`,
and machine-specific investigations in `hardware/<hostname>/`. Keep supporting
files beside their reports and retain dates on investigation notes.
