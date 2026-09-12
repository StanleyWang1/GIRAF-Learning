# IMU collection

Schema `giraf-replay-v2` records IMU without changing the RGB + 15D state inputs
used by existing training and deployment. Set `imu.enabled: true` in collection
YAML (enabled in both supplied configs). Use a fresh output Zarr: the writer
refuses to append v2 episodes to v1 datasets. Existing v1 datasets remain readable.

## Signals

The BNO086 shares the existing DepthAI device pipeline with RGB. A separate host
thread drains IMU so image acquisition never waits for an IMU report.
Camera alignment runs in its own process, separately from teleoperation and the
saver. A teleop thread/GIL stall therefore cannot prevent it draining images.

| Report | Values | Meaning |
| --- | --- | --- |
| `ACCELEROMETER_CALIBRATED` | XYZ float32, m/s² | Includes gravity; DepthAI IMU frame with stored affine calibration |
| `GYROSCOPE_CALIBRATED` | XYZ float32, rad/s | DepthAI IMU frame with stored affine calibration |
| `GAME_ROTATION_VECTOR` | XYZW float32 (`i,j,k,real`) | Native fused game quaternion; gravity-referenced tilt, arbitrary heading with yaw drift |

There is no host filtering, averaging, gravity subtraction, quaternion sign
adjustment, or robot/world transformation. Do not assume the quaternion's native
frame matches the calibrated vector frame. Session metadata preserves the SDK's
extrinsics and calibration for future conversion. Identity affine matrices do
not establish a custom calibration. Controlled physical-axis verification remains
a manual check; metadata explicitly states it was not performed automatically.

All reports request 100 Hz. On the tested OAK-D-S2-AF/BNO086, firmware 3.9.9 and
DepthAI 3.9.0, the combined packet stream delivers approximately 100 Hz per report.
Acceleration counters advance at approximately 125 Hz: roughly one in five
generated acceleration reports is absent from delivered packets. DepthAI permits
only one IMU node on this device. This observed device/SDK limitation is not a
host saver overrun. We retain **every delivered report**, with its sequence gaps;
this is not a lossless archive of the internal 125 Hz acceleration output.
Reformat using timestamps, never row number or an assumed uniform 10 ms period.

## Zarr layout

`data/` keeps the image-step time axis. Additional arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `imu` | `[T,10]` | acceleration XYZ, gyro XYZ, quaternion XYZW |
| `imu_timestamp_ns` | `[T,3]` | selected report generation time |
| `imu_sequence_num` | `[T,3]` | selected report identity |
| `imu_age_ns` | `[T,3]` | camera capture minus generation time |
| `imu_available` | `[T,3]` | causal report found |
| `imu_sensor_valid` | `[T,3]` | available, fresh and valid report |
| `imu_valid` | `[T]` | all three sensor-valid flags true |

Three-column arrays use accelerometer, gyro, game rotation order; root attributes
`imu_fields` and `imu_sensor_order` explicitly define the columns.
`max_imu_age_ms` is stored and defaults to 25 ms. Valid reports must be finite;
quaternion norm must be within 0.9–1.1. Values are never renormalized.
Sensor accuracy is saved separately and does not gate freshness.

Each image selects the latest already-received report generated at or before
capture. Missing reports have NaN values, -1 timestamp/sequence/age, and zero
availability/validity. Stale reports retain values and age with zero validity.
Existing `alignment_valid` semantics and training sample selection are unchanged.

Independent full-rate groups are `imu/accelerometer`, `imu/gyroscope`, and
`imu/game_rotation_vector`. Each stores:

- `values`, `timestamp_ns`, `device_timestamp_ns`, `receive_timestamp_ns`;
- `sequence_num`, `sequence_gap`, `report_valid`, `producer_dropped_total`;
- `accuracy` (-1 unavailable, 0 unreliable, 1 low, 2 medium, 3 high) and
  `rotation_accuracy_rad` (SDK field where exposed, NaN when absent).

Generation/receive timestamps are int64 nanoseconds in the host monotonic clock;
device timestamps retain the device clock. A rotation accuracy field does not
establish absolute-heading accuracy for the game quaternion.

Per-stream `episode_ends` delimit reports independently from the image commit
marker `meta/episode_ends`. An episode includes up to one second of available
pre-roll and reports through its stop timestamp. Stop allows a bounded one-second
grace period for pending reports; reports beyond stop are trimmed before commit.
If a delayed stop request arrives after frames were already saved, the aligner
acknowledges a boundary covering those frames. Episode JSON records both
`requested_stop_monotonic_ns` and the actual `stop_monotonic_ns`, so a delayed
request does not leave the video, aligned arrays and raw IMU with different ends.

Per-stream `episode_saver_dropped` and `episode_producer_dropped` record detected
host buffer loss. Producer loss covers episode start through final draining;
raw cumulative counters also permit inspection of retained pre-roll. Sequence
gaps are counter-inferred missing reports, without attribution to a particular
transport stage. The unknown prefix before the first report and unreceived
reports after the final report cannot be counted from sequences.

`meta/episode_stop_monotonic_ns`, `meta/episode_imu_valid_steps`, and
`meta/episode_imu_session_id` record boundaries and quality. Session IDs refer to
root attribute `imu_sessions` containing device identity, firmware/SDK, USB,
calibration/extrinsics, requested rate, and frame conventions. Episode JSON
artifacts also include measured report rates and loss summaries.

## Failures, compatibility, and later use

Enabled startup requires the supported device, available calibration parameters,
verified host clock mapping, and valid reports from all three sensors. Recording
starts only while reports are recent. Later IMU gaps do not reject images or
episodes. Device/process or storage failure retains existing error handling.
Control/motor updates that become stale during a teleop pause still invalidate
`alignment_valid`; the system does not fabricate fresh robot state. Camera/IMU
continue recording through such pauses. Genuine overwritten frames during an
active episode still reject it. Unrecorded frames from idle time or slow commits
are discarded without a false overrun, and catch-up copies use bounded batches.
Disabled collection uses the same schema with empty full-rate streams and missing
aligned values.

All streams are staged before commit. Raw offsets are appended before the image
commit marker; recovery truncates uncommitted tails. Rejected recordings retain
staged IMU. Pruning and cleaning preserve reports/pre-roll inside each original
episode and rebuild offsets. For split episodes, host loss counts conservatively
refer to the source episode because exact loss locations are unavailable.

Future preprocessing can regenerate 30 Hz observations with causal filtering or
different freshness thresholds. IMU-aware training should filter whole observation
windows requiring invalid IMU, without deleting rows and joining across gaps.
Online filtering must match training. Dedicated viewer indicators and policy
conditioning are deferred. The generic viewer accepts v2 and renders missing
numeric values as gaps.

## Verification

Hardware-free source check:

```bash
uvx ruff check src scripts
```

Explicit live comparison (never opens motors):

```bash
uv run --extra hardware python scripts/imu_hardware_smoke.py --seconds 15
```

Add `--parent-stall-s 3` to reproduce a three-second teleop-process stall while
the camera, alignment, and saver processes continue independently. The smoke
test never opens motors, including when a stall is requested.

This writes RGB, video and synthetic control into fresh `/tmp` folders, first
without IMU, then with IMU. These artifacts are **not demonstrations**. Results
include image rate/gaps/latency, IMU rate/gaps/validity, and host loss counters.

An initial 15-second comparison on 2026-09-07 saved 450 images in each case at
30.01 Hz with zero image gaps and zero host IMU drops. Median image receive
latency was 30.26 ms without IMU and 30.55 ms with IMU; p99 was 31.12 vs 31.32 ms.
All 450 aligned IMU observations were valid. Quaternion norms ranged from
0.999966 to 1.000034. These are stationary checks with synthetic control, not
robot-motion or long-duration load validation.

After isolating alignment from teleop, a live 15-second test with an injected
three-second parent-process GIL stall saved all 450 images at 30.01 Hz with zero
image gaps or host IMU losses. All 450 IMU observations remained valid. The 89
observations with stale synthetic control were correctly marked alignment-invalid
and preserved. Earlier regression checks also covered a slow commit followed by
a second episode, bounded catch-up copies, and rejection of genuine
active-episode loss.
