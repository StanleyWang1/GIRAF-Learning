# IMU collection health audit — 2026-09-08

Audited `data/imu_test/imu_test.zarr` and all three corresponding videos in
`data/imu_test/imu_test_videos`. Dataset and videos were read only. This report
describes the three committed episodes at offsets `[1539, 2175, 2802]`, not the
earlier preserved rejected episode. Recording integrity looks good; physical IMU
accuracy and long-session reliability are not fully established by these trials.

| Episode (zero-based) | Recording duration | Images | Camera rate | Control/motor valid | IMU valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 51.31 s | 1,539 | 30.010 Hz | 1,532 (99.55%) | 1,539 (100%) |
| 1 | 21.22 s | 636 | 30.010 Hz | 627 (98.58%) | 636 (100%) |
| 2 | 20.95 s | 627 | 30.010 Hz | 620 (98.88%) | 627 (100%) |

## Recording integrity

- All 2,802 camera frames are contiguous: no sequence gaps, reversals, duplicate
  sequence numbers, or identical adjacent stored images. Frame intervals are
  approximately 33.322 ms. Maximum observed camera receive latency is 32.47 ms.
- Every video frame was decoded. Counts exactly match Zarr, and video timestamps
  increase. Decoded RGB frames match the corresponding Zarr images within expected
  lossy compression differences (median mean absolute pixel error 1.61–1.66/255).
  Visual spot checks show the gripper and objects without obvious image corruption;
  this is not a demonstration-success or task-quality review.
- No missing data/metadata chunks, malformed episode boundaries, nonfinite
  state/actions, or mismatches between stored validity and freshness rules.
- All aligned IMU values, sequence numbers, and timestamps match their saved raw
  source reports exactly. All ages recompute correctly; no future reports were used.
- Each raw sensor stream contains 9,652 delivered reports across the three trials,
  including approximately one second of preroll per trial. Raw timestamps increase,
  stop bounds are respected, and producer/saver loss counters are zero. Raw stream
  shapes, episode offsets, and chunks are consistent.
- Worst IMU ages at camera timestamps: accelerometer 15.755 ms, gyro 10.540 ms,
  rotation vector 10.351 ms, all below the configured 25 ms limit. Quaternion norms
  range from 0.999960 to 1.000043.

## Qualifications and follow-ups

**Control/motor freshness:** 23/2,802 rows (0.82%) exceed the 50 ms limit: nine for
control and fourteen for motor-command age. Each invalid run is one frame long.
Maximum control age is 69.48 ms; maximum motor age is 81.51 ms. These rows are
correctly retained with `alignment_valid=0`. No motor-command rejection was found.
The rest of the recordings remain usable; no reason to discard these episodes was
found in the integrity audit.

**Training windows:** `src/giraf/learning/dataset.py:episode_windows` currently
filters only the anchor's validity. With the default two observations and sixteen
predicted actions, the 2,779 accepted windows include 23 with flagged observation
history and 332 with flagged action rows (these categories overlap). This does not
prove every such target is wrong, but it means anchor filtering does not exclude
all stale rows. Before training, enforce validity across required observation and
action indices. Preserve time spacing rather than deleting individual source rows.
The saved flags allow this correction without recollection. This loader behavior
predates the IMU changes. Existing training does not consume IMU.

**Delivered IMU rate:** all three reports arrive at approximately 100.06 Hz.
Accelerometer sequence gaps total 2,378, consistent with the previously observed
internal approximately 125 Hz counter and approximately 100 Hz combined output.
Acceleration intervals are generally approximately 8 or 16 ms, not uniform 10 ms.
Gyro and rotation-vector sequences have no gaps. Zero host loss counters do not
mean lossless capture of every internally generated accelerometer report. Use saved
timestamps for future resampling.

**Accuracy status:** rotation-vector accuracy is 3 (HIGH) throughout; accelerometer
and gyro accuracy are 0 throughout. The SDK defines 0 as UNRELIABLE and also uses
it as the field's default. The recordings cannot establish whether firmware leaves
this field unset for these report types or is reporting unreliable accuracy.
See the [SDK report definitions](https://github.com/luxonis/depthai-shared/blob/main/include/depthai-shared/datatype/RawIMUData.hpp).
`imu_valid` checks freshness and numerical validity, not calibration accuracy.
Before relying on IMU for learning, investigate that status and perform a stationary
gravity/gyro-bias check plus known-axis rotations to confirm scale, signs, and frame
conventions. Dynamic readings alone are insufficient to certify calibration.

**Scaling collection:** these are approximately 93.48 seconds across two device
sessions. They support continuing collection in batches, but do not establish
long-session reliability. A practical next validation is 10–20 additional trials
spread over a normal 30–60 minute session, including the usual recording stops,
saves, and process restarts, followed by the same continuity/freshness audit.
Continue reviewing demonstration success separately from recording integrity.

## Scope of the implementation

This was a moderate refactor localized to collection and dataset handling. The
recorder now has separate acquisition, alignment, and saver processes. Moving
alignment out of the teleop process addresses the camera-ring failure exposed by a
long teleop-side pause. Actual active-recording data loss still rejects an episode.

The other substantial work was adding timestamped raw IMU streams, camera-aligned
IMU arrays, validity/loss metadata, schema v2, and corresponding save/recovery,
prune/clean, and reader compatibility support. Controller calculations, motor
command semantics, existing state/action dimensions, and policy architecture were
not redesigned. Training and deployment continue using existing inputs; consuming
IMU in a policy remains future work.

Prior validation included 17 regression tests and a real camera/IMU recording with
an artificial three-second parent-process stall: 450 contiguous images, all IMU
rows valid, and stale control rows correctly flagged. That stress test and these
real trials support the fix, but do not identify the original pause's exact trigger.

Detailed audit results are in `imu_test_health_2026-09-08.json`; core structural
and timing results are in `imu_test_core_health_2026-09-08.json`. No collection or
training code was changed during this audit.
