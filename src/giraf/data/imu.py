"""IMU report semantics and causal alignment, independent of DepthAI imports."""

from __future__ import annotations

import numpy as np

IMU_STREAMS = ("accelerometer", "gyroscope", "game_rotation_vector")
IMU_REPORTS = (
    "ACCELEROMETER_CALIBRATED",
    "GYROSCOPE_CALIBRATED",
    "GAME_ROTATION_VECTOR",
)
IMU_PACKET_FIELDS = ("acceleroMeter", "gyroscope", "rotationVector")
IMU_FIELDS = (
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
)
IMU_PRE_ROLL_NS = 1_000_000_000
IMU_STOP_GRACE_S = 1.0


def report_example(stream: str) -> dict[str, np.ndarray]:
    return {
        "values": np.zeros(4 if stream == "game_rotation_vector" else 3, np.float32),
        "timestamp_ns": np.int64(0),
        "device_timestamp_ns": np.int64(0),
        "receive_timestamp_ns": np.int64(0),
        "sequence_num": np.int64(0),
        "sequence_gap": np.int64(0),
        "producer_dropped_total": np.int64(0),
        "accuracy": np.int8(-1),
        "rotation_accuracy_rad": np.float32(np.nan),
        "report_valid": np.uint8(0),
    }


def aligned_imu_example() -> dict[str, np.ndarray]:
    return {
        "imu": np.full(10, np.nan, np.float32),
        "imu_timestamp_ns": np.full(3, -1, np.int64),
        "imu_sequence_num": np.full(3, -1, np.int64),
        "imu_age_ns": np.full(3, -1, np.int64),
        "imu_available": np.zeros(3, np.uint8),
        "imu_sensor_valid": np.zeros(3, np.uint8),
        "imu_valid": np.uint8(0),
    }


IMU_ALIGNED_KEYS = tuple(aligned_imu_example())


def timestamp_ns(value) -> int:
    return int(round(value.total_seconds() * 1_000_000_000))


class ReportDecoder:
    """Deduplicate packet-carried reports without assuming synchronized sensors."""

    def __init__(self) -> None:
        self.previous: dict[str, tuple[int, int]] = {}

    def decode(self, stream: str, report, receive_ns: int):
        generation = timestamp_ns(report.getTimestamp())
        device = timestamp_ns(report.getTimestampDevice())
        sequence = int(report.getSequenceNum()) & 0xFFFFFFFF
        previous = self.previous.get(stream)
        # Empty fields and duplicates occur in packets driven by another sensor.
        if device <= 0 or (previous and device == previous[0]):
            return None
        if previous and device < previous[0]:
            raise RuntimeError(f"{stream} device timestamp moved backwards")
        if abs(receive_ns - generation) > 10_000_000_000:
            raise RuntimeError(
                f"{stream} host timestamp is not monotonic-clock aligned"
            )
        values = (
            [report.i, report.j, report.k, report.real]
            if stream == "game_rotation_vector"
            else [report.x, report.y, report.z]
        )
        values = np.asarray(values, dtype=np.float32)
        valid = bool(np.isfinite(values).all())
        if stream == "game_rotation_vector":
            valid = valid and 0.9 <= float(np.linalg.norm(values)) <= 1.1
        gap = 0
        if previous:
            # DepthAI extends the native BNO counter to a 32-bit report counter.
            delta = (sequence - previous[1]) % 2**32
            gap = max(0, delta - 1)
        self.previous[stream] = (device, sequence)
        return {
            "values": values,
            "timestamp_ns": np.int64(generation),
            "device_timestamp_ns": np.int64(device),
            "receive_timestamp_ns": np.int64(receive_ns),
            "sequence_num": np.int64(sequence),
            "sequence_gap": np.int64(gap),
            "producer_dropped_total": np.int64(0),
            "accuracy": np.int8(int(report.accuracy)),
            "rotation_accuracy_rad": np.float32(
                getattr(report, "rotationVectorAccuracy", np.nan)
            ),
            "report_valid": np.uint8(valid),
        }


def align_imu(batches: dict, capture_ns: int, max_age_ns: int) -> dict:
    """Select each latest causal report from a snapshot of already-received data."""
    result = aligned_imu_example()
    for index, stream in enumerate(IMU_STREAMS):
        batch = batches.get(stream)
        if batch is None:
            continue
        candidates = np.flatnonzero(batch["timestamp_ns"] <= capture_ns)
        if not candidates.size:
            continue
        selected = int(candidates[-1])
        source_ns = int(batch["timestamp_ns"][selected])
        age = capture_ns - source_ns
        start, stop = ((0, 3), (3, 6), (6, 10))[index]
        result["imu"][start:stop] = batch["values"][selected]
        result["imu_timestamp_ns"][index] = source_ns
        result["imu_sequence_num"][index] = batch["sequence_num"][selected]
        result["imu_age_ns"][index] = age
        result["imu_available"][index] = 1
        result["imu_sensor_valid"][index] = 0 <= age <= max_age_ns and bool(
            batch["report_valid"][selected]
        )
    result["imu_valid"] = np.uint8(result["imu_sensor_valid"].all())
    return result
