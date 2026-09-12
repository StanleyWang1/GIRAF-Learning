"""Explicit opt-in camera/IMU recording check; never commands motors.

Run: ``uv run --extra hardware python scripts/imu_hardware_smoke.py --seconds 15``.
Temporary recordings contain synthetic control state, not demonstrations.
"""

import argparse
import ctypes
import json
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from giraf.data.config import CollectorConfig, ImuConfig
from giraf.data.imu import IMU_STREAMS
from giraf.data.imu_storage import validate_streams
from giraf.data.pipeline import DataCollectionPipeline


def record(seconds, enabled, output, parent_stall_s=0):
    base = CollectorConfig()
    config = replace(
        base,
        imu=ImuConfig(enabled=enabled),
        dataset=replace(base.dataset, output_dir=output),
    )
    collector = DataCollectionPipeline(config, hardware_enabled=False)
    stop = threading.Event()
    errors = []

    def publish():
        try:
            while not stop.is_set():
                collector.publish_control(
                    timestamp_ns=time.monotonic_ns(),
                    task_twist=np.zeros(6),
                    joint_velocity_command=np.zeros(6),
                    joint_position_command=np.zeros(6),
                    state=np.zeros(15),
                    grasp=False,
                    clutch=True,
                    tracking=True,
                )
                stop.wait(0.01)
        except BaseException as exc:
            errors.append(str(exc))

    publisher = threading.Thread(target=publish)
    try:
        collector.start(lambda: {"record_toggle_count": 0})
        publisher.start()
        time.sleep(1)
        if not collector.start_episode():
            raise RuntimeError("sources not ready")
        started = time.monotonic()
        end = started + seconds
        stalled = False
        while time.monotonic() < end:
            if collector.error:
                raise RuntimeError(collector.error)
            if parent_stall_s and not stalled and time.monotonic() - started >= 1:
                libc = ctypes.PyDLL(None)
                libc.usleep.argtypes = [ctypes.c_uint]
                libc.usleep(round(parent_stall_s * 1e6))
                stalled = True
            time.sleep(0.05)
        result = collector.end_episode()
        if not result or result.get("rejected"):
            raise RuntimeError("recording was rejected")
    finally:
        stop.set()
        if publisher.ident is not None:
            publisher.join(2)
        collector.stop()
    if errors:
        raise RuntimeError(errors)
    root = zarr.open_group(str(config.zarr_path), mode="r")
    validate_streams(root, 1)
    data = root["data"]
    timestamps = data["timestamp_ns"][:]
    seq = data["camera_sequence_num"][:]
    latency = data["camera_receive_latency_ns"][:] / 1e6
    summary = {
        "imu_enabled": enabled,
        "synthetic_control_only": True,
        "injected_parent_stall_s": parent_stall_s,
        "path": str(config.zarr_path),
        "frames": len(timestamps),
        "camera_hz": float(
            (len(timestamps) - 1) * 1e9 / (timestamps[-1] - timestamps[0])
        ),
        "camera_dropped": int(np.maximum(np.diff(seq) - 1, 0).sum()),
        "camera_latency_ms_p50_p95_p99": np.percentile(latency, [50, 95, 99]).tolist(),
        "control_alignment_valid_fraction": float(data["alignment_valid"][:].mean()),
        "imu_valid_fraction": float(data["imu_valid"][:].mean()),
        "imu": result["imu"],
        "zarr_bytes": sum(
            p.stat().st_size for p in config.zarr_path.rglob("*") if p.is_file()
        ),
    }
    for name in IMU_STREAMS:
        values = root[f"imu/{name}/values"][:]
        if len(values):
            summary["imu"]["streams"][name]["mean_values"] = values.mean(
                axis=0
            ).tolist()
            if name == "game_rotation_vector":
                summary["imu"]["streams"][name]["quaternion_norm_range"] = [
                    float(np.linalg.norm(values, axis=1).min()),
                    float(np.linalg.norm(values, axis=1).max()),
                ]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument(
        "--parent-stall-s",
        type=float,
        default=0,
        help="hold the parent process GIL while acquisition continues",
    )
    args = parser.parse_args()
    if args.seconds < 2:
        parser.error("use at least two seconds")
    if not 0 <= args.parent_stall_s <= args.seconds - 2:
        parser.error("parent stall must be between zero and seconds minus two")
    directory = Path(tempfile.mkdtemp(prefix="giraf-imu-smoke-"))
    summaries = []
    for enabled in (False, True):
        summaries.append(
            record(
                args.seconds,
                enabled,
                directory / ("imu" if enabled else "baseline"),
                args.parent_stall_s,
            )
        )
        print(json.dumps(summaries[-1], indent=2), flush=True)
        time.sleep(2)
    result = directory / "summary.json"
    result.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {result}")


if __name__ == "__main__":
    main()
