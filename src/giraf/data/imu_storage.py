"""Independent IMU time axes, committed by the RGB episode marker."""

from __future__ import annotations

import numpy as np

from .imu import IMU_PRE_ROLL_NS, IMU_REPORTS, IMU_STREAMS, report_example


def initialize_streams(root, chunk_length: int, compressor, *, committed=False):
    imu = root.require_group("imu")
    imu.attrs.update(
        {
            "pre_roll_ns": IMU_PRE_ROLL_NS,
            "values_semantics": "original delivered reports, not host filtered",
            "timestamp_clock": "host monotonic ns (DepthAI synchronized)",
            "accuracy_codes": {
                "-1": "unavailable",
                "0": "unreliable",
                "1": "low",
                "2": "medium",
                "3": "high",
            },
            "sequence_gap_semantics": "counter-inferred missing reports; initial prefix unknown",
        }
    )
    for index, name in enumerate(IMU_STREAMS):
        group = imu.require_group(name)
        group.attrs.update(
            {
                "report": IMU_REPORTS[index],
                "units": ("m/s^2 including gravity", "rad/s", "unit quaternion XYZW")[
                    index
                ],
            }
        )
        for key, example in report_example(name).items():
            value = np.asarray(example)
            if key not in group:
                group.create_dataset(
                    key,
                    shape=(0,) + value.shape,
                    chunks=(chunk_length,) + value.shape,
                    dtype=value.dtype,
                    compressor=compressor,
                )
        if committed:
            for key in (
                "episode_ends",
                "episode_saver_dropped",
                "episode_producer_dropped",
            ):
                if key not in group:
                    group.create_dataset(
                        key, shape=(0,), chunks=(1024,), dtype=np.int64, compressor=None
                    )


def validate_streams(root, n_episodes: int) -> None:
    for name in IMU_STREAMS:
        group = root[f"imu/{name}"]
        ends = np.asarray(group["episode_ends"][:], dtype=np.int64)
        if len(ends) != n_episodes or np.any(np.diff(ends, prepend=0) < 0):
            raise RuntimeError(f"invalid {name} episode offsets")
        total = int(ends[-1]) if len(ends) else 0
        for key in report_example(name):
            if group[key].shape[0] != total:
                raise RuntimeError(
                    f"{name}/{key} length differs from committed offsets"
                )
        if group["episode_saver_dropped"].shape != (n_episodes,):
            raise RuntimeError(f"invalid {name} saver-loss metadata")
        if group["episode_producer_dropped"].shape != (n_episodes,):
            raise RuntimeError(f"invalid {name} producer-loss metadata")


def recover_streams(root, n_episodes: int) -> None:
    for name in IMU_STREAMS:
        group = root[f"imu/{name}"]
        ends = group["episode_ends"]
        if len(ends) < n_episodes:
            raise RuntimeError(f"{name} offsets shorter than RGB commit marker")
        committed = np.asarray(ends[:n_episodes], dtype=np.int64)
        if np.any(np.diff(committed, prepend=0) < 0):
            raise RuntimeError(f"invalid committed {name} offsets")
        total = int(committed[-1]) if n_episodes else 0
        for key in report_example(name):
            array = group[key]
            if len(array) < total:
                raise RuntimeError(f"{name}/{key} shorter than committed length")
            if len(array) > total:
                array.resize((total,) + array.shape[1:])
        for key in (
            "episode_ends",
            "episode_saver_dropped",
            "episode_producer_dropped",
        ):
            if len(group[key]) < n_episodes:
                raise RuntimeError(f"{name}/{key} shorter than commit marker")
            group[key].resize(n_episodes)


def append_streams(source_root, target_root, episode_index: int) -> None:
    # Check every report array before appending any stream.
    for name in IMU_STREAMS:
        source = source_root[f"imu/{name}"]
        target = target_root[f"imu/{name}"]
        count = len(source["timestamp_ns"])
        for key in report_example(name):
            if (
                source[key].shape != (count,) + target[key].shape[1:]
                or source[key].dtype != target[key].dtype
            ):
                raise RuntimeError(f"staged IMU schema mismatch: {name}/{key}")
    for name in IMU_STREAMS:
        source = source_root[f"imu/{name}"]
        target = target_root[f"imu/{name}"]
        current = len(target["timestamp_ns"])
        count = len(source["timestamp_ns"])
        for key in report_example(name):
            array = target[key]
            array.resize((current + count,) + array.shape[1:])
            for start in range(0, count, source[key].chunks[0]):
                stop = min(count, start + source[key].chunks[0])
                array[current + start : current + stop] = source[key][start:stop]
        for key, value in (
            ("episode_ends", current + count),
            ("episode_saver_dropped", source.attrs.get("saver_dropped", 0)),
            ("episode_producer_dropped", source.attrs.get("producer_dropped", 0)),
        ):
            target[key].resize(episode_index + 1)
            target[key][episode_index] = value


def segment_time_bounds(root, segment) -> tuple[int, int]:
    """Retain the source recording boundaries when an entire edge is kept."""
    episode = segment.source_episode
    ends = root["meta/episode_ends"]
    source_start = int(ends[episode - 1]) if episode else 0
    source_stop = int(ends[episode])
    start = int(root["data/timestamp_ns"][segment.start])
    stop = int(root["data/timestamp_ns"][segment.stop - 1])
    if segment.start == source_start and "episode_start_monotonic_ns" in root["meta"]:
        start = int(root["meta/episode_start_monotonic_ns"][episode])
    if segment.stop < source_stop:
        stop = int(root["data/timestamp_ns"][segment.stop]) - 1
    elif "episode_stop_monotonic_ns" in root["meta"]:
        stop = int(root["meta/episode_stop_monotonic_ns"][episode])
    return start, stop


def copy_pruned_streams(source_root, target_root, segments) -> None:
    """Copy each segment's reports/pre-roll only from its original episode."""
    if "imu" not in source_root:
        return
    from .prune import _create_array_like

    target_imu = target_root.require_group("imu")
    target_imu.attrs.update(dict(source_root["imu"].attrs))
    for name in IMU_STREAMS:
        source = source_root[f"imu/{name}"]
        target = target_imu.require_group(name)
        target.attrs.update(dict(source.attrs))
        intervals = []
        for segment in segments:
            episode = segment.source_episode
            low = int(source["episode_ends"][episode - 1]) if episode else 0
            high = int(source["episode_ends"][episode])
            timestamps = source["timestamp_ns"][low:high]
            first_ns, last_ns = segment_time_bounds(source_root, segment)
            start = low + int(np.searchsorted(timestamps, first_ns - IMU_PRE_ROLL_NS))
            stop = low + int(np.searchsorted(timestamps, last_ns, side="right"))
            intervals.append((start, stop))
        total = sum(stop - start for start, stop in intervals)
        for key in report_example(name):
            array = _create_array_like(target, key, source[key], total)
            cursor = 0
            for start, stop in intervals:
                for begin in range(start, stop, source[key].chunks[0]):
                    end = min(stop, begin + source[key].chunks[0])
                    array[cursor : cursor + end - begin] = source[key][begin:end]
                    cursor += end - begin
        offsets = _create_array_like(
            target, "episode_ends", source["episode_ends"], len(segments)
        )
        offsets[:] = np.cumsum([b - a for a, b in intervals], dtype=np.int64)
        for key in ("episode_saver_dropped", "episode_producer_dropped"):
            losses = _create_array_like(target, key, source[key], len(segments))
            losses[:] = [source[key][s.source_episode] for s in segments]
        target.attrs["host_loss_scope"] = "source episode (loss location unavailable)"
