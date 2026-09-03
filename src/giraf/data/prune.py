"""Remove inactive intervals from a GIRAF ReplayBuffer without editing the source."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zarr

from .schema import GRASP_INDEX


@dataclass(frozen=True, slots=True)
class EpisodeSegment:
    """One retained half-open interval in the source ReplayBuffer."""

    source_episode: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class PruneReport:
    """Summary of a prune preview or completed copy."""

    source: str
    output: str | None
    input_episodes: int
    output_episodes: int
    dropped_episodes: int
    input_steps: int
    kept_steps: int
    removed_steps: int


def _episode_ends(root) -> np.ndarray:
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise RuntimeError("dataset is missing meta/episode_ends")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1:
        raise RuntimeError("meta/episode_ends must be one-dimensional")
    if len(ends) and (ends[0] <= 0 or np.any(np.diff(ends) <= 0)):
        raise RuntimeError("meta/episode_ends must be strictly increasing")
    return ends


def _validate_dataset(root, ends: np.ndarray) -> int:
    if "data" not in root or "action" not in root["data"]:
        raise RuntimeError("dataset is missing data/action")
    if "timestamp_ns" not in root["data"]:
        raise RuntimeError("dataset is missing data/timestamp_ns")
    action = root["data/action"]
    if action.ndim != 2 or action.shape[1] <= GRASP_INDEX:
        raise RuntimeError(f"data/action must have at least {GRASP_INDEX + 1} columns")
    n_steps = int(action.shape[0])
    expected_steps = int(ends[-1]) if len(ends) else 0
    if n_steps != expected_steps:
        raise RuntimeError("data/action length does not match meta/episode_ends")
    for name, array in root["data"].arrays():
        if array.ndim == 0 or array.shape[0] != n_steps:
            raise RuntimeError(f"data/{name} is not aligned to the step dimension")
    for name, array in root["meta"].arrays():
        if name == "episode_ends":
            continue
        if array.ndim == 0 or array.shape[0] != len(ends):
            raise RuntimeError(f"meta/{name} is not aligned to the episode dimension")
    return n_steps


def _pad_mask(mask: np.ndarray, padding_steps: int) -> np.ndarray:
    if padding_steps == 0 or not mask.any():
        return mask
    active = np.flatnonzero(mask)
    changes = np.zeros(len(mask) + 1, dtype=np.int64)
    starts = np.maximum(active - padding_steps, 0)
    stops = np.minimum(active + padding_steps + 1, len(mask))
    np.add.at(changes, starts, 1)
    np.add.at(changes, stops, -1)
    return np.cumsum(changes[:-1]) > 0


def find_active_segments(
    root,
    *,
    action_epsilon: float = 1e-6,
    padding_steps: int = 0,
    min_segment_steps: int = 1,
    keep_grasp_transitions: bool = True,
    grasp_cooldown_s: float = 2.5,
) -> tuple[list[EpisodeSegment], int, int]:
    """Find contiguous runs containing motion commands or grasp transitions."""

    if not np.isfinite(action_epsilon) or action_epsilon < 0:
        raise ValueError("action_epsilon must be finite and non-negative")
    if padding_steps < 0:
        raise ValueError("padding_steps must be non-negative")
    if min_segment_steps <= 0:
        raise ValueError("min_segment_steps must be positive")
    if not np.isfinite(grasp_cooldown_s) or grasp_cooldown_s < 0:
        raise ValueError("grasp_cooldown_s must be finite and non-negative")

    ends = _episode_ends(root)
    n_steps = _validate_dataset(root, ends)
    segments: list[EpisodeSegment] = []
    dropped_episodes = 0
    episode_start = 0
    for episode, episode_stop_value in enumerate(ends):
        episode_stop = int(episode_stop_value)
        action = np.asarray(root["data/action"][episode_start:episode_stop])
        motion = action[:, :GRASP_INDEX]
        if not np.all(np.isfinite(motion)):
            raise RuntimeError(f"episode {episode} contains non-finite motion commands")
        active = np.any(np.abs(motion) > action_epsilon, axis=1)
        if keep_grasp_transitions and len(action) > 1:
            grasp = action[:, GRASP_INDEX] >= 0.5
            transitions = np.flatnonzero(grasp[1:] != grasp[:-1]) + 1
            if len(transitions):
                timestamps = np.asarray(
                    root["data/timestamp_ns"][episode_start:episode_stop],
                    dtype=np.int64,
                )
                if np.any(np.diff(timestamps) <= 0):
                    raise RuntimeError(
                        f"episode {episode} timestamps are not strictly increasing"
                    )
                cooldown_ns = round(grasp_cooldown_s * 1_000_000_000)
                for transition in transitions:
                    cooldown_end = int(timestamps[transition]) + cooldown_ns
                    stop = int(np.searchsorted(timestamps, cooldown_end, side="right"))
                    active[transition:stop] = True
        active = _pad_mask(active, padding_steps)

        edges = np.diff(
            np.concatenate(
                (
                    np.zeros(1, dtype=np.int8),
                    active.astype(np.int8),
                    np.zeros(1, dtype=np.int8),
                )
            )
        )
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        episode_segments = 0
        for start, stop in zip(starts, stops, strict=True):
            if int(stop - start) < min_segment_steps:
                continue
            segments.append(
                EpisodeSegment(
                    source_episode=episode,
                    start=episode_start + int(start),
                    stop=episode_start + int(stop),
                )
            )
            episode_segments += 1
        if episode_segments == 0:
            dropped_episodes += 1
        episode_start = episode_stop
    return segments, n_steps, dropped_episodes


def _create_array_like(group, name: str, source, length: int):
    target = group.create_dataset(
        name,
        shape=(length,) + source.shape[1:],
        chunks=source.chunks,
        dtype=source.dtype,
        compressor=source.compressor,
        fill_value=source.fill_value,
        order=source.order,
        filters=source.filters,
    )
    target.attrs.update(dict(source.attrs))
    return target


def _copy_data(source, target, segments: list[EpisodeSegment]) -> None:
    output_start = 0
    for segment in segments:
        remaining_start = segment.start
        while remaining_start < segment.stop:
            copy_length = min(int(source.chunks[0]), segment.stop - remaining_start)
            output_stop = output_start + copy_length
            target[output_start:output_stop] = source[
                remaining_start : remaining_start + copy_length
            ]
            output_start = output_stop
            remaining_start += copy_length


def _copy_episode_metadata(source_root, target_root, segments) -> None:
    source_meta = source_root["meta"]
    target_meta = target_root["meta"]
    n_output_episodes = len(segments)
    lengths = np.asarray([segment.length for segment in segments], dtype=np.int64)
    ends = np.cumsum(lengths, dtype=np.int64)
    source_ends = source_meta["episode_ends"]
    target_ends = _create_array_like(
        target_meta, "episode_ends", source_ends, n_output_episodes
    )
    target_ends[:] = ends

    for name, source in source_meta.arrays():
        if name == "episode_ends":
            continue
        target = _create_array_like(target_meta, name, source, n_output_episodes)
        for output_episode, segment in enumerate(segments):
            target[output_episode] = source[segment.source_episode]

    data = source_root["data"]
    if "timestamp_ns" in data and "episode_start_monotonic_ns" in target_meta:
        starts = np.asarray(
            [data["timestamp_ns"][segment.start] for segment in segments],
            dtype=np.int64,
        )
        if (
            "episode_start_wall_time_ns" in target_meta
            and "episode_start_wall_time_ns" in source_meta
        ):
            wall_starts = np.asarray(
                [
                    int(
                        source_meta["episode_start_wall_time_ns"][
                            segment.source_episode
                        ]
                    )
                    + int(starts[index])
                    - int(
                        source_meta["episode_start_monotonic_ns"][
                            segment.source_episode
                        ]
                    )
                    for index, segment in enumerate(segments)
                ],
                dtype=np.int64,
            )
            target_meta["episode_start_wall_time_ns"][:] = wall_starts
        target_meta["episode_start_monotonic_ns"][:] = starts

    if "alignment_valid" in data:
        valid_counts = np.asarray(
            [
                np.count_nonzero(data["alignment_valid"][segment.start : segment.stop])
                for segment in segments
            ],
            dtype=np.int64,
        )
        if "episode_valid_steps" in target_meta:
            target_meta["episode_valid_steps"][:] = valid_counts
        if "episode_invalid_steps" in target_meta:
            target_meta["episode_invalid_steps"][:] = lengths - valid_counts


def _write_pruned_copy(
    source_root,
    output: Path,
    segments: list[EpisodeSegment],
    *,
    report_attributes: dict[str, object],
) -> None:
    source_path = Path(source_root.store.path).resolve()
    output = output.resolve()
    if output == source_path or source_path in output.parents:
        raise ValueError("output must not be the source dataset or inside it")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        target_root = zarr.open_group(str(temporary), mode="w")
        target_root.attrs.update(dict(source_root.attrs))
        target_root.attrs["prune"] = report_attributes
        target_data = target_root.require_group("data")
        target_meta = target_root.require_group("meta")
        target_data.attrs.update(dict(source_root["data"].attrs))
        target_meta.attrs.update(dict(source_root["meta"].attrs))
        kept_steps = sum(segment.length for segment in segments)
        for name, source in source_root["data"].arrays():
            target = _create_array_like(target_data, name, source, kept_steps)
            _copy_data(source, target, segments)
        _copy_episode_metadata(source_root, target_root, segments)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prune_replay_buffer(
    source: str | Path,
    output: str | Path | None = None,
    *,
    action_epsilon: float = 1e-6,
    padding_steps: int = 0,
    min_segment_steps: int = 1,
    keep_grasp_transitions: bool = True,
    grasp_cooldown_s: float = 2.5,
) -> PruneReport:
    """Preview or write a copy containing only contiguous active intervals.

    Supplying no ``output`` performs a read-only preview. The source is never
    modified, and an existing output path is never overwritten.
    """

    source_path = Path(source).resolve()
    source_root = zarr.open_group(str(source_path), mode="r")
    segments, input_steps, dropped_episodes = find_active_segments(
        source_root,
        action_epsilon=action_epsilon,
        padding_steps=padding_steps,
        min_segment_steps=min_segment_steps,
        keep_grasp_transitions=keep_grasp_transitions,
        grasp_cooldown_s=grasp_cooldown_s,
    )
    kept_steps = sum(segment.length for segment in segments)
    input_episodes = len(_episode_ends(source_root))
    output_path = Path(output).resolve() if output is not None else None
    report = PruneReport(
        source=str(source_path),
        output=str(output_path) if output_path is not None else None,
        input_episodes=input_episodes,
        output_episodes=len(segments),
        dropped_episodes=dropped_episodes,
        input_steps=input_steps,
        kept_steps=kept_steps,
        removed_steps=input_steps - kept_steps,
    )
    if output_path is not None:
        attributes = {
            **asdict(report),
            "action_epsilon": action_epsilon,
            "padding_steps": padding_steps,
            "min_segment_steps": min_segment_steps,
            "keep_grasp_transitions": keep_grasp_transitions,
            "grasp_cooldown_s": grasp_cooldown_s,
        }
        _write_pruned_copy(
            source_root,
            output_path,
            segments,
            report_attributes=attributes,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="source ReplayBuffer .zarr")
    parser.add_argument(
        "--output",
        help="new .zarr path; omit to preview without writing",
    )
    parser.add_argument(
        "--action-epsilon",
        type=float,
        default=1e-6,
        help="motion magnitudes at or below this threshold are inactive",
    )
    parser.add_argument(
        "--padding-steps",
        type=int,
        default=0,
        help="retain this many inactive samples around active samples",
    )
    parser.add_argument(
        "--min-segment-steps",
        type=int,
        default=1,
        help="discard retained runs shorter than this many samples",
    )
    parser.add_argument(
        "--ignore-grasp-transitions",
        action="store_true",
        help="do not retain grasp transitions or their cooldown windows",
    )
    parser.add_argument(
        "--grasp-cooldown-s",
        type=float,
        default=2.5,
        help="retain samples for this many seconds after grasp transitions",
    )
    args = parser.parse_args()
    report = prune_replay_buffer(
        args.dataset,
        args.output,
        action_epsilon=args.action_epsilon,
        padding_steps=args.padding_steps,
        min_segment_steps=args.min_segment_steps,
        keep_grasp_transitions=not args.ignore_grasp_transitions,
        grasp_cooldown_s=args.grasp_cooldown_s,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    if args.output is None:
        print("Preview only: pass --output to write a pruned copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
