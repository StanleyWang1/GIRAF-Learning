"""Audit a GIRAF ReplayBuffer and copy healthy episodes to a clean Zarr."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .data.schema import GRASP_INDEX
from .diagnostics import (
    array_missing_chunk_rows,
    diagnose_episode,
    episode_missing_chunk_rows,
)

CORE_DATA_KEYS = ("camera_rgb", "action", "state")
REQUIRED_DATA_KEYS = CORE_DATA_KEYS + ("timestamp_ns", "alignment_valid")


@dataclass(frozen=True, slots=True)
class Segment:
    source_episode: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class EpisodeAudit:
    episode: int
    start: int
    stop: int
    healthy: bool
    reasons: tuple[str, ...]
    missing_data_rows: dict[str, int]
    missing_metadata: tuple[str, ...]

    @property
    def steps(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class CleanReport:
    source: str
    output: str
    written: bool
    input_episodes: int
    healthy_episodes: tuple[int, ...]
    rejected_episodes: tuple[int, ...]
    input_steps: int
    healthy_steps: int
    output_episodes: int
    output_steps: int
    removed_unhealthy_steps: int
    removed_inactive_steps: int
    prune_inactive: bool
    audits: tuple[EpisodeAudit, ...]


def default_clean_path(source: str | Path) -> Path:
    path = Path(source).expanduser().resolve()
    stem = path.name.removesuffix(".zarr")
    return path.with_name(f"{stem}_cleaned.zarr")


def _episode_ends(root) -> np.ndarray:
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise RuntimeError("dataset is missing meta/episode_ends")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1:
        raise RuntimeError("meta/episode_ends must be one-dimensional")
    if not len(ends):
        raise RuntimeError("dataset contains no episodes")
    if ends[0] <= 0 or np.any(np.diff(ends) <= 0):
        raise RuntimeError("meta/episode_ends must be strictly increasing")
    return ends


def _validate_dataset(root, ends: np.ndarray) -> None:
    if "data" not in root:
        raise RuntimeError("dataset is missing data group")
    data = root["data"]
    for name in REQUIRED_DATA_KEYS:
        if name not in data:
            raise RuntimeError(f"dataset is missing data/{name}")
    action = data["action"]
    if action.ndim != 2 or action.shape[1] <= GRASP_INDEX:
        raise RuntimeError(f"data/action must have at least {GRASP_INDEX + 1} columns")
    expected_steps = int(ends[-1])
    for name, array in data.arrays():
        if array.ndim == 0 or array.shape[0] != expected_steps:
            raise RuntimeError(f"data/{name} is not aligned to the step dimension")
    for name, array in root["meta"].arrays():
        if array.ndim == 0 or array.shape[0] != len(ends):
            raise RuntimeError(f"meta/{name} is not aligned to the episode dimension")


def _metadata_chunk_gaps(root, episode: int) -> tuple[str, ...]:
    return tuple(
        name
        for name, array in root["meta"].arrays()
        if array_missing_chunk_rows(array, episode, episode + 1)
    )


def _diagnostic_reasons(report: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    metadata = report["metadata_counts"]
    if metadata is None:
        reasons.append("episode validity metadata is missing")
    elif not metadata["matches_data"]:
        reasons.append("episode metadata disagrees with data/alignment_valid")
    if report["non_binary_alignment_values"]:
        reasons.append("data/alignment_valid contains values other than 0 or 1")
    if report["timestamp_non_increasing_intervals"]:
        reasons.append("camera timestamps are not strictly increasing")
    sequence = report["camera_sequence"]
    if sequence is not None and sequence["discontinuities"]:
        reasons.append("camera sequence numbers are discontinuous")
    for name, count in report["non_finite_rows"].items():
        if count:
            reasons.append(f"data/{name} contains {count} non-finite rows")
    for name, consistency in report["stored_timing_consistency"].items():
        if consistency is not None and consistency["mismatch_steps"]:
            reasons.append(f"stored {name} disagrees with source timestamps")
    if report["source_field_health"]["camera_receive_timestamp_zero_filled"]:
        reasons.append("camera receive timestamps are zero-filled")
    if report["source_field_health"]["camera_sequence_constant"]:
        reasons.append("camera sequence numbers do not advance")

    rules = [rule for rule in report["validity_rules"].values() if rule is not None]
    if rules and not any(rule["stored_mismatch_steps"] == 0 for rule in rules):
        reasons.append("alignment validity matches neither saved collection rule")
    return reasons


def audit_replay_buffer(
    source: str | Path,
) -> tuple[Any, np.ndarray, list[EpisodeAudit]]:
    """Audit every episode without changing the source dataset."""

    source_path = Path(source).expanduser().resolve()
    root = zarr.open_group(str(source_path), mode="r")
    ends = _episode_ends(root)
    _validate_dataset(root, ends)
    audits: list[EpisodeAudit] = []
    start = 0
    for episode, stop_value in enumerate(ends):
        stop = int(stop_value)
        missing_rows = {
            name: count
            for name, count in episode_missing_chunk_rows(root, start, stop).items()
            if count
        }
        missing_metadata = _metadata_chunk_gaps(root, episode)
        reasons: list[str] = []
        if missing_rows:
            reasons.append("one or more physical data chunks are missing")
        if missing_metadata:
            reasons.append("one or more physical metadata chunks are missing")
        try:
            diagnostics = diagnose_episode(source_path, episode)
        except (IndexError, KeyError, OSError, RuntimeError, ValueError) as exc:
            reasons.append(f"diagnostics failed: {type(exc).__name__}: {exc}")
        else:
            reasons.extend(_diagnostic_reasons(diagnostics))
        audits.append(
            EpisodeAudit(
                episode=episode,
                start=start,
                stop=stop,
                healthy=not reasons,
                reasons=tuple(dict.fromkeys(reasons)),
                missing_data_rows=missing_rows,
                missing_metadata=missing_metadata,
            )
        )
        start = stop
    return root, ends, audits


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


def _active_segments(
    root,
    audits: list[EpisodeAudit],
    *,
    action_epsilon: float,
    padding_steps: int,
    min_segment_steps: int,
    keep_grasp_transitions: bool,
    grasp_cooldown_s: float,
) -> list[Segment]:
    segments: list[Segment] = []
    cooldown_ns = round(grasp_cooldown_s * 1_000_000_000)
    for audit in audits:
        if not audit.healthy:
            continue
        action = np.asarray(root["data/action"][audit.start : audit.stop])
        active = np.any(np.abs(action[:, :GRASP_INDEX]) > action_epsilon, axis=1)
        timestamps = np.asarray(
            root["data/timestamp_ns"][audit.start : audit.stop], dtype=np.int64
        )
        if keep_grasp_transitions and len(action) > 1:
            grasp = action[:, GRASP_INDEX] >= 0.5
            transitions = np.flatnonzero(grasp[1:] != grasp[:-1]) + 1
            for transition in transitions:
                cooldown_end = int(timestamps[transition]) + cooldown_ns
                segment_stop = int(
                    np.searchsorted(timestamps, cooldown_end, side="right")
                )
                active[transition:segment_stop] = True
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
        for start, stop in zip(starts, stops, strict=True):
            if int(stop - start) >= min_segment_steps:
                segments.append(
                    Segment(
                        source_episode=audit.episode,
                        start=audit.start + int(start),
                        stop=audit.start + int(stop),
                    )
                )
    return segments


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


def _copy_segments(source, target, segments: list[Segment]) -> None:
    output_start = 0
    for segment in segments:
        source_start = segment.start
        while source_start < segment.stop:
            count = min(int(source.chunks[0]), segment.stop - source_start)
            target[output_start : output_start + count] = source[
                source_start : source_start + count
            ]
            output_start += count
            source_start += count


def _copy_episode_metadata(source_root, target_root, segments: list[Segment]) -> None:
    source_meta = source_root["meta"]
    target_meta = target_root["meta"]
    lengths = np.asarray([segment.length for segment in segments], dtype=np.int64)
    episode_ends = _create_array_like(
        target_meta, "episode_ends", source_meta["episode_ends"], len(segments)
    )
    episode_ends[:] = np.cumsum(lengths, dtype=np.int64)

    for name, source in source_meta.arrays():
        if name == "episode_ends":
            continue
        target = _create_array_like(target_meta, name, source, len(segments))
        for output_episode, segment in enumerate(segments):
            target[output_episode] = source[segment.source_episode]

    source_ends = np.asarray(source_meta["episode_ends"][:], dtype=np.int64)
    source_starts = np.concatenate((np.zeros(1, dtype=np.int64), source_ends[:-1]))
    if "timestamp_ns" in source_root["data"]:
        timestamps = source_root["data/timestamp_ns"]
        for output_episode, segment in enumerate(segments):
            if segment.start == int(source_starts[segment.source_episode]):
                continue
            timestamp = int(timestamps[segment.start])
            if "episode_start_monotonic_ns" in target_meta:
                target_meta["episode_start_monotonic_ns"][output_episode] = timestamp
            if (
                "episode_start_wall_time_ns" in target_meta
                and "episode_start_monotonic_ns" in source_meta
            ):
                source_wall = int(
                    source_meta["episode_start_wall_time_ns"][segment.source_episode]
                )
                source_monotonic = int(
                    source_meta["episode_start_monotonic_ns"][segment.source_episode]
                )
                target_meta["episode_start_wall_time_ns"][output_episode] = (
                    source_wall + timestamp - source_monotonic
                )

    if "alignment_valid" in source_root["data"]:
        valid_counts = np.asarray(
            [
                np.count_nonzero(
                    source_root["data/alignment_valid"][segment.start : segment.stop]
                )
                for segment in segments
            ],
            dtype=np.int64,
        )
        if "episode_valid_steps" in target_meta:
            target_meta["episode_valid_steps"][:] = valid_counts
        if "episode_invalid_steps" in target_meta:
            target_meta["episode_invalid_steps"][:] = lengths - valid_counts


def _write_clean_copy(
    source_root,
    output: Path,
    segments: list[Segment],
    report_attributes: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        target_root = zarr.open_group(str(temporary), mode="w")
        target_root.attrs.update(dict(source_root.attrs))
        target_root.attrs["clean"] = report_attributes
        target_data = target_root.require_group("data")
        target_meta = target_root.require_group("meta")
        target_data.attrs.update(dict(source_root["data"].attrs))
        target_meta.attrs.update(dict(source_root["meta"].attrs))
        output_steps = sum(segment.length for segment in segments)
        for name, source in source_root["data"].arrays():
            target = _create_array_like(target_data, name, source, output_steps)
            _copy_segments(source, target, segments)
        _copy_episode_metadata(source_root, target_root, segments)

        _target_root, _target_ends, target_audits = audit_replay_buffer(temporary)
        unhealthy = [audit.episode for audit in target_audits if not audit.healthy]
        if unhealthy:
            raise RuntimeError(
                f"cleaned copy failed verification for episodes {unhealthy}"
            )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def clean_replay_buffer(
    source: str | Path,
    output: str | Path | None = None,
    *,
    dry_run: bool = False,
    prune_inactive: bool = False,
    action_epsilon: float = 1e-6,
    padding_steps: int = 0,
    min_segment_steps: int = 1,
    keep_grasp_transitions: bool = True,
    grasp_cooldown_s: float = 2.5,
) -> CleanReport:
    """Audit the source and atomically write only healthy episode data."""

    if not np.isfinite(action_epsilon) or action_epsilon < 0:
        raise ValueError("action_epsilon must be finite and non-negative")
    if padding_steps < 0:
        raise ValueError("padding_steps must be non-negative")
    if min_segment_steps <= 0:
        raise ValueError("min_segment_steps must be positive")
    if not np.isfinite(grasp_cooldown_s) or grasp_cooldown_s < 0:
        raise ValueError("grasp_cooldown_s must be finite and non-negative")

    source_path = Path(source).expanduser().resolve()
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else default_clean_path(source_path)
    )
    if output_path == source_path or source_path in output_path.parents:
        raise ValueError("output must not be the source dataset or inside it")
    if not dry_run and output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    root, ends, audits = audit_replay_buffer(source_path)
    healthy = [audit for audit in audits if audit.healthy]
    if not healthy:
        raise RuntimeError("no healthy episodes were found")
    if prune_inactive:
        segments = _active_segments(
            root,
            audits,
            action_epsilon=action_epsilon,
            padding_steps=padding_steps,
            min_segment_steps=min_segment_steps,
            keep_grasp_transitions=keep_grasp_transitions,
            grasp_cooldown_s=grasp_cooldown_s,
        )
    else:
        segments = [
            Segment(audit.episode, audit.start, audit.stop) for audit in healthy
        ]
    if not segments:
        raise RuntimeError("no samples remain after inactive pruning")

    healthy_steps = sum(audit.steps for audit in healthy)
    output_steps = sum(segment.length for segment in segments)
    report = CleanReport(
        source=str(source_path),
        output=str(output_path),
        written=False,
        input_episodes=len(audits),
        healthy_episodes=tuple(audit.episode for audit in healthy),
        rejected_episodes=tuple(audit.episode for audit in audits if not audit.healthy),
        input_steps=int(ends[-1]),
        healthy_steps=healthy_steps,
        output_episodes=len(segments),
        output_steps=output_steps,
        removed_unhealthy_steps=int(ends[-1]) - healthy_steps,
        removed_inactive_steps=healthy_steps - output_steps,
        prune_inactive=prune_inactive,
        audits=tuple(audits),
    )
    if dry_run:
        return report

    attributes = {
        "source": report.source,
        "healthy_source_episodes": list(report.healthy_episodes),
        "rejected_source_episodes": list(report.rejected_episodes),
        "prune_inactive": prune_inactive,
        "segments": [asdict(segment) for segment in segments],
    }
    if prune_inactive:
        attributes.update(
            {
                "action_epsilon": action_epsilon,
                "padding_steps": padding_steps,
                "min_segment_steps": min_segment_steps,
                "keep_grasp_transitions": keep_grasp_transitions,
                "grasp_cooldown_s": grasp_cooldown_s,
            }
        )
    _write_clean_copy(root, output_path, segments, attributes)
    return replace(report, written=True)


def _ranges(values: tuple[int, ...]) -> str:
    if not values:
        return "none"
    ranges: list[str] = []
    start = stop = values[0]
    for value in values[1:]:
        if value == stop + 1:
            stop = value
            continue
        ranges.append(str(start) if start == stop else f"{start}-{stop}")
        start = stop = value
    ranges.append(str(start) if start == stop else f"{start}-{stop}")
    return ", ".join(ranges)


def format_clean_report(report: CleanReport) -> str:
    lines = [
        "GIRAF dataset cleaning",
        f"source: {report.source}",
        f"output: {report.output}",
        f"healthy episodes: {_ranges(report.healthy_episodes)}",
        f"rejected episodes: {_ranges(report.rejected_episodes)}",
        (
            f"steps: input={report.input_steps} healthy={report.healthy_steps} "
            f"output={report.output_steps}"
        ),
        (
            f"removed: unhealthy={report.removed_unhealthy_steps} "
            f"inactive={report.removed_inactive_steps}"
        ),
        f"inactive pruning: {'enabled' if report.prune_inactive else 'disabled'}",
    ]
    rejected = [audit for audit in report.audits if not audit.healthy]
    if rejected:
        lines.append("rejection details:")
        for audit in rejected:
            core = {
                name: audit.missing_data_rows[name]
                for name in CORE_DATA_KEYS
                if name in audit.missing_data_rows
            }
            detail = "; ".join(audit.reasons)
            if core:
                missing = ", ".join(f"{name}={count}" for name, count in core.items())
                detail += f"; missing core rows: {missing}"
            lines.append(f"  episode {audit.episode}: {detail}")
    lines.append(
        "status: cleaned copy written" if report.written else "status: dry run only"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="source ReplayBuffer .zarr")
    parser.add_argument(
        "--output",
        help="output .zarr; defaults to <source_stem>_cleaned.zarr",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="audit and report without writing"
    )
    parser.add_argument(
        "--prune-inactive",
        action="store_true",
        help="also split and remove inactive action intervals",
    )
    parser.add_argument("--action-epsilon", type=float, default=1e-6)
    parser.add_argument("--padding-steps", type=int, default=0)
    parser.add_argument("--min-segment-steps", type=int, default=1)
    parser.add_argument("--grasp-cooldown-s", type=float, default=2.5)
    parser.add_argument("--ignore-grasp-transitions", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = clean_replay_buffer(
        args.dataset,
        args.output,
        dry_run=args.dry_run,
        prune_inactive=args.prune_inactive,
        action_epsilon=args.action_epsilon,
        padding_steps=args.padding_steps,
        min_segment_steps=args.min_segment_steps,
        keep_grasp_transitions=not args.ignore_grasp_transitions,
        grasp_cooldown_s=args.grasp_cooldown_s,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(format_clean_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
