"""Print read-only diagnostics for one GIRAF ReplayBuffer episode."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import zarr

NS_PER_MS = 1_000_000.0
NS_PER_S = 1_000_000_000.0


def array_missing_chunk_rows(array, start: int, stop: int) -> int:
    """Count selected rows backed by at least one missing physical chunk."""

    chunk_length = int(array.chunks[0])
    first_chunk = start // chunk_length
    last_chunk = (stop - 1) // chunk_length
    trailing_grid = tuple(
        range(math.ceil(size / chunk))
        for size, chunk in zip(array.shape[1:], array.chunks[1:], strict=True)
    )
    missing_time_chunks: set[int] = set()
    for time_chunk in range(first_chunk, last_chunk + 1):
        trailing_coordinates = itertools.product(*trailing_grid)
        if any(
            array._chunk_key((time_chunk, *coordinates)) not in array.chunk_store
            for coordinates in trailing_coordinates
        ):
            missing_time_chunks.add(time_chunk)
    return sum(
        max(
            0,
            min(stop, (chunk + 1) * chunk_length)
            - max(start, chunk * chunk_length),
        )
        for chunk in missing_time_chunks
    )


def episode_missing_chunk_rows(root, start: int, stop: int) -> dict[str, int]:
    """Count rows backed by missing physical chunks for each data array."""

    return {
        name: array_missing_chunk_rows(array, start, stop)
        for name, array in root["data"].arrays()
    }


def _episode_bounds(root, requested_episode: int) -> tuple[int, int, int, int]:
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise RuntimeError("dataset is missing meta/episode_ends")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if ends.ndim != 1:
        raise RuntimeError("meta/episode_ends must be one-dimensional")
    if len(ends) and (ends[0] <= 0 or np.any(np.diff(ends) <= 0)):
        raise RuntimeError("meta/episode_ends must be strictly increasing")
    if not len(ends):
        raise IndexError("dataset contains no episodes")

    episode = requested_episode
    if episode < 0:
        episode += len(ends)
    if episode < 0 or episode >= len(ends):
        raise IndexError(f"episode {requested_episode} is outside 0..{len(ends) - 1}")
    start = 0 if episode == 0 else int(ends[episode - 1])
    stop = int(ends[episode])
    return episode, len(ends), start, stop


def _load_episode_array(
    data,
    name: str,
    selection: slice,
    steps: int,
    *,
    required: bool = False,
) -> np.ndarray | None:
    if name not in data:
        if required:
            raise RuntimeError(f"dataset is missing data/{name}")
        return None
    values = np.asarray(data[name][selection])
    if values.ndim == 0 or values.shape[0] != steps:
        raise RuntimeError(f"data/{name} is not aligned to the selected episode")
    return values


def _stats(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float] | None:
    if not values.size:
        return None
    scaled = np.asarray(values, dtype=np.float64) / scale
    return {
        "min": float(np.min(scaled)),
        "p50": float(np.percentile(scaled, 50)),
        "p95": float(np.percentile(scaled, 95)),
        "max": float(np.max(scaled)),
    }


def _saved_limit_ms(attrs: Mapping[str, Any], name: str) -> float | None:
    config = attrs.get("last_collector_config")
    if not isinstance(config, Mapping):
        return None
    alignment = config.get("alignment")
    if not isinstance(alignment, Mapping):
        return None
    try:
        value = float(alignment[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _count_true(values: np.ndarray | None) -> int | None:
    return None if values is None else int(np.count_nonzero(values))


def _finite_row_failures(values: np.ndarray | None) -> int | None:
    if values is None:
        return None
    if values.ndim == 1:
        return int(np.count_nonzero(~np.isfinite(values)))
    axes = tuple(range(1, values.ndim))
    return int(np.count_nonzero(~np.all(np.isfinite(values), axis=axes)))


def _stored_timing_consistency(
    data,
    name: str,
    selection: slice,
    steps: int,
    recomputed_ns: np.ndarray | None,
) -> dict[str, int | float] | None:
    stored = _load_episode_array(data, name, selection, steps)
    if stored is None or recomputed_ns is None:
        return None
    difference = np.asarray(stored, dtype=np.int64) - np.asarray(
        recomputed_ns, dtype=np.int64
    )
    return {
        "mismatch_steps": int(np.count_nonzero(difference)),
        "max_abs_error_ms": float(np.max(np.abs(difference)) / NS_PER_MS)
        if difference.size
        else 0.0,
    }


def diagnose_episode(dataset: str | Path, episode: int) -> dict[str, Any]:
    """Return JSON-serializable, read-only diagnostics for one episode."""

    source = Path(dataset).expanduser().resolve()
    root = zarr.open_group(str(source), mode="r")
    if "data" not in root:
        raise RuntimeError("dataset is missing data group")
    resolved_episode, episode_count, start, stop = _episode_bounds(root, episode)
    selection = slice(start, stop)
    steps = stop - start
    data = root["data"]

    timestamps = _load_episode_array(
        data, "timestamp_ns", selection, steps, required=True
    )
    alignment_valid_raw = _load_episode_array(
        data, "alignment_valid", selection, steps, required=True
    )
    assert timestamps is not None and alignment_valid_raw is not None
    timestamps = np.asarray(timestamps, dtype=np.int64)
    alignment_valid = np.asarray(alignment_valid_raw) != 0
    invalid = ~alignment_valid

    control_timestamps = _load_episode_array(
        data, "control_timestamp_ns", selection, steps
    )
    motor_timestamps = _load_episode_array(
        data, "motor_timestamp_ns", selection, steps
    )
    camera_receive_timestamps = _load_episode_array(
        data, "camera_receive_timestamp_ns", selection, steps
    )
    motor_accepted_raw = _load_episode_array(
        data, "motor_command_accepted", selection, steps
    )
    tracking_raw = _load_episode_array(data, "tracking", selection, steps)
    clutch_raw = _load_episode_array(data, "clutch", selection, steps)
    sequences = _load_episode_array(data, "camera_sequence_num", selection, steps)
    actions = _load_episode_array(data, "action", selection, steps)
    states = _load_episode_array(data, "state", selection, steps)

    control_age_ns = (
        timestamps - np.asarray(control_timestamps, dtype=np.int64)
        if control_timestamps is not None
        else None
    )
    motor_age_ns = (
        timestamps - np.asarray(motor_timestamps, dtype=np.int64)
        if motor_timestamps is not None
        else None
    )
    camera_receive_latency_ns = (
        np.asarray(camera_receive_timestamps, dtype=np.int64) - timestamps
        if camera_receive_timestamps is not None
        else None
    )
    camera_receive_timestamps_zero_filled = bool(
        camera_receive_timestamps is not None
        and steps
        and not np.any(camera_receive_timestamps)
        and np.any(timestamps)
    )
    motor_accepted = (
        np.asarray(motor_accepted_raw) != 0
        if motor_accepted_raw is not None
        else None
    )

    attrs = dict(root.attrs)
    max_control_age_ms = _saved_limit_ms(attrs, "max_control_age_ms")
    max_motor_age_ms = _saved_limit_ms(attrs, "max_motor_age_ms")

    control_within_limit = None
    if control_age_ns is not None and max_control_age_ms is not None:
        control_within_limit = (control_age_ns >= 0) & (
            control_age_ns <= max_control_age_ms * NS_PER_MS
        )
    motor_within_limit = None
    if motor_age_ns is not None and max_motor_age_ms is not None:
        motor_within_limit = (motor_age_ns >= 0) & (
            motor_age_ns <= max_motor_age_ms * NS_PER_MS
        )

    dry_run_expected = control_within_limit
    hardware_expected = None
    if control_within_limit is not None and motor_within_limit is not None:
        if motor_accepted is not None:
            hardware_expected = (
                control_within_limit & motor_within_limit & motor_accepted
            )

    def rule_summary(expected: np.ndarray | None) -> dict[str, int] | None:
        if expected is None:
            return None
        return {
            "expected_valid_steps": int(np.count_nonzero(expected)),
            "stored_mismatch_steps": int(
                np.count_nonzero(expected != alignment_valid)
            ),
        }

    failure_signals: dict[str, int | None] = {
        "control_timestamp_in_future": None,
        "control_too_old": None,
        "motor_timestamp_in_future": None,
        "motor_too_old": None,
        "motor_command_not_accepted": None,
    }
    failure_masks: list[np.ndarray] = []
    if control_age_ns is not None:
        mask = control_age_ns < 0
        failure_signals["control_timestamp_in_future"] = int(np.count_nonzero(mask))
        failure_masks.append(mask)
        if max_control_age_ms is not None:
            mask = control_age_ns > max_control_age_ms * NS_PER_MS
            failure_signals["control_too_old"] = int(np.count_nonzero(mask))
            failure_masks.append(mask)
    if motor_age_ns is not None:
        mask = motor_age_ns < 0
        failure_signals["motor_timestamp_in_future"] = int(np.count_nonzero(mask))
        failure_masks.append(mask)
        if max_motor_age_ms is not None:
            mask = motor_age_ns > max_motor_age_ms * NS_PER_MS
            failure_signals["motor_too_old"] = int(np.count_nonzero(mask))
            failure_masks.append(mask)
    if motor_accepted is not None:
        mask = ~motor_accepted
        failure_signals["motor_command_not_accepted"] = int(np.count_nonzero(mask))
        failure_masks.append(mask)

    unexplained_invalid = None
    if (
        failure_masks
        and control_within_limit is not None
        and motor_within_limit is not None
        and motor_accepted is not None
    ):
        failure_union = np.logical_or.reduce(failure_masks)
        unexplained_invalid = int(np.count_nonzero(invalid & ~failure_union))

    intervals_ns = np.diff(timestamps)
    duration_s = (
        float((timestamps[-1] - timestamps[0]) / NS_PER_S) if steps > 1 else 0.0
    )
    effective_hz = (
        float((steps - 1) / duration_s) if steps > 1 and duration_s > 0 else None
    )

    sequence_health = None
    if sequences is not None:
        sequence_deltas = np.diff(np.asarray(sequences, dtype=np.int64))
        sequence_health = {
            "discontinuities": int(np.count_nonzero(sequence_deltas != 1)),
            "duplicates": int(np.count_nonzero(sequence_deltas == 0)),
            "backward_jumps": int(np.count_nonzero(sequence_deltas < 0)),
            "estimated_missing_frames": int(
                np.sum(np.maximum(sequence_deltas - 1, 0), dtype=np.int64)
            ),
        }

    metadata_counts = None
    meta = root["meta"]
    if "episode_valid_steps" in meta and "episode_invalid_steps" in meta:
        metadata_counts = {
            "valid_steps": int(meta["episode_valid_steps"][resolved_episode]),
            "invalid_steps": int(meta["episode_invalid_steps"][resolved_episode]),
        }
        metadata_counts["matches_data"] = bool(
            metadata_counts["valid_steps"] == int(np.count_nonzero(alignment_valid))
            and metadata_counts["invalid_steps"] == int(np.count_nonzero(invalid))
        )

    timing_consistency = {
        "control_age_ns": _stored_timing_consistency(
            data, "control_age_ns", selection, steps, control_age_ns
        ),
        "motor_age_ns": _stored_timing_consistency(
            data, "motor_age_ns", selection, steps, motor_age_ns
        ),
        "camera_receive_latency_ns": _stored_timing_consistency(
            data,
            "camera_receive_latency_ns",
            selection,
            steps,
            None
            if camera_receive_timestamps_zero_filled
            else camera_receive_latency_ns,
        ),
    }

    valid_steps = int(np.count_nonzero(alignment_valid))
    invalid_steps = steps - valid_steps
    missing_chunk_rows = episode_missing_chunk_rows(root, start, stop)
    report: dict[str, Any] = {
        "dataset": str(source),
        "schema_version": attrs.get("schema_version"),
        "git_revisions": list(attrs.get("git_revisions", [])),
        "episode": resolved_episode,
        "episode_count": episode_count,
        "global_step_start": start,
        "global_step_stop": stop,
        "steps": steps,
        "duration_s": duration_s,
        "effective_hz": effective_hz,
        "valid_steps": valid_steps,
        "invalid_steps": invalid_steps,
        "non_binary_alignment_values": int(
            np.count_nonzero(
                (np.asarray(alignment_valid_raw) != 0)
                & (np.asarray(alignment_valid_raw) != 1)
            )
        ),
        "metadata_counts": metadata_counts,
        "limits_ms": {
            "max_control_age": max_control_age_ms,
            "max_motor_age": max_motor_age_ms,
        },
        "validity_rules": {
            "dry_run": rule_summary(dry_run_expected),
            "hardware": rule_summary(hardware_expected),
        },
        "hardware_failure_signals": failure_signals,
        "invalid_without_hardware_failure_signal": unexplained_invalid,
        "timing_ms": {
            "camera_interval": _stats(intervals_ns, scale=NS_PER_MS),
            "camera_receive_latency": _stats(
                camera_receive_latency_ns, scale=NS_PER_MS
            )
            if (
                camera_receive_latency_ns is not None
                and not camera_receive_timestamps_zero_filled
            )
            else None,
            "control_age": _stats(control_age_ns, scale=NS_PER_MS)
            if control_age_ns is not None
            else None,
            "motor_age": _stats(motor_age_ns, scale=NS_PER_MS)
            if motor_age_ns is not None
            else None,
        },
        "timestamp_non_increasing_intervals": int(
            np.count_nonzero(intervals_ns <= 0)
        ),
        "camera_sequence": sequence_health,
        "status_counts": {
            "motor_command_accepted": _count_true(motor_accepted),
            "tracking": _count_true(
                np.asarray(tracking_raw) != 0 if tracking_raw is not None else None
            ),
            "clutch": _count_true(
                np.asarray(clutch_raw) != 0 if clutch_raw is not None else None
            ),
        },
        "non_finite_rows": {
            "action": _finite_row_failures(actions),
            "state": _finite_row_failures(states),
        },
        "stored_timing_consistency": timing_consistency,
        "missing_chunk_rows": missing_chunk_rows,
        "source_field_health": {
            "camera_receive_timestamp_zero_filled": (
                camera_receive_timestamps_zero_filled
            ),
            "camera_sequence_constant": bool(
                sequences is not None
                and steps > 1
                and np.all(np.asarray(sequences) == np.asarray(sequences)[0])
            ),
        },
    }

    findings: list[str] = []
    core_missing = {
        name: missing_chunk_rows.get(name, steps)
        for name in ("camera_rgb", "action", "state")
        if missing_chunk_rows.get(name, steps)
    }
    if core_missing:
        detail = ", ".join(f"{name}={count}" for name, count in core_missing.items())
        findings.append(
            "Missing physical Zarr chunks affect core training rows: " + detail + "."
        )
    hardware_rule = report["validity_rules"]["hardware"]
    dry_run_rule = report["validity_rules"]["dry_run"]
    not_accepted = failure_signals["motor_command_not_accepted"]
    metadata_matches = metadata_counts is None or metadata_counts["matches_data"]
    if metadata_counts is not None and not metadata_counts["matches_data"]:
        findings.append(
            "Dataset inconsistency: episode metadata reports "
            f"valid={metadata_counts['valid_steps']} "
            f"invalid={metadata_counts['invalid_steps']}, but data/alignment_valid "
            f"reports valid={valid_steps} invalid={invalid_steps}."
        )
    if (
        steps > 0
        and invalid_steps == steps
        and not_accepted == steps
        and hardware_rule is not None
        and hardware_rule["stored_mismatch_steps"] == 0
    ):
        if metadata_matches:
            findings.append(
                "All samples have motor_command_accepted=0; this alone makes every "
                "sample invalid under the hardware-mode alignment rule."
            )
        else:
            findings.append(
                "data/motor_command_accepted is 0 for every sample, but the episode "
                "metadata reports valid samples. Treat this as inconsistent stored "
                "data, not confirmed motor-command failure."
            )
    if (
        metadata_matches
        and hardware_rule is not None
        and dry_run_rule is not None
        and hardware_rule["stored_mismatch_steps"] == 0
        and dry_run_rule["stored_mismatch_steps"] > 0
    ):
        findings.append(
            "The stored validity flags match the hardware-mode rule, not the "
            "dry-run rule."
        )
    if camera_receive_timestamps_zero_filled:
        findings.append(
            "data/camera_receive_timestamp_ns is zero-filled, so camera receive "
            "latency cannot be diagnosed for this episode."
        )
    if report["source_field_health"]["camera_sequence_constant"]:
        findings.append(
            "data/camera_sequence_num does not advance within the episode; it is "
            "likely missing or zero-filled."
        )
    for name, consistency in timing_consistency.items():
        if consistency is not None and consistency["mismatch_steps"]:
            findings.append(
                f"Stored {name} disagrees with timestamps on "
                f"{consistency['mismatch_steps']} steps; use the recomputed timing "
                "statistics above."
            )
    report["findings"] = findings
    return report


def _format_count(count: int | None, total: int) -> str:
    if count is None:
        return "unavailable"
    percentage = 100.0 * count / total if total else 0.0
    return f"{count}/{total} ({percentage:.1f}%)"


def _format_stats(stats: Mapping[str, float] | None) -> str:
    if stats is None:
        return "unavailable"
    return (
        f"min={stats['min']:.3f} p50={stats['p50']:.3f} "
        f"p95={stats['p95']:.3f} max={stats['max']:.3f}"
    )


def format_report(report: Mapping[str, Any]) -> str:
    """Format a diagnostics report for a terminal."""

    steps = int(report["steps"])
    lines = [
        "GIRAF episode diagnostics",
        f"dataset: {report['dataset']}",
        f"schema: {report['schema_version'] or 'unknown'}",
        (
            f"episode: {report['episode']} of {report['episode_count']} "
            f"(global steps [{report['global_step_start']}, "
            f"{report['global_step_stop']}))"
        ),
        f"steps: {steps}",
        f"duration: {report['duration_s']:.3f} s",
        (
            f"effective rate: {report['effective_hz']:.3f} Hz"
            if report["effective_hz"] is not None
            else "effective rate: unavailable"
        ),
        "",
        "Zarr chunk coverage",
    ]
    missing_chunks = {
        name: count for name, count in report["missing_chunk_rows"].items() if count
    }
    if missing_chunks:
        lines.extend(
            f"  {name}: {count}/{steps} rows missing"
            for name, count in sorted(missing_chunks.items())
        )
        lines.append("  Missing chunks are returned by Zarr as fill values.")
    else:
        lines.append("  complete")
    lines.extend(
        [
            "",
            "Validity",
            f"  stored valid:   {_format_count(report['valid_steps'], steps)}",
            f"  stored invalid: {_format_count(report['invalid_steps'], steps)}",
        ]
    )
    metadata = report["metadata_counts"]
    if metadata is not None:
        lines.append(
            "  episode metadata: "
            f"valid={metadata['valid_steps']} invalid={metadata['invalid_steps']} "
            f"matches_data={metadata['matches_data']}"
        )

    limits = report["limits_ms"]
    control_limit = limits["max_control_age"]
    motor_limit = limits["max_motor_age"]
    lines.extend(
        [
            "  saved limits: "
            f"control={control_limit:.3f} ms"
            if control_limit is not None
            else "  saved limits: control=unavailable",
        ]
    )
    lines[-1] += (
        f" motor={motor_limit:.3f} ms"
        if motor_limit is not None
        else " motor=unavailable"
    )

    for label, name in (("dry-run", "dry_run"), ("hardware", "hardware")):
        rule = report["validity_rules"][name]
        if rule is None:
            lines.append(f"  {label} rule reconstruction: unavailable")
        else:
            lines.append(
                f"  {label} rule: "
                f"expected_valid={rule['expected_valid_steps']}/{steps} "
                f"stored_mismatches={rule['stored_mismatch_steps']}"
            )

    labels = {
        "control_timestamp_in_future": "control timestamp in future",
        "control_too_old": "control too old",
        "motor_timestamp_in_future": "motor timestamp in future",
        "motor_too_old": "motor too old",
        "motor_command_not_accepted": "motor command not accepted",
    }
    lines.append("  hardware failure signals (counts can overlap):")
    for name, label in labels.items():
        lines.append(
            f"    {label}: "
            f"{_format_count(report['hardware_failure_signals'][name], steps)}"
        )
    unexplained = report["invalid_without_hardware_failure_signal"]
    lines.append(
        "    invalid without listed signal: "
        + (str(unexplained) if unexplained is not None else "unavailable")
    )

    timing = report["timing_ms"]
    lines.extend(
        [
            "",
            "Timing (ms; recomputed from timestamps)",
            f"  camera interval:        {_format_stats(timing['camera_interval'])}",
            "  camera receive latency: "
            f"{_format_stats(timing['camera_receive_latency'])}",
            f"  control age:            {_format_stats(timing['control_age'])}",
            f"  motor age:              {_format_stats(timing['motor_age'])}",
            "",
            "Stream health",
            "  non-increasing camera timestamps: "
            f"{report['timestamp_non_increasing_intervals']}",
        ]
    )
    sequence = report["camera_sequence"]
    if sequence is None:
        lines.append("  camera sequence: unavailable")
    else:
        lines.append(
            "  camera sequence: "
            f"discontinuities={sequence['discontinuities']} "
            f"duplicates={sequence['duplicates']} "
            f"backward={sequence['backward_jumps']} "
            f"estimated_missing={sequence['estimated_missing_frames']}"
        )
    status = report["status_counts"]
    lines.extend(
        [
            "  motor command accepted: "
            f"{_format_count(status['motor_command_accepted'], steps)}",
            f"  tracking true: {_format_count(status['tracking'], steps)}",
            f"  clutch true:   {_format_count(status['clutch'], steps)}",
            "  non-finite rows: "
            f"action={report['non_finite_rows']['action']} "
            f"state={report['non_finite_rows']['state']}",
            "",
            "Stored timing consistency",
        ]
    )
    for name, consistency in report["stored_timing_consistency"].items():
        if consistency is None:
            lines.append(f"  {name}: unavailable")
        else:
            lines.append(
                f"  {name}: mismatches={consistency['mismatch_steps']} "
                f"max_abs_error={consistency['max_abs_error_ms']:.3f} ms"
            )

    findings = report["findings"]
    if findings:
        lines.extend(("", "Findings"))
        lines.extend(f"  - {finding}" for finding in findings)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="source ReplayBuffer .zarr")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of the terminal report",
    )
    args = parser.parse_args()
    report = diagnose_episode(args.dataset, args.episode)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
