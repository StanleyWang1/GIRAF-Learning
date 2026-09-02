"""Diffusion Policy-compatible Zarr ReplayBuffer writer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numcodecs
import numpy as np
import zarr

from .config import CollectorConfig
from .schema import (
    ACTION_FIELDS,
    EPISODE_META_KEYS,
    JOINT_FIELDS,
    SCHEMA_VERSION,
    STATE_FIELDS,
    TIME_DATA_KEYS,
)


def disk_compressor():
    return numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)


def git_revision() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


class ReplayBufferWriter:
    """Append staged episodes using Diffusion Policy's data/meta layout."""

    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        config.dataset.output_dir.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(config.zarr_path), mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        if "episode_ends" not in self.meta:
            self.meta.create_dataset(
                "episode_ends",
                shape=(0,),
                chunks=(1024,),
                dtype=np.int64,
                compressor=None,
            )
        self._set_schema_attributes()
        self.recover_uncommitted_tail()

    def _set_schema_attributes(self) -> None:
        attrs = self.root.attrs
        existing = attrs.get("schema_version")
        if existing is not None and existing != SCHEMA_VERSION:
            raise RuntimeError(
                f"dataset schema is {existing!r}, expected {SCHEMA_VERSION!r}"
            )
        expected = {
            "schema_version": SCHEMA_VERSION,
            "action_fields": list(ACTION_FIELDS),
            "joint_fields": list(JOINT_FIELDS),
            "state_fields": list(STATE_FIELDS),
            "state_semantics": "command-derived; not measured hardware feedback",
            "grasp_semantics": "operator command; not contact sensing",
            "image_layout": "THWC RGB uint8",
            "aligned_hz": float(self.config.dataset.aligned_hz),
            "resize_dim": list(self.config.dataset.resize_dim),
            "resize_mode": self.config.dataset.resize_mode,
            "motor_command_accepted_semantics": (
                "host dispatch calls returned successfully; not hardware feedback"
            ),
        }
        for key in ("aligned_hz", "resize_dim", "resize_mode"):
            if key in attrs and attrs[key] != expected[key]:
                raise RuntimeError(
                    f"dataset attribute {key}={attrs[key]!r} does not match "
                    f"collector setting {expected[key]!r}"
                )
        attrs.update(expected)
        attrs["last_collector_config"] = self.config.as_dict()
        revision = git_revision()
        revisions = list(attrs.get("git_revisions", []))
        if revision not in revisions:
            revisions.append(revision)
        attrs["git_revisions"] = revisions

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def n_episodes(self) -> int:
        return int(self.episode_ends.shape[0])

    @property
    def n_steps(self) -> int:
        if self.n_episodes == 0:
            return 0
        return int(self.episode_ends[-1])

    def recover_uncommitted_tail(self) -> None:
        """Truncate arrays not covered by the commit-marker episode ends."""

        committed_steps = self.n_steps
        for _name, array in self.data.arrays():
            if array.shape[0] < committed_steps:
                raise RuntimeError("ReplayBuffer data is shorter than episode_ends")
            if array.shape[0] > committed_steps:
                array.resize((committed_steps,) + array.shape[1:])
        committed_episodes = self.n_episodes
        for key in EPISODE_META_KEYS:
            if key in self.meta:
                array = self.meta[key]
                if array.shape[0] < committed_episodes:
                    raise RuntimeError(f"metadata {key} is shorter than episode_ends")
                if array.shape[0] > committed_episodes:
                    array.resize((committed_episodes,))

    def append_stage(
        self,
        stage_group,
        *,
        start_wall_time_ns: int,
        start_monotonic_ns: int,
        valid_steps: int,
        invalid_steps: int,
    ) -> tuple[int, int]:
        stage_data = stage_group["data"]
        keys = set(stage_data.array_keys())
        if keys != set(TIME_DATA_KEYS):
            missing = set(TIME_DATA_KEYS) - keys
            extra = keys - set(TIME_DATA_KEYS)
            raise RuntimeError(
                f"staged data keys differ from schema: missing={missing}, extra={extra}"
            )
        episode_length = int(stage_data["timestamp_ns"].shape[0])
        if episode_length <= 0:
            raise ValueError("cannot commit an empty episode")
        for key in TIME_DATA_KEYS:
            if stage_data[key].shape[0] != episode_length:
                raise RuntimeError(f"staged array {key} has a different time length")

        current = self.n_steps
        new_length = current + episode_length
        compressor = disk_compressor()

        # Validate every existing array before mutating any of them.
        for key in TIME_DATA_KEYS:
            source = stage_data[key]
            if key in self.data:
                target = self.data[key]
                if target.shape[1:] != source.shape[1:] or target.dtype != source.dtype:
                    raise RuntimeError(
                        f"existing ReplayBuffer schema mismatch for {key}"
                    )

        for key in TIME_DATA_KEYS:
            source = stage_data[key]
            if key not in self.data:
                chunks = (self.config.dataset.zarr_chunk_length,) + source.shape[1:]
                self.data.create_dataset(
                    key,
                    shape=(current,) + source.shape[1:],
                    chunks=chunks,
                    dtype=source.dtype,
                    compressor=compressor,
                )
            target = self.data[key]
            target.resize((new_length,) + target.shape[1:])
            copy_step = max(1, int(source.chunks[0]))
            for start in range(0, episode_length, copy_step):
                stop = min(episode_length, start + copy_step)
                target[current + start : current + stop] = source[start:stop]

        episode_meta = {
            "episode_start_wall_time_ns": np.int64(start_wall_time_ns),
            "episode_start_monotonic_ns": np.int64(start_monotonic_ns),
            "episode_valid_steps": np.int64(valid_steps),
            "episode_invalid_steps": np.int64(invalid_steps),
        }
        episode_index = self.n_episodes
        for key, value in episode_meta.items():
            if key not in self.meta:
                self.meta.create_dataset(
                    key,
                    shape=(episode_index,),
                    chunks=(1024,),
                    dtype=np.int64,
                    compressor=None,
                )
            target = self.meta[key]
            target.resize((episode_index + 1,))
            target[episode_index] = value

        # Commit marker: written only after every time-major and episode-level
        # array is durable enough for restart recovery.
        self.episode_ends.resize((episode_index + 1,))
        self.episode_ends[episode_index] = np.int64(new_length)
        return episode_index, episode_length
