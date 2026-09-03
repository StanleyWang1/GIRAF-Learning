"""Read an explicit deployment start pose from a replay dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from giraf.data.schema import SCHEMA_VERSION, STATE_DIM

from .safety import state_from_joints


@dataclass(frozen=True, slots=True)
class ReferenceStart:
    dataset: Path
    episode: int
    step: int
    joints: np.ndarray
    state: np.ndarray
    camera_rgb: np.ndarray


def load_reference_start(dataset: str | Path, episode: int) -> ReferenceStart:
    """Load and cross-check the first command-derived state of one episode."""

    path = Path(dataset)
    if not path.is_dir():
        raise FileNotFoundError(f"reference dataset does not exist: {path}")
    if episode < 0:
        raise ValueError("reference episode must be non-negative")

    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported reference dataset schema: {path}")
    for key in (
        "meta/episode_ends",
        "data/joint_position_command",
        "data/state",
        "data/camera_rgb",
    ):
        if key not in root:
            raise ValueError(f"reference dataset is missing {key}")

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("reference dataset has no complete episodes")
    if np.any(np.diff(episode_ends) <= 0) or episode_ends[0] <= 0:
        raise ValueError("reference episode boundaries are invalid")
    if episode >= episode_ends.size:
        raise ValueError(
            f"reference episode {episode} is out of range; "
            f"dataset contains {episode_ends.size} episodes"
        )

    step = 0 if episode == 0 else int(episode_ends[episode - 1])
    joints = np.asarray(root["data/joint_position_command"][step], dtype=np.float32)
    stored_state = np.asarray(root["data/state"][step], dtype=np.float32)
    camera_rgb = np.asarray(root["data/camera_rgb"][step], dtype=np.uint8)
    if (
        joints.shape != (6,)
        or stored_state.shape != (STATE_DIM,)
        or camera_rgb.ndim != 3
        or camera_rgb.shape[-1] != 3
    ):
        raise ValueError("reference start arrays have unexpected shapes")
    calculated_state = state_from_joints(joints)
    if not np.allclose(calculated_state, stored_state, rtol=1e-5, atol=1e-5):
        raise ValueError("reference state does not match its joint command")

    return ReferenceStart(
        dataset=path,
        episode=episode,
        step=step,
        joints=joints,
        state=calculated_state,
        camera_rgb=camera_rgb,
    )
