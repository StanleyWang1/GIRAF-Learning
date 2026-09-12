"""Window a GIRAF ReplayBuffer into training batches of raw physical units."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import zarr

from giraf.data.schema import ACTION_DIM, ACTION_SPACES, GRASP_INDEX, STATE_DIM

from .normalize import Normalizer
from .policy import Batch
from .preprocess import IMU_INPUT_DIMS


def open_replay_group(path: str | Path) -> tuple[zarr.Group, zarr.ZipStore | None]:
    """Open a read-only directory or ZIP; the caller owns the returned ZIP store."""

    path = Path(path)
    if path.suffix.lower() != ".zip":
        return zarr.open_group(str(path), mode="r"), None

    store = zarr.ZipStore(str(path), mode="r")
    try:
        group_path = None
        if ".zgroup" not in store:
            candidates = {
                key.split("/", 1)[0]
                for key in store.keys()
                if key.count("/") == 1 and key.endswith("/.zgroup")
            }
            if len(candidates) != 1:
                raise ValueError(
                    "ZIP must contain a Zarr group at archive root or under "
                    "exactly one top-level directory"
                )
            group_path = candidates.pop()
        return zarr.open_group(store=store, mode="r", path=group_path), store
    except Exception:
        store.close()
        raise


def episode_windows(
    episode_ends: np.ndarray,
    valid: np.ndarray | None,
    *,
    observation_horizon: int,
    prediction_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step index tables ``obs_idx [W, To]`` and ``act_idx [W, Tp]``.

    One window is anchored at every step ``t``. Observations are the ``To``
    steps ending at ``t``; actions are the ``Tp`` steps starting at ``t - To + 1``
    (Diffusion Policy convention, so the executable chunk begins at ``t``).
    Indices are clamped inside the episode: the first observation and the last
    action repeat at the boundaries. Anchors with ``valid == 0`` are dropped.
    """

    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or (np.diff(ends) <= 0).any() or (ends.size and ends[0] <= 0):
        raise ValueError("episode_ends must be strictly increasing and positive")
    if observation_horizon <= 0 or prediction_horizon < observation_horizon:
        raise ValueError("horizons must satisfy 0 < To <= Tp")
    obs_offsets = np.arange(-observation_horizon + 1, 1)
    act_offsets = np.arange(
        -observation_horizon + 1, prediction_horizon - observation_horizon + 1
    )
    obs_blocks, act_blocks = [], []
    start = 0
    for end in ends.tolist():
        anchors = np.arange(start, end)
        if valid is not None:
            anchors = anchors[np.asarray(valid[start:end]) != 0]
        obs_blocks.append(np.clip(anchors[:, None] + obs_offsets, start, end - 1))
        act_blocks.append(np.clip(anchors[:, None] + act_offsets, start, end - 1))
        start = end
    if not obs_blocks:
        return (
            np.zeros((0, observation_horizon), np.int64),
            np.zeros((0, prediction_horizon), np.int64),
        )
    return np.concatenate(obs_blocks), np.concatenate(act_blocks)


def split_episodes(
    n_episodes: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Shuffle episode indices into disjoint sorted train and validation lists."""

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
    n_val = round(n_episodes * val_fraction)
    if val_fraction > 0 and n_episodes >= 2:
        n_val = max(1, min(n_val, n_episodes - 1))
    elif val_fraction == 0:
        n_val = 0
    order = np.random.default_rng(seed).permutation(n_episodes)
    val = sorted(int(i) for i in order[:n_val])
    train = sorted(int(i) for i in order[n_val:])
    return train, val


class ReplayDataset:
    """Re-iterable batch source over ``replay_buffer.zarr``.

    Low-dimensional arrays live in RAM. Images are read from Zarr per batch
    unless ``preload_images`` is set or ``preloaded_camera`` is supplied.
    Each ``__iter__`` reshuffles with
    ``seed + epoch`` so ``train(..., epochs=1)`` per epoch is deterministic.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int,
        observation_horizon: int = 2,
        prediction_horizon: int = 16,
        shuffle: bool = True,
        seed: int = 0,
        start_epoch: int = 0,
        preload_images: bool = False,
        preloaded_camera: np.ndarray | None = None,
        require_alignment_valid: bool = True,
        episodes: Sequence[int] | None = None,
        action_space: str = "twist",
        imu_input: str = "none",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_epoch < 0:
            raise ValueError("start_epoch must be non-negative")
        if action_space not in ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ACTION_SPACES}")
        if imu_input not in IMU_INPUT_DIMS:
            raise ValueError(f"imu_input must be one of {tuple(IMU_INPUT_DIMS)}")
        self.imu_dim = IMU_INPUT_DIMS[imu_input]
        self.path = Path(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = start_epoch

        root, self._store = open_replay_group(self.path)
        data = root["data"]
        if action_space == "twist":
            self.actions = np.asarray(data["action"][:], dtype=np.float32)
        else:
            joint_position = np.asarray(
                data["joint_position_command"][:, :6], dtype=np.float32
            )
            grasp = np.asarray(
                data["action"][:, GRASP_INDEX : GRASP_INDEX + 1], dtype=np.float32
            )
            self.actions = np.concatenate([joint_position, grasp], axis=1)
        self.states = np.asarray(data["state"][:], dtype=np.float32)
        n_steps = self.actions.shape[0]
        if self.actions.shape != (n_steps, ACTION_DIM):
            raise ValueError(f"data/action must be [T, {ACTION_DIM}]")
        if self.states.shape != (n_steps, STATE_DIM):
            raise ValueError(f"data/state must be [T, {STATE_DIM}]")
        self._camera = data["camera_rgb"]
        if (
            self._camera.ndim != 4
            or self._camera.shape[0] != n_steps
            or self._camera.shape[-1] != 3
        ):
            raise ValueError("data/camera_rgb must be [T, H, W, 3]")
        self.image_shape = tuple(int(v) for v in self._camera.shape[1:])
        if preloaded_camera is not None:
            if not isinstance(preloaded_camera, np.ndarray):
                raise TypeError("preloaded_camera must be a NumPy ndarray")
            if preloaded_camera.shape != self._camera.shape:
                raise ValueError("preloaded_camera shape must match data/camera_rgb")
            if preloaded_camera.dtype != np.uint8:
                raise TypeError("preloaded_camera dtype must be uint8")
            self._camera = preloaded_camera
        elif preload_images:
            self._camera = np.asarray(self._camera[:], dtype=np.uint8)

        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        if episode_ends.size and int(episode_ends[-1]) != n_steps:
            raise ValueError("meta/episode_ends does not cover data arrays")
        self._episode_ends = episode_ends
        valid = None
        if require_alignment_valid and "alignment_valid" in data:
            valid = np.asarray(data["alignment_valid"][:]) != 0
        self._episode_of_step = np.repeat(
            np.arange(len(episode_ends)), np.diff(episode_ends, prepend=0)
        )
        if episodes is None:
            self.episodes = list(range(len(episode_ends)))
        else:
            out_of_range = [e for e in episodes if not 0 <= e < len(episode_ends)]
            if out_of_range:
                raise ValueError(f"episode indices out of range: {out_of_range}")
            self.episodes = sorted(int(e) for e in episodes)
            episode_mask = np.isin(self._episode_of_step, self.episodes)
            valid = episode_mask if valid is None else valid & episode_mask
        self.obs_idx, self.act_idx = episode_windows(
            episode_ends,
            valid,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        self.imu = None
        if self.imu_dim:
            if "imu" not in data or "imu_sensor_valid" not in data:
                raise ValueError(
                    "IMU input requires data/imu and data/imu_sensor_valid"
                )
            self.imu = np.asarray(data["imu"][:], dtype=np.float32)
            sensor_valid = np.asarray(data["imu_sensor_valid"][:])
            if self.imu.shape != (n_steps, 10) or sensor_valid.shape != (n_steps, 3):
                raise ValueError("IMU arrays must have shapes [T, 10] and [T, 3]")
            sensor_count = 2 if self.imu_dim == 6 else 3
            imu_valid = (sensor_valid[:, :sensor_count] == 1).all(axis=1)
            imu_valid &= np.isfinite(self.imu[:, : self.imu_dim]).all(axis=1)
            if self.imu_dim == 10:
                norms = np.linalg.norm(self.imu[:, 6:], axis=1)
                imu_valid &= (norms >= 0.9) & (norms <= 1.1)
            # Drop entire observation windows, preserving the original time axis.
            keep = imu_valid[self.obs_idx].all(axis=1)
            self.obs_idx, self.act_idx = self.obs_idx[keep], self.act_idx[keep]
        if len(self.obs_idx) == 0:
            raise ValueError("dataset contains no usable training windows")
        # Lazy cameras need the ZIP open; all other arrays are already in RAM.
        if self._store is not None and self.preloaded_camera is not None:
            self._store.close()

    @property
    def preloaded_camera(self) -> np.ndarray | None:
        """The shared image array, or None when images are read lazily."""

        return self._camera if isinstance(self._camera, np.ndarray) else None

    @property
    def n_windows(self) -> int:
        """Total number of training windows kept after filtering."""

        return len(self.obs_idx)

    @property
    def n_episodes(self) -> int:
        """Total number of episodes in the underlying replay buffer."""

        return len(self._episode_ends)

    def __len__(self) -> int:
        return -(-self.n_windows // self.batch_size)

    def fit_normalizer(self) -> Normalizer:
        """Fit on training episodes; IMU uses only retained observation rows."""

        mask = np.isin(self._episode_of_step, self.episodes)
        imu = None
        if self.imu is not None:
            imu = self.imu[np.unique(self.obs_idx), : self.imu_dim]
        return Normalizer.fit(self.actions[mask], self.states[mask], imu)

    def _images(self, indices: np.ndarray) -> np.ndarray:
        flat = indices.reshape(-1)
        if isinstance(self._camera, np.ndarray):
            return self._camera[flat].reshape(*indices.shape, *self.image_shape)
        unique, inverse = np.unique(flat, return_inverse=True)
        frames = self._camera.oindex[unique]
        return frames[inverse].reshape(*indices.shape, *self.image_shape)

    def __iter__(self):
        order = np.arange(self.n_windows)
        if self.shuffle:
            order = np.random.default_rng(self.seed + self._epoch).permutation(order)
        self._epoch += 1
        for start in range(0, self.n_windows, self.batch_size):
            rows = order[start : start + self.batch_size]
            obs_idx, act_idx = self.obs_idx[rows], self.act_idx[rows]
            observations = {
                "camera_rgb": self._images(obs_idx),
                "state": self.states[obs_idx],
            }
            if self.imu is not None:
                observations["imu"] = self.imu[obs_idx]
            yield Batch(
                observations=observations,
                actions=self.actions[act_idx],
            )
