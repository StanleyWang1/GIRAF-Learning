"""Window a GIRAF ReplayBuffer into training batches of raw physical units."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from giraf.data.schema import ACTION_DIM, STATE_DIM

from .normalize import Normalizer
from .policy import Batch


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


class ReplayDataset:
    """Re-iterable batch source over ``replay_buffer.zarr``.

    Low-dimensional arrays live in RAM. Images are read from Zarr per batch
    unless ``preload_images`` is set. Each ``__iter__`` reshuffles with
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
        preload_images: bool = False,
        require_alignment_valid: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.path = Path(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        root = zarr.open_group(str(self.path), mode="r")
        data = root["data"]
        self.actions = np.asarray(data["action"][:], dtype=np.float32)
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
        if preload_images:
            self._camera = np.asarray(self._camera[:], dtype=np.uint8)

        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        if episode_ends.size and int(episode_ends[-1]) != n_steps:
            raise ValueError("meta/episode_ends does not cover data arrays")
        valid = None
        if require_alignment_valid and "alignment_valid" in data:
            valid = np.asarray(data["alignment_valid"][:])
        self.obs_idx, self.act_idx = episode_windows(
            episode_ends,
            valid,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        if len(self.obs_idx) == 0:
            raise ValueError("dataset contains no usable training windows")

    @property
    def n_windows(self) -> int:
        return len(self.obs_idx)

    def __len__(self) -> int:
        return -(-self.n_windows // self.batch_size)

    def fit_normalizer(self) -> Normalizer:
        return Normalizer.fit(self.actions, self.states)

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
            yield Batch(
                observations={
                    "camera_rgb": self._images(obs_idx),
                    "state": self.states[obs_idx],
                },
                actions=self.actions[act_idx],
            )
