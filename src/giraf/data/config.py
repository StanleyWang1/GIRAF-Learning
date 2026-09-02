"""Configuration loading and validation for the collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    output_dir: Path = Path("data/demos")
    zarr_name: str = "replay_buffer.zarr"
    episode_dir_name: str = "videos"
    aligned_hz: float = 30.0
    resize_dim: tuple[int, int] = (224, 224)
    resize_mode: str = "stretch"
    save_raw_video: bool = True
    raw_video_codec: str = "libx264"
    raw_video_crf: int = 21
    zarr_chunk_length: int = 16
    saver_batch_size: int = 16


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    max_control_age_ms: float = 50.0
    max_motor_age_ms: float = 50.0


@dataclass(frozen=True, slots=True)
class SharedMemoryConfig:
    get_time_budget_s: float = 0.10
    safety_margin: float = 1.5
    camera_history: int = 64
    control_history: int = 256
    motor_history: int = 256
    aligned_history: int = 128


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    shared_memory: SharedMemoryConfig = field(default_factory=SharedMemoryConfig)

    def validate(self) -> None:
        camera, dataset, shm = self.camera, self.dataset, self.shared_memory
        if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
            raise ValueError("camera dimensions and fps must be positive")
        if dataset.aligned_hz <= 0 or dataset.aligned_hz > camera.fps:
            raise ValueError("aligned_hz must be positive and no greater than camera fps")
        if dataset.resize_mode != "stretch":
            # TODO: letterbox/crop resize modes if aspect ratio matters for training.
            raise ValueError("only resize_mode=stretch is implemented")
        if len(dataset.resize_dim) != 2 or min(dataset.resize_dim) <= 0:
            raise ValueError("resize_dim must contain two positive integers")
        if not dataset.zarr_name.endswith(".zarr"):
            raise ValueError("zarr_name must end in .zarr")
        if dataset.zarr_chunk_length <= 0 or dataset.saver_batch_size <= 0:
            raise ValueError("Zarr chunk and saver batch sizes must be positive")
        if not 0 <= dataset.raw_video_crf <= 51:
            raise ValueError("raw_video_crf must be between 0 and 51")
        if min(self.alignment.max_control_age_ms, self.alignment.max_motor_age_ms) <= 0:
            raise ValueError("alignment age limits must be positive")
        if shm.get_time_budget_s <= 0 or shm.safety_margin < 1.0:
            raise ValueError("invalid shared-memory timing configuration")
        histories = (
            shm.camera_history,
            shm.control_history,
            shm.motor_history,
            shm.aligned_history,
        )
        if min(histories) <= 0:
            raise ValueError("shared-memory history sizes must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration snapshot."""

        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    @property
    def zarr_path(self) -> Path:
        return self.dataset.output_dir / self.dataset.zarr_name

    @property
    def video_dir(self) -> Path:
        return self.dataset.output_dir / self.dataset.episode_dir_name


_SECTIONS = {
    "camera": CameraConfig,
    "dataset": DatasetConfig,
    "alignment": AlignmentConfig,
    "shared_memory": SharedMemoryConfig,
}


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: {sorted(unknown)}")


def _section(raw: dict[str, Any], name: str, cls: type) -> dict[str, Any]:
    section = raw.get(name) or {}
    if not isinstance(section, dict):
        raise TypeError(f"configuration section {name!r} must be a mapping")
    _reject_unknown(section, set(cls.__dataclass_fields__), name)
    return dict(section)


def load_config(path: str | Path) -> CollectorConfig:
    """Load a strict YAML configuration file; unknown keys are errors."""

    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError("collector configuration root must be a mapping")
    _reject_unknown(raw, set(_SECTIONS), "root")

    sections = {name: _section(raw, name, cls) for name, cls in _SECTIONS.items()}
    dataset = sections["dataset"]
    if "output_dir" in dataset:
        dataset["output_dir"] = Path(dataset["output_dir"])
    if "resize_dim" in dataset:
        dataset["resize_dim"] = tuple(int(value) for value in dataset["resize_dim"])

    config = CollectorConfig(
        **{name: cls(**sections[name]) for name, cls in _SECTIONS.items()}
    )
    config.validate()
    return config
