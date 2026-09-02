"""Configuration loading and validation for the collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    color_mode: str = "rgb"
    include_depth: bool = False


@dataclass(frozen=True, slots=True)
class StreamConfig:
    action_hz: float = 100.0
    state_hz: float = 100.0
    grasp_hz: float = 100.0
    motor_hz: float = 100.0


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    output_dir: Path = Path("data/giraf_demos")
    zarr_name: str = "replay_buffer.zarr"
    episode_dir_name: str = "videos"
    aligned_hz: float = 30.0
    resize_dim: tuple[int, int] = (224, 224)
    resize_mode: str = "stretch"
    normalization: str = "none"
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
    streams: StreamConfig = field(default_factory=StreamConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    shared_memory: SharedMemoryConfig = field(default_factory=SharedMemoryConfig)

    def validate(self) -> None:
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ValueError("camera dimensions and fps must be positive")
        if self.camera.color_mode != "rgb":
            raise ValueError(
                "v1 collection stores RGB; convert grayscale during training"
            )
        if self.camera.include_depth:
            raise ValueError("depth collection is reserved but not implemented in v1")
        if self.dataset.aligned_hz <= 0 or self.dataset.aligned_hz > self.camera.fps:
            raise ValueError(
                "aligned_hz must be positive and no greater than camera fps"
            )
        if self.dataset.resize_mode != "stretch":
            raise ValueError(
                "this configuration currently supports resize_mode=stretch"
            )
        if self.dataset.normalization != "none":
            raise ValueError("store uint8 RGB and normalize in the training loader")
        if len(self.dataset.resize_dim) != 2 or min(self.dataset.resize_dim) <= 0:
            raise ValueError("resize_dim must contain two positive integers")
        if not self.dataset.zarr_name.endswith(".zarr"):
            raise ValueError("zarr_name must end in .zarr")
        if self.dataset.zarr_chunk_length <= 0 or self.dataset.saver_batch_size <= 0:
            raise ValueError("Zarr chunk and saver batch sizes must be positive")
        if not 0 <= self.dataset.raw_video_crf <= 51:
            raise ValueError("raw_video_crf must be between 0 and 51")
        if (
            min(
                self.streams.action_hz,
                self.streams.state_hz,
                self.streams.grasp_hz,
                self.streams.motor_hz,
            )
            <= 0
        ):
            raise ValueError("stream rates must be positive")
        if (
            min(
                self.alignment.max_control_age_ms,
                self.alignment.max_motor_age_ms,
            )
            <= 0
        ):
            raise ValueError("alignment age limits must be positive")
        shm = self.shared_memory
        if shm.get_time_budget_s <= 0 or shm.safety_margin < 1.0:
            raise ValueError("invalid shared-memory timing configuration")
        if (
            min(
                shm.camera_history,
                shm.control_history,
                shm.motor_history,
                shm.aligned_history,
            )
            <= 0
        ):
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


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"configuration section {name!r} must be a mapping")
    return value


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: {sorted(unknown)}")


def load_config(path: str | Path) -> CollectorConfig:
    """Load a strict YAML configuration file."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("data collection requires PyYAML") from exc

    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("collector configuration root must be a mapping")
    _reject_unknown(
        raw,
        {"camera", "streams", "dataset", "alignment", "shared_memory"},
        "root",
    )

    camera_raw = _section(raw, "camera")
    streams_raw = _section(raw, "streams")
    dataset_raw = _section(raw, "dataset")
    alignment_raw = _section(raw, "alignment")
    shm_raw = _section(raw, "shared_memory")
    _reject_unknown(camera_raw, set(CameraConfig.__dataclass_fields__), "camera")
    _reject_unknown(streams_raw, set(StreamConfig.__dataclass_fields__), "streams")
    _reject_unknown(dataset_raw, set(DatasetConfig.__dataclass_fields__), "dataset")
    _reject_unknown(
        alignment_raw, set(AlignmentConfig.__dataclass_fields__), "alignment"
    )
    _reject_unknown(
        shm_raw, set(SharedMemoryConfig.__dataclass_fields__), "shared_memory"
    )

    if "output_dir" in dataset_raw:
        dataset_raw["output_dir"] = Path(dataset_raw["output_dir"])
    if "resize_dim" in dataset_raw:
        dataset_raw["resize_dim"] = tuple(
            int(value) for value in dataset_raw["resize_dim"]
        )

    config = CollectorConfig(
        camera=CameraConfig(**camera_raw),
        streams=StreamConfig(**streams_raw),
        dataset=DatasetConfig(**dataset_raw),
        alignment=AlignmentConfig(**alignment_raw),
        shared_memory=SharedMemoryConfig(**shm_raw),
    )
    config.validate()
    return config
