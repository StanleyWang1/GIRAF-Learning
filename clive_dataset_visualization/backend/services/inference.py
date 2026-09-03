from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import cv2
import numpy as np
from omegaconf import OmegaConf

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.key_registry import parse_camera_key
from utils.helper import load_hydra_config_with_defaults


PNG_COMPRESSION = 1
DEFAULT_SERVER_PORT = 8767
MAX_BATCH_SIZE = 32
TRAIN_TIME_INFERENCE_MODE = "train_time"
DEFAULT_INFERENCE_MODE = TRAIN_TIME_INFERENCE_MODE
_VALID_INFERENCE_MODES = {TRAIN_TIME_INFERENCE_MODE}

_TASK_CONFIG_CACHE: Dict[str, tuple[int, Dict[str, Any], bool, str]] = {}


class InferenceConfigError(ValueError):
    pass


@dataclass
class InferenceTaskConfig:
    path: str
    action_downsample_steps: int
    obs_specs: Dict[str, Dict[str, Any]]
    temporal_profiles: Dict[str, Dict[str, "TemporalRangeSpec"]]
    branches: tuple[str, ...]
    max_horizon_by_branch: Dict[str, int]
    max_obs_steps_by_branch: Dict[str, int]
    obs_pose_repr: str = ""


@dataclass(frozen=True)
class TemporalRangeSpec:
    horizon: int
    down_sample_steps: int


class ZmqInferenceTransport:
    def __init__(self, server_host: str, server_port: int):
        try:
            import zmq
        except Exception as exc:  # pragma: no cover - exercised only in runtime envs missing pyzmq
            raise RuntimeError("pyzmq is required for inference overlay support") from exc

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 10_000)
        self._socket.setsockopt(zmq.SNDTIMEO, 10_000)
        self._socket.connect(f"tcp://{server_host}:{server_port}")

    def request(self, payload: Mapping[str, Any]) -> Any:
        self._socket.send_pyobj(dict(payload))
        return self._socket.recv_pyobj()

    def close(self) -> None:
        self._socket.close()
        self._context.term()


def create_zmq_transport(server_host: str, server_port: int) -> ZmqInferenceTransport:
    return ZmqInferenceTransport(server_host=server_host, server_port=server_port)


def _normalize_yaml_path(yaml_path: str) -> Path:
    if not yaml_path or not str(yaml_path).strip():
        raise InferenceConfigError("yaml_path is required")

    path = Path(str(yaml_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"inference yaml not found: {path}")
    if not path.is_file():
        raise InferenceConfigError(f"inference yaml is not a file: {path}")
    return path


def _load_task_config_dict(yaml_path: str) -> tuple[Dict[str, Any], bool, str]:
    path = _normalize_yaml_path(yaml_path)
    cache_key = str(path)
    mtime_ns = int(path.stat().st_mtime_ns)
    cached = _TASK_CONFIG_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1], cached[2], cached[3]

    cfg = None
    try:
        cfg = load_hydra_config_with_defaults(str(path))
    except Exception:
        cfg = OmegaConf.load(str(path))

    train_time_short_enabled = OmegaConf.select(
        cfg,
        "policy.obs_encoder.short_range.obs_window",
        default=None,
    ) is not None

    obs_pose_repr = str(OmegaConf.select(cfg, "task.pose_repr.obs_pose_repr", default="") or "")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise InferenceConfigError(f"failed to load inference yaml: {path}")

    task_cfg = cfg_dict.get("task")
    if not isinstance(task_cfg, dict):
        raise InferenceConfigError(f"missing task section in inference yaml: {path}")

    _TASK_CONFIG_CACHE[cache_key] = (mtime_ns, task_cfg, bool(train_time_short_enabled), obs_pose_repr)
    return task_cfg, bool(train_time_short_enabled), obs_pose_repr


def _require_positive_int(value: Any, field_name: str, path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        raise InferenceConfigError(f"invalid {field_name} in inference yaml: {path}")
    return parsed


def _resolve_obs_shape(spec: Mapping[str, Any]) -> Sequence[int]:
    raw_shape = spec.get("raw_shape")
    if isinstance(raw_shape, (list, tuple)) and raw_shape:
        return [int(v) for v in raw_shape]
    shape = spec.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        return [int(v) for v in shape]
    return []


def _resolve_downsample_steps(key: str, spec: Mapping[str, Any], field_name: str) -> int:
    try:
        downsample = int(spec.get(field_name) or 0)
    except (TypeError, ValueError):
        downsample = 0
    if downsample <= 0:
        raise InferenceConfigError(f"missing valid {field_name} for observation key: {key}")
    return downsample


def _resolve_optional_positive_int(key: str, spec: Mapping[str, Any], field_name: str) -> int | None:
    if field_name not in spec or spec.get(field_name) is None:
        return None
    try:
        parsed = int(spec.get(field_name))
    except (TypeError, ValueError) as exc:
        raise InferenceConfigError(f"invalid {field_name} for observation key: {key}") from exc
    if parsed <= 0:
        raise InferenceConfigError(f"invalid {field_name} for observation key: {key}")
    return parsed


def _validate_downsample_field(task_cfg: InferenceTaskConfig, field_name: str) -> None:
    for key, spec in task_cfg.obs_specs.items():
        _resolve_downsample_steps(key, spec, field_name)


def _resolve_temporal_profiles(
    key: str,
    spec: Mapping[str, Any],
) -> Dict[str, TemporalRangeSpec]:
    temporal_profile = spec.get("temporal_profile")
    if temporal_profile is None:
        horizon = spec.get("horizon")
        short_dss = spec.get("short_dss")
        long_dss = spec.get("long_dss")
        if horizon is not None and short_dss is not None and long_dss is not None:
            shared_horizon = _require_positive_int(horizon, f"obs.{key}.horizon", Path(key))
            return {
                "short": TemporalRangeSpec(
                    horizon=shared_horizon,
                    down_sample_steps=_require_positive_int(short_dss, f"obs.{key}.short_dss", Path(key)),
                ),
                "long": TemporalRangeSpec(
                    horizon=shared_horizon,
                    down_sample_steps=_require_positive_int(long_dss, f"obs.{key}.long_dss", Path(key)),
                ),
            }
        if horizon is not None and spec.get("down_sample_steps") is not None:
            return {
                "short": TemporalRangeSpec(
                    horizon=_require_positive_int(horizon, f"obs.{key}.horizon", Path(key)),
                    down_sample_steps=_require_positive_int(
                        spec.get("down_sample_steps"),
                        f"obs.{key}.down_sample_steps",
                        Path(key),
                    ),
                )
            }
        raise InferenceConfigError(
            f"observation key '{key}' must define temporal_profile or legacy horizon/down_sample_steps"
        )

    if not isinstance(temporal_profile, Mapping):
        raise InferenceConfigError(f"obs.{key}.temporal_profile must be a mapping")
    if not temporal_profile:
        raise InferenceConfigError(f"obs.{key}.temporal_profile must be non-empty")

    resolved: Dict[str, TemporalRangeSpec] = {}
    for raw_branch, raw_spec in temporal_profile.items():
        branch = str(raw_branch)
        if branch not in {"short", "long"}:
            raise InferenceConfigError(
                f"observation key '{key}' uses unsupported temporal branch '{branch}'"
            )
        if not isinstance(raw_spec, Mapping):
            raise InferenceConfigError(f"obs.{key}.temporal_profile.{branch} must be a mapping")
        if "horizon" not in raw_spec or "down_sample_steps" not in raw_spec:
            raise InferenceConfigError(
                f"observation key '{key}' branch '{branch}' must define horizon and down_sample_steps"
            )
        horizon = _require_positive_int(raw_spec.get("horizon"), f"obs.{key}.temporal_profile.{branch}.horizon", Path(key))
        down_sample_steps = _require_positive_int(
            raw_spec.get("down_sample_steps"),
            f"obs.{key}.temporal_profile.{branch}.down_sample_steps",
            Path(key),
        )
        resolved[branch] = TemporalRangeSpec(horizon=horizon, down_sample_steps=down_sample_steps)

    return resolved


def load_inference_task_config(yaml_path: str) -> InferenceTaskConfig:
    path = _normalize_yaml_path(yaml_path)
    loaded_cfg, _, obs_pose_repr = _load_task_config_dict(str(path))

    task_cfg = loaded_cfg.get("task") if isinstance(loaded_cfg.get("task"), Mapping) else loaded_cfg

    shape_meta = task_cfg.get("shape_meta", {})
    obs_specs = shape_meta.get("obs")
    if not isinstance(obs_specs, dict) or not obs_specs:
        raise InferenceConfigError(f"missing task.shape_meta.obs in inference yaml: {path}")

    action_spec = shape_meta.get("action")
    if not isinstance(action_spec, Mapping):
        raise InferenceConfigError(f"missing task.shape_meta.action in inference yaml: {path}")
    action_downsample_steps = _require_positive_int(
        action_spec.get("down_sample_steps", action_spec.get("dss")),
        "task.shape_meta.action.down_sample_steps",
        path,
    )

    normalized_obs_specs: Dict[str, Dict[str, Any]] = {}
    temporal_profiles: Dict[str, Dict[str, TemporalRangeSpec]] = {}
    branch_usage: set[str] = set()
    for key, value in obs_specs.items():
        if not isinstance(value, Mapping):
            raise InferenceConfigError(f"invalid task.shape_meta.obs.{key} in inference yaml: {path}")
        spec = dict(value)
        key_profiles = _resolve_temporal_profiles(str(key), spec)
        temporal_profiles[str(key)] = key_profiles
        branch_usage.update(key_profiles.keys())
        normalized_obs_specs[str(key)] = spec

    if "short" not in branch_usage:
        raise InferenceConfigError(f"task.shape_meta.obs in inference yaml must define a short temporal branch: {path}")
    default_temporal_profile = task_cfg.get("temporal_profile")
    if isinstance(default_temporal_profile, Mapping) and default_temporal_profile:
        allowed_branches = tuple(branch for branch in ("short", "long") if branch in default_temporal_profile)
        if not allowed_branches:
            raise InferenceConfigError(f"task.temporal_profile must define at least one branch: {path}")
        if not branch_usage.issubset(set(allowed_branches)):
            raise InferenceConfigError(
                f"task.shape_meta.obs uses branches {sorted(branch_usage)} not present in task.temporal_profile: {path}"
            )
        unused_branches = [branch for branch in allowed_branches if branch not in branch_usage]
        if unused_branches:
            raise InferenceConfigError(
                f"task.temporal_profile branches unused by any observation key: {unused_branches}"
            )

    branches = tuple(branch for branch in ("short", "long") if branch in branch_usage)

    max_horizon_by_branch: Dict[str, int] = {}
    max_obs_steps_by_branch: Dict[str, int] = {}
    for branch in branches:
        max_horizon = 0
        max_obs_steps = 0
        for key_profiles in temporal_profiles.values():
            if branch not in key_profiles:
                continue
            spec = key_profiles[branch]
            max_horizon = max(max_horizon, int(spec.horizon))
            max_obs_steps = max(max_obs_steps, max(0, (int(spec.horizon) - 1) * int(spec.down_sample_steps)))
        max_horizon_by_branch[branch] = max_horizon
        max_obs_steps_by_branch[branch] = max_obs_steps

    return InferenceTaskConfig(
        path=str(path),
        action_downsample_steps=action_downsample_steps,
        obs_specs=normalized_obs_specs,
        temporal_profiles=temporal_profiles,
        branches=branches,
        max_horizon_by_branch=max_horizon_by_branch,
        max_obs_steps_by_branch=max_obs_steps_by_branch,
        obs_pose_repr=obs_pose_repr,
    )


def _normalize_inference_mode(inference_mode: str | None) -> str:
    normalized = str(inference_mode or DEFAULT_INFERENCE_MODE).strip().lower()
    if normalized not in _VALID_INFERENCE_MODES:
        valid_modes = ", ".join(sorted(_VALID_INFERENCE_MODES))
        raise InferenceConfigError(f"invalid inference_mode {inference_mode!r}; expected one of: {valid_modes}")
    return normalized


def _is_pose_obs_key(key: str) -> bool:
    return key.endswith("eef_pos") or key.endswith("eef_rpy")


def _effective_horizon(key: str, horizon: int, obs_pose_repr: str) -> int:
    if obs_pose_repr == "relative_mod" and _is_pose_obs_key(key):
        return horizon + 1
    return horizon


def _fmt_indices(indices: List[int]) -> str:
    if not indices:
        return "[]"
    if len(indices) == 1:
        return f"[{indices[0]}]"
    step = indices[1] - indices[0] if len(indices) > 1 else 1
    is_uniform = all(indices[i] - indices[i - 1] == step for i in range(1, len(indices)))
    if is_uniform and step == 1:
        return f"[{indices[0]}..{indices[-1]}]"
    if is_uniform:
        return f"[{indices[0]}..{indices[-1]}:{step}]"
    return str(indices)


def _sample_frame_indices(frame_index: int, horizon: int, downsample_steps: int) -> List[int]:
    if horizon <= 0:
        return []
    return [max(0, frame_index - (horizon - 1 - offset) * downsample_steps) for offset in range(horizon)]


def _resolve_observation_horizon(
    key: str,
    spec: Mapping[str, Any],
    default_horizon: int,
    horizon_field: str | None = None,
    camera_horizon: int | None = None,
    numeric_horizon: int | None = None,
) -> int:
    if horizon_field:
        resolved = _resolve_optional_positive_int(key, spec, horizon_field)
        if resolved is not None:
            return resolved

    parsed = parse_camera_key(key)
    horizon = default_horizon
    if parsed is not None:
        if camera_horizon is not None:
            horizon = camera_horizon
    elif numeric_horizon is not None:
        horizon = numeric_horizon

    try:
        parsed_horizon = int(horizon)
    except (TypeError, ValueError) as exc:
        raise InferenceConfigError(f"invalid observation horizon for key {key}: {horizon!r}") from exc
    if parsed_horizon <= 0:
        raise InferenceConfigError(f"invalid observation horizon for key {key}: {parsed_horizon}")
    return parsed_horizon


def _observation_sampling_details(
    task_cfg: InferenceTaskConfig,
    frame_index: int,
    horizon: int,
    downsample_field: str,
    horizon_field: str | None = None,
    camera_horizon: int | None = None,
    numeric_horizon: int | None = None,
    branch: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    details_by_key: Dict[str, Dict[str, Any]] = {}
    if task_cfg.temporal_profiles:
        branch_name = branch or (task_cfg.branches[0] if task_cfg.branches else "short")
        for key, spec in task_cfg.temporal_profiles.items():
            branch_spec = spec.get(branch_name)
            if branch_spec is None:
                continue
            horizon_used = _effective_horizon(key, branch_spec.horizon, task_cfg.obs_pose_repr)
            frame_indices = [int(idx) for idx in _sample_frame_indices(frame_index, horizon_used, branch_spec.down_sample_steps)]
            details_by_key[key] = {
                "down_sample_steps": int(branch_spec.down_sample_steps),
                "horizon_used": int(horizon_used),
                "frame_indices": frame_indices,
                "branch": branch_name,
            }
        return details_by_key

    for key, spec in task_cfg.obs_specs.items():
        downsample_steps = _resolve_downsample_steps(key, spec, downsample_field)
        horizon_used = _resolve_observation_horizon(
            key,
            spec,
            default_horizon=horizon,
            horizon_field=horizon_field,
            camera_horizon=camera_horizon,
            numeric_horizon=numeric_horizon,
        )
        horizon_used = _effective_horizon(key, horizon_used, task_cfg.obs_pose_repr)
        frame_indices = [int(idx) for idx in _sample_frame_indices(frame_index, horizon_used, downsample_steps)]
        details_by_key[key] = {
            "down_sample_steps": int(downsample_steps),
            "horizon_used": int(horizon_used),
            "frame_indices": frame_indices,
            "downsample_field": str(downsample_field),
        }
    return details_by_key


def _fetch_numeric_sample(adapter: BaseAdapter, episode_index: int, key: str, frame_index: int) -> np.ndarray:
    sample = np.asarray(adapter.signal_window(episode_index, key, frame_index, frame_index + 1, 1))
    if sample.shape[0] <= 0:
        raise ValueError(f"empty observation sample for key={key} at frame={frame_index}")
    return np.asarray(sample[0])


def _reshape_numeric_sample(key: str, sample: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)
    if not target_shape:
        return arr.astype(np.float32, copy=False)
    try:
        return arr.reshape(tuple(target_shape)).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"observation key {key} has sample shape {tuple(arr.shape)} but expected {tuple(target_shape)}"
        ) from exc


def _encode_png(frame: np.ndarray) -> np.ndarray:
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(frame), [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION])
    if not ok:
        raise RuntimeError("failed to encode frame as png")
    return encoded


def _build_rgb_observation(
    adapter: BaseAdapter,
    episode_index: int,
    stream_id: str,
    frame_indices: Sequence[int],
    spec: Mapping[str, Any],
) -> List[np.ndarray]:
    target_shape = _resolve_obs_shape(spec)
    if len(target_shape) != 3:
        raise InferenceConfigError(f"RGB observation expects shape [C,H,W], got {target_shape} for {stream_id}")

    _, height, width = target_shape
    encoded_frames: List[np.ndarray] = []
    for frame_index in frame_indices:
        frame = np.asarray(adapter.frame_rgb(episode_index, stream_id, frame_index))
        if frame.ndim != 3:
            raise ValueError(f"RGB frame for stream {stream_id} has invalid shape {frame.shape}")
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        encoded_frames.append(_encode_png(resized))
    return encoded_frames


def _build_depth_observation(
    adapter: BaseAdapter,
    episode_index: int,
    stream_id: str,
    frame_indices: Sequence[int],
    spec: Mapping[str, Any],
) -> List[np.ndarray]:
    target_shape = _resolve_obs_shape(spec)
    if len(target_shape) == 3:
        _, height, width = target_shape
    elif len(target_shape) == 2:
        height, width = target_shape
    else:
        raise InferenceConfigError(f"Depth observation expects shape [H,W] or [1,H,W], got {target_shape} for {stream_id}")

    encoded_frames: List[np.ndarray] = []
    for frame_index in frame_indices:
        frame = np.asarray(adapter.frame_depth(episode_index, stream_id, frame_index))
        if frame.ndim != 2:
            raise ValueError(f"depth frame for stream {stream_id} has invalid shape {frame.shape}")
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        encoded_frames.append(_encode_png(resized))
    return encoded_frames


def build_inference_observation(
    adapter: BaseAdapter,
    episode_index: int,
    frame_index: int,
    task_cfg: InferenceTaskConfig,
    k: int | None = None,
    frame_indices_override: Mapping[str, Sequence[int]] | None = None,
    downsample_field: str = "short_dss",
    add_numeric_batch_dim: bool = True,
    horizon_field: str | None = None,
    camera_horizon: int | None = None,
    numeric_horizon: int | None = None,
    branch: str | None = None,
) -> Dict[str, Any]:
    schema = adapter.episode_schema(episode_index)
    if frame_index < 0 or frame_index >= schema.length:
        raise IndexError(f"frame index {frame_index} out of range [0, {max(0, schema.length - 1)}]")

    dataset_keys = set(adapter.all_keys())
    stream_ids = {camera.stream_id for camera in schema.cameras}
    depth_stream_ids = {camera.stream_id for camera in schema.cameras if camera.depth_source is not None}

    obs: Dict[str, Any] = {}
    default_horizon = int(k or 1)
    if task_cfg.temporal_profiles:
        branch_name = branch or (task_cfg.branches[0] if task_cfg.branches else "short")
        for key, spec in task_cfg.obs_specs.items():
            parsed = parse_camera_key(key)
            branch_spec = task_cfg.temporal_profiles.get(key, {}).get(branch_name)
            if branch_spec is None:
                continue
            if frame_indices_override is not None and key in frame_indices_override:
                frame_indices = [int(idx) for idx in frame_indices_override[key]]
                if len(frame_indices) <= 0:
                    raise InferenceConfigError(f"frame_indices_override for key {key} is empty")
            else:
                horizon_used = _effective_horizon(key, branch_spec.horizon, task_cfg.obs_pose_repr)
                frame_indices = _sample_frame_indices(frame_index, horizon_used, branch_spec.down_sample_steps)

            if parsed is not None:
                stream_id, modality = parsed
                if modality == "rgb":
                    if stream_id not in stream_ids:
                        raise KeyError(f"camera stream missing for observation key: {key}")
                    obs[key] = _build_rgb_observation(adapter, episode_index, stream_id, frame_indices, spec)
                    continue
                if modality == "depth":
                    if stream_id not in depth_stream_ids:
                        raise KeyError(f"depth stream missing for observation key: {key}")
                    obs[key] = _build_depth_observation(adapter, episode_index, stream_id, frame_indices, spec)
                    continue

            if key not in dataset_keys:
                raise KeyError(f"dataset does not contain required observation key: {key}")

            target_shape = _resolve_obs_shape(spec)
            samples = [
                _reshape_numeric_sample(key, _fetch_numeric_sample(adapter, episode_index, key, idx), target_shape)
                for idx in frame_indices
            ]
            numeric_obs = np.stack(samples, axis=0).astype(np.float32, copy=False)
            if add_numeric_batch_dim:
                numeric_obs = np.expand_dims(numeric_obs, axis=0)
            obs[key] = numeric_obs
        return obs

    if default_horizon <= 0:
        raise InferenceConfigError("observation horizon must be positive")

    for key, spec in task_cfg.obs_specs.items():
        parsed = parse_camera_key(key)
        if frame_indices_override is not None and key in frame_indices_override:
            frame_indices = [int(idx) for idx in frame_indices_override[key]]
            if len(frame_indices) <= 0:
                raise InferenceConfigError(f"frame_indices_override for key {key} is empty")
        else:
            downsample_steps = _resolve_downsample_steps(key, spec, downsample_field)
            key_horizon = _resolve_observation_horizon(
                key,
                spec,
                default_horizon=default_horizon,
                horizon_field=horizon_field,
                camera_horizon=camera_horizon,
                numeric_horizon=numeric_horizon,
            )
            key_horizon = _effective_horizon(key, key_horizon, task_cfg.obs_pose_repr)
            frame_indices = _sample_frame_indices(frame_index, key_horizon, downsample_steps)

        if parsed is not None:
            stream_id, modality = parsed
            if modality == "rgb":
                if stream_id not in stream_ids:
                    raise KeyError(f"camera stream missing for observation key: {key}")
                obs[key] = _build_rgb_observation(adapter, episode_index, stream_id, frame_indices, spec)
                continue
            if modality == "depth":
                if stream_id not in depth_stream_ids:
                    raise KeyError(f"depth stream missing for observation key: {key}")
                obs[key] = _build_depth_observation(adapter, episode_index, stream_id, frame_indices, spec)
                continue

        if key not in dataset_keys:
            raise KeyError(f"dataset does not contain required observation key: {key}")

        target_shape = _resolve_obs_shape(spec)
        samples = [
            _reshape_numeric_sample(key, _fetch_numeric_sample(adapter, episode_index, key, idx), target_shape)
            for idx in frame_indices
        ]
        numeric_obs = np.stack(samples, axis=0).astype(np.float32, copy=False)
        if add_numeric_batch_dim:
            numeric_obs = np.expand_dims(numeric_obs, axis=0)
        obs[key] = numeric_obs

    return obs


def _extract_action_payload(action_response: Any) -> np.ndarray:
    # Some servers return traceback text when inference fails; surface that
    # directly instead of trying to coerce it into a float array.
    if isinstance(action_response, (str, bytes)):
        raise RuntimeError(f"inference server returned an error:\n{action_response}")

    if isinstance(action_response, Mapping):
        status = action_response.get("status")
        if isinstance(status, str) and status.lower() == "error":
            detail = action_response.get("detail") or action_response.get("error") or action_response.get("message")
            raise RuntimeError(f"inference server returned an error:\n{detail or action_response}")
        for key in ("action", "actions"):
            if key in action_response:
                action_response = action_response[key]
                break

    if isinstance(action_response, (str, bytes)):
        raise RuntimeError(f"inference server returned an error:\n{action_response}")

    try:
        action_array = np.asarray(action_response, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"predicted action payload has unsupported type {type(action_response).__name__}: {action_response!r}"
        ) from exc
    if action_array.ndim == 0:
        raise ValueError("predicted action payload is empty")
    if action_array.ndim not in {1, 2, 3}:
        raise ValueError(f"predicted action must be rank 1, 2, or 3, got shape {tuple(action_array.shape)}")
    return action_array


def _extract_action_batches(action_response: Any, no_gripper: bool = False) -> tuple[np.ndarray, List[np.ndarray]]:
    action_array = _extract_action_payload(action_response)
    if action_array.ndim == 1:
        action_batches = [action_array.reshape(1, -1)]
    elif action_array.ndim == 2:
        action_batches = [action_array]
    else:
        action_batches = [np.asarray(action_array[idx], dtype=np.float32) for idx in range(action_array.shape[0])]

    for batch_array in action_batches:
        if batch_array.ndim != 2:
            raise ValueError(f"batched predicted action must contain 2D arrays, got shape {tuple(batch_array.shape)}")
        valid_dims = {6, 12} if no_gripper else {7, 14}
        if batch_array.shape[1] not in valid_dims:
            raise ValueError(
                "unsupported predicted action shape "
                f"{tuple(batch_array.shape)}; expected rows of {sorted(valid_dims)}D pose actions"
            )

    return action_array, action_batches


def _decode_action_positions(action_array: np.ndarray, no_gripper: bool = False) -> Dict[str, Dict[str, Any]]:
    robots: Dict[str, Dict[str, Any]] = {}
    cols = action_array.shape[1]
    robot_stride = 6 if no_gripper else 7
    if cols >= robot_stride:
        robot0_pos = action_array[:, 0:3].astype(float)
        robots["robot0"] = {
            "predicted_pos": robot0_pos.tolist(),
            "current_predicted_pos": robot0_pos[0].tolist(),
        }
    if cols >= robot_stride * 2:
        robot1_pos = action_array[:, robot_stride:robot_stride + 3].astype(float)
        robots["robot1"] = {
            "predicted_pos": robot1_pos.tolist(),
            "current_predicted_pos": robot1_pos[0].tolist(),
        }
    return robots


def _build_predict_request_payload(
    obs: Mapping[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    return dict(obs, batch_size=batch_size)


def _sampling_horizon_summary(details: Mapping[str, Mapping[str, Any]]) -> int:
    if not details:
        return 0
    return max(int(detail.get("horizon_used") or 0) for detail in details.values())


def run_inference_overlay(
    adapter: BaseAdapter,
    episode_index: int,
    frame_index: int,
    yaml_path: str,
    server_host: str = "localhost",
    server_port: int = DEFAULT_SERVER_PORT,
    warmup_steps: int = 1,
    batch_size: int = 1,
    inference_mode: str = DEFAULT_INFERENCE_MODE,
    no_gripper: bool = False,
    transport_factory: Callable[[str, int], Any] | None = None,
) -> Dict[str, Any]:
    task_cfg = load_inference_task_config(yaml_path)
    schema = adapter.episode_schema(episode_index)
    if frame_index < 0 or frame_index >= schema.length:
        raise IndexError(f"frame index {frame_index} out of range [0, {max(0, schema.length - 1)}]")

    normalized_inference_mode = _normalize_inference_mode(inference_mode)
    requested_batch_size = max(1, min(MAX_BATCH_SIZE, int(batch_size)))
    predict_obs_horizon_used = 0
    predict_sampling_details: Dict[str, Dict[str, Any]] = {}
    long_obs_horizon_used = 0
    long_sampling_details: Dict[str, Dict[str, Any]] = {}
    request_type_sent = ""
    diagnostics_message = ""
    obs_payload: Dict[str, Any] = {}
    active_branches: List[str] = []

    transport = (transport_factory or create_zmq_transport)(server_host, int(server_port))
    try:
        active_branches = list(task_cfg.branches) if task_cfg.branches else ["short"]
        all_sampling: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for branch in active_branches:
            branch_obs = build_inference_observation(
                adapter=adapter,
                episode_index=episode_index,
                frame_index=frame_index,
                task_cfg=task_cfg,
                add_numeric_batch_dim=False,
                branch=branch,
            )
            if not branch_obs:
                continue
            obs_payload[branch] = branch_obs
            branch_sampling_details = _observation_sampling_details(
                task_cfg=task_cfg,
                frame_index=frame_index,
                horizon=0,
                downsample_field="down_sample_steps",
                branch=branch,
            )
            all_sampling[branch] = branch_sampling_details
            if branch == "short":
                predict_sampling_details = branch_sampling_details
                predict_obs_horizon_used = _sampling_horizon_summary(branch_sampling_details)
            elif branch == "long":
                long_sampling_details = branch_sampling_details
                long_obs_horizon_used = _sampling_horizon_summary(branch_sampling_details)

        if not obs_payload:
            raise InferenceConfigError("no observation branches could be resolved from task yaml")

        print(f"[inference] obs_pose_repr={task_cfg.obs_pose_repr!r}")
        obs_summary_parts = []
        for br in sorted(all_sampling.keys()):
            key_parts = [f"    {k}: {_fmt_indices(d['frame_indices'])} (n={len(d['frame_indices'])})" for k, d in sorted(all_sampling[br].items())]
            obs_summary_parts.append(f"  {br}:\n" + "\n".join(key_parts))
        print("[inference] obs indices:\n" + "\n".join(obs_summary_parts))

        request_type_sent = "predict_action"
        print(
            "[inference train_time] sending "
            f"request_type={request_type_sent} branches={sorted(obs_payload.keys())} "
            f"batch_size={requested_batch_size}"
        )
        action_response = transport.request(
            _build_predict_request_payload(obs=obs_payload, batch_size=requested_batch_size)
        )
        diagnostics_message = (
            f"Sent 1 {request_type_sent} call with branches={sorted(obs_payload.keys())}, "
            f"batch_size={requested_batch_size} to {server_host}:{int(server_port)}"
        )
    finally:
        close_fn = getattr(transport, "close", None)
        if callable(close_fn):
            close_fn()

    action_array, action_batches = _extract_action_batches(action_response, no_gripper=no_gripper)
    batch_payloads = [
        {
            "batch_index": batch_index,
            "action_shape": [int(v) for v in batch_array.shape],
            "robots": _decode_action_positions(batch_array, no_gripper=no_gripper),
        }
        for batch_index, batch_array in enumerate(action_batches)
    ]
    robots = batch_payloads[0]["robots"] if batch_payloads else {}
    return {
        "status": "ok",
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "yaml_path": task_cfg.path,
        "server_host": str(server_host),
        "server_port": int(server_port),
        "inference_mode": normalized_inference_mode,
        "request_type_sent": request_type_sent,
        "warmup_steps_requested": 0,
        "warmup_steps_effective": 0,
        "warmup_update_calls_requested": 0,
        "warmup_update_calls_effective": 0,
        "warmup_update_reference_frames": [],
        "warmup_update_frames": [],
        "warmup_update_snapshots": [],
        "warmup_update_skip_reason": "deprecated",
        "predict_obs_horizon_used": int(predict_obs_horizon_used),
        "long_obs_horizon_used": int(long_obs_horizon_used),
        "predict_sampling_details": predict_sampling_details,
        "long_sampling_details": long_sampling_details,
        "batch_size_requested": requested_batch_size,
        "batch_count_returned": len(action_batches),
        "action_shape": [int(v) for v in action_array.shape],
        "action_downsample_steps": int(task_cfg.action_downsample_steps),
        "batch_action_shapes": [[int(v) for v in batch_array.shape] for batch_array in action_batches],
        "batches": batch_payloads,
        "robots": robots,
        "ground_truth": None,
        "diagnostics": {
            "message": diagnostics_message
        },
    }


def _extract_progress_scalar(response: Any) -> float:
    """Pull a single progress scalar out of a ProgressInferenceNode response."""
    if isinstance(response, (str, bytes)):
        raise RuntimeError(f"inference server returned an error:\n{response!r}")

    if not isinstance(response, Mapping):
        raise ValueError(
            f"progress inference response has unsupported type {type(response).__name__}: {response!r}"
        )

    status = response.get("status")
    if isinstance(status, str) and status.lower() == "error":
        detail = response.get("detail") or response.get("error") or response.get("message")
        raise RuntimeError(f"inference server returned an error:\n{detail or response}")

    if "progress" not in response:
        raise ValueError(f"progress inference response missing 'progress' key: {response!r}")

    value = response["progress"]
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("progress inference response 'progress' list is empty")
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"progress inference response 'progress' is not a number: {value!r}"
        ) from exc


def run_progress_graph(
    adapter: BaseAdapter,
    episode_index: int,
    yaml_path: str,
    server_host: str = "localhost",
    server_port: int = DEFAULT_SERVER_PORT,
    eval_every: int = 10,
    transport_factory: Callable[[str, int], Any] | None = None,
) -> Dict[str, Any]:
    """Run a SARM-style progress estimator across an episode.

    Iterates frame indices at stride ``eval_every`` and queries the progress
    inference server once per frame, reusing a single ZMQ socket for the whole
    loop. Returns parallel ``frames`` / ``progress`` arrays suitable for a
    full-episode line plot.
    """
    task_cfg = load_inference_task_config(yaml_path)
    schema = adapter.episode_schema(episode_index)
    episode_length = int(schema.length)
    if episode_length <= 0:
        raise InferenceConfigError(f"episode {episode_index} has no frames")

    step = max(1, int(eval_every))
    frame_indices = list(range(0, episode_length, step))
    if not frame_indices:
        raise InferenceConfigError("no frames to evaluate (eval_every too large?)")

    active_branches = list(task_cfg.branches) if task_cfg.branches else ["short"]

    transport = (transport_factory or create_zmq_transport)(server_host, int(server_port))
    collected_progress: List[float] = []
    try:
        for frame_index in frame_indices:
            obs_payload: Dict[str, Any] = {}
            for branch in active_branches:
                branch_obs = build_inference_observation(
                    adapter=adapter,
                    episode_index=episode_index,
                    frame_index=frame_index,
                    task_cfg=task_cfg,
                    add_numeric_batch_dim=False,
                    branch=branch,
                )
                if branch_obs:
                    obs_payload[branch] = branch_obs

            if not obs_payload:
                raise InferenceConfigError(
                    "no observation branches could be resolved from progress yaml"
                )

            response = transport.request(
                _build_predict_request_payload(obs=obs_payload, batch_size=1)
            )
            collected_progress.append(_extract_progress_scalar(response))
    finally:
        close_fn = getattr(transport, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "status": "ok",
        "episode_index": int(episode_index),
        "episode_length": episode_length,
        "eval_every": step,
        "frames": [int(f) for f in frame_indices],
        "progress": [float(p) for p in collected_progress],
        "yaml_path": task_cfg.path,
        "server_host": str(server_host),
        "server_port": int(server_port),
        "branches": sorted(active_branches),
    }
