from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.adapters.factory import create_adapter
from dataset_visualization.backend.adapters.raw_sidecar import RawSidecarAdapter
from dataset_visualization.backend.config import AppConfig
from dataset_visualization.backend.services.events import compute_events
from dataset_visualization.backend.services.inference import (
    DEFAULT_INFERENCE_MODE,
    DEFAULT_SERVER_PORT,
    MAX_BATCH_SIZE,
    run_inference_overlay,
    run_progress_graph,
)
from dataset_visualization.backend.services.media import MediaService
from dataset_visualization.backend.services.mutations import (
    delete_and_compact_sidecar_episode,
    validate_sidecar_episode_layout,
)
from dataset_visualization.backend.services.signals import build_signal_payload
from dataset_visualization.backend.services.timing import compute_episode_timing
from dataset_visualization.backend.services.trajectory import build_trajectory_payload
from utils.helper import ReplayBuffer


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "frontend"


def _ensure_supported(adapter: BaseAdapter) -> None:
    if adapter.supported:
        return
    raise HTTPException(status_code=400, detail=adapter.unsupported_reason or "Unsupported dataset format")


def _resolve_sidecar_root(adapter: BaseAdapter, config: AppConfig) -> Optional[Path]:
    adapter_root = getattr(adapter, "videos_root", None)
    if isinstance(adapter_root, Path):
        return adapter_root
    if config.videos_root is not None:
        return config.videos_root
    return None


class InferenceRunRequest(BaseModel):
    episode_index: int = Field(..., ge=0)
    frame_index: int = Field(..., ge=0)
    yaml_path: str
    server_host: str = "localhost"
    server_port: int = Field(DEFAULT_SERVER_PORT, ge=1, le=65535)
    warmup_steps: int = Field(1, ge=1)
    batch_size: int = Field(1, ge=1, le=MAX_BATCH_SIZE)
    inference_mode: str = DEFAULT_INFERENCE_MODE
    no_gripper: bool = False


class ProgressGraphRequest(BaseModel):
    episode_index: int = Field(..., ge=0)
    yaml_path: str
    server_host: str = "localhost"
    server_port: int = Field(DEFAULT_SERVER_PORT, ge=1, le=65535)
    eval_every: int = Field(10, ge=1)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Dataset Visualization", version="0.1.0")
    adapter = create_adapter(config.input_path, videos_root=config.videos_root, depth_shape=config.depth_shape)
    media_service = MediaService(adapter=adapter, prefetch=config.prefetch)
    events_cache: Dict[int, List[Dict[str, object]]] = {}
    mutation_lock = Lock()

    frontend_dir = _frontend_dir()
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    def _close_runtime_state() -> None:
        nonlocal adapter, media_service
        media_service.shutdown()
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn):
            close_fn()

    def _reopen_runtime_state() -> None:
        nonlocal adapter, media_service
        adapter = create_adapter(config.input_path, videos_root=config.videos_root, depth_shape=config.depth_shape)
        media_service = MediaService(adapter=adapter, prefetch=config.prefetch)
        events_cache.clear()

    @app.middleware("http")
    async def disable_asset_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(frontend_dir / "index.html"))

    @app.on_event("shutdown")
    def shutdown_services() -> None:
        _close_runtime_state()

    @app.get("/api/health")
    def api_health() -> Dict[str, object]:
        return {
            "status": "ok",
            "format": adapter.format_name,
            "supported": adapter.supported,
            "reason": adapter.unsupported_reason,
        }

    @app.get("/api/dataset/summary")
    def api_dataset_summary() -> Dict[str, object]:
        summary = adapter.dataset_summary()
        payload = asdict(summary)
        payload["requested_episode"] = config.episode
        payload["videos_root"] = str(config.videos_root) if config.videos_root is not None else None
        payload["history_frames"] = config.history_frames
        payload["prefetch"] = config.prefetch
        payload["depth_shape"] = list(config.depth_shape) if config.depth_shape is not None else None
        payload["all_keys"] = adapter.all_keys() if adapter.supported else []
        return payload

    @app.get("/api/episodes")
    def api_episodes() -> Dict[str, object]:
        _ensure_supported(adapter)

        episodes: List[Dict[str, object]] = []
        ep_count = adapter.episode_count()
        for ep in range(ep_count):
            start, end = adapter.episode_bounds(ep)
            length = end - start

            duration = 0.0
            if length >= 2:
                ts = adapter.episode_timestamps(ep, 0, length, max(1, length - 1))
                if len(ts) >= 2:
                    duration = float(ts[-1] - ts[0])

            stream_ids = adapter.list_stream_ids(ep)
            has_depth = any(adapter.has_depth(ep, sid) for sid in stream_ids)
            episodes.append(
                {
                    "episode_index": ep,
                    "length": length,
                    "start": start,
                    "end": end,
                    "duration_sec": duration,
                    "stream_ids": stream_ids,
                    "has_depth": has_depth,
                }
            )

        return {"episode_count": ep_count, "episodes": episodes}

    @app.get("/api/episode/{episode_index}/schema")
    def api_episode_schema(episode_index: int) -> Dict[str, object]:
        _ensure_supported(adapter)
        try:
            schema = adapter.episode_schema(episode_index)
            return asdict(schema)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/episode/{episode_index}/timing")
    def api_episode_timing(
        episode_index: int,
        fps_cap: float = Query(30.0),
    ) -> Dict[str, object]:
        _ensure_supported(adapter)
        try:
            return compute_episode_timing(adapter, episode_index, fps_cap=fps_cap)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/episode/{episode_index}/signals")
    def api_episode_signals(
        episode_index: int,
        keys: str = Query(..., description="Comma-separated keys"),
        start: int = Query(0),
        end: int = Query(0),
        stride: int = Query(1),
    ) -> Dict[str, object]:
        _ensure_supported(adapter)
        selected_keys = [k.strip() for k in keys.split(",") if k.strip()]
        if not selected_keys:
            raise HTTPException(status_code=400, detail="No keys provided")

        try:
            return build_signal_payload(adapter, episode_index, selected_keys, start, end, stride)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/episode/{episode_index}/frame")
    def api_episode_frame(
        episode_index: int,
        camera: str,
        idx: int,
        modality: str = Query("rgb", pattern="^(rgb|depth)$"),
        colormap: str = Query("turbo"),
        profile: str = Query("full", pattern="^(full|scrub|preview)$"),
    ) -> Response:
        _ensure_supported(adapter)
        try:
            if modality == "rgb":
                payload = media_service.get_rgb(episode_index, camera, idx, profile=profile)
            else:
                payload = media_service.get_depth(episode_index, camera, idx, colormap=colormap, profile=profile)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return Response(content=payload, media_type="image/jpeg")

    @app.get("/api/episode/{episode_index}/events")
    def api_episode_events(episode_index: int) -> Dict[str, object]:
        _ensure_supported(adapter)
        if episode_index not in events_cache:
            events_cache[episode_index] = compute_events(adapter, episode_index)
        return {"episode_index": episode_index, "events": events_cache[episode_index]}

    @app.get("/api/episode/{episode_index}/trajectory3d")
    def api_episode_trajectory(
        episode_index: int,
        idx: int = Query(0),
        window: int = Query(180),
    ) -> Dict[str, object]:
        _ensure_supported(adapter)
        try:
            return build_trajectory_payload(adapter, episode_index, idx, window)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/inference/run")
    def api_inference_run(request: InferenceRunRequest) -> Dict[str, object]:
        _ensure_supported(adapter)
        try:
            return run_inference_overlay(
                adapter=adapter,
                episode_index=request.episode_index,
                frame_index=request.frame_index,
                yaml_path=request.yaml_path,
                server_host=request.server_host,
                server_port=request.server_port,
                warmup_steps=request.warmup_steps,
                batch_size=request.batch_size,
                inference_mode=request.inference_mode,
                no_gripper=request.no_gripper,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/inference/progress_graph")
    def api_inference_progress_graph(request: ProgressGraphRequest) -> Dict[str, object]:
        _ensure_supported(adapter)
        try:
            return run_progress_graph(
                adapter=adapter,
                episode_index=request.episode_index,
                yaml_path=request.yaml_path,
                server_host=request.server_host,
                server_port=request.server_port,
                eval_every=request.eval_every,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/episode/{episode_index}")
    def api_delete_episode(
        episode_index: int,
        delete_videos: bool = Query(True, description="Delete sidecar episode folders when applicable"),
    ) -> Dict[str, object]:
        nonlocal adapter, media_service
        _ensure_supported(adapter)

        with mutation_lock:
            _ensure_supported(adapter)
            episode_count_before = adapter.episode_count()
            if episode_index < 0 or episode_index >= episode_count_before:
                raise HTTPException(
                    status_code=400,
                    detail=f"episode index {episode_index} out of range [0, {max(0, episode_count_before - 1)}]",
                )

            sidecar_applicable = isinstance(adapter, RawSidecarAdapter) or config.videos_root is not None
            videos_root = _resolve_sidecar_root(adapter, config)
            videos_root_str = str(videos_root) if videos_root is not None else None
            video_ops: Dict[str, object] = {"deleted_dir": None, "renamed": []}

            if sidecar_applicable:
                if videos_root is None:
                    raise HTTPException(status_code=409, detail="sidecar videos root is required but not configured")
                if not delete_videos:
                    raise HTTPException(
                        status_code=400,
                        detail="delete_videos=false is not allowed for sidecar datasets under strict consistency policy",
                    )
                try:
                    validate_sidecar_episode_layout(videos_root, episode_count_before)
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc

            # Ensure open video readers/caches release file handles before sidecar mutation.
            _close_runtime_state()
            try:
                replay_buffer = ReplayBuffer.create_from_path(str(config.input_path), mode="a")
                replay_buffer.drop_episode_by_index(int(episode_index))
            except Exception as exc:
                _reopen_runtime_state()
                raise HTTPException(status_code=500, detail=f"failed deleting episode from zarr: {exc}") from exc

            if sidecar_applicable and delete_videos:
                assert videos_root is not None
                try:
                    video_ops = delete_and_compact_sidecar_episode(videos_root, int(episode_index), episode_count_before)
                except Exception as exc:
                    _reopen_runtime_state()
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"episode deleted from zarr but sidecar compaction failed: {exc}. "
                            "Please reconcile sidecar folders manually."
                        ),
                    ) from exc

            _reopen_runtime_state()
            episode_count_after = adapter.episode_count()
            suggested_episode_index: Optional[int] = None
            if episode_count_after > 0:
                suggested_episode_index = min(max(int(episode_index), 0), episode_count_after - 1)

            return {
                "deleted_episode_index": int(episode_index),
                "episode_count_before": int(episode_count_before),
                "episode_count_after": int(episode_count_after),
                "suggested_episode_index": suggested_episode_index,
                "videos_applied": bool(sidecar_applicable and delete_videos),
                "videos_root": videos_root_str,
                "video_ops": video_ops,
            }

    return app
