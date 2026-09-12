"""Thread-safe session and per-rollout deployment recording."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._file = path.open("x", encoding="utf-8")
        self._closed = False

    def write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            return
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.close()


class _VideoWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: float) -> None:
        import av

        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=int(round(fps)))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": "21"}
        self._shape = (height, width, 3)
        self._closed = False

    def write(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("video writer is closed")
        if rgb.shape != self._shape or rgb.dtype != np.uint8:
            raise ValueError(
                f"video frame must be uint8 with shape {self._shape}, "
                f"got {rgb.dtype} {rgb.shape}"
            )
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()


@dataclass(frozen=True, slots=True)
class VideoSettings:
    enabled: bool
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("video enabled must be a boolean")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video dimensions must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video fps must be finite and positive")


class _Episode:
    def __init__(
        self,
        directory: Path,
        *,
        index: int,
        generation: int,
        started: Mapping[str, Any],
        video_settings: VideoSettings,
    ) -> None:
        directory.mkdir(parents=False, exist_ok=False)
        self.directory = directory
        self.index = index
        self.generation = generation
        self.started = dict(started)
        self.video_settings = video_settings
        self.data = _JsonlWriter(directory / "data.jsonl")
        self.video: _VideoWriter | None = None
        self.video_error: str | None = None
        self.frames = 0
        self.ended: dict[str, Any] | None = None

    def write(self, record: Mapping[str, Any]) -> None:
        self.data.write(record)

    def write_frame(self, rgb: np.ndarray) -> None:
        if not self.video_settings.enabled or self.video_error is not None:
            return
        if self.ended is not None:
            return
        try:
            if self.video is None:
                settings = self.video_settings
                self.video = _VideoWriter(
                    self.directory / "camera.mp4",
                    width=settings.width,
                    height=settings.height,
                    fps=settings.fps,
                )
            self.video.write(rgb)
            self.frames += 1
        except Exception as exc:
            self.video_error = str(exc)
            self._close_video()
            raise

    def finish(self, record: Mapping[str, Any]) -> str | None:
        if self.ended is not None:
            return None
        self.ended = dict(record)
        close_error = self._close_video()
        if close_error is not None:
            self.video_error = close_error
        video_path = self.directory / "camera.mp4"
        metadata = {
            "schema_version": 1,
            "episode": self.index,
            "generation": self.generation,
            "complete": True,
            "started_wall_time_ns": self.started["wall_time_ns"],
            "started_monotonic_ns": self.started["monotonic_ns"],
            "ended_wall_time_ns": self.ended["wall_time_ns"],
            "ended_monotonic_ns": self.ended["monotonic_ns"],
            "reason": self.ended.get("reason", self.ended["event"]),
            "initial_joints": self.started.get("joints"),
            "initial_grasp": self.started.get("grasp"),
            "final_joints": self.ended.get("joints"),
            "final_grasp": self.ended.get("grasp"),
            "duration_seconds": (
                self.ended["monotonic_ns"] - self.started["monotonic_ns"]
            )
            / 1_000_000_000,
            "data": "data.jsonl",
            "frames": self.frames,
            "video": "camera.mp4" if video_path.is_file() else None,
            "video_complete": (
                not self.video_settings.enabled
                or (
                    self.frames > 0
                    and video_path.is_file()
                    and self.video_error is None
                )
            ),
            "video_error": self.video_error,
        }
        temporary = self.directory / ".episode.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "episode.json")
        return close_error

    def _close_video(self) -> str | None:
        video, self.video = self.video, None
        if video is None:
            return None
        try:
            video.close()
        except Exception as exc:
            return str(exc)
        return None

    def close(self) -> None:
        self.data.close()


class DeploymentRecorder:
    """Record one deployment session and split policy activations into episodes."""

    def __init__(
        self,
        directory: Path,
        *,
        config: Mapping[str, Any],
        video: VideoSettings,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self._lock = threading.RLock()
        self._closed = False
        self._video_settings = video
        self._session = _JsonlWriter(directory / "events.jsonl")
        self._episodes_directory = directory / "episodes"
        self._episodes_directory.mkdir()
        self._episodes: dict[int, _Episode] = {}
        self._current: _Episode | None = None
        self._pending: list[tuple[_Episode, dict[str, Any]]] = []
        (directory / "config.json").write_text(
            json.dumps(dict(config), indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _record(event: str, values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }

    def write(self, event: str, **values: Any) -> None:
        with self._lock:
            if self._closed:
                return
            record = self._record(event, values)

            if event == "activated" and values.get("source") == "policy":
                if self._current is not None:
                    superseded = self._record("paused", {"reason": "superseded"})
                    self._current.write(superseded)
                    self._finish_current(superseded)
                generation = int(values["generation"])
                episode = _Episode(
                    self._episodes_directory / f"episode_{len(self._episodes):04d}",
                    index=len(self._episodes),
                    generation=generation,
                    started=record,
                    video_settings=self._video_settings,
                )
                self._episodes[generation] = episode
                self._current = episode

            generation = values.get("generation")
            episode = self._episodes.get(generation) if generation is not None else None
            if episode is None and event in {
                "error",
                "finished",
                "paused",
                "stop_requested",
            }:
                episode = self._current
            if episode is not None:
                record["episode"] = episode.index
                episode.write(record)

            if event not in {
                "control",
                "inference_finished",
                "inference_started",
                "policy",
                "state_distribution",
            }:
                self._session.write(record)

            if event in {"finished", "paused", "stop_requested"}:
                self._finish_current(record)

    def _finish_current(self, record: Mapping[str, Any]) -> None:
        episode, self._current = self._current, None
        if episode is None:
            return
        self._pending.append((episode, dict(record)))

    def finish_pending(self) -> None:
        """Finalize ended rollouts on the camera/inference thread."""

        with self._lock:
            while self._pending:
                episode, record = self._pending.pop(0)
                self._finalize_episode(episode, record)

    def _finalize_episode(self, episode: _Episode, record: Mapping[str, Any]) -> None:
        try:
            close_error = episode.finish(record)
            if close_error is not None:
                error = self._record(
                    "video_error",
                    {"generation": episode.generation, "detail": close_error},
                )
                episode.write(error)
                self._session.write(error)
        finally:
            episode.close()

    def write_frame(self, rgb: np.ndarray, *, generation: int) -> None:
        with self._lock:
            if self._closed:
                return
            episode = self._episodes.get(generation)
            if episode is None or episode is not self._current:
                return
            try:
                episode.write_frame(rgb)
            except Exception as exc:
                record = self._record(
                    "video_error", {"generation": generation, "detail": str(exc)}
                )
                self._session.write(record)
                episode.write(record)
                raise RuntimeError("rollout video recording failed") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._current is not None:
                record = self._record("stop_requested", {"reason": "session_finished"})
                self._session.write(record)
                self._current.write(record)
                self._finish_current(record)
            self.finish_pending()
            for episode in self._episodes.values():
                episode.close()
            self._session.close()
            self._closed = True
