from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable, Set, Tuple

import cv2
import numpy as np

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.services.cache import LRUCache


COLORMAP_MAP = {
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "gray": -1,
}

FRAME_PROFILES = {
    "full": {"max_long_edge": None, "jpeg_quality": 88, "prefetch_steps": 1},
    "scrub": {"max_long_edge": 960, "jpeg_quality": 60, "prefetch_steps": 0},
    "preview": {"max_long_edge": 480, "jpeg_quality": 45, "prefetch_steps": 0},
}


class MediaService:
    def __init__(self, adapter: BaseAdapter, max_cache_items: int = 1024, prefetch: int = 60):
        self.adapter = adapter
        self.prefetch = max(0, int(prefetch))
        self.cache: LRUCache[Tuple[object, ...], bytes] = LRUCache(max_items=max_cache_items)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-prefetch")
        self._prefetch_tasks_lock = Lock()
        self._prefetch_tasks: Set[Tuple[object, ...]] = set()

    @staticmethod
    def _profile_config(profile: str) -> dict:
        config = FRAME_PROFILES.get(str(profile))
        if config is None:
            raise ValueError(f"Unsupported frame profile: {profile}")
        return config

    @classmethod
    def _prefetch_steps(cls, profile: str, configured_prefetch: int) -> int:
        profile_cfg = cls._profile_config(profile)
        return min(max(0, int(configured_prefetch)), int(profile_cfg["prefetch_steps"]))

    @staticmethod
    def _resize_to_long_edge(frame: np.ndarray, max_long_edge: int | None, interpolation: int) -> np.ndarray:
        if max_long_edge is None:
            return frame
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= 0 or longest <= int(max_long_edge):
            return frame
        scale = float(max_long_edge) / float(longest)
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        if out_w == w and out_h == h:
            return frame
        return cv2.resize(frame, (out_w, out_h), interpolation=interpolation)

    @staticmethod
    def _encode_rgb_jpeg(frame_rgb: np.ndarray, profile: str = "full") -> bytes:
        if frame_rgb.ndim != 3:
            raise ValueError(f"Expected RGB frame [H,W,C], got {frame_rgb.shape}")
        profile_cfg = MediaService._profile_config(profile)
        resized_rgb = MediaService._resize_to_long_edge(
            frame_rgb,
            max_long_edge=profile_cfg["max_long_edge"],
            interpolation=cv2.INTER_AREA,
        )
        frame_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(profile_cfg["jpeg_quality"])],
        )
        if not ok:
            raise RuntimeError("Failed encoding RGB frame")
        return enc.tobytes()

    @staticmethod
    def _normalize_depth(depth: np.ndarray) -> np.ndarray:
        depth = depth.astype(np.float32)
        valid = depth[depth > 0]
        if valid.size == 0:
            return np.zeros_like(depth, dtype=np.uint8)

        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        if hi <= lo:
            hi = lo + 1.0

        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    @classmethod
    def _encode_depth(cls, depth_raw: np.ndarray, colormap: str = "turbo", profile: str = "full") -> bytes:
        profile_cfg = cls._profile_config(profile)
        resized_depth = cls._resize_to_long_edge(
            depth_raw,
            max_long_edge=profile_cfg["max_long_edge"],
            interpolation=cv2.INTER_NEAREST,
        )
        norm = cls._normalize_depth(resized_depth)
        cmap = COLORMAP_MAP.get(colormap, cv2.COLORMAP_TURBO)
        if cmap == -1:
            vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        else:
            vis = cv2.applyColorMap(norm, cmap)

        ok, enc = cv2.imencode(
            ".jpg",
            vis,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(profile_cfg["jpeg_quality"])],
        )
        if not ok:
            raise RuntimeError("Failed encoding depth frame")
        return enc.tobytes()

    def get_rgb(self, episode_index: int, stream_id: str, idx: int, profile: str = "full") -> bytes:
        profile_cfg = self._profile_config(profile)
        key = ("rgb", episode_index, stream_id, int(idx), str(profile))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        frame = self.adapter.frame_rgb(episode_index, stream_id, idx)
        encoded = self._encode_rgb_jpeg(frame, profile=str(profile))
        self.cache.set(key, encoded)
        if int(profile_cfg["prefetch_steps"]) > 0:
            self._schedule_prefetch_rgb(episode_index, stream_id, idx, str(profile))
        return encoded

    def get_depth(self, episode_index: int, stream_id: str, idx: int, colormap: str = "turbo", profile: str = "full") -> bytes:
        profile_cfg = self._profile_config(profile)
        key = ("depth", episode_index, stream_id, int(idx), colormap, str(profile))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        frame = self.adapter.frame_depth(episode_index, stream_id, idx)
        encoded = self._encode_depth(frame, colormap=colormap, profile=str(profile))
        self.cache.set(key, encoded)
        if int(profile_cfg["prefetch_steps"]) > 0:
            self._schedule_prefetch_depth(episode_index, stream_id, idx, colormap, str(profile))
        return encoded

    def _schedule_prefetch(self, task_key: Tuple[object, ...], work_fn: Callable[[], None]) -> None:
        if self.prefetch <= 0:
            return

        with self._prefetch_tasks_lock:
            if task_key in self._prefetch_tasks:
                return
            self._prefetch_tasks.add(task_key)

        def _job() -> None:
            try:
                work_fn()
            finally:
                with self._prefetch_tasks_lock:
                    self._prefetch_tasks.discard(task_key)

        self._prefetch_executor.submit(_job)

    def _schedule_prefetch_rgb(self, episode_index: int, stream_id: str, idx: int, profile: str = "full") -> None:
        if self._prefetch_steps(profile, self.prefetch) <= 0:
            return
        self._schedule_prefetch(
            task_key=("rgb", int(episode_index), str(stream_id), int(idx), str(profile)),
            work_fn=lambda: self._prefetch_rgb(episode_index, stream_id, idx, str(profile)),
        )

    def _schedule_prefetch_depth(self, episode_index: int, stream_id: str, idx: int, colormap: str, profile: str = "full") -> None:
        if self._prefetch_steps(profile, self.prefetch) <= 0:
            return
        self._schedule_prefetch(
            task_key=("depth", int(episode_index), str(stream_id), int(idx), str(colormap), str(profile)),
            work_fn=lambda: self._prefetch_depth(episode_index, stream_id, idx, colormap, str(profile)),
        )

    def _prefetch_rgb(self, episode_index: int, stream_id: str, idx: int, profile: str = "full") -> None:
        max_steps = self._prefetch_steps(profile, self.prefetch)
        for k in range(1, max_steps + 1):
            pre_idx = idx + k
            key = ("rgb", episode_index, stream_id, int(pre_idx), str(profile))
            if self.cache.get(key) is not None:
                continue
            try:
                frame = self.adapter.frame_rgb(episode_index, stream_id, pre_idx)
                self.cache.set(key, self._encode_rgb_jpeg(frame, profile=str(profile)))
            except Exception:
                break

    def _prefetch_depth(self, episode_index: int, stream_id: str, idx: int, colormap: str, profile: str = "full") -> None:
        max_steps = self._prefetch_steps(profile, self.prefetch)
        for k in range(1, max_steps + 1):
            pre_idx = idx + k
            key = ("depth", episode_index, stream_id, int(pre_idx), colormap, str(profile))
            if self.cache.get(key) is not None:
                continue
            try:
                frame = self.adapter.frame_depth(episode_index, stream_id, pre_idx)
                self.cache.set(key, self._encode_depth(frame, colormap=colormap, profile=str(profile)))
            except Exception:
                break

    def shutdown(self) -> None:
        self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
