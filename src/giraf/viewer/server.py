"""Small dependency-free HTTP server for the GIRAF viewer."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import cv2

from .dataset import DatasetFormatError, GirafDataset


EPISODE_ROUTE = re.compile(
    r"^/api/episode/(?P<episode>\d+)/(?P<resource>schema|metrics|events|signals|frame)$"
)


class _FrameCache:
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max(1, int(max_items))
        self._values: OrderedDict[tuple[int, int, str], bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[int, int, str]) -> bytes | None:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def set(self, key: tuple[int, int, str], value: bytes) -> None:
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.max_items:
                self._values.popitem(last=False)


def _encode_frame(dataset: GirafDataset, episode: int, index: int, profile: str) -> bytes:
    frame_rgb = dataset.frame(episode, index)
    if profile == "scrub" and max(frame_rgb.shape[:2]) > 640:
        height, width = frame_rgb.shape[:2]
        scale = 640.0 / max(height, width)
        frame_rgb = cv2.resize(
            frame_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    quality = 62 if profile == "scrub" else 88
    ok, encoded = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("failed to encode camera frame")
    return encoded.tobytes()


def create_server(
    dataset: GirafDataset,
    host: str = "127.0.0.1",
    port: int = 8080,
    requested_episode: int | None = None,
) -> ThreadingHTTPServer:
    """Create a server instance; useful for both the CLI and tests."""

    static_dir = Path(__file__).resolve().parent / "static"
    frame_cache = _FrameCache()

    class ViewerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self, payload: Any, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            encoded = json.dumps(
                payload, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            self._send_bytes(encoded, "application/json; charset=utf-8", status)

        @staticmethod
        def _integer_query(
            query: dict[str, list[str]], key: str, default: int
        ) -> int:
            raw = query.get(key, [str(default)])[0]
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"query parameter {key} must be an integer") from exc

        def _serve_static(self, filename: str) -> None:
            if filename not in {"index.html", "app.js", "styles.css"}:
                self._send_json({"detail": "not found"}, HTTPStatus.NOT_FOUND)
                return
            path = static_dir / filename
            if not path.is_file():
                self._send_json(
                    {"detail": f"viewer asset missing: {filename}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), content_type)

        def _serve_frame(
            self, episode: int, query: dict[str, list[str]]
        ) -> None:
            index = self._integer_query(query, "idx", 0)
            profile = query.get("profile", ["full"])[0]
            if profile not in {"full", "scrub"}:
                raise ValueError("profile must be full or scrub")
            cache_key = (episode, index, profile)
            payload = frame_cache.get(cache_key)
            if payload is None:
                payload = _encode_frame(dataset, episode, index, profile)
                frame_cache.set(cache_key, payload)
            self._send_bytes(payload, "image/jpeg")

        def _serve_episode_resource(
            self,
            episode: int,
            resource: str,
            query: dict[str, list[str]],
        ) -> None:
            if resource == "schema":
                self._send_json(dataset.schema(episode))
                return
            if resource == "metrics":
                self._send_json(dataset.episode_metrics(episode))
                return
            if resource == "events":
                self._send_json(
                    {"episode_index": episode, "events": dataset.events(episode)}
                )
                return
            if resource == "frame":
                self._serve_frame(episode, query)
                return

            keys = [
                key.strip()
                for key in query.get("keys", [""])[0].split(",")
                if key.strip()
            ]
            if not keys:
                raise ValueError("query parameter keys must not be empty")
            length = dataset.episode_length(episode)
            start = self._integer_query(query, "start", 0)
            end = self._integer_query(query, "end", length)
            stride = self._integer_query(query, "stride", 1)
            self._send_json(
                dataset.signal_payload(episode, keys, start, end, stride)
            )

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._serve_static("index.html")
                    return
                if parsed.path.startswith("/static/"):
                    self._serve_static(parsed.path.removeprefix("/static/"))
                    return
                if parsed.path == "/favicon.ico":
                    self._send_bytes(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
                    return
                if parsed.path == "/api/health":
                    self._send_json(
                        {"status": "ok", "format": "giraf_zarr", "read_only": True}
                    )
                    return
                if parsed.path == "/api/dataset/summary":
                    self._send_json(dataset.summary(requested_episode))
                    return
                if parsed.path == "/api/episodes":
                    episodes = dataset.episodes()
                    self._send_json(
                        {"episode_count": len(episodes), "episodes": episodes}
                    )
                    return

                match = EPISODE_ROUTE.match(parsed.path)
                if match is None:
                    self._send_json({"detail": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._serve_episode_resource(
                    int(match.group("episode")), match.group("resource"), query
                )
            except (DatasetFormatError, FileNotFoundError, IndexError, ValueError) as exc:
                self._send_json({"detail": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
                self._send_json(
                    {"detail": f"viewer error: {type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: object) -> None:
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

    server = ThreadingHTTPServer((host, int(port)), ViewerHandler)
    server.daemon_threads = True
    return server


def run_server(
    dataset: GirafDataset,
    host: str,
    port: int,
    requested_episode: int | None,
    open_browser: bool,
) -> None:
    server = create_server(dataset, host, port, requested_episode)
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}"
    print(
        f"GIRAF viewer: {dataset.episode_count} episodes, "
        f"{dataset.total_steps} steps\n{url}\nPress Ctrl-C to stop."
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
