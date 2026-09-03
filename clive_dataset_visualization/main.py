from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from dataset_visualization.backend.config import AppConfig, parse_depth_shape
from dataset_visualization.server import run_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset Visualization Tool")
    parser.add_argument("--input", required=True, type=str, help="Path to input .zarr dataset")
    parser.add_argument("--videos-root", type=str, default=None, help="Optional sidecar videos root")
    parser.add_argument("--episode", type=int, default=None, help="Default episode index to load")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host bind address")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--history-frames", type=int, default=180, help="Default history frames")
    parser.add_argument("--prefetch", type=int, default=60, help="Frame prefetch hint")
    parser.add_argument(
        "--depth-shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional override for depth bin frame shape",
    )
    return parser


def _to_config(args: argparse.Namespace) -> AppConfig:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    videos_root = Path(args.videos_root).expanduser().resolve() if args.videos_root else None
    depth_shape = parse_depth_shape(tuple(args.depth_shape)) if args.depth_shape is not None else None

    return AppConfig(
        input_path=input_path,
        videos_root=videos_root,
        episode=args.episode,
        host=args.host,
        port=int(args.port),
        history_frames=max(1, int(args.history_frames)),
        prefetch=max(0, int(args.prefetch)),
        depth_shape=depth_shape,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _to_config(args)
    run_server(config)


if __name__ == "__main__":
    main()
