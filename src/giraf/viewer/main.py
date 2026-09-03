"""Command-line entry point for the GIRAF dataset viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .dataset import GirafDataset
from .server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browse a GIRAF replay-buffer Zarr dataset in a local web UI."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Input .zarr path")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode initially selected in the browser (default: 0)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the viewer URL in the default browser",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = GirafDataset(args.dataset)
    if dataset.episode_count == 0:
        raise SystemExit("dataset has no committed episodes")
    if args.episode < 0 or args.episode >= dataset.episode_count:
        raise SystemExit(
            f"episode {args.episode} is outside [0, {dataset.episode_count - 1}]"
        )
    if args.port < 0 or args.port > 65535:
        raise SystemExit("port must be in [0, 65535]")
    run_server(
        dataset,
        host=str(args.host),
        port=int(args.port),
        requested_episode=int(args.episode),
        open_browser=bool(args.open),
    )


if __name__ == "__main__":
    main()
