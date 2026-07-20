"""Command-line sanity launcher for the GIRAF simulation package."""

from __future__ import annotations

import argparse

from .runner import run_headless, run_viewer
from .scenes import available_scenes
from .simulation import GirafSimulation


def build_parser() -> argparse.ArgumentParser:
    scene_names = [scene.name for scene in available_scenes()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=scene_names, default="arm")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of step calls in headless mode",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional simulated seconds before closing the viewer",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not throttle viewer stepping to simulated time",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with GirafSimulation(args.scene, frame_skip=args.frame_skip) as simulation:
        print(
            f"Loaded {simulation.scene.name!r}: "
            f"nq={simulation.model.nq}, nv={simulation.model.nv}, "
            f"nu={simulation.model.nu}, dt={simulation.physics_dt:g}s"
        )
        if args.headless:
            run_headless(simulation, args.steps)
        else:
            run_viewer(
                simulation,
                duration=args.duration,
                realtime=not args.no_realtime,
            )
        print(f"Finished at simulation time {simulation.data.time:.3f}s")


if __name__ == "__main__":
    main()
