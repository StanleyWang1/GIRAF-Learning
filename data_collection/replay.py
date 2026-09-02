"""Inspect, replay, or extract one episode from a GIRAF ReplayBuffer."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def episode_slice(root, episode: int) -> slice:
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode < 0:
        episode += len(ends)
    if episode < 0 or episode >= len(ends):
        raise IndexError(f"episode {episode} is outside 0..{len(ends) - 1}")
    start = 0 if episode == 0 else int(ends[episode - 1])
    return slice(start, int(ends[episode]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="path to replay_buffer.zarr")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--show", action="store_true", help="display synchronized samples"
    )
    parser.add_argument(
        "--extract-dir", default=None, help="write RGB frames as PNG files"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="display playback rate")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("replay requires zarr<3") from exc

    root = zarr.open_group(args.dataset, mode="r")
    selection = episode_slice(root, args.episode)
    images = root["data/camera_rgb"][selection]
    actions = root["data/action"][selection]
    states = root["data/state"][selection]
    timestamps = root["data/timestamp_ns"][selection]
    valid = root["data/alignment_valid"][selection]
    print(
        f"episode={args.episode} steps={len(images)} valid={int(valid.sum())} "
        f"invalid={int(len(valid) - valid.sum())} action_shape={actions.shape} "
        f"state_shape={states.shape}"
    )

    extract_dir = Path(args.extract_dir) if args.extract_dir else None
    if extract_dir is not None:
        extract_dir.mkdir(parents=True, exist_ok=True)

    delay_ms = max(1, int(round(1000.0 / args.fps)))
    for index, (rgb, action, timestamp_ns, is_valid) in enumerate(
        zip(images, actions, timestamps, valid)
    ):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if extract_dir is not None:
            path = extract_dir / f"{index:06d}.png"
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"could not write {path}")
        if args.show:
            preview = bgr.copy()
            text = (
                f"step={index} t={int(timestamp_ns)} valid={bool(is_valid)} "
                f"grasp={int(action[-1] >= 0.5)}"
            )
            cv2.putText(
                preview,
                text,
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0) if is_valid else (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("GIRAF episode replay", preview)
            if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), 27):
                break
    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
