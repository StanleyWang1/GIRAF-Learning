"""Lightweight entry point that initializes the replay window before Zarr."""

from __future__ import annotations

import importlib
import sys

PREVIEW_WINDOW = "GIRAF episode replay"


def main() -> int:
    # On this Qt5/XWayland stack, importing the data package first initializes
    # NumPy/Zarr worker libraries and can leave HighGUI spinning without ever
    # mapping a window. Initialize the native window before that package import.
    if "--show" in sys.argv[1:]:
        cv2 = importlib.import_module("cv2")
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    replay = importlib.import_module("giraf.data.replay")
    return replay.main()


if __name__ == "__main__":
    raise SystemExit(main())
