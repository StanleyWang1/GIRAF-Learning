from __future__ import annotations

import unittest
from unittest.mock import patch

from giraf.data.replay import (
    PREVIEW_WINDOW,
    _open_preview_window,
    _resize_preview_window,
)


class ReplayWindowTests(unittest.TestCase):
    @patch("giraf.data.replay.cv2")
    def test_preview_window_is_explicitly_initialized(self, cv2) -> None:
        cv2.WINDOW_NORMAL = 0

        _open_preview_window()
        _resize_preview_window((224, 320, 3))

        cv2.namedWindow.assert_called_once_with(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow.assert_called_once_with(PREVIEW_WINDOW, 640, 448)


if __name__ == "__main__":
    unittest.main()
