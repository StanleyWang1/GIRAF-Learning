from __future__ import annotations

import unittest
from unittest.mock import patch

from giraf import replay_cli
from giraf.data.replay import PREVIEW_WINDOW, _resize_preview_window


class ReplayWindowTests(unittest.TestCase):
    def test_window_name_matches_cli_wrapper(self) -> None:
        # replay_cli creates the window before the data package is imported, so
        # the two modules must agree on its name without importing each other.
        self.assertEqual(PREVIEW_WINDOW, replay_cli.PREVIEW_WINDOW)

    @patch("giraf.data.replay.cv2")
    def test_preview_window_is_resized_to_double_frame_size(self, cv2) -> None:
        _resize_preview_window((224, 320, 3))
        cv2.resizeWindow.assert_called_once_with(PREVIEW_WINDOW, 640, 448)


if __name__ == "__main__":
    unittest.main()
