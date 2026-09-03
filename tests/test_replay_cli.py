from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from giraf.replay_cli import PREVIEW_WINDOW, main


class ReplayCliTests(unittest.TestCase):
    @patch.object(sys, "argv", ["giraf-replay", "--show"])
    def test_show_initializes_qt_before_importing_data_package(self) -> None:
        events: list[str] = []
        cv2 = Mock(WINDOW_NORMAL=0)
        cv2.namedWindow.side_effect = lambda *_: events.append("namedWindow")
        replay = SimpleNamespace(main=lambda: events.append("replay") or 0)

        def load(name: str):
            events.append(f"import:{name}")
            return cv2 if name == "cv2" else replay

        with patch("giraf.replay_cli.importlib.import_module", side_effect=load):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(
            events,
            [
                "import:cv2",
                "namedWindow",
                "import:giraf.data.replay",
                "replay",
            ],
        )
        cv2.namedWindow.assert_called_once_with(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

    @patch.object(sys, "argv", ["giraf-replay"])
    def test_no_show_does_not_import_opencv(self) -> None:
        replay = SimpleNamespace(main=lambda: 0)
        with patch(
            "giraf.replay_cli.importlib.import_module", return_value=replay
        ) as load:
            result = main()

        self.assertEqual(result, 0)
        load.assert_called_once_with("giraf.data.replay")


if __name__ == "__main__":
    unittest.main()
