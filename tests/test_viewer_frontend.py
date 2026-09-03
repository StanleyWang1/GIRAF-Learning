from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


class ViewerFrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for frontend tests")
    def test_playback_regressions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                "node",
                str(root / "tests/js/viewer_playback_test.js"),
                str(root / "src/giraf/viewer/static/app.js"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
