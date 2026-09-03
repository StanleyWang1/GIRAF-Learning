from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from giraf.diagnostics import diagnose_episode, format_report, main


def create_diagnostic_replay(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "schema_version": "giraf-replay-v1",
            "git_revisions": ["test-revision"],
            "last_collector_config": {
                "alignment": {
                    "max_control_age_ms": 50.0,
                    "max_motor_age_ms": 50.0,
                }
            },
        }
    )
    data = root.require_group("data")
    meta = root.require_group("meta")

    timestamps = np.asarray(
        [1_000_000_000, 1_100_000_000, 1_200_000_000, 1_300_000_000],
        dtype=np.int64,
    )
    control_ages = np.asarray(
        [10_000_000, 60_000_000, -5_000_000, 10_000_000], dtype=np.int64
    )
    motor_ages = np.asarray(
        [5_000_000, 10_000_000, 70_000_000, 10_000_000], dtype=np.int64
    )
    arrays = {
        "timestamp_ns": timestamps,
        "control_timestamp_ns": timestamps - control_ages,
        "motor_timestamp_ns": timestamps - motor_ages,
        "camera_receive_timestamp_ns": timestamps + 2_000_000,
        "control_age_ns": np.asarray([10_000_000, 0, -5_000_000, 10_000_000]),
        "motor_age_ns": motor_ages,
        "camera_receive_latency_ns": np.full(4, 2_000_000, dtype=np.int64),
        "alignment_valid": np.asarray([1, 0, 1, 0], dtype=np.uint8),
        "motor_command_accepted": np.asarray([1, 1, 1, 0], dtype=np.uint8),
        "tracking": np.asarray([1, 1, 0, 1], dtype=np.uint8),
        "clutch": np.asarray([0, 1, 1, 0], dtype=np.uint8),
        "camera_sequence_num": np.asarray([10, 11, 13, 13], dtype=np.int64),
        "action": np.zeros((4, 7), dtype=np.float32),
        "state": np.zeros((4, 15), dtype=np.float32),
    }
    for name, values in arrays.items():
        data.create_dataset(name, data=values, chunks=(2,) + values.shape[1:])
    meta.create_dataset("episode_ends", data=np.asarray([2, 4], dtype=np.int64))
    meta.create_dataset("episode_valid_steps", data=np.asarray([1, 1], dtype=np.int64))
    meta.create_dataset(
        "episode_invalid_steps", data=np.asarray([1, 1], dtype=np.int64)
    )


class EpisodeDiagnosticsTests(unittest.TestCase):
    def test_reports_validity_failures_and_timing_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            create_diagnostic_replay(source)

            report = diagnose_episode(source, 0)

            self.assertEqual(report["steps"], 2)
            self.assertEqual(report["valid_steps"], 1)
            self.assertEqual(report["hardware_failure_signals"]["control_too_old"], 1)
            self.assertEqual(
                report["validity_rules"]["hardware"]["stored_mismatch_steps"], 0
            )
            self.assertEqual(
                report["stored_timing_consistency"]["control_age_ns"]["mismatch_steps"],
                1,
            )
            self.assertTrue(report["metadata_counts"]["matches_data"])

            output = format_report(report)
            self.assertIn("hardware failure signals", output)
            self.assertIn("control too old: 1/2 (50.0%)", output)

    def test_negative_episode_and_sequence_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            create_diagnostic_replay(source)

            report = diagnose_episode(source, -1)

            self.assertEqual(report["episode"], 1)
            self.assertEqual(report["global_step_start"], 2)
            self.assertEqual(report["camera_sequence"]["discontinuities"], 1)
            self.assertEqual(report["camera_sequence"]["duplicates"], 1)
            self.assertEqual(report["camera_sequence"]["estimated_missing_frames"], 0)
            self.assertEqual(report["hardware_failure_signals"]["motor_too_old"], 1)
            self.assertEqual(
                report["hardware_failure_signals"]["motor_command_not_accepted"],
                1,
            )

    def test_json_cli_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            create_diagnostic_replay(source)
            stdout = StringIO()
            with (
                patch.object(
                    __import__("sys"),
                    "argv",
                    [
                        "giraf-diagnose",
                        "--dataset",
                        str(source),
                        "--episode",
                        "0",
                        "--json",
                    ],
                ),
                redirect_stdout(stdout),
            ):
                result = main()

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["episode"], 0)

    def test_metadata_disagreement_is_reported_as_inconsistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.zarr"
            create_diagnostic_replay(source)
            root = zarr.open_group(str(source), mode="r+")
            root["meta/episode_valid_steps"][0] = 2
            root["meta/episode_invalid_steps"][0] = 0

            report = diagnose_episode(source, 0)

            self.assertFalse(report["metadata_counts"]["matches_data"])
            self.assertTrue(
                any(
                    finding.startswith("Dataset inconsistency:")
                    for finding in report["findings"]
                )
            )


if __name__ == "__main__":
    unittest.main()
