"""Small replay fixtures for ZIP loading and single train/validation preloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import zarr

from giraf.data.schema import ACTION_DIM, STATE_DIM
from giraf.learning import train_cli
from giraf.learning.dataset import ReplayDataset, open_replay_group


class ReplayDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "replay.zarr"
        root = zarr.open_group(str(self.path), mode="w")
        self.images = np.arange(8 * 4 * 5 * 3, dtype=np.uint8).reshape(8, 4, 5, 3)
        self.actions = np.arange(8 * ACTION_DIM, dtype=np.float32).reshape(
            8, ACTION_DIM
        )
        self.states = np.arange(8 * STATE_DIM, dtype=np.float32).reshape(8, STATE_DIM)
        root.create_dataset("data/camera_rgb", data=self.images, chunks=(2, 4, 5, 3))
        root.create_dataset("data/action", data=self.actions)
        root.create_dataset("data/state", data=self.states)
        root.create_dataset("data/joint_position_command", data=self.actions + 100)
        root.create_dataset("data/alignment_valid", data=[1, 0, 1, 1, 1, 1, 1, 1])
        root.create_dataset("meta/episode_ends", data=[4, 8])

    def archive(self, wrapper=""):
        path = self.directory / ("wrapped.ZIP" if wrapper else "root.zip")
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            for source in self.path.rglob("*"):
                if source.is_file():
                    archive.write(
                        source, wrapper + source.relative_to(self.path).as_posix()
                    )
        return path

    def dataset(self, path=None, **kwargs):
        dataset = ReplayDataset(
            path or self.path,
            batch_size=3,
            observation_horizon=2,
            prediction_horizon=4,
            shuffle=False,
            **kwargs,
        )
        if dataset._store is not None:
            self.addCleanup(dataset._store.close)
        return dataset

    def test_directory_batches(self):
        dataset = self.dataset()
        self.assertEqual(dataset.n_episodes, 2)
        self.assertEqual(dataset.n_windows, 7)
        self.assertEqual(dataset.image_shape, (4, 5, 3))
        self.assertIsNone(dataset.preloaded_camera)
        batch = next(iter(dataset))
        np.testing.assert_array_equal(
            batch.observations["camera_rgb"], self.images[[[0, 0], [1, 2], [2, 3]]]
        )
        np.testing.assert_array_equal(
            batch.observations["state"], self.states[[[0, 0], [1, 2], [2, 3]]]
        )
        np.testing.assert_array_equal(
            batch.actions, self.actions[[[0, 0, 1, 2], [1, 2, 3, 3], [2, 3, 3, 3]]]
        )

    def test_zip_parity(self):
        for wrapper in ("", "replay.zarr/"):
            path = self.archive(wrapper)
            for preload in (False, True):
                for action_space in ("twist", "joint_position"):
                    with self.subTest(
                        wrapper=wrapper, preload=preload, action=action_space
                    ):
                        directory = self.dataset(
                            preload_images=preload, action_space=action_space
                        )
                        zipped = self.dataset(
                            path, preload_images=preload, action_space=action_space
                        )
                        self.assertEqual(zipped.n_episodes, directory.n_episodes)
                        self.assertEqual(zipped.n_windows, directory.n_windows)
                        self.assertEqual(zipped.image_shape, directory.image_shape)
                        self.assertEqual(len(zipped), len(directory))
                        np.testing.assert_array_equal(zipped.states, directory.states)
                        np.testing.assert_array_equal(zipped.actions, directory.actions)
                        self.assertEqual(zipped._store.zf.fp is None, preload)
                        for actual, expected in zip(zipped, directory, strict=True):
                            np.testing.assert_array_equal(
                                actual.actions, expected.actions
                            )
                            for key in expected.observations:
                                np.testing.assert_array_equal(
                                    actual.observations[key], expected.observations[key]
                                )

    def test_shared_preload_identity_and_one_read(self):
        for path in (self.path, self.archive(), self.archive("replay.zarr/")):
            with self.subTest(path=path), patch.object(
                zarr.Array,
                "__getitem__",
                autospec=True,
                side_effect=zarr.Array.__getitem__,
            ) as reads:
                train = self.dataset(path, preload_images=True, episodes=[0])
                val = self.dataset(
                    path, preloaded_camera=train.preloaded_camera, episodes=[1]
                )
                self.assertIs(val.preloaded_camera, train.preloaded_camera)
                np.testing.assert_array_equal(train.preloaded_camera, self.images)
                self.assertTrue((train.obs_idx < 4).all())
                self.assertTrue((val.obs_idx >= 4).all())
                list(train)
                list(val)
                camera_reads = [
                    call
                    for call in reads.call_args_list
                    if call.args[0].path.endswith("data/camera_rgb")
                ]
                self.assertEqual(len(camera_reads), 1)
                self.assertEqual(camera_reads[0].args[1], slice(None))

    def test_invalid_shared_camera(self):
        for camera, error, message in (
            (self.images[:-1], ValueError, "shape must match"),
            (self.images.astype(np.float32), TypeError, "dtype must be uint8"),
            (self.images.tolist(), TypeError, "must be a NumPy ndarray"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                self.dataset(preloaded_camera=camera, preload_images=True)

    def test_invalid_zip_layouts_close_store(self):
        for keys in (
            ["readme.txt"],
            ["a/.zgroup", "b/.zgroup"],
            ["nested/replay.zarr/.zgroup"],
        ):
            with self.subTest(keys=keys):
                path = self.directory / "invalid.zip"
                with ZipFile(path, "w") as archive:
                    for key in keys:
                        archive.writestr(key, '{"zarr_format": 2}')
                store = zarr.ZipStore(str(path), mode="r")
                self.addCleanup(store.close)
                with patch("giraf.learning.dataset.zarr.ZipStore", return_value=store):
                    with self.assertRaisesRegex(ValueError, "exactly one top-level"):
                        open_replay_group(path)
                self.assertIsNone(store.zf.fp)

    def test_cli_shares_preload(self):
        # Exercise CLI setup and batch loops with a lightweight policy double.
        for path in (self.path, self.archive(), self.archive("replay.zarr/")):
            for preload in (False, True):
                with self.subTest(path=path, preload=preload):
                    config = train_cli.parse_config(
                        [
                            "--dataset", str(path),
                            "--output-dir", str(self.directory / "run"),
                            "--epochs", "1", "--batch-size", "3",
                            "--val-fraction", "0.05",
                            *(["--preload-images"] if preload else []),
                        ]
                    )
                    datasets, stores = [], []

                    def make_dataset(*args, **kwargs):
                        dataset = ReplayDataset(*args, **kwargs)
                        datasets.append(dataset)
                        if dataset._store is not None:
                            self.addCleanup(dataset._store.close)
                        return dataset

                    def open_group(path):
                        root, store = open_replay_group(path)
                        if store is not None:
                            stores.append(store)
                            self.addCleanup(store.close)
                        return root, store

                    def make_policy(policy_config, *, normalizer):
                        return Mock(
                            config=policy_config,
                            normalizer=normalizer,
                            device="cpu",
                            train_step=Mock(return_value={"loss": 1.0}),
                            evaluate=Mock(
                                return_value={"loss": 1.0, "action_mse": 1.0}
                            ),
                        )

                    with (
                        patch.object(
                            train_cli, "ReplayDataset", side_effect=make_dataset
                        ),
                        patch.object(
                            train_cli, "open_replay_group", side_effect=open_group
                        ),
                        patch.object(
                            train_cli, "DiffusionPolicy", side_effect=make_policy
                        ),
                        patch.object(train_cli, "_build_lr_scheduler") as scheduler,
                        patch.object(
                            zarr.Array, "__getitem__", autospec=True,
                            side_effect=zarr.Array.__getitem__,
                        ) as reads,
                    ):
                        scheduler.return_value.get_last_lr.return_value = [1e-4]
                        self.assertEqual(
                            train_cli.run(config), config.output_dir / "policy.pt"
                        )
                    train, val = datasets
                    self.assertIs(train.preloaded_camera, val.preloaded_camera)
                    self.assertEqual(train.preloaded_camera is not None, preload)
                    self.assertTrue(set(train.episodes).isdisjoint(val.episodes))
                    camera_reads = [
                        call
                        for call in reads.call_args_list
                        if call.args[0].path.endswith("data/camera_rgb")
                    ]
                    self.assertEqual(len(camera_reads), int(preload))
                    for store in stores:
                        self.assertIsNone(store.zf.fp)


if __name__ == "__main__":
    unittest.main()
