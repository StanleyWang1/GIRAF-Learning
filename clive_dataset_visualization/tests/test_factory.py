from pathlib import Path

import pytest

pytest.importorskip("zarr")
import zarr

from dataset_visualization.backend.adapters.factory import create_adapter
from dataset_visualization.backend.adapters.full_zarr import FullZarrAdapter
from dataset_visualization.backend.adapters.raw_sidecar import RawSidecarAdapter


def _create_base_dataset(path: Path):
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    data.array("action", [[0.0] * 14, [0.1] * 14], dtype="f4", chunks=(2, 14), overwrite=True)
    data.array("timestamps", [1.0, 2.0], dtype="f8", chunks=(2,), overwrite=True)
    meta.array("episode_ends", [2], dtype="i8", chunks=(1,), overwrite=True)


def test_factory_detects_full_zarr(tmp_path: Path):
    zpath = tmp_path / "full_like.zarr"
    _create_base_dataset(zpath)

    root = zarr.open(str(zpath), mode="a")
    data = root["data"]
    data.array("camera_123_rgb", [[[ [0, 0, 0] ]], [[[1, 1, 1]]]], dtype="u1", chunks=(1, 1, 1, 3), overwrite=True)

    adapter = create_adapter(zpath)
    assert isinstance(adapter, FullZarrAdapter)


def test_factory_detects_raw_sidecar(tmp_path: Path):
    zpath = tmp_path / "raw_like.zarr"
    _create_base_dataset(zpath)

    adapter = create_adapter(zpath)
    assert isinstance(adapter, RawSidecarAdapter)


def test_factory_rejects_index_based(tmp_path: Path):
    zpath = tmp_path / "legacy_indices.zarr"
    _create_base_dataset(zpath)

    root = zarr.open(str(zpath), mode="a")
    root["data"].array("camera_123_indices", [0, 1], dtype="i8", chunks=(2,), overwrite=True)

    adapter = create_adapter(zpath)
    assert not adapter.supported
    summary = adapter.dataset_summary()
    assert summary.supported is False
    assert "camera_*_indices" in (summary.unsupported_reason or "")
