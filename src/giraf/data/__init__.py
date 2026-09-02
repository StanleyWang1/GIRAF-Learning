"""Demonstration recording, storage, and replay."""

from .config import CollectorConfig, load_config
from .pipeline import DataCollectionPipeline
from .schema import ACTION_DIM, STATE_DIM, diffusion_shape_meta

__all__ = [
    "ACTION_DIM",
    "CollectorConfig",
    "DataCollectionPipeline",
    "STATE_DIM",
    "diffusion_shape_meta",
    "load_config",
]
