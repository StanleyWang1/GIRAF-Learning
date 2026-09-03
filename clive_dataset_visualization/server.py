from __future__ import annotations

import uvicorn

from dataset_visualization.backend.app import create_app
from dataset_visualization.backend.config import AppConfig


def run_server(config: AppConfig) -> None:
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
