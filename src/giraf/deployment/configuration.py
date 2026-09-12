"""Validated command-line configuration for deployment."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from giraf.drivers.optitrack import DEFAULT_RIGID_BODY_ID, DEFAULT_SERVER_IP


class DeploymentMode(str, Enum):
    SHADOW = "shadow"
    DRY_RUN = "dry-run"
    HARDWARE = "hardware"


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    checkpoint: Path
    collector_config: Path = Path("config/data_collection.yaml")
    mode: DeploymentMode = DeploymentMode.SHADOW
    device: str = "cuda"
    action_scale: float = 0.2
    inference_steps: int | None = None
    server_ip: str = DEFAULT_SERVER_IP
    client_ip: str | None = None
    rigid_id: int = DEFAULT_RIGID_BODY_ID
    action_timeout_s: float = 0.5
    max_frame_age_s: float = 0.15
    state_margin_fraction: float = 0.05
    enforce_training_bounds: bool = False
    allow_grasp: bool = False
    seed: int = 0
    log_dir: Path | None = None
    save_video: bool = True
    hardware_confirmed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DeploymentMode):
            raise TypeError("mode must be a DeploymentMode")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint}")
        if not self.collector_config.is_file():
            raise FileNotFoundError(
                f"collector config does not exist: {self.collector_config}"
            )
        if self.inference_steps is not None and self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive when provided")
        if self.mode is DeploymentMode.HARDWARE and not self.hardware_confirmed:
            raise ValueError(
                "hardware mode requires confirmation that the robot is physically "
                "at the teleop home pose"
            )
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.action_timeout_s, self.max_frame_age_s)
        ):
            raise ValueError("timeout values must be finite and positive")
        if not math.isfinite(self.action_scale) or not 0 <= self.action_scale <= 1:
            raise ValueError("action_scale must be finite and in [0, 1]")
        if (
            not math.isfinite(self.state_margin_fraction)
            or self.state_margin_fraction < 0
        ):
            raise ValueError("state_margin_fraction must be finite and non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("config/data_collection.yaml")
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in DeploymentMode],
        default=DeploymentMode.SHADOW.value,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="override checkpoint diffusion inference steps",
    )
    parser.add_argument("--server-ip", default=DEFAULT_SERVER_IP)
    parser.add_argument("--client-ip", default=None)
    parser.add_argument("--rigid-id", type=int, default=DEFAULT_RIGID_BODY_ID)
    parser.add_argument("--action-timeout", type=float, default=0.5, help="seconds")
    parser.add_argument("--max-frame-age", type=float, default=0.15, help="seconds")
    parser.add_argument("--state-margin", type=float, default=0.05)
    parser.add_argument(
        "--enforce-training-bounds",
        action="store_true",
        help="pause policy when state leaves expanded training-data bounds",
    )
    parser.add_argument("--allow-grasp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help=(
            "confirm hardware control and that the robot is physically at the "
            "teleop home pose"
        ),
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> DeploymentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = DeploymentMode(args.mode)
    if mode is DeploymentMode.HARDWARE and not args.confirm_hardware:
        parser.error("--mode hardware requires --confirm-hardware")
    return DeploymentConfig(
        checkpoint=args.checkpoint,
        collector_config=args.config,
        mode=mode,
        device=args.device,
        action_scale=args.action_scale,
        inference_steps=args.inference_steps,
        server_ip=args.server_ip,
        client_ip=args.client_ip,
        rigid_id=args.rigid_id,
        action_timeout_s=args.action_timeout,
        max_frame_age_s=args.max_frame_age,
        state_margin_fraction=args.state_margin,
        enforce_training_bounds=args.enforce_training_bounds,
        allow_grasp=args.allow_grasp,
        seed=args.seed,
        log_dir=args.log_dir,
        save_video=not args.no_video,
        hardware_confirmed=args.confirm_hardware,
    )
