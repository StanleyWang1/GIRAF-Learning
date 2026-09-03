"""Train a DiffusionPolicy on a GIRAF ReplayBuffer with checkpoints and logging."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .dataset import ReplayDataset
from .diffusion import DiffusionPolicy, DiffusionPolicyConfig
from .pipeline import train


@dataclass(frozen=True)
class TrainConfig:
    dataset: Path
    output_dir: Path
    epochs: int = 100
    batch_size: int = 64
    checkpoint_every: int = 10
    seed: int = 0
    preload_images: bool = False
    wandb: bool = False
    wandb_project: str = "giraf"
    policy: DiffusionPolicyConfig = DiffusionPolicyConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="replay_buffer.zarr"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run directory (default: checkpoints/<timestamp>)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="in epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preload-images", action="store_true", help="hold all frames in RAM"
    )
    parser.add_argument(
        "--down-dims", type=int, nargs="+", default=None, help="U-Net widths"
    )
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="giraf")
    return parser


def parse_config(argv: Sequence[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.checkpoint_every <= 0:
        raise SystemExit("epochs, batch-size and checkpoint-every must be positive")
    policy = DiffusionPolicyConfig(learning_rate=args.learning_rate, device=args.device)
    if args.down_dims is not None:
        policy = replace(policy, down_dims=tuple(args.down_dims))
    if args.diffusion_steps is not None:
        policy = replace(
            policy,
            diffusion_steps=args.diffusion_steps,
            inference_steps=min(policy.inference_steps, args.diffusion_steps),
        )
    output_dir = args.output_dir or Path("checkpoints") / time.strftime("%Y%m%d-%H%M%S")
    return TrainConfig(
        dataset=args.dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        preload_images=args.preload_images,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        policy=policy,
    )


class _Logger:
    """Append one JSON line per epoch to metrics.jsonl and optionally mirror to wandb."""

    def __init__(self, config: TrainConfig) -> None:
        self._file = (config.output_dir / "metrics.jsonl").open("a")
        self._run = None
        if config.wandb:
            import wandb  # optional dependency: uv sync --extra train

            # Offline by default so training works without an account; set
            # WANDB_MODE=online to upload.
            self._run = wandb.init(
                project=config.wandb_project,
                dir=str(config.output_dir),
                mode=os.environ.get("WANDB_MODE", "offline"),
                config=_json_safe(asdict(config)),
            )

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        record = {"epoch": epoch, **metrics}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        print(json.dumps(record), flush=True)
        if self._run is not None:
            self._run.log(metrics, step=epoch)

    def close(self) -> None:
        self._file.close()
        if self._run is not None:
            self._run.finish()


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean_metrics(history: list[dict[str, float]]) -> dict[str, float]:
    keys = history[0].keys()
    return {key: float(np.mean([step[key] for step in history])) for key in keys}


def run(config: TrainConfig) -> Path:
    """Train, checkpoint, and return the path of the final policy."""

    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ReplayDataset(
        config.dataset,
        batch_size=config.batch_size,
        observation_horizon=config.policy.observation_horizon,
        prediction_horizon=config.policy.prediction_horizon,
        seed=config.seed,
        preload_images=config.preload_images,
    )
    normalizer = dataset.fit_normalizer()
    policy = DiffusionPolicy(config.policy, normalizer=normalizer)
    (config.output_dir / "config.json").write_text(
        json.dumps(_json_safe(asdict(config)), indent=2) + "\n"
    )
    (config.output_dir / "normalizer.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2) + "\n"
    )
    print(
        f"[TRAIN] {dataset.n_windows} windows, {len(dataset)} batches/epoch, "
        f"device={policy.device}",
        flush=True,
    )

    logger = _Logger(config)
    latest = config.output_dir / "policy.pt"
    try:
        for epoch in range(1, config.epochs + 1):
            started = time.monotonic()
            metrics = _mean_metrics(train(policy, dataset, epochs=1))
            metrics["epoch_seconds"] = time.monotonic() - started
            logger.log(epoch, metrics)
            policy.save(latest)
            if epoch % config.checkpoint_every == 0:
                policy.save(config.output_dir / f"policy_epoch_{epoch:04d}.pt")
    finally:
        logger.close()
    return latest


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_config(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
