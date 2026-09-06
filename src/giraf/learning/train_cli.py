"""Train a DiffusionPolicy on a GIRAF ReplayBuffer with checkpoints and logging."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import zarr

from giraf.data.schema import ACTION_SPACES

from .dataset import ReplayDataset, split_episodes
from .diffusion import DiffusionPolicy, DiffusionPolicyConfig
from .pipeline import evaluate, train


@dataclass(frozen=True)
class TrainConfig:
    dataset: Path
    output_dir: Path
    resume: Path | None = None
    start_epoch: int = 0
    epochs: int = 100
    batch_size: int = 64
    checkpoint_every: int = 10
    seed: int = 0
    val_fraction: float = 0.1
    preload_images: bool = False
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="restore model and optimizer from an existing policy checkpoint",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="last completed epoch in --resume (required when resuming)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10, help="in epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of episodes held out for validation",
    )
    parser.add_argument(
        "--preload-images", action="store_true", help="hold all frames in RAM"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="linear learning-rate warmup, in optimizer steps",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="LR floor at the end of cosine decay, as a fraction of the peak",
    )
    parser.add_argument(
        "--down-dims", type=int, nargs="+", default=None, help="U-Net widths"
    )
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument(
        "--encoder",
        choices=("conv", "resnet18", "dinov2"),
        default="conv",
        help="image encoder; dinov2 downloads weights through torch.hub on first use",
    )
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="giraf")
    parser.add_argument(
        "--action-space",
        choices=ACTION_SPACES,
        default="twist",
        help="action representation: twist (m/s, rad/s) or joint_position",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=16,
        help="length of the predicted action chunk",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=8,
        help="executed steps per replan",
    )
    parser.add_argument(
        "--no-temporal-ensemble",
        action="store_true",
        help="disable averaging overlapping action-chunk predictions in act()",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.checkpoint_every <= 0:
        raise SystemExit("epochs, batch-size and checkpoint-every must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("warmup-steps must be non-negative")
    if not 0 < args.min_lr_ratio <= 1:
        raise SystemExit("min-lr-ratio must satisfy 0 < min-lr-ratio <= 1")
    if not 0 <= args.val_fraction < 1:
        raise SystemExit("val-fraction must satisfy 0 <= val-fraction < 1")
    if args.resume is None and args.start_epoch != 0:
        raise SystemExit("--start-epoch requires --resume")
    if args.resume is not None and args.start_epoch <= 0:
        raise SystemExit("--resume requires a positive --start-epoch")
    if args.resume is not None and args.epochs <= args.start_epoch:
        raise SystemExit("--epochs must be greater than --start-epoch")
    policy = DiffusionPolicyConfig(
        learning_rate=args.learning_rate,
        device=args.device,
        encoder=args.encoder,
        action_space=args.action_space,
        prediction_horizon=args.prediction_horizon,
        action_horizon=args.action_horizon,
        temporal_ensemble=not args.no_temporal_ensemble,
    )
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
        resume=args.resume,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        val_fraction=args.val_fraction,
        preload_images=args.preload_images,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        policy=policy,
    )


class _Logger:
    """Append one JSON line per epoch to metrics.jsonl, optionally mirrored to wandb."""

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


def _build_lr_scheduler(
    policy: DiffusionPolicy, config: TrainConfig, batches_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a warmup-then-cosine LR schedule, fast-forwarded to a resumed epoch."""

    total_steps = config.epochs * batches_per_epoch
    warmup_steps = config.warmup_steps
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        """Linear warmup to 1.0 over warmup_steps, then cosine decay to min_lr_ratio."""

        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(policy.optimizer, lr_lambda)
    with warnings.catch_warnings():
        # Fast-forwarding replays past steps without matching optimizer.step()
        # calls, which torch otherwise (harmlessly) warns about.
        warnings.filterwarnings(
            "ignore", message=r"Detected call of `lr_scheduler\.step\(\)`"
        )
        for _ in range(config.start_epoch * batches_per_epoch):
            scheduler.step()
    return scheduler


_RESUME_WATCHED_FIELDS = (
    "encoder",
    "action_space",
    "prediction_horizon",
    "action_horizon",
    "temporal_ensemble",
    "learning_rate",
    "down_dims",
    "diffusion_steps",
)


def _ignored_resume_flags(
    cli_policy: DiffusionPolicyConfig, checkpoint_policy: DiffusionPolicyConfig
) -> list[str]:
    """Return watched policy fields where the CLI value differs from the checkpoint."""

    return [
        field
        for field in _RESUME_WATCHED_FIELDS
        if getattr(cli_policy, field) != getattr(checkpoint_policy, field)
    ]


def run(config: TrainConfig) -> Path:
    """Train, checkpoint, and return the path of the final policy."""

    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if (
        config.resume is not None
        and config.resume.resolve().parent == config.output_dir.resolve()
    ):
        raise ValueError(
            "a resumed legacy checkpoint must use a new --output-dir so the "
            "original metrics and checkpoints remain untouched"
        )
    if config.resume is None:
        policy = None
        policy_config = config.policy
    else:
        policy = DiffusionPolicy.load(config.resume, device=config.policy.device)
        policy_config = policy.config
        ignored = _ignored_resume_flags(config.policy, policy_config)
        if ignored:
            print(
                f"[TRAIN] resuming from checkpoint; ignoring CLI flags: {ignored}",
                flush=True,
            )

    root = zarr.open_group(str(config.dataset), mode="r")
    dataset_episode_count = len(root["meta/episode_ends"])
    train_episodes, val_episodes = split_episodes(
        dataset_episode_count, config.val_fraction, config.seed
    )
    train_dataset = ReplayDataset(
        config.dataset,
        batch_size=config.batch_size,
        observation_horizon=policy_config.observation_horizon,
        prediction_horizon=policy_config.prediction_horizon,
        seed=config.seed,
        start_epoch=config.start_epoch,
        preload_images=config.preload_images,
        episodes=train_episodes,
        action_space=policy_config.action_space,
    )
    val_dataset = None
    if val_episodes:
        val_dataset = ReplayDataset(
            config.dataset,
            batch_size=config.batch_size,
            observation_horizon=policy_config.observation_horizon,
            prediction_horizon=policy_config.prediction_horizon,
            shuffle=False,
            preload_images=config.preload_images,
            episodes=val_episodes,
            action_space=policy_config.action_space,
        )
    if config.resume is not None and val_dataset is not None:
        print(
            "[TRAIN] warning: validation episodes may have been trained on by "
            "the resumed checkpoint",
            flush=True,
        )

    normalizer = train_dataset.fit_normalizer()
    if policy is None:
        policy = DiffusionPolicy(policy_config, normalizer=normalizer)
    elif policy.normalizer is None:
        raise ValueError("resumed checkpoint has no normalizer")
    config = replace(config, policy=policy.config)
    config_payload = _json_safe(asdict(config))
    config_payload["train_episodes"] = train_episodes
    config_payload["val_episodes"] = val_episodes
    (config.output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n"
    )
    (config.output_dir / "normalizer.json").write_text(
        json.dumps(policy.normalizer.to_dict(), indent=2) + "\n"
    )
    print(
        f"[TRAIN] {train_dataset.n_windows} windows, {len(train_dataset)} "
        f"batches/epoch, device={policy.device}, "
        f"train_episodes={train_episodes}, val_episodes={val_episodes}",
        flush=True,
    )
    if config.resume is not None:
        print(
            f"[TRAIN] warm-started {config.resume} after epoch {config.start_epoch}; "
            f"target={config.epochs}",
            flush=True,
        )

    lr_scheduler = _build_lr_scheduler(policy, config, len(train_dataset))

    logger = _Logger(config)
    latest = config.output_dir / "policy.pt"
    best = config.output_dir / "best.pt"
    best_val_action_mse = math.inf
    try:
        for epoch in range(config.start_epoch + 1, config.epochs + 1):
            started = time.monotonic()
            metrics = _mean_metrics(
                train(
                    policy,
                    train_dataset,
                    epochs=1,
                    on_step=lambda _: lr_scheduler.step(),
                )
            )
            metrics["lr"] = lr_scheduler.get_last_lr()[0]
            if val_dataset is not None:
                val_metrics = evaluate(policy, val_dataset)
                metrics["val_loss"] = val_metrics["loss"]
                metrics["val_action_mse"] = val_metrics["action_mse"]
                if metrics["val_action_mse"] < best_val_action_mse:
                    best_val_action_mse = metrics["val_action_mse"]
                    policy.save(best)
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
