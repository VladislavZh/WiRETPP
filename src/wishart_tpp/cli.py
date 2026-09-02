"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from lightning.fabric import Fabric

from wishart_tpp.config import ExperimentConfig
from wishart_tpp.training.pure_experiment import PureExperimentRunner
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner


def _configure_determinism(enabled: bool) -> None:
    """Select reproducible CUDA kernels before Fabric initializes the device."""

    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/all12.yaml"))
    parser.add_argument("--backbone", choices=("cotic", "thp", "nhp", "rmtpp"))
    parser.add_argument("--integral-method", choices=("gauss_legendre", "monte_carlo"))
    parser.add_argument("--integral-samples", type=int)
    parser.add_argument("--method", choices=("wishart", "pure"))
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--dataset")
    parser.add_argument("--accelerator")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--shared-checkpoint-root", type=Path)
    parser.add_argument("--replication-seed", type=int)
    parser.add_argument("--compute-batch-size", type=int)
    parser.add_argument("--shared-compute-batch-size", type=int)
    parser.add_argument("--packed-filename")
    return parser


def _configured_experiment(args: argparse.Namespace) -> ExperimentConfig:
    # Command-line values override only the corresponding typed config fields.
    config = ExperimentConfig.from_yaml(args.config)
    model = config.model
    training = config.training
    runtime = config.runtime
    if args.backbone:
        model = replace(model, backbone=args.backbone)
    if args.integral_method:
        model = replace(model, integral_method=args.integral_method)
    if args.integral_samples is not None:
        model = replace(model, integral_samples=args.integral_samples)
    if args.method:
        training = replace(training, method=args.method)
    if args.fixed_alpha is not None:
        training = replace(training, fixed_alpha=args.fixed_alpha)
    if args.compute_batch_size is not None:
        if args.compute_batch_size < 2:
            raise ValueError("--compute-batch-size must be at least 2")
        training = replace(
            training,
            batch_size=args.compute_batch_size,
            evaluation_batch_size=args.compute_batch_size // 2,
            path_shard_size=4 * args.compute_batch_size,
        )
    if args.shared_compute_batch_size is not None:
        if args.shared_compute_batch_size < 2:
            raise ValueError("--shared-compute-batch-size must be at least 2")
        training = replace(
            training,
            shared_batch_size=args.shared_compute_batch_size,
            shared_evaluation_batch_size=args.shared_compute_batch_size // 2,
            shared_path_shard_size=4 * args.shared_compute_batch_size,
        )
    if args.replication_seed is not None:
        model = replace(model, integral_seed=args.replication_seed)
    runtime_overrides = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "accelerator": args.accelerator,
            "output_root": args.output_root,
            "shared_checkpoint_root": args.shared_checkpoint_root,
            "packed_filename": args.packed_filename,
        }.items()
        if value is not None
    }
    if args.replication_seed is not None:
        runtime_overrides.update(
            optimization_seed=args.replication_seed,
            monte_carlo_seed=args.replication_seed,
        )
    config = replace(
        config,
        model=model,
        training=training,
        runtime=replace(runtime, **runtime_overrides),
    )
    return config


def main() -> None:
    config = _configured_experiment(_parser().parse_args())
    _configure_determinism(config.runtime.deterministic)

    # Fabric owns device and precision; the experiment retains the explicit loop.
    fabric = Fabric(
        accelerator=config.runtime.accelerator,
        devices=1,
        precision=config.runtime.precision,
    )
    fabric.launch()
    runner = {
        "pure": PureExperimentRunner,
        "wishart": WishartExperimentRunner,
    }[config.training.method]
    runner(config, fabric).run()


if __name__ == "__main__":
    main()
