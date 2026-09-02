#!/usr/bin/env python3
"""Run the fixed-q matched COTIC/Pure/Wishart real-data benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.fabric import Fabric

from wishart_tpp.data import DatasetSplit
from wishart_tpp.real_protocol import ComputeShards, RealProtocol
from wishart_tpp.training.artifact_io import (
    sha256,
    tensor_tree_equal,
    write_json,
)
from wishart_tpp.training.experiment import ExperimentContext
from wishart_tpp.training.pure_branch import fit_pure_branch
from wishart_tpp.training.selected_evaluation import (
    METHODS,
    combine_group_curves,
    evaluate_selected,
)
from wishart_tpp.training.shared_pretrain import SharedPretrainer
from wishart_tpp.training.wishart_branch import fit_wishart_branch
from wishart_tpp.training.wishart_experiment import WishartExperimentRunner

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "configs/real_data.yaml"
DEFAULT_ROOT = WORKSPACE / "runs/real_data_fixed_q_50cycle"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=RealProtocol.datasets, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--path-shard-size", type=int)
    parser.add_argument("--evaluation-sample-shard-size", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    for name in (
        "batch_size",
        "evaluation_batch_size",
        "path_shard_size",
        "evaluation_sample_shard_size",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def subset_context(context: ExperimentContext) -> ExperimentContext:
    """Reduce every split for a fast end-to-end smoke run."""

    return ExperimentContext(
        context.dataset,
        DatasetSplit(
            context.split.train.select(np.arange(64)),
            context.split.validation.select(np.arange(16)),
            context.split.test.select(np.arange(16)),
        ),
        context.batch_size,
        context.evaluation_batch_size,
        context.evaluator,
    )


def verify_aligned(left: ExperimentContext, right: ExperimentContext) -> None:
    """Reject method contexts that do not contain identical trajectories."""

    for name in ("train", "validation", "test"):
        left_ids = getattr(left.split, name).source_ids
        right_ids = getattr(right.split, name).source_ids
        if not np.array_equal(left_ids, right_ids):
            raise RuntimeError(f"Pure/Wishart {name} source IDs differ")


def requested_shards(args: argparse.Namespace) -> ComputeShards:
    """Merge explicit compute overrides with dataset-safe defaults."""

    default = RealProtocol.default_shards(args.dataset)
    return ComputeShards(
        args.batch_size or default.batch,
        args.evaluation_batch_size or default.evaluation_batch,
        args.path_shard_size or default.path,
        args.evaluation_sample_shard_size or default.evaluation_samples,
    )


def manifest_payload(args, pure_config, wishart_config, context) -> dict:
    """Describe the complete scientific and computational run contract."""

    training = wishart_config.training
    return {
        "complete": False,
        "smoke": args.smoke,
        "dataset": args.dataset,
        "seed": args.seed,
        "methods": list(METHODS),
        "train_paths": len(context.split.train.sequences),
        "validation_paths": len(context.split.validation.sequences),
        "test_paths": len(context.split.test.sequences),
        "time_normalization": context.dataset.normalization.as_dict(),
        "shared_pretrain_steps": training.shared_pretrain_steps,
        "main_cycles": training.active_cycles,
        "updates_per_cycle": training.neural_steps_per_cycle,
        "effective_batch_size": training.effective_batch_size,
        "fixed_q_elbo": True,
        "common_shared_checkpoint_all_methods": True,
        "cotic_continuous_optimizer_rng_from_pretrain": True,
        "wishart_contract": {
            "active_block_likelihood": "convex_intensity_v1",
            "population_df": training.population_df,
            "population_df_fixed": not training.learn_population_df,
            "initial_alpha": training.initial_alpha,
            "em_batch_size": training.em_batch_size,
            "w_to_pi_routing": False,
            "correlation_squared_attention": False,
            "block_traces": False,
            "full_kc_matrix": False,
            "direct_prior_shortcut": False,
            "posterior_cache": False,
            "stored_cross_block_responsibilities": False,
        },
        "pure_config": pure_config.as_dict(),
        "wishart_config": wishart_config.as_dict(),
        "test_read": False,
    }


def verify_curves(group: Path, cycles: int, pure_steps: int, interval: int) -> None:
    """Require complete matched validation grids before marking completion."""

    pretrain = pd.read_csv(group / "shared/validation_curve.csv")
    expected_pretrain = list(range(len(pretrain)))
    if pretrain.cycle.astype(int).tolist() != expected_pretrain:
        raise RuntimeError("shared pretrain curve is incomplete")
    for method in METHODS:
        curve = pd.read_csv(group / method / "validation_curve.csv")
        curve = curve.loc[curve.method == method]
        expected = (
            list(range(cycles + 1))
            if method == "wishart_k5"
            else list(range(pure_steps // interval + 1))
        )
        if curve.cycle.astype(int).tolist() != expected:
            raise RuntimeError(f"{method} validation curve is incomplete")


def main() -> None:
    args = arguments()
    root = args.output_root.resolve()
    if args.smoke:
        root = root.with_name(root.name + "_smoke")
    group = root / args.dataset / f"seed_{args.seed}"
    completion = group / "completion.json"
    if completion.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("complete") and bool(payload.get("smoke")) == args.smoke:
            print(f"[real] dataset={args.dataset} seed={args.seed} reused")
            return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    pure_config, wishart_config = RealProtocol(args.config).configurations(
        dataset=args.dataset,
        seed=args.seed,
        output=group,
        smoke=args.smoke,
        shards=requested_shards(args),
    )
    fabric = Fabric(accelerator="cuda", devices=1, precision="32-true")
    fabric.launch()
    pure_runner = WishartExperimentRunner(pure_config, fabric)
    wishart_runner = WishartExperimentRunner(wishart_config, fabric)
    pure_context = pure_runner._prepare(args.dataset)
    wishart_context = wishart_runner._prepare(args.dataset)
    if args.smoke:
        pure_context = subset_context(pure_context)
        wishart_context = subset_context(wishart_context)
    verify_aligned(pure_context, wishart_context)
    group.mkdir(parents=True, exist_ok=True)
    manifest = manifest_payload(args, pure_config, wishart_config, pure_context)
    write_json(group / "manifest.json", manifest)

    shared_output = group / "shared"
    shared = SharedPretrainer(pure_runner).fit(
        pure_context,
        shared_output,
        steps=pure_config.training.shared_pretrain_steps,
    )
    interval = 1 if args.smoke else wishart_config.training.neural_steps_per_cycle
    fit_pure_branch(
        pure_runner,
        pure_context,
        shared,
        group / "cotic_k1",
        method="cotic_k1",
        components=1,
        validation_interval=interval,
        seed_scheduler_with_initial_validation=True,
        continuous_optimizer_checkpoint=(
            shared_output / "shared_pretrain_checkpoint.pt"
        ),
    )
    fit_pure_branch(
        pure_runner,
        pure_context,
        shared,
        group / "pure_k5",
        method="pure_k5",
        components=5,
        validation_interval=interval,
        seed_scheduler_with_initial_validation=True,
    )
    fit_wishart_branch(wishart_runner, wishart_context, shared, group / "wishart_k5")

    active = torch.load(
        group / "wishart_k5/active_cycle_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected = torch.load(
        group / "wishart_k5/selected_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not tensor_tree_equal(selected["model_state"], active["best"]["model_state"]):
        raise RuntimeError("selected Wishart state mismatches the EM checkpoint")
    evaluated = evaluate_selected(
        pure_runner,
        pure_context,
        shared,
        group,
        skip_test=args.smoke or args.skip_test,
    )
    combine_group_curves(group)
    verify_curves(
        group,
        wishart_config.training.active_cycles,
        pure_config.training.pure_steps,
        interval,
    )

    result = {
        "complete": True,
        "smoke": args.smoke,
        "dataset": args.dataset,
        "seed": args.seed,
        "shared_checkpoint_sha256": sha256(shared_output / "shared_checkpoint.pt"),
        "selected_state_matches_checkpoint": True,
        "methods": evaluated,
        "test_read": not (args.smoke or args.skip_test),
    }
    write_json(group / "result.json", result)
    manifest.update(
        {
            "complete": True,
            "test_read": result["test_read"],
            "selected_state_matches_checkpoint": True,
            "shared_checkpoint_sha256": result["shared_checkpoint_sha256"],
        }
    )
    write_json(group / "manifest.json", manifest)
    write_json(
        completion,
        {
            "complete": True,
            "smoke": args.smoke,
            "dataset": args.dataset,
            "seed": args.seed,
            "test_read": result["test_read"],
        },
    )


if __name__ == "__main__":
    main()
