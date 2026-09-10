"""Command line for a single dataset/seed under the fixed paper protocol."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from lightning.fabric import Fabric

from active_wishart_tpp.config import ExperimentConfig
from active_wishart_tpp.training.artifact_io import write_json
from active_wishart_tpp.training.runner import ExperimentRunner


def main():
    """Run Pure, Wishart, or their sequential matched pair from one shared pretrain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("pure", "wishart", "both"), default="both")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    changes = {
        key: getattr(args, key)
        for key in ("seed", "dataset", "output_root")
        if getattr(args, key) is not None
    }
    config = replace(config, runtime=replace(config.runtime, **changes))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    fabric = Fabric(
        accelerator=config.runtime.accelerator, devices=1, precision="32-true"
    )
    fabric.launch()
    runner = ExperimentRunner(config, fabric)
    shared = runner.shared()
    methods = ("pure", "wishart") if args.method == "both" else (args.method,)
    for method in methods:
        runner.fit(method, shared)
    write_json(
        runner.output / "completion.json", dict(completed=True, methods=list(methods))
    )


if __name__ == "__main__":
    main()
