#!/usr/bin/env python3
"""Sweep only the overcomplete THP-backbone LR in the corrected big protocol.

The component heads stay at 1e-3.  While K exceeds the target, the shared THP
backbone uses the swept rate (or is exactly frozen for the zero reference).
After the final physical removal returns the model to K=5, the backbone is
linearly restored to 1e-3 over the original 200-step ramp.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from run_sin_k5_thp_frozen_backbone_lr_warmup_sweep import (
    _gpu_sample,
    _run_with_vram_guard,
    _write_csv,
)
from run_two_phase_all12_benchmark import _write_json


DATASET = "sin_K5_C5"
SEED = 0
SPLIT_SEED = 42
HEAD_LEARNING_RATE = 1e-3
POST_TARGET_BACKBONE_LEARNING_RATE = 1e-3
BACKBONE_RESTORE_RAMP_STEPS = 200
LOGICAL_EFFECTIVE_BATCH_SIZE = 128


def _lr_tag(value: float) -> str:
    if value == 0.0:
        return "lr_0_exact_frozen"
    return f"lr_{value:.0e}".replace("e-", "em").replace("e+", "ep")


def _prepare_pretrain(source_dir: Path, run_dir: Path) -> None:
    mappings = (
        (
            source_dir / "checkpoint_cotic_k1.pt",
            run_dir / "checkpoint_cotic_k1.pt",
        ),
        (
            source_dir / "pretrain_history.csv",
            run_dir / "pretrain_history.csv",
        ),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for source, target in mappings:
        if not source.is_file():
            raise FileNotFoundError(f"missing big-protocol K=1 artifact: {source}")
        if not target.is_file():
            shutil.copy2(source, target)


def _command(
    root: Path,
    *,
    run_dir: Path,
    overcomplete_backbone_learning_rate: float,
    batch_size: int,
    evaluation_batch_size: int,
    pruning_batch_size: int,
    gradient_accumulation_steps: int,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "run_wishart_overcomplete_backward_elimination.py"),
        "--architecture", "thp",
        "--data-root", str(root / "data"),
        "--dataset", DATASET,
        "--outdir", str(run_dir),
        "--seed", str(SEED),
        "--split-seed", str(SPLIT_SEED),
        "--initial-components", "10",
        "--target-components", "5",
        "--warmup-steps", "300",
        "--prune-interval", "100",
        "--post-pruning-steps", "600",
        "--batch-size", str(batch_size),
        "--evaluation-batch-size", str(evaluation_batch_size),
        "--pruning-batch-size", str(pruning_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--neural-learning-rate", str(HEAD_LEARNING_RATE),
        "--backbone-learning-rate", str(POST_TARGET_BACKBONE_LEARNING_RATE),
        "--backbone-ramp-steps", str(BACKBONE_RESTORE_RAMP_STEPS),
        "--omega-learning-rate", "0.01",
        "--alpha-learning-rate", "0.0001",
        "--weight-decay", "1e-5",
        "--gradient-clip", "20",
        "--train-samples", "2",
        "--validation-samples", "8",
        "--test-samples", "16",
        "--pruning-samples", "8",
        "--alpha-temperature", "1",
        "--initial-alpha", "0.1",
        "--alpha-parameterization", "projected",
        "--interaction-mode", "convex",
        "--mean-hyperprior-strength", "1.566",
        "--initial-beta", "0.9",
        "--beta-decay-rate", "5",
        "--exploration-nu", "200",
        "--initial-rho", "5",
        "--final-rho", "5",
        "--dual-learning-rate", "0.01",
        "--nu-multiplier", "1.5",
        "--pretrain-steps", "10",
        "--head-initialization-noise", "0.01",
        "--dpp-strength", "0",
        "--dpp-bandwidth", "1",
        "--dpp-jitter", "1e-4",
        "--dpp-decay-steps", "200",
        "--device", "cuda",
    ]
    if overcomplete_backbone_learning_rate == 0.0:
        command.append("--freeze-encoder-until-target")
    else:
        command.extend((
            "--overcomplete-backbone-learning-rate",
            str(overcomplete_backbone_learning_rate),
        ))
    return command


def _load_vram_samples(run_dir: Path) -> list[dict]:
    path = run_dir / "vram_monitor.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["memory_used_mib"] = int(row["memory_used_mib"])
    return rows


def _summary_row(
    result: dict,
    *,
    overcomplete_backbone_learning_rate: float,
    vram_samples: list[dict],
) -> dict:
    return {
        "seed": result["seed"],
        "split_seed": result["split_seed"],
        "overcomplete_backbone_learning_rate": (
            overcomplete_backbone_learning_rate
        ),
        "exact_frozen_reference": overcomplete_backbone_learning_rate == 0.0,
        "post_target_backbone_learning_rate": result[
            "post_target_backbone_learning_rate"
        ],
        "backbone_restore_ramp_steps": result["backbone_ramp_steps"],
        "head_learning_rate": result["head_learning_rate"],
        "best_validation_purity": result["best_validation_purity"],
        "best_validation_nll_per_exposure": result[
            "best_validation_nll_per_exposure"
        ],
        "test_purity_descriptive_only": result["test_purity"],
        "test_ari_descriptive_only": result["test_ari"],
        "test_nmi_descriptive_only": result["test_nmi"],
        "best_step": result["best_step"],
        "best_alpha": result["best_alpha"],
        "alpha_min_during_training": result["alpha_min_during_training"],
        "alpha_max_during_training": result["alpha_max_during_training"],
        "alpha_gradient_norm_median": result["alpha_gradient_norm_median"],
        "alpha_gradient_norm_maximum": result["alpha_gradient_norm_maximum"],
        "gradient_clipped_step_fraction": result[
            "gradient_clipped_step_fraction"
        ],
        "runtime_seconds": result["runtime_seconds"],
        "peak_sampled_dedicated_vram_mib": max(
            (sample["memory_used_mib"] for sample in vram_samples),
            default=None,
        ),
        "external_k1_checkpoint_reused": True,
    }


def _write_selection(outdir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["best_validation_purity"]),
            float(row["best_validation_nll_per_exposure"]),
            float(row["overcomplete_backbone_learning_rate"]),
        ),
    )
    ranking = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["validation_selection_rank"] = rank
        item["selected_by_validation"] = rank == 1
        ranking.append(item)
    _write_csv(outdir / "sweep_summary.csv", ranking)
    _write_json(outdir / "selection.json", {
        "selection_policy": (
            "maximize best validation purity; tie-break by lower validation "
            "NLL, then lower overcomplete-backbone LR"
        ),
        "test_used_for_selection": False,
        "selected_overcomplete_backbone_learning_rate": ranking[0][
            "overcomplete_backbone_learning_rate"
        ],
        "selected_best_validation_purity": ranking[0][
            "best_validation_purity"
        ],
        "selected_test_purity_descriptive_only": ranking[0][
            "test_purity_descriptive_only"
        ],
        "ranking": ranking,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/"
            "dan_sin_k5_thp_big_protocol_overcomplete_backbone_lr_seed0_20260807"
        ),
    )
    parser.add_argument(
        "--overcomplete-backbone-learning-rates",
        type=float,
        nargs="+",
        default=(0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4),
    )
    parser.add_argument(
        "--pretrain-source-dir",
        type=Path,
        default=Path(
            "artifacts/dan_all12_best_elimination_corrected_seed0_20260805/"
            "sin_K5_C5/thp/wishart"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--pruning-batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=16
    )
    parser.add_argument("--vram-limit-mib", type=int, default=7000)
    parser.add_argument("--vram-poll-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value < 0.0 for value in args.overcomplete_backbone_learning_rates):
        parser.error("overcomplete backbone learning rates must be nonnegative")
    if len(set(args.overcomplete_backbone_learning_rates)) != len(
        args.overcomplete_backbone_learning_rates
    ):
        parser.error("overcomplete backbone learning rates must be unique")
    if any(value <= 0 for value in (
        args.batch_size,
        args.evaluation_batch_size,
        args.pruning_batch_size,
        args.gradient_accumulation_steps,
        args.vram_limit_mib,
    )):
        parser.error("batch, accumulation, and VRAM values must be positive")
    if (
        args.batch_size * args.gradient_accumulation_steps
        != LOGICAL_EFFECTIVE_BATCH_SIZE
    ):
        parser.error(
            "batch-size * gradient-accumulation-steps must preserve the "
            f"big-protocol effective batch {LOGICAL_EFFECTIVE_BATCH_SIZE}"
        )
    if args.vram_poll_seconds <= 0.0:
        parser.error("VRAM polling interval must be positive")

    root = Path(__file__).resolve().parents[1]
    outdir = args.outdir.resolve()
    source_dir = (
        args.pretrain_source_dir
        if args.pretrain_source_dir.is_absolute()
        else root / args.pretrain_source_dir
    ).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rates = list(args.overcomplete_backbone_learning_rates)
    if args.dry_run:
        commands = [
            _command(
                root,
                run_dir=outdir / _lr_tag(rate),
                overcomplete_backbone_learning_rate=rate,
                batch_size=args.batch_size,
                evaluation_batch_size=args.evaluation_batch_size,
                pruning_batch_size=args.pruning_batch_size,
                gradient_accumulation_steps=(
                    args.gradient_accumulation_steps
                ),
            )
            for rate in rates
        ]
        _write_json(outdir / "dry_run_commands.json", commands)
        return

    initial_gpu = _gpu_sample()
    if initial_gpu is None:
        raise RuntimeError("could not inspect NVIDIA GPU")
    if initial_gpu["gpu_index"] != 0 or "RTX 4060" not in initial_gpu["gpu_name"]:
        raise RuntimeError(f"unexpected CUDA device: {initial_gpu}")
    if initial_gpu["memory_total_mib"] < args.vram_limit_mib:
        raise RuntimeError("VRAM safety limit exceeds physical dedicated VRAM")
    _write_json(outdir / "protocol.json", {
        "basis": "corrected big all-12 sin_K5_C5 THP/Wishart protocol",
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "initial_components": 10,
        "target_components": 5,
        "warmup_steps": 300,
        "prune_interval": 100,
        "post_pruning_steps": 600,
        "pretrain_steps": 10,
        "original_physical_batch_size": 16,
        "memory_safe_physical_batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "pruning_batch_size": args.pruning_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "physical_batch_adaptation_preserves_effective_batch": True,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "post_target_backbone_learning_rate": (
            POST_TARGET_BACKBONE_LEARNING_RATE
        ),
        "backbone_restore_ramp_steps": BACKBONE_RESTORE_RAMP_STEPS,
        "overcomplete_backbone_learning_rates": rates,
        "zero_rate_is_exact_frozen_reference": True,
        "external_k1_checkpoint_reused": True,
        "pretrain_source_dir": str(source_dir),
        "test_used_for_selection": False,
        "cuda_visible_devices": "0",
        "vram_limit_mib": args.vram_limit_mib,
        "vram_poll_seconds": args.vram_poll_seconds,
        "physical_gpu": initial_gpu,
    })

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)

    for index, rate in enumerate(rates, start=1):
        run_dir = outdir / _lr_tag(rate)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                rows.append(_summary_row(
                    result,
                    overcomplete_backbone_learning_rate=rate,
                    vram_samples=_load_vram_samples(run_dir),
                ))
                _write_csv(outdir / "summary.partial.csv", rows)
                _write_selection(outdir, rows)
                continue

        _prepare_pretrain(source_dir, run_dir)
        command = _command(
            root,
            run_dir=run_dir,
            overcomplete_backbone_learning_rate=rate,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            pruning_batch_size=args.pruning_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        _write_json(run_dir / "command.json", command)
        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(rates),
            "queue_index": index,
            "current": {
                "overcomplete_backbone_learning_rate": rate,
                "exact_frozen_reference": rate == 0.0,
                "post_target_backbone_learning_rate": (
                    POST_TARGET_BACKBONE_LEARNING_RATE
                ),
                "head_learning_rate": HEAD_LEARNING_RATE,
                "backbone_restore_ramp_steps": BACKBONE_RESTORE_RAMP_STEPS,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
            "test_used_for_selection": False,
            "vram_limit_mib": args.vram_limit_mib,
        })
        started = time.time()
        returncode, samples, exceeded = _run_with_vram_guard(
            command,
            root=root,
            run_dir=run_dir,
            environment=environment,
            vram_limit_mib=args.vram_limit_mib,
            poll_seconds=args.vram_poll_seconds,
        )
        if exceeded or returncode != 0 or not result_path.is_file():
            failure = {
                "overcomplete_backbone_learning_rate": rate,
                "reason": (
                    "dedicated_vram_safety_limit_exceeded"
                    if exceeded
                    else "benchmark_process_failed"
                ),
                "returncode": returncode,
                "runtime_seconds": time.time() - started,
                "peak_sampled_dedicated_vram_mib": max(
                    (sample["memory_used_mib"] for sample in samples),
                    default=None,
                ),
                "run_dir": str(run_dir),
            }
            failures.append(failure)
            _write_json(outdir / "failures.json", failures)
            _write_json(outdir / "suite_progress.json", {
                "status": (
                    "stopped_vram_limit" if exceeded else "failed"
                ),
                "completed": len(rows),
                "failed": len(failures),
                "total": len(rates),
                "current": failure,
                "last_completed": rows[-1] if rows else None,
                "test_used_for_selection": False,
            })
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(_summary_row(
            result,
            overcomplete_backbone_learning_rate=rate,
            vram_samples=samples,
        ))
        _write_csv(outdir / "summary.partial.csv", rows)
        _write_selection(outdir, rows)

    _write_csv(outdir / "summary.csv", rows)
    _write_selection(outdir, rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass",
        "completed": len(rows),
        "failed": 0,
        "total": len(rates),
        "current": None,
        "last_completed": rows[-1] if rows else None,
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
