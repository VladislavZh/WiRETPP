#!/usr/bin/env python3
"""Resumable THP/sin_K5_C5 sweep over K=1 pretrain length and alpha LR."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from run_sin_k5_thp_frozen_backbone_lr_warmup_sweep import (
    _gpu_sample,
    _replace_argument,
    _row,
    _run_with_vram_guard,
    _write_csv,
)
from run_two_phase_all12_benchmark import _command, _write_json


DATASET = "sin_K5_C5"
ARCHITECTURE = "thp"
SEED = 0
HEAD_LEARNING_RATE = 1e-3
FIXED_BACKBONE_LEARNING_RATE = 3e-4
DEFAULT_ALPHA_LEARNING_RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
DEFAULT_PRETRAIN_STEPS = (300, 100, 50, 25, 10)


def _alpha_tag(value: float) -> str:
    return f"lr_alpha_{value:.0e}".replace("e-", "em").replace("e+", "ep")


def _run_tag(alpha_learning_rate: float, pretrain_steps: int) -> str:
    return f"pretrain_{pretrain_steps}/{_alpha_tag(alpha_learning_rate)}"


def _load_vram_samples(run_dir: Path) -> list[dict]:
    path = run_dir / "vram_monitor.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        samples = list(csv.DictReader(stream))
    for sample in samples:
        sample["memory_used_mib"] = int(sample["memory_used_mib"])
    return samples


def _result_row(
    result: dict,
    *,
    alpha_learning_rate: float,
    pretrain_steps: int,
    warmup_steps: int,
    vram_samples: list[dict],
) -> dict:
    if int(result["pretrain_steps"]) != pretrain_steps:
        raise RuntimeError("result pretrain_steps does not match request")
    row = _row(
        result,
        learning_rate=FIXED_BACKBONE_LEARNING_RATE,
        warmup_steps=warmup_steps,
        vram_samples=vram_samples,
    )
    row.update({
        "pretrain_steps": pretrain_steps,
        "alpha_learning_rate": alpha_learning_rate,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "post_target_recovery_steps": result["post_pruning_steps"],
        "alpha_min_during_training": result["alpha_min_during_training"],
        "alpha_max_during_training": result["alpha_max_during_training"],
        "alpha_gradient_norm_median": result["alpha_gradient_norm_median"],
        "alpha_gradient_norm_maximum": result["alpha_gradient_norm_maximum"],
        "k1_checkpoint_reused_from_external_run": False,
    })
    return row


def _write_selection(outdir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ranked = sorted(rows, key=lambda row: (
        -float(row["best_validation_purity"]),
        float(row["best_validation_nll_per_exposure"]),
        int(row["pretrain_steps"]),
        float(row["alpha_learning_rate"]),
    ))
    ranking = []
    for rank, row in enumerate(ranked, start=1):
        ranked_row = dict(row)
        ranked_row["validation_selection_rank"] = rank
        ranked_row["selected_by_validation"] = rank == 1
        ranking.append(ranked_row)
    _write_csv(outdir / "sweep_summary.csv", ranking)
    _write_json(outdir / "selection.json", {
        "selection_policy": (
            "maximize best_validation_purity; tie-break by lower validation "
            "NLL, fewer K=1 pretrain steps, then lower alpha learning rate"
        ),
        "test_used_for_selection": False,
        "selected_alpha_learning_rate": ranking[0]["alpha_learning_rate"],
        "selected_pretrain_steps": ranking[0]["pretrain_steps"],
        "fixed_backbone_learning_rate": FIXED_BACKBONE_LEARNING_RATE,
        "fixed_overcomplete_warmup_steps": ranking[0]["warmup_steps"],
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
            "artifacts/dan_sin_k5_thp_frozen_lr_alpha_pretrain_"
            "seed0_20260806"
        ),
    )
    parser.add_argument(
        "--alpha-learning-rates",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_LEARNING_RATES,
    )
    parser.add_argument(
        "--pretrain-steps-grid",
        type=int,
        nargs="+",
        default=DEFAULT_PRETRAIN_STEPS,
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=FIXED_BACKBONE_LEARNING_RATE,
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--post-target-recovery-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--pruning-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--vram-limit-mib", type=int, default=7000)
    parser.add_argument("--vram-poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.backbone_learning_rate != FIXED_BACKBONE_LEARNING_RATE:
        parser.error("this sweep fixes --backbone-learning-rate at 3e-4")
    if any(value <= 0.0 for value in args.alpha_learning_rates):
        parser.error("all alpha learning rates must be positive")
    if len(set(args.alpha_learning_rates)) != len(args.alpha_learning_rates):
        parser.error("alpha learning rates must not contain duplicates")
    if any(value <= 0 for value in args.pretrain_steps_grid):
        parser.error("all K=1 pretrain step values must be positive")
    if any(value <= 0 for value in (
        args.warmup_steps,
        args.post_target_recovery_steps,
        args.batch_size,
        args.evaluation_batch_size,
        args.pruning_batch_size,
        args.gradient_accumulation_steps,
        args.vram_limit_mib,
    )):
        parser.error("step, batch, accumulation, and VRAM values must be positive")
    if args.vram_poll_seconds <= 0.0:
        parser.error("VRAM polling interval must be positive")

    root = Path(__file__).resolve().parents[1]
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    initial_gpu = _gpu_sample()
    if initial_gpu is None:
        raise RuntimeError("could not inspect NVIDIA GPU")
    if initial_gpu["gpu_index"] != 0 or "RTX 4060" not in initial_gpu["gpu_name"]:
        raise RuntimeError(f"unexpected CUDA device: {initial_gpu}")
    if initial_gpu["memory_total_mib"] < args.vram_limit_mib:
        raise RuntimeError("VRAM safety limit exceeds physical dedicated VRAM")

    _write_json(outdir / "gpu_policy.json", {
        "cuda_visible_devices": "0",
        "physical_gpu": initial_gpu,
        "vram_limit_mib": args.vram_limit_mib,
        "stop_before_wddm_shared_memory_paging": True,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "pruning_batch_size": args.pruning_batch_size,
        "pruning_trace_padded_event_budget": 512,
        "pruning_rng_group_preserved": True,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "encoder_frozen_until_target": True,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "fixed_backbone_learning_rate": FIXED_BACKBONE_LEARNING_RATE,
        "alpha_learning_rates": list(args.alpha_learning_rates),
        "k1_pretrain_steps_grid": list(args.pretrain_steps_grid),
        "fixed_overcomplete_warmup_steps": args.warmup_steps,
        "post_target_recovery_steps": args.post_target_recovery_steps,
        "external_k1_checkpoint_reused": False,
        "test_used_for_selection": False,
    })

    specifications = [
        (alpha_learning_rate, pretrain_steps)
        for pretrain_steps in args.pretrain_steps_grid
        for alpha_learning_rate in args.alpha_learning_rates
    ]
    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)

    for queue_index, (alpha_learning_rate, pretrain_steps) in enumerate(
        specifications, start=1
    ):
        run_dir = outdir / _run_tag(alpha_learning_rate, pretrain_steps)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                rows.append(_result_row(
                    result,
                    alpha_learning_rate=alpha_learning_rate,
                    pretrain_steps=pretrain_steps,
                    warmup_steps=args.warmup_steps,
                    vram_samples=_load_vram_samples(run_dir),
                ))
                _write_csv(outdir / "summary.partial.csv", rows)
                _write_selection(outdir, rows)
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = _command(
            root,
            seed=SEED,
            dataset=DATASET,
            architecture=ARCHITECTURE,
            variant="wishart",
            outdir=run_dir,
            wishart_interaction_mode="convex",
            wishart_initial_alpha=0.1,
            wishart_alpha_parameterization="projected",
            wishart_alpha_temperature=1.0,
        )
        replacements = {
            "--warmup-steps": str(args.warmup_steps),
            "--post-pruning-steps": str(args.post_target_recovery_steps),
            "--batch-size": str(args.batch_size),
            "--evaluation-batch-size": str(args.evaluation_batch_size),
            "--gradient-accumulation-steps": str(
                args.gradient_accumulation_steps
            ),
            "--neural-learning-rate": str(HEAD_LEARNING_RATE),
            "--dual-learning-rate": "0.05",
            "--backbone-learning-rate": str(FIXED_BACKBONE_LEARNING_RATE),
            "--omega-learning-rate": "0.010096",
            "--alpha-learning-rate": str(alpha_learning_rate),
            "--pretrain-steps": str(pretrain_steps),
            "--final-rho": "50",
            "--dpp-strength": "0.1",
            "--initial-beta": "0.9",
        }
        for flag, value in replacements.items():
            _replace_argument(command, flag, value)
        command.extend(("--pruning-batch-size", str(args.pruning_batch_size)))
        command.append("--freeze-encoder-until-target")

        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(specifications),
            "queue_index": queue_index,
            "current": {
                "seed": SEED,
                "split_seed": 42,
                "dataset": DATASET,
                "architecture": ARCHITECTURE,
                "variant": "wishart_frozen_encoder_until_target",
                "head_learning_rate": HEAD_LEARNING_RATE,
                "backbone_learning_rate": FIXED_BACKBONE_LEARNING_RATE,
                "alpha_learning_rate": alpha_learning_rate,
                "pretrain_steps": pretrain_steps,
                "warmup_steps": args.warmup_steps,
                "post_target_recovery_steps": (
                    args.post_target_recovery_steps
                ),
                "external_k1_checkpoint_reused": False,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
            "test_used_for_selection": False,
            "vram_limit_mib": args.vram_limit_mib,
        })
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        started = time.time()
        returncode, vram_samples, exceeded = _run_with_vram_guard(
            command,
            root=root,
            run_dir=run_dir,
            environment=environment,
            vram_limit_mib=args.vram_limit_mib,
            poll_seconds=args.vram_poll_seconds,
        )
        if exceeded:
            failure = {
                "alpha_learning_rate": alpha_learning_rate,
                "pretrain_steps": pretrain_steps,
                "reason": "dedicated_vram_safety_limit_exceeded",
                "vram_limit_mib": args.vram_limit_mib,
                "peak_sampled_dedicated_vram_mib": max(
                    sample["memory_used_mib"] for sample in vram_samples
                ),
                "run_dir": str(run_dir),
            }
            failures.append(failure)
            _write_json(outdir / "failures.json", failures)
            _write_json(outdir / "suite_progress.json", {
                "status": "stopped_vram_limit",
                "completed": len(rows),
                "failed": len(failures),
                "total": len(specifications),
                "current": failure,
                "last_completed": rows[-1] if rows else None,
                "test_used_for_selection": False,
            })
            return
        if returncode != 0 or not result_path.is_file():
            failures.append({
                "alpha_learning_rate": alpha_learning_rate,
                "pretrain_steps": pretrain_steps,
                "returncode": returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(_result_row(
            result,
            alpha_learning_rate=alpha_learning_rate,
            pretrain_steps=pretrain_steps,
            warmup_steps=args.warmup_steps,
            vram_samples=vram_samples,
        ))
        _write_csv(outdir / "summary.partial.csv", rows)
        _write_selection(outdir, rows)

    _write_csv(outdir / "summary.csv", rows)
    _write_selection(outdir, rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(specifications),
        "current": None,
        "last_completed": rows[-1] if rows else None,
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
