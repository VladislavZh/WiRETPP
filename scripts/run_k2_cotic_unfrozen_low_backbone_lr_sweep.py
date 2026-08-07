#!/usr/bin/env python3
"""Resumable COTIC/K2 sweep with an unfrozen low-LR shared backbone.

Component heads remain at the neural learning rate while the shared COTIC
encoder uses a separate lower rate from the first elimination step.  Every run
has a fixed post-target recovery phase and is selected on validation only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from run_k2_cotic_frozen_thp_top5_sweep import _prepare_pretrain
from run_sin_k5_thp_frozen_backbone_lr_warmup_sweep import (
    _gpu_sample,
    _lr_tag,
    _replace_argument,
    _row,
    _run_with_vram_guard,
    _write_csv,
    _write_selection,
)
from run_two_phase_all12_benchmark import _command, _write_json


DATASET = "K2_C5"
ARCHITECTURE = "cotic"
SEED = 0
HEAD_LEARNING_RATE = 1e-3
DEFAULT_BACKBONE_LEARNING_RATES = (3e-4, 1e-4, 3e-5, 1e-5, 1e-3)


def _run_tag(learning_rate: float) -> str:
    return _lr_tag(learning_rate)


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
    learning_rate: float,
    warmup_steps: int,
    recovery_steps: int,
    vram_samples: list[dict],
) -> dict:
    row = _row(
        result,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        vram_samples=vram_samples,
    )
    row["head_learning_rate"] = HEAD_LEARNING_RATE
    row["post_target_recovery_steps"] = recovery_steps
    row["separate_backbone_learning_rate"] = result[
        "separate_backbone_learning_rate"
    ]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/dan_k2_cotic_unfrozen_low_backbone_lr_seed0_20260806"
        ),
    )
    parser.add_argument(
        "--backbone-learning-rates",
        type=float,
        nargs="+",
        default=DEFAULT_BACKBONE_LEARNING_RATES,
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
    if any(value <= 0.0 for value in args.backbone_learning_rates):
        parser.error("all backbone learning rates must be positive")
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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "encoder_frozen_until_target": False,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "backbone_learning_rates": list(args.backbone_learning_rates),
        "warmup_steps": args.warmup_steps,
        "post_target_recovery_steps": args.post_target_recovery_steps,
        "test_used_for_selection": False,
    })

    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)

    for queue_index, learning_rate in enumerate(
        args.backbone_learning_rates, start=1
    ):
        run_dir = outdir / _run_tag(learning_rate)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                rows.append(_result_row(
                    result,
                    learning_rate=learning_rate,
                    warmup_steps=args.warmup_steps,
                    recovery_steps=args.post_target_recovery_steps,
                    vram_samples=_load_vram_samples(run_dir),
                ))
                _write_csv(outdir / "summary.partial.csv", rows)
                _write_selection(outdir, rows)
                continue

        pretrain_source = _prepare_pretrain(root, run_dir)
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
            "--backbone-learning-rate": str(learning_rate),
            "--dual-learning-rate": "0.05",
            "--omega-learning-rate": "0.010096",
            "--alpha-learning-rate": "0.00059",
            "--pretrain-steps": "300",
            "--final-rho": "50",
            "--dpp-strength": "0.1",
            "--initial-beta": "0.9",
        }
        for flag, value in replacements.items():
            _replace_argument(command, flag, value)
        command.extend(("--pruning-batch-size", str(args.pruning_batch_size)))
        command.append("--separate-backbone-learning-rate")

        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(args.backbone_learning_rates),
            "queue_index": queue_index,
            "current": {
                "seed": SEED,
                "split_seed": 42,
                "dataset": DATASET,
                "architecture": ARCHITECTURE,
                "variant": "wishart_unfrozen_separate_backbone_lr",
                "head_learning_rate": HEAD_LEARNING_RATE,
                "backbone_learning_rate": learning_rate,
                "warmup_steps": args.warmup_steps,
                "post_target_recovery_steps": args.post_target_recovery_steps,
                "run_dir": str(run_dir),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
            "last_completed": rows[-1] if rows else None,
            "test_used_for_selection": False,
            "vram_limit_mib": args.vram_limit_mib,
            "pretrain_source": str(pretrain_source),
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
                "backbone_learning_rate": learning_rate,
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
                "total": len(args.backbone_learning_rates),
                "current": failure,
                "last_completed": rows[-1] if rows else None,
                "test_used_for_selection": False,
            })
            return
        if returncode != 0 or not result_path.is_file():
            failures.append({
                "backbone_learning_rate": learning_rate,
                "returncode": returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(_result_row(
            result,
            learning_rate=learning_rate,
            warmup_steps=args.warmup_steps,
            recovery_steps=args.post_target_recovery_steps,
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
        "total": len(args.backbone_learning_rates),
        "current": None,
        "last_completed": rows[-1] if rows else None,
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
