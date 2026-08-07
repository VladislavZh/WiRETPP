#!/usr/bin/env python3
"""Run the THP validation top-5 frozen-backbone settings on COTIC/K2_C5.

The five (backbone learning rate, warmup) pairs are transferred unchanged
from the completed sin_K5_C5 THP sweep.  Runs are sequential and resumable;
test metrics are recorded only after validation-based model selection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

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

# Validation ranks from the completed THP/sin_K5_C5 seed-0 sweep.
TOP_CONFIGURATIONS = (
    (1, 3e-4, 50),
    (2, 3e-4, 300),
    (3, 1e-4, 300),
    (4, 1e-5, 300),
    (5, 3e-4, 10),
)


def _run_tag(source_rank: int, learning_rate: float, warmup_steps: int) -> str:
    return (
        f"rank_{source_rank:02d}_warmup_{warmup_steps}_"
        f"{_lr_tag(learning_rate)}"
    )


def _pretrain_source(root: Path) -> Path:
    return (
        root
        / "artifacts"
        / "dan_two_phase_all12_projected_alpha0p1_seed0_20260803"
        / DATASET
        / ARCHITECTURE
        / "no_wishart"
    )


def _prepare_pretrain(root: Path, run_dir: Path) -> Path:
    source = _pretrain_source(root)
    mappings = (
        (source / "checkpoint_k1.pt", run_dir / "checkpoint_cotic_k1.pt"),
        (source / "pretrain_history.csv", run_dir / "pretrain_history.csv"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for source_path, target_path in mappings:
        if not source_path.is_file():
            raise FileNotFoundError(f"missing frozen COTIC pretrain: {source_path}")
        if not target_path.is_file():
            shutil.copy2(source_path, target_path)
    return source


def _load_vram_samples(run_dir: Path) -> list[dict]:
    path = run_dir / "vram_monitor.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        samples = list(csv.DictReader(stream))
    for sample in samples:
        sample["memory_used_mib"] = int(sample["memory_used_mib"])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/dan_k2_cotic_frozen_thp_top5_seed0_20260806"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--pruning-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--vram-limit-mib", type=int, default=7000)
    parser.add_argument("--vram-poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if any(value <= 0 for value in (
        args.batch_size,
        args.evaluation_batch_size,
        args.pruning_batch_size,
        args.gradient_accumulation_steps,
        args.vram_limit_mib,
    )):
        parser.error("batch, accumulation, and VRAM values must be positive")
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

    pretrain_source = _pretrain_source(root)
    for required in ("checkpoint_k1.pt", "pretrain_history.csv"):
        if not (pretrain_source / required).is_file():
            raise FileNotFoundError(
                f"missing frozen COTIC pretrain: {pretrain_source / required}"
            )
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
        "pretrain_source": str(pretrain_source),
        "transferred_thp_validation_top5": [
            {
                "source_validation_rank": rank,
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
            }
            for rank, learning_rate, warmup_steps in TOP_CONFIGURATIONS
        ],
    })

    rows: list[dict] = []
    failures: list[dict] = []
    _write_json(outdir / "failures.json", failures)

    for queue_index, (source_rank, learning_rate, warmup_steps) in enumerate(
        TOP_CONFIGURATIONS, start=1
    ):
        run_dir = outdir / _run_tag(
            source_rank, learning_rate, warmup_steps
        )
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "pass":
                row = _row(
                    result,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    vram_samples=_load_vram_samples(run_dir),
                )
                row["source_validation_rank"] = source_rank
                rows.append(row)
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
            "--warmup-steps": str(warmup_steps),
            "--batch-size": str(args.batch_size),
            "--evaluation-batch-size": str(args.evaluation_batch_size),
            "--gradient-accumulation-steps": str(
                args.gradient_accumulation_steps
            ),
            "--dual-learning-rate": "0.05",
            "--backbone-learning-rate": str(learning_rate),
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
        command.append("--freeze-encoder-until-target")

        _write_json(outdir / "suite_progress.json", {
            "status": "running",
            "completed": len(rows),
            "failed": len(failures),
            "total": len(TOP_CONFIGURATIONS),
            "queue_index": queue_index,
            "current": {
                "source_validation_rank": source_rank,
                "seed": SEED,
                "split_seed": 42,
                "dataset": DATASET,
                "architecture": ARCHITECTURE,
                "variant": "wishart_frozen_encoder_until_target",
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
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
                "source_validation_rank": source_rank,
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
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
                "total": len(TOP_CONFIGURATIONS),
                "current": failure,
                "last_completed": rows[-1] if rows else None,
                "test_used_for_selection": False,
            })
            return
        if returncode != 0 or not result_path.is_file():
            failures.append({
                "source_validation_rank": source_rank,
                "backbone_learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "returncode": returncode,
                "runtime_seconds": time.time() - started,
                "run_dir": str(run_dir),
            })
            _write_json(outdir / "failures.json", failures)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row = _row(
            result,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            vram_samples=vram_samples,
        )
        row["source_validation_rank"] = source_rank
        rows.append(row)
        _write_csv(outdir / "summary.partial.csv", rows)
        _write_selection(outdir, rows)

    _write_csv(outdir / "summary.csv", rows)
    _write_selection(outdir, rows)
    _write_json(outdir / "suite_progress.json", {
        "status": "pass" if not failures else "completed_with_failures",
        "completed": len(rows),
        "failed": len(failures),
        "total": len(TOP_CONFIGURATIONS),
        "current": None,
        "last_completed": rows[-1] if rows else None,
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
