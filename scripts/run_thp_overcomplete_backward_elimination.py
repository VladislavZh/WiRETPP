#!/usr/bin/env python3
"""Overcomplete THP mixture with full-train backward elimination."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lal_wishart.experiment import clustering_row
from lal_wishart.metrics import cluster_purity
from lal_wishart.models.reference_output_mixtures import (
    ReferenceCOTICOutputMixture,
    ReferenceNHPOutputMixture,
    ReferenceTHPOutputMixture,
)
from lal_wishart.models.latent_wishart_attention_nhp import (
    base_nhp_component_scores_from_trace,
)
from lal_wishart.reproduction.dan_synthetic import (
    load_dataset,
    shuffled_split,
)
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    component_removal_marginal_nll_deltas,
    evaluate_direct_nhp_mixture,
    unbalanced_ot_dual_free_energy,
)
from run_corrected_shared_wishart_architectures import _set_seed


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sample_batch(sequences, rng, batch_size):
    indices = rng.choice(len(sequences), size=batch_size, replace=False)
    return tuple(sequences[int(index)] for index in indices)


def _shared_encoder(model):
    if isinstance(model, ReferenceCOTICOutputMixture):
        return model.encoder
    if isinstance(model, ReferenceTHPOutputMixture):
        return model.backbone
    raise TypeError("two-phase protocol requires THP or COTIC")


def _is_shared_encoder_parameter(model, name: str) -> bool:
    if isinstance(model, ReferenceCOTICOutputMixture):
        return name.startswith("encoder.")
    if isinstance(model, ReferenceTHPOutputMixture):
        return name.startswith("backbone.")
    return False


def _make_two_phase_optimizer(
    model,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
):
    backbone = []
    head = []
    mixture = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "mixture_logits":
            mixture.append(parameter)
        elif _is_shared_encoder_parameter(model, name):
            backbone.append(parameter)
        else:
            head.append(parameter)
    groups = []
    if head:
        groups.append({
            "params": head,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "group_name": "head",
        })
    if mixture:
        groups.append({
            "params": mixture,
            "lr": head_lr,
            "weight_decay": 0.0,
            "group_name": "mixture",
        })
    if backbone:
        groups.append({
            "params": backbone,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "group_name": "backbone",
        })
    return torch.optim.Adam(groups)


def _set_optimizer_group_lr(optimizer, group_name: str, value: float) -> None:
    for group in optimizer.param_groups:
        if group.get("group_name") == group_name:
            group["lr"] = value


def _functional_dpp_penalty(
    trace,
    *,
    bandwidth: float,
    jitter: float,
):
    if trace.event_base_intensities.shape[0] == 0:
        return trace.event_base_intensities.new_zeros(())
    marked = trace.event_base_intensities.gather(
        2,
        trace.event_marks[:, None, None].expand(
            -1, trace.n_components, 1
        ),
    ).squeeze(2)
    signatures = torch.log(marked.clamp_min(1e-12)).transpose(0, 1)
    signatures = signatures - signatures.mean(dim=1, keepdim=True)
    signatures = signatures / signatures.square().mean(
        dim=1, keepdim=True
    ).add(1e-8).sqrt()
    squared_distances = (
        signatures[:, None, :] - signatures[None, :, :]
    ).square().mean(dim=-1)
    kernel = torch.exp(
        -squared_distances / (2.0 * bandwidth * bandwidth)
    )
    kernel = kernel + jitter * torch.eye(
        trace.n_components,
        dtype=kernel.dtype,
        device=kernel.device,
    )
    cholesky = torch.linalg.cholesky(kernel)
    return -2.0 * torch.log(cholesky.diagonal()).sum() / trace.n_components


def _dpp_weight_at_step(
    step: int,
    *,
    final_prune_step: int,
    initial_strength: float,
    decay_steps: int,
) -> float:
    if initial_strength == 0.0:
        return 0.0
    if step <= final_prune_step:
        return initial_strength
    if decay_steps == 0 or step >= final_prune_step + decay_steps:
        return 0.0
    fraction = (step - final_prune_step) / decay_steps
    return initial_strength * 0.5 * (1.0 + math.cos(math.pi * fraction))


def _pretrain_k1(
    model,
    train_sequences,
    validation_sequences,
    *,
    steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    rng,
    outdir: Path,
):
    checkpoint = outdir / "checkpoint_k1.pt"
    completed_history = outdir / "pretrain_history.csv"
    if checkpoint.is_file() and completed_history.is_file():
        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=model.device,
                weights_only=True,
            )
        )
        return
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimized = tuple(model.parameters())
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_nll = math.inf
    validation_exposure = sum(
        sequence.horizon for sequence in validation_sequences
    )
    progress = tqdm(
        range(1, steps + 1),
        desc=f"{model.__class__.__name__} K=1 pretrain",
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(gradient_accumulation_steps):
            batch = _sample_batch(train_sequences, rng, batch_size)
            loss = model.negative_log_likelihood(batch) / len(batch)
            (loss / gradient_accumulation_steps).backward()
            losses.append(loss.detach())
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized, gradient_clip
            ).detach().cpu()
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_nll = float(
                -model.log_likelihoods(validation_sequences).sum()
                / validation_exposure
            )
        model.train()
        is_best = validation_nll < best_validation_nll
        if is_best:
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "step": step,
            "train_nll_per_path": float(torch.stack(losses).mean().cpu()),
            "validation_nll_per_exposure": validation_nll,
            "gradient_norm": gradient_norm,
            "gradient_clip_coefficient": min(
                1.0, gradient_clip / max(gradient_norm, 1e-30)
            ),
            "is_best_checkpoint": is_best,
        })
        if step % 25 == 0 or step == steps:
            pd.DataFrame(history).to_csv(
                outdir / "pretrain_history.partial.csv", index=False
            )
        progress.set_postfix({
            "val_nll": f"{validation_nll:.4f}",
            "best": f"{best_validation_nll:.4f}",
        }, refresh=False)
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(
        outdir / "pretrain_history.csv", index=False
    )
    torch.save(model.state_dict(), checkpoint)


def independent_head_negative_log_likelihood(component_scores):
    """Mean NLL when every head must explain every sequence on its own."""

    if component_scores.ndim != 2 or component_scores.shape[1] <= 1:
        raise ValueError("independent-head scores must have shape (paths, K>1)")
    return -component_scores.mean()


def _pretrain_independent_heads(
    model,
    train_sequences,
    validation_sequences,
    *,
    steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    rng,
    outdir: Path,
):
    """Fit every K head to the complete marginal process without a mixture."""

    checkpoint = outdir / "checkpoint_independent_heads.pt"
    completed_history = outdir / "independent_pretrain_history.csv"
    if checkpoint.is_file() and completed_history.is_file():
        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=model.device,
                weights_only=True,
            )
        )
        return
    optimized = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if name != "mixture_logits"
    )
    optimizer = torch.optim.Adam(
        optimized,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_nll = math.inf
    validation_exposure = sum(
        sequence.horizon for sequence in validation_sequences
    )
    progress = tqdm(
        range(1, steps + 1),
        desc=f"{model.__class__.__name__} K={model.n_components} independent",
    )
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(gradient_accumulation_steps):
            batch = _sample_batch(train_sequences, rng, batch_size)
            component_scores = model.component_scores(batch)
            loss = independent_head_negative_log_likelihood(component_scores)
            (loss / gradient_accumulation_steps).backward()
            losses.append(loss.detach())
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized, gradient_clip
            ).detach().cpu()
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_scores = _full_component_scores(
                model,
                validation_sequences,
                batch_size=batch_size,
            )
            per_head_validation_nll = (
                -validation_scores.sum(dim=0) / validation_exposure
            )
            validation_nll = float(per_head_validation_nll.mean())
        is_best = validation_nll < best_validation_nll
        if is_best:
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "step": step,
            "train_mean_independent_nll_per_path": float(
                torch.stack(losses).mean().cpu()
            ),
            "validation_mean_independent_nll_per_exposure": validation_nll,
            "validation_best_head_nll_per_exposure": float(
                per_head_validation_nll.min()
            ),
            "validation_worst_head_nll_per_exposure": float(
                per_head_validation_nll.max()
            ),
            "validation_head_nll_standard_deviation": float(
                per_head_validation_nll.std(unbiased=False)
            ),
            "gradient_norm": gradient_norm,
            "gradient_clip_coefficient": min(
                1.0, gradient_clip / max(gradient_norm, 1e-30)
            ),
            "is_best_checkpoint": is_best,
        })
        if step % 25 == 0 or step == steps:
            pd.DataFrame(history).to_csv(
                outdir / "independent_pretrain_history.partial.csv",
                index=False,
            )
        progress.set_postfix({
            "val_mean": f"{validation_nll:.4f}",
            "head_sd": f"{float(per_head_validation_nll.std(unbiased=False)):.4f}",
            "best": f"{best_validation_nll:.4f}",
        }, refresh=False)
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(completed_history, index=False)
    torch.save(model.state_dict(), checkpoint)


def _cotic_parameter_groups(model):
    groups = {
        "classical": [],
        "continuous_affine": [],
        "projection": [],
        "scale": [],
        "mixture": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "mixture_logits":
            group = "mixture"
        elif "kernel_network" in name:
            group = "continuous_affine"
        elif name.startswith("encoder."):
            group = "classical"
        elif name.startswith("intensity_head.layer"):
            group = "projection"
        elif name == "intensity_head.softplus_params":
            group = "scale"
        else:
            raise ValueError(f"unclassified COTIC parameter: {name}")
        groups[group].append(parameter)
    if any(not parameters for parameters in groups.values()):
        raise RuntimeError("every COTIC optimizer group must be non-empty")
    return groups


def _make_neural_optimizer(
    model,
    *,
    learning_rate,
    weight_decay,
    cotic_classical_learning_rate,
    cotic_kernel_learning_rate,
    cotic_projection_learning_rate,
    structured_cotic,
):
    if isinstance(model, ReferenceCOTICOutputMixture) and structured_cotic:
        groups = _cotic_parameter_groups(model)
        optimizer_groups = [
            {
                "params": groups["classical"],
                "lr": cotic_classical_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": groups["continuous_affine"],
                "lr": cotic_kernel_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": groups["projection"],
                "lr": cotic_projection_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": groups["scale"],
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": groups["mixture"],
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
        ]
    else:
        neural = [
            parameter
            for name, parameter in model.named_parameters()
            if name != "mixture_logits"
        ]
        optimizer_groups = [
            {
                "params": neural,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": (model.mixture_logits,),
                "lr": learning_rate,
                "weight_decay": 0.0,
            },
        ]
    return torch.optim.Adam(optimizer_groups)


def _make_optimizers(
    model,
    *,
    learning_rate,
    weight_decay,
    dual_lr,
    cotic_classical_learning_rate,
    cotic_kernel_learning_rate,
    cotic_projection_learning_rate,
    structured_cotic,
):
    optimizer = _make_neural_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        cotic_classical_learning_rate=cotic_classical_learning_rate,
        cotic_kernel_learning_rate=cotic_kernel_learning_rate,
        cotic_projection_learning_rate=cotic_projection_learning_rate,
        structured_cotic=structured_cotic,
    )
    dual = torch.nn.Parameter(
        torch.zeros(
            model.n_components,
            device=model.device,
            dtype=model.dtype,
        )
    )
    dual_optimizer = torch.optim.Adam((dual,), lr=dual_lr, maximize=True)
    return optimizer, dual, dual_optimizer


def _clip_model_gradients(
    model,
    *,
    global_clip,
    cotic_kernel_clip,
    cotic_projection_clip,
    structured_cotic,
):
    if not (
        isinstance(model, ReferenceCOTICOutputMixture)
        and structured_cotic
    ):
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                tuple(model.parameters()), global_clip
            ).detach().cpu()
        )
        coefficient = min(
            1.0, global_clip / max(gradient_norm, 1e-30)
        )
        return gradient_norm, coefficient, {}
    groups = _cotic_parameter_groups(model)
    thresholds = {
        "classical": global_clip,
        "continuous_affine": cotic_kernel_clip,
        "projection": cotic_projection_clip,
        "scale": min(global_clip, 5.0),
        "mixture": global_clip,
    }
    statistics = {}
    squared = 0.0
    coefficients = []
    for group, parameters in groups.items():
        raw_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, thresholds[group]
            ).detach().cpu()
        )
        coefficient = min(
            1.0, thresholds[group] / max(raw_norm, 1e-30)
        )
        statistics[f"gradient_norm_{group}"] = raw_norm
        statistics[f"gradient_clip_coefficient_{group}"] = coefficient
        squared += raw_norm * raw_norm
        coefficients.append(coefficient)
    return math.sqrt(squared), min(coefficients), statistics


@torch.no_grad()
def _full_component_scores(model, sequences, *, batch_size):
    model.eval()
    rows = []
    for start in range(0, len(sequences), batch_size):
        rows.append(
            model.component_scores(
                sequences[start : start + batch_size]
            ).cpu()
        )
    return torch.cat(rows)


def _rho_at_step(
    step: int,
    *,
    final_prune_step: int,
    post_pruning_steps: int,
    initial_rho: float,
    final_rho: float,
) -> float:
    if step <= final_prune_step:
        return initial_rho
    fraction = min(
        (step - final_prune_step) / post_pruning_steps,
        1.0,
    )
    return initial_rho + fraction * (final_rho - initial_rho)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="K5_C5")
    parser.add_argument(
        "--architecture",
        choices=("thp", "nhp", "cotic"),
        default="thp",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/thp_overcomplete_backward_elimination"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--initial-components", type=int, default=10)
    parser.add_argument("--target-components", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--prune-interval", type=int, default=100)
    parser.add_argument("--post-pruning-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--cotic-classical-learning-rate", type=float, default=3e-4
    )
    parser.add_argument(
        "--cotic-kernel-learning-rate", type=float, default=3e-4
    )
    parser.add_argument(
        "--cotic-projection-learning-rate", type=float, default=1e-4
    )
    parser.add_argument("--cotic-calibration-steps", type=int, default=150)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument("--cotic-kernel-gradient-clip", type=float, default=20.0)
    parser.add_argument(
        "--cotic-projection-gradient-clip", type=float, default=10.0
    )
    parser.add_argument("--ot-temperature", type=float, default=1.0)
    parser.add_argument("--initial-rho", type=float, default=5.0)
    parser.add_argument("--final-rho", type=float, default=50.0)
    parser.add_argument("--dual-learning-rate", type=float, default=0.05)
    parser.add_argument("--nu-multiplier", type=float, default=1.5)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument(
        "--independent-head-pretrain-steps", type=int, default=0
    )
    parser.add_argument("--freeze-encoder-until-target", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-ramp-steps", type=int, default=200)
    parser.add_argument("--head-initialization-noise", type=float, default=0.01)
    parser.add_argument("--dpp-strength", type=float, default=0.0)
    parser.add_argument("--dpp-bandwidth", type=float, default=1.0)
    parser.add_argument("--dpp-jitter", type=float, default=1e-4)
    parser.add_argument("--dpp-decay-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not (
        args.initial_components >= args.target_components >= 2
        and args.warmup_steps >= 0
        and (
            args.initial_components == args.target_components
            or args.warmup_steps > 0
        )
        and args.prune_interval > 0
        and args.post_pruning_steps > 0
        and args.batch_size > 0
        and args.evaluation_batch_size > 0
        and args.gradient_accumulation_steps > 0
        and args.learning_rate > 0.0
        and args.cotic_classical_learning_rate > 0.0
        and args.cotic_kernel_learning_rate > 0.0
        and args.cotic_projection_learning_rate > 0.0
        and args.cotic_calibration_steps >= 0
        and (
            args.architecture != "cotic"
            or (
                args.initial_components > args.target_components
                and args.cotic_calibration_steps < args.warmup_steps
            )
            or (
                args.initial_components == args.target_components
                and args.cotic_calibration_steps < args.post_pruning_steps
            )
        )
        and args.weight_decay >= 0.0
        and args.gradient_clip > 0.0
        and args.cotic_kernel_gradient_clip > 0.0
        and args.cotic_projection_gradient_clip > 0.0
        and args.ot_temperature > 0.0
        and args.initial_rho > 0.0
        and args.final_rho >= args.initial_rho
        and args.dual_learning_rate > 0.0
        and args.nu_multiplier >= 1.0
        and args.pretrain_steps >= 0
        and args.independent_head_pretrain_steps >= 0
        and args.backbone_learning_rate > 0.0
        and args.backbone_ramp_steps >= 0
        and args.head_initialization_noise >= 0.0
        and args.dpp_strength >= 0.0
        and args.dpp_bandwidth > 0.0
        and args.dpp_jitter > 0.0
        and args.dpp_decay_steps >= 0
    ):
        raise ValueError("invalid overcomplete training configuration")
    if args.pretrain_steps > 0 and args.independent_head_pretrain_steps > 0:
        raise ValueError("choose K=1 or independent-head pretraining, not both")
    pretraining_active = (
        args.pretrain_steps > 0
        or args.independent_head_pretrain_steps > 0
    )
    two_phase_optimizer_active = pretraining_active and (
        args.freeze_encoder_until_target or args.dpp_strength > 0.0
    )
    if args.architecture == "nhp" and pretraining_active:
        raise ValueError("two-phase benchmark currently targets THP/COTIC")
    if not pretraining_active and (
        args.freeze_encoder_until_target or args.dpp_strength > 0.0
    ):
        raise ValueError("frozen/DPP protocol requires K=1 pretraining")

    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_root, args.dataset)
    if args.target_components != dataset.n_components:
        raise ValueError("target K must equal the dataset's true K")
    split = shuffled_split(dataset, seed=args.split_seed)
    device = torch.device(args.device)
    architecture_index = ("thp", "nhp", "cotic").index(
        args.architecture
    )
    initialization_seed = (
        2026121100
        + args.seed * 1_000_000
        + architecture_index * 1_000
    )
    _set_seed(initialization_seed)
    common = {
        "horizon": max(
            sequence.horizon for sequence in dataset.sequences
        ),
        "quadrature_order": 4,
        "initialization_seed": initialization_seed,
    }
    if args.architecture == "thp":
        model = ReferenceTHPOutputMixture(
            1 if args.pretrain_steps > 0 else args.initial_components,
            dataset.n_marks,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            **common,
        ).to(device)
    elif args.architecture == "nhp":
        initial_total_rate = sum(
            sequence.count for sequence in split.train
        ) / sum(sequence.horizon for sequence in split.train)
        model = ReferenceNHPOutputMixture(
            args.initial_components,
            dataset.n_marks,
            hidden_size=16,
            initial_total_rate=initial_total_rate,
            **common,
        ).to(device)
    else:
        model = ReferenceCOTICOutputMixture(
            1 if args.pretrain_steps > 0 else args.initial_components,
            dataset.n_marks,
            input_channels=32,
            hidden_size=64,
            num_layers=7,
            kernel_size=3,
            dropout=0.1,
            dilation_factor=1.29,
            **common,
        ).to(device)
    if args.pretrain_steps > 0:
        _pretrain_k1(
            model,
            split.train,
            split.validation,
            steps=args.pretrain_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            rng=np.random.default_rng(initialization_seed + 101),
            outdir=args.outdir,
        )
        model.expand_components(
            args.initial_components,
            noise_scale=args.head_initialization_noise,
            initialization_seed=initialization_seed + 17,
        )
    elif args.independent_head_pretrain_steps > 0:
        _pretrain_independent_heads(
            model,
            split.train,
            split.validation,
            steps=args.independent_head_pretrain_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            rng=np.random.default_rng(initialization_seed + 101),
            outdir=args.outdir,
        )
    if args.freeze_encoder_until_target:
        for parameter in _shared_encoder(model).parameters():
            parameter.requires_grad_(False)
    if two_phase_optimizer_active:
        optimizer = _make_two_phase_optimizer(
            model,
            head_lr=args.learning_rate,
            backbone_lr=args.backbone_learning_rate,
            weight_decay=args.weight_decay,
        )
        assignment_dual = torch.nn.Parameter(
            torch.zeros(
                model.n_components,
                device=model.device,
                dtype=model.dtype,
            )
        )
        dual_optimizer = torch.optim.Adam(
            (assignment_dual,),
            lr=args.dual_learning_rate,
            maximize=True,
        )
    else:
        optimizer, assignment_dual, dual_optimizer = _make_optimizers(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dual_lr=args.dual_learning_rate,
            cotic_classical_learning_rate=(
                args.cotic_classical_learning_rate
            ),
            cotic_kernel_learning_rate=args.cotic_kernel_learning_rate,
            cotic_projection_learning_rate=(
                args.cotic_projection_learning_rate
            ),
            structured_cotic=not (
                args.architecture == "cotic"
                and args.cotic_calibration_steps > 0
            ),
        )
    rng = np.random.default_rng(initialization_seed + 301)
    removals = args.initial_components - args.target_components
    prune_steps = tuple(
        args.warmup_steps + index * args.prune_interval
        for index in range(removals)
    )
    # In the fixed-K ablation there is no pruning event.  With warmup_steps=0,
    # the separate K=1 pretraining phase is followed immediately by target-K
    # training and checkpoint selection.
    final_prune_step = (
        prune_steps[-1] if prune_steps else args.warmup_steps
    )
    total_steps = final_prune_step + args.post_pruning_steps
    lineage = list(range(args.initial_components))
    pruning_events = []
    history = []
    best_state = None
    best_dual = None
    best_step = -1
    best_purity = -math.inf
    best_validation_nll = math.inf
    best_rho = math.nan
    progress = tqdm(
        range(1, total_steps + 1),
        desc=f"{args.architecture.upper()} overcomplete",
    )

    for step in progress:
        structured_cotic = not (
            args.architecture == "cotic"
            and (
                two_phase_optimizer_active
                or step <= args.cotic_calibration_steps
            )
        )
        if (
            not two_phase_optimizer_active
            and
            args.architecture == "cotic"
            and args.cotic_calibration_steps > 0
            and step == args.cotic_calibration_steps + 1
        ):
            optimizer = _make_neural_optimizer(
                model,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                cotic_classical_learning_rate=(
                    args.cotic_classical_learning_rate
                ),
                cotic_kernel_learning_rate=(
                    args.cotic_kernel_learning_rate
                ),
                cotic_projection_learning_rate=(
                    args.cotic_projection_learning_rate
                ),
                structured_cotic=True,
            )
        current_backbone_lr = (
            args.learning_rate
            if not args.freeze_encoder_until_target
            else 0.0
        )
        if (
            args.freeze_encoder_until_target
            and step > final_prune_step
        ):
            if args.backbone_ramp_steps == 0:
                ramp = 1.0
            else:
                ramp = min(
                    1.0,
                    (step - final_prune_step) / args.backbone_ramp_steps,
                )
            current_backbone_lr = args.backbone_learning_rate * ramp
            _set_optimizer_group_lr(
                optimizer, "backbone", current_backbone_lr
            )
        rho = _rho_at_step(
            step,
            final_prune_step=final_prune_step,
            post_pruning_steps=args.post_pruning_steps,
            initial_rho=args.initial_rho,
            final_rho=args.final_rho,
        )
        dpp_weight = _dpp_weight_at_step(
            step,
            final_prune_step=final_prune_step,
            initial_strength=args.dpp_strength,
            decay_steps=args.dpp_decay_steps,
        )
        batches = tuple(
            _sample_batch(split.train, rng, args.batch_size)
            for _ in range(args.gradient_accumulation_steps)
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        dual_optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        detached_dpp_penalties = []
        detached_assignments = []
        reference_marginal = None
        for batch in batches:
            if dpp_weight > 0.0:
                trace = model.build_trace(batch)
                component = base_nhp_component_scores_from_trace(trace)
                dpp_penalty = _functional_dpp_penalty(
                    trace,
                    bandwidth=args.dpp_bandwidth,
                    jitter=args.dpp_jitter,
                )
            else:
                component = model.component_scores(batch)
                dpp_penalty = component.new_zeros(())
            scores = component + model.mixture_log_weights()[None, :]
            free_energy, assignments, reference_marginal = (
                unbalanced_ot_dual_free_energy(
                    scores,
                    assignment_dual,
                    temperature=args.ot_temperature,
                    marginal_penalty=rho,
                )
            )
            loss = free_energy + dpp_weight * dpp_penalty
            (loss / args.gradient_accumulation_steps).backward()
            detached_losses.append(loss.detach())
            detached_dpp_penalties.append(dpp_penalty.detach())
            detached_assignments.append(assignments.detach())
        gradient_norm, gradient_clip_coefficient, gradient_statistics = (
            _clip_model_gradients(
                model,
                global_clip=args.gradient_clip,
                cotic_kernel_clip=args.cotic_kernel_gradient_clip,
                cotic_projection_clip=args.cotic_projection_gradient_clip,
                structured_cotic=structured_cotic,
            )
        )
        optimizer.step()
        dual_optimizer.step()
        with torch.no_grad():
            assignment_dual.sub_(assignment_dual.mean())
        train_objective = float(torch.stack(detached_losses).mean().cpu())
        train_dpp_penalty = float(
            torch.stack(detached_dpp_penalties).mean().cpu()
        )
        batch_assignments = torch.cat(detached_assignments)

        pruning_event = None
        if step in prune_steps:
            train_scores = _full_component_scores(
                model,
                split.train,
                batch_size=args.evaluation_batch_size,
            )
            full_nll, removal_deltas = (
                component_removal_marginal_nll_deltas(
                    train_scores,
                    model.mixture_logits.detach().cpu(),
                )
            )
            removed_local = int(torch.argmin(removal_deltas))
            removed_lineage = lineage[removed_local]
            weights_before = torch.softmax(
                model.mixture_logits.detach().cpu(), dim=0
            )
            components_before = model.n_components
            kept_local = model.prune_component(removed_local)
            lineage = [lineage[index] for index in kept_local]
            pruning_event = {
                "step": step,
                "components_before": components_before,
                "components_after": model.n_components,
                "removed_local_component": removed_local,
                "removed_initial_component": removed_lineage,
                "kept_initial_components": lineage.copy(),
                "full_train_marginal_nll_per_path": float(full_nll),
                "removal_nll_deltas_per_path": removal_deltas.tolist(),
                "selected_removal_delta_per_path": float(
                    removal_deltas[removed_local]
                ),
                "mixture_weights_before": weights_before.tolist(),
                "optimizer_reset": True,
                "ot_dual_reset": True,
                "uniform_target_after": 1.0 / model.n_components,
            }
            pruning_events.append(pruning_event)
            _write_json(
                args.outdir / "pruning_events.partial.json",
                pruning_events,
            )
            torch.save(
                model.state_dict(),
                args.outdir
                / f"checkpoint_post_prune_step_{step}_K{model.n_components}.pt",
            )
            if (
                step == final_prune_step
                and args.freeze_encoder_until_target
            ):
                for parameter in _shared_encoder(model).parameters():
                    parameter.requires_grad_(True)
            if two_phase_optimizer_active:
                optimizer = _make_two_phase_optimizer(
                    model,
                    head_lr=args.learning_rate,
                    backbone_lr=args.backbone_learning_rate,
                    weight_decay=args.weight_decay,
                )
                assignment_dual = torch.nn.Parameter(
                    torch.zeros(
                        model.n_components,
                        device=model.device,
                        dtype=model.dtype,
                    )
                )
                dual_optimizer = torch.optim.Adam(
                    (assignment_dual,),
                    lr=args.dual_learning_rate,
                    maximize=True,
                )
            else:
                optimizer, assignment_dual, dual_optimizer = _make_optimizers(
                    model,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    dual_lr=args.dual_learning_rate,
                    cotic_classical_learning_rate=(
                        args.cotic_classical_learning_rate
                    ),
                    cotic_kernel_learning_rate=(
                        args.cotic_kernel_learning_rate
                    ),
                    cotic_projection_learning_rate=(
                        args.cotic_projection_learning_rate
                    ),
                    structured_cotic=structured_cotic,
                )

        model.eval()
        validation = evaluate_direct_nhp_mixture(
            model,
            split.validation,
            cutoff=None,
            batch_size=args.evaluation_batch_size,
            assignment_dual=assignment_dual.detach(),
            assignment_temperature=args.ot_temperature,
        )
        validation_purity = cluster_purity(
            split.validation_labels,
            validation.full_cluster_probabilities.argmax(dim=1).numpy(),
        )
        validation_exposure = sum(
            sequence.horizon for sequence in split.validation
        )
        validation_nll = float(
            -validation.full_marginal_scores.sum() / validation_exposure
        )
        selection_eligible = (
            model.n_components == args.target_components
            and (removals > 0 or step > final_prune_step)
        )
        is_better = selection_eligible and (
            validation_purity > best_purity + 1e-12
            or (
                abs(validation_purity - best_purity) <= 1e-12
                and validation_nll < best_validation_nll
            )
        )
        if is_better:
            best_state = copy.deepcopy(model.state_dict())
            best_dual = assignment_dual.detach().cpu().clone()
            best_step = step
            best_purity = validation_purity
            best_validation_nll = validation_nll
            best_rho = rho
        history.append({
            "step": step,
            "optimizer_steps": step,
            "n_components": model.n_components,
            "rho": rho,
            "dpp_weight": dpp_weight,
            "functional_dpp_penalty": train_dpp_penalty,
            "backbone_learning_rate": current_backbone_lr,
            "train_objective_per_path": train_objective,
            "gradient_norm": gradient_norm,
            "gradient_clip_coefficient": gradient_clip_coefficient,
            "optimizer_regime": (
                "structured_cotic"
                if structured_cotic
                and args.architecture == "cotic"
                else "legacy_global"
            ),
            **gradient_statistics,
            "batch_assignment_stationarity_error": float(
                (
                    batch_assignments.mean(dim=0)
                    - reference_marginal.detach()
                ).abs().max().cpu()
            ) if pruning_event is None else float("nan"),
            "validation_purity": validation_purity,
            "validation_nll_per_exposure": validation_nll,
            "selection_eligible": selection_eligible,
            "is_best_checkpoint": is_better,
            "pruned_component": (
                pruning_event["removed_local_component"]
                if pruning_event is not None
                else -1
            ),
            "dual_max_abs": float(
                assignment_dual.detach().abs().max().cpu()
            ),
        })
        if step % 25 == 0 or pruning_event is not None:
            pd.DataFrame(history).to_csv(
                args.outdir / "history.partial.csv", index=False
            )
            _write_json(args.outdir / "progress.json", {
                "step": step,
                "total_steps": total_steps,
                "n_components": model.n_components,
                "rho": rho,
                "dpp_weight": dpp_weight,
                "functional_dpp_penalty": train_dpp_penalty,
                "backbone_learning_rate": current_backbone_lr,
                "validation_purity": validation_purity,
                "selection_eligible": selection_eligible,
                "best_step": best_step,
                "best_validation_purity": (
                    best_purity if best_state is not None else None
                ),
            })
        progress.set_postfix({
            "K": model.n_components,
            "rho": f"{rho:.1f}",
            "val": f"{validation_purity:.3f}",
            "best": f"{best_purity:.3f}" if best_state is not None else "-",
            "clip": f"{gradient_clip_coefficient:.2f}",
        }, refresh=False)

    if best_state is None or best_dual is None:
        raise RuntimeError("no K-target checkpoint was eligible for selection")
    model.load_state_dict(best_state)
    assignment_dual = best_dual
    model.eval()
    train_evaluation = evaluate_direct_nhp_mixture(
        model,
        split.train,
        cutoff=None,
        batch_size=args.evaluation_batch_size,
        assignment_dual=assignment_dual,
        assignment_temperature=args.ot_temperature,
    )
    test_evaluation = evaluate_direct_nhp_mixture(
        model,
        split.test,
        cutoff=None,
        batch_size=args.evaluation_batch_size,
        assignment_dual=assignment_dual,
        assignment_temperature=args.ot_temperature,
    )
    cluster, predictions = clustering_row(
        split.test_labels,
        test_evaluation.full_cluster_probabilities,
        model=args.architecture,
        degrees_of_freedom=None,
        representation="overcomplete_backward_elimination_no_wishart",
    )
    train_probabilities = train_evaluation.full_cluster_probabilities
    train_soft_marginal = train_probabilities.mean(dim=0)
    train_counts = torch.bincount(
        train_probabilities.argmax(dim=1),
        minlength=args.target_components,
    )
    target = torch.full_like(train_soft_marginal, 1.0 / args.target_components)
    reference = torch.softmax(
        torch.log(target) + assignment_dual / best_rho,
        dim=0,
    )
    test_exposure = sum(sequence.horizon for sequence in split.test)
    result = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "initialization_seed": initialization_seed,
        "architecture": args.architecture,
        "variant": (
            "no_wishart_overcomplete_backward_elimination"
            if removals > 0
            else "no_wishart_fixed_k_warmup"
        ),
        "initial_components": args.initial_components,
        "target_components": args.target_components,
        "final_component_lineage": lineage,
        "pretrain_steps": args.pretrain_steps,
        "independent_head_pretrain_steps": (
            args.independent_head_pretrain_steps
        ),
        "pretrain_mode": (
            "k1_then_expand"
            if args.pretrain_steps > 0
            else (
                "independent_heads_without_mixture"
                if args.independent_head_pretrain_steps > 0
                else "none"
            )
        ),
        "encoder_frozen_until_target": (
            args.freeze_encoder_until_target
        ),
        "backbone_learning_rate": args.backbone_learning_rate,
        "backbone_ramp_steps": args.backbone_ramp_steps,
        "head_initialization_noise": args.head_initialization_noise,
        "dpp_initial_strength": args.dpp_strength,
        "dpp_bandwidth": args.dpp_bandwidth,
        "dpp_jitter": args.dpp_jitter,
        "dpp_decay_steps": args.dpp_decay_steps,
        "warmup_steps": args.warmup_steps,
        "prune_interval": args.prune_interval,
        "post_pruning_steps": args.post_pruning_steps,
        "total_optimizer_steps": total_steps,
        "best_step": best_step,
        "best_validation_purity": best_purity,
        "best_validation_nll_per_exposure": best_validation_nll,
        "test_purity": cluster["purity"],
        "test_ari": cluster["ari"],
        "test_nmi": cluster["nmi"],
        "test_cluster_sizes": cluster["cluster_sizes"],
        "test_nll_per_exposure": float(
            -test_evaluation.full_marginal_scores.sum() / test_exposure
        ),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "cotic_kernel_learning_rate": (
            args.cotic_kernel_learning_rate
            if args.architecture == "cotic"
            else None
        ),
        "cotic_classical_learning_rate": (
            args.cotic_classical_learning_rate
            if args.architecture == "cotic"
            else None
        ),
        "cotic_projection_learning_rate": (
            args.cotic_projection_learning_rate
            if args.architecture == "cotic"
            else None
        ),
        "cotic_calibration_steps": (
            args.cotic_calibration_steps
            if args.architecture == "cotic"
            else None
        ),
        "gradient_clip": args.gradient_clip,
        "cotic_kernel_gradient_clip": (
            args.cotic_kernel_gradient_clip
            if args.architecture == "cotic"
            else None
        ),
        "cotic_projection_gradient_clip": (
            args.cotic_projection_gradient_clip
            if args.architecture == "cotic"
            else None
        ),
        "initial_rho": args.initial_rho,
        "final_rho": args.final_rho,
        "best_rho": best_rho,
        "assignment_dual": assignment_dual.tolist(),
        "train_true_cluster_counts": np.bincount(
            split.train_labels, minlength=args.target_components
        ).tolist(),
        "train_predicted_component_counts": train_counts.tolist(),
        "train_soft_assignment_marginal": train_soft_marginal.tolist(),
        "train_soft_marginal_l1_to_uniform": float(
            (train_soft_marginal - target).abs().sum()
        ),
        "assignment_ot_reference_marginal": reference.tolist(),
        "assignment_ot_full_train_stationarity_error": float(
            (train_soft_marginal - reference).abs().max()
        ),
        "nu_multiplier": args.nu_multiplier,
        "initial_wishart_nu_if_enabled": int(
            args.nu_multiplier * args.initial_components * dataset.n_marks
        ),
        "wishart_enabled": False,
        "runtime_seconds": time.time() - started,
    }
    pd.DataFrame(history).to_csv(args.outdir / "history.csv", index=False)
    pd.DataFrame(pruning_events).to_csv(
        args.outdir / "pruning_events.csv", index=False
    )
    _write_json(args.outdir / "pruning_events.json", pruning_events)
    _write_json(args.outdir / "result.json", result)
    _write_json(args.outdir / "config.json", vars(args) | {
        "data_root": str(args.data_root),
        "outdir": str(args.outdir),
        "prune_steps": prune_steps,
        "total_steps": total_steps,
    })
    torch.save(model.state_dict(), args.outdir / "checkpoint_best.pt")
    np.save(args.outdir / "assignment_dual.npy", assignment_dual.numpy())
    probabilities = test_evaluation.full_cluster_probabilities.numpy()
    pd.DataFrame({
        "source_id": split.test_ids,
        "true_cluster": split.test_labels,
        "predicted_cluster": predictions,
        **{
            f"probability_{component}": probabilities[:, component]
            for component in range(args.target_components)
        },
    }).to_csv(args.outdir / "predictions.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
