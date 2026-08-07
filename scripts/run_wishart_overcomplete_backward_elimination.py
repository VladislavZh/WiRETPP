#!/usr/bin/env python3
"""Signed-Wishart overcomplete THP/COTIC with physical backward elimination."""

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
    ReferenceTHPOutputMixture,
)
from lal_wishart.models.signed_wishart import SignedWishartTPP
from lal_wishart.models.wishart_math import (
    cluster_log_weights_from_matrices,
    monte_carlo_marginal_scores,
)
from lal_wishart.reproduction.dan_synthetic import load_dataset, shuffled_split
from lal_wishart.train.fit_latent_wishart_attention_nhp import (
    evaluate_latent_wishart_nhp,
    integrated_wishart_component_scores,
    unbalanced_ot_dual_free_energy,
)
from run_corrected_shared_wishart_architectures import _set_seed
from run_thp_overcomplete_backward_elimination import (
    _rho_at_step,
    _sample_batch,
    _write_json,
)


PRUNING_TRACE_EVENT_BUDGET = 512


def _nu(multiplier: float, n_components: int, n_marks: int) -> int:
    value = int(multiplier * n_components * n_marks)
    if value < n_components * n_marks:
        raise ValueError("Wishart nu must be at least K*C")
    return value


def _beta_at_step(
    step: int,
    *,
    total_steps: int,
    initial_beta: float,
    decay_rate: float,
) -> float:
    if total_steps <= 1:
        return 0.0
    fraction = (step - 1) / (total_steps - 1)
    endpoint = math.exp(-decay_rate)
    multiplier = (
        math.exp(-decay_rate * fraction) - endpoint
    ) / (1.0 - endpoint)
    return initial_beta * max(multiplier, 0.0)


def _linear_alpha_at_step(step: int, *, schedule_steps: int) -> float:
    """Linearly move alpha from zero at step 1 to one at schedule_steps."""

    if schedule_steps <= 1:
        raise ValueError("alpha schedule steps must exceed one")
    return min(max((step - 1) / (schedule_steps - 1), 0.0), 1.0)


def _is_shared_encoder_parameter(name: str) -> bool:
    return (
        name.startswith("backbone.encoder.")
        or name.startswith("backbone.backbone.")
    )


def _shared_encoder(backbone):
    if isinstance(backbone, ReferenceCOTICOutputMixture):
        return backbone.encoder
    if isinstance(backbone, ReferenceTHPOutputMixture):
        return backbone.backbone
    raise TypeError("two-phase protocol requires THP or COTIC")


def _make_optimizers(
    model,
    *,
    neural_lr,
    backbone_lr=None,
    omega_lr,
    alpha_lr,
    weight_decay,
    dual_lr,
):
    named_neural = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.")
        and name != "backbone.mixture_logits"
        and parameter.requires_grad
    ]
    backbone = [
        parameter
        for name, parameter in named_neural
        if _is_shared_encoder_parameter(name)
    ]
    head = [
        parameter
        for name, parameter in named_neural
        if not _is_shared_encoder_parameter(name)
    ]
    neural = backbone + head
    distribution = [model.raw_mean_cholesky]
    interaction = list(model.interaction_parameters())
    groups = []
    if head:
        groups.append({
            "params": head,
            "lr": neural_lr,
            "weight_decay": weight_decay,
            "group_name": "head",
        })
    if backbone:
        groups.append({
            "params": backbone,
            "lr": neural_lr if backbone_lr is None else backbone_lr,
            "weight_decay": weight_decay,
            "group_name": "backbone",
        })
    groups.append({
            "params": distribution,
            "lr": omega_lr,
            "weight_decay": 0.0,
            "group_name": "omega",
        })
    if interaction:
        groups.append({
            "params": interaction,
            "lr": alpha_lr,
            "weight_decay": 0.0,
            "group_name": "alpha",
        })
    optimizer = torch.optim.Adam(groups)
    dual = torch.nn.Parameter(
        torch.zeros(
            model.n_components,
            device=model.device,
            dtype=model.dtype,
        )
    )
    dual_optimizer = torch.optim.Adam((dual,), lr=dual_lr, maximize=True)
    optimized = neural + distribution + interaction
    parameter_groups = {
        "neural": neural,
        "backbone": backbone,
        "head": head,
        "omega": distribution,
        "alpha": interaction,
    }
    return optimizer, dual, dual_optimizer, optimized, parameter_groups


def _gradient_norm(parameters) -> float:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().cpu())


def _functional_dpp_penalty(
    trace,
    *,
    bandwidth: float,
    jitter: float,
) -> Tensor:
    """Repel component functions using marked-event log-rate signatures."""

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
    # Divide by K so the strength retains a comparable meaning while pruning.
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


def _set_optimizer_group_lr(optimizer, group_name: str, value: float) -> None:
    for group in optimizer.param_groups:
        if group.get("group_name") == group_name:
            group["lr"] = value


def _backbone_learning_rate_at_step(
    step: int,
    *,
    final_prune_step: int,
    overcomplete_learning_rate: float,
    post_target_learning_rate: float,
    ramp_steps: int,
) -> float:
    """Return the shared-backbone LR without changing component-head LR.

    The overcomplete rate is used through the optimizer step that performs the
    final physical removal.  Subsequent target-K steps linearly restore the
    backbone to its post-target rate.  This also subsumes the historical
    freeze-then-ramp schedule when ``overcomplete_learning_rate`` is zero.
    """

    if step <= final_prune_step:
        return overcomplete_learning_rate
    if ramp_steps == 0:
        return post_target_learning_rate
    fraction = min(1.0, (step - final_prune_step) / ramp_steps)
    return overcomplete_learning_rate + fraction * (
        post_target_learning_rate - overcomplete_learning_rate
    )


def _pretrain_k1(
    backbone,
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
) -> list[dict[str, float | int | bool]]:
    """Fit the ordinary marginal K=1 process before output expansion."""

    checkpoint = outdir / "checkpoint_cotic_k1.pt"
    completed_history = outdir / "pretrain_history.csv"
    if checkpoint.is_file() and completed_history.is_file():
        backbone.load_state_dict(
            torch.load(
                checkpoint,
                map_location=backbone.device,
                weights_only=True,
            )
        )
        return pd.read_csv(completed_history).to_dict("records")
    optimizer = torch.optim.Adam(
        backbone.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimized = [
        parameter for parameter in backbone.parameters()
        if parameter.requires_grad
    ]
    history = []
    best_state = copy.deepcopy(backbone.state_dict())
    best_validation_nll = math.inf
    validation_exposure = sum(
        sequence.horizon for sequence in validation_sequences
    )
    progress = tqdm(
        range(1, steps + 1),
        desc=f"{backbone.__class__.__name__} K=1 pretrain",
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(gradient_accumulation_steps):
            batch = _sample_batch(train_sequences, rng, batch_size)
            loss = backbone.negative_log_likelihood(batch) / len(batch)
            (loss / gradient_accumulation_steps).backward()
            losses.append(loss.detach())
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized, gradient_clip
            ).detach().cpu()
        )
        optimizer.step()
        backbone.eval()
        with torch.no_grad():
            validation_nll = float(
                -backbone.log_likelihoods(validation_sequences).sum()
                / validation_exposure
            )
        backbone.train()
        is_best = validation_nll < best_validation_nll
        if is_best:
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(backbone.state_dict())
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
    backbone.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(
        outdir / "pretrain_history.csv", index=False
    )
    torch.save(backbone.state_dict(), checkpoint)
    return history


@torch.no_grad()
def _full_marginal_nll(model, sequences, *, n_samples, sample_seed, batch_size):
    """Score fixed RNG groups with memory-bounded CUDA trace microbatches.

    ``batch_size`` deliberately continues to define the random-number groups:
    one Wishart draw tensor is sampled for the whole group with the historical
    seed, then sliced per sequence.  Building the neural trace for the complete
    group, however, pads every path to its longest member.  The synthetic DAN
    split mixes paths from 10 to 449 events, so a two-path pruning group can
    nearly double the largest trace and briefly push WDDM beyond dedicated
    VRAM.  Greedy trace microbatches are therefore capped by padded event count;
    this preserves candidate scores and common random numbers while retaining
    the original vectorized path for ordinary short groups.
    """

    was_training = model.training
    beta = model.exploration_beta
    model.eval()
    model.exploration_beta = 0.0
    scores = []
    try:
        for start in range(0, len(sequences), batch_size):
            batch = tuple(sequences[start : start + batch_size])
            matrices = model.sample_matrices(
                len(batch),
                n_samples,
                sample_seed=sample_seed + start,
            )
            gates = cluster_log_weights_from_matrices(
                matrices,
                n_components=model.n_components,
                n_marks=model.n_marks,
            )
            first = 0
            while first < len(batch):
                last = first + 1
                while last < len(batch):
                    candidate = batch[first : last + 1]
                    padded_events = len(candidate) * max(
                        len(sequence.times) for sequence in candidate
                    )
                    if padded_events > PRUNING_TRACE_EVENT_BUDGET:
                        break
                    last += 1
                trace_batch = batch[first:last]
                trace = model.backbone.build_trace(trace_batch)
                component = model.component_scores_from_trace(
                    trace,
                    matrices[first:last],
                )
                score = monte_carlo_marginal_scores(
                    component,
                    gates[first:last],
                )
                scores.append(score.cpu())
                del trace_batch, trace, component, score
                first = last
            del batch, matrices, gates
    finally:
        model.exploration_beta = beta
        model.train(was_training)
    return -torch.cat(scores).mean()


def _physical_removal_deltas(
    model,
    sequences,
    *,
    nu_multiplier,
    n_samples,
    sample_seed,
    batch_size,
):
    full_nll = _full_marginal_nll(
        model,
        sequences,
        n_samples=n_samples,
        sample_seed=sample_seed,
        batch_size=batch_size,
    )
    removed_nlls = []
    for component in range(model.n_components):
        candidate = copy.deepcopy(model)
        candidate.prune_component(
            component,
            degrees_of_freedom=_nu(
                nu_multiplier,
                model.n_components - 1,
                model.n_marks,
            ),
        )
        removed_nlls.append(
            _full_marginal_nll(
                candidate,
                sequences,
                n_samples=n_samples,
                sample_seed=sample_seed,
                batch_size=batch_size,
            )
        )
        del candidate
        if next(model.parameters()).is_cuda:
            # Each candidate is a full GPU model copy. Return its released
            # blocks before constructing the next candidate instead of
            # retaining every candidate peak in the caching allocator.
            torch.cuda.empty_cache()
    removed = torch.stack(removed_nlls)
    return full_nll, removed - full_nll, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="K5_C5")
    parser.add_argument(
        "--architecture", choices=("thp", "cotic"), default="thp"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/wishart_overcomplete_backward_elimination"),
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
    parser.add_argument("--pruning-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--neural-learning-rate", type=float, default=1e-3)
    parser.add_argument("--omega-learning-rate", type=float, default=0.010096)
    parser.add_argument("--alpha-learning-rate", type=float, default=0.00059)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--pruning-samples", type=int, default=8)
    parser.add_argument("--alpha-temperature", type=float, default=1.943)
    parser.add_argument("--initial-alpha", type=float, default=0.5)
    parser.add_argument(
        "--alpha-schedule",
        choices=("learned", "fixed", "linear_0_to_1"),
        default="learned",
    )
    parser.add_argument("--alpha-schedule-steps", type=int, default=1000)
    parser.add_argument(
        "--alpha-parameterization",
        choices=("sigmoid", "projected"),
        default="sigmoid",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=("residual", "convex"),
        default="residual",
    )
    parser.add_argument("--mean-hyperprior-strength", type=float, default=1.566)
    parser.add_argument("--initial-beta", type=float, default=0.9)
    parser.add_argument("--beta-decay-rate", type=float, default=5.0)
    parser.add_argument("--exploration-nu", type=int, default=200)
    parser.add_argument("--initial-rho", type=float, default=5.0)
    parser.add_argument("--final-rho", type=float, default=50.0)
    parser.add_argument("--dual-learning-rate", type=float, default=0.05)
    parser.add_argument("--nu-multiplier", type=float, default=1.5)
    parser.add_argument("--cotic-pretrain-steps", type=int, default=0)
    parser.add_argument(
        "--freeze-cotic-encoder-until-target", action="store_true"
    )
    parser.add_argument(
        "--cotic-backbone-learning-rate", type=float, default=1e-5
    )
    parser.add_argument("--cotic-backbone-ramp-steps", type=int, default=200)
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--freeze-encoder-until-target", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument(
        "--overcomplete-backbone-learning-rate",
        type=float,
        default=None,
        help=(
            "Use this LR only for the unfrozen shared encoder while K exceeds "
            "the target. After the final removal, restore it to "
            "--backbone-learning-rate over --backbone-ramp-steps. Component "
            "heads remain at --neural-learning-rate throughout."
        ),
    )
    parser.add_argument(
        "--separate-backbone-learning-rate",
        action="store_true",
        help=(
            "Honor --backbone-learning-rate for an unfrozen shared encoder "
            "while keeping component heads at --neural-learning-rate."
        ),
    )
    parser.add_argument("--backbone-ramp-steps", type=int, default=None)
    parser.add_argument("--head-initialization-noise", type=float, default=0.01)
    parser.add_argument("--dpp-strength", type=float, default=0.0)
    parser.add_argument("--dpp-bandwidth", type=float, default=1.0)
    parser.add_argument("--dpp-jitter", type=float, default=1e-4)
    parser.add_argument("--dpp-decay-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pretrain_steps = (
        args.cotic_pretrain_steps
        if args.pretrain_steps is None
        else args.pretrain_steps
    )
    freeze_encoder = (
        args.freeze_cotic_encoder_until_target
        or args.freeze_encoder_until_target
    )
    backbone_learning_rate = (
        args.cotic_backbone_learning_rate
        if args.backbone_learning_rate is None
        else args.backbone_learning_rate
    )
    backbone_ramp_steps = (
        args.cotic_backbone_ramp_steps
        if args.backbone_ramp_steps is None
        else args.backbone_ramp_steps
    )
    if (
        freeze_encoder
        and args.overcomplete_backbone_learning_rate is not None
    ):
        raise ValueError(
            "--freeze-encoder-until-target and "
            "--overcomplete-backbone-learning-rate are mutually exclusive"
        )
    scheduled_backbone_restore = (
        freeze_encoder
        or args.overcomplete_backbone_learning_rate is not None
    )
    overcomplete_backbone_learning_rate = (
        0.0
        if freeze_encoder
        else args.overcomplete_backbone_learning_rate
        if args.overcomplete_backbone_learning_rate is not None
        else backbone_learning_rate
        if args.separate_backbone_learning_rate
        else args.neural_learning_rate
    )
    use_separate_backbone_lr = (
        scheduled_backbone_restore
        or args.separate_backbone_learning_rate
    )
    pruning_batch_size = (
        args.evaluation_batch_size
        if args.pruning_batch_size is None
        else args.pruning_batch_size
    )
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
        and pruning_batch_size > 0
        and args.gradient_accumulation_steps > 0
        and args.neural_learning_rate > 0.0
        and args.omega_learning_rate > 0.0
        and args.alpha_learning_rate > 0.0
        and args.weight_decay >= 0.0
        and args.gradient_clip > 0.0
        and args.train_samples > 0
        and args.validation_samples > 0
        and args.test_samples > 0
        and args.pruning_samples > 0
        and args.alpha_temperature > 0.0
        and args.mean_hyperprior_strength >= 0.0
        and 0.0 <= args.initial_beta <= 1.0
        and args.beta_decay_rate > 0.0
        and args.exploration_nu
        >= args.initial_components * 5
        and args.initial_rho > 0.0
        and args.final_rho >= args.initial_rho
        and args.dual_learning_rate > 0.0
        and args.nu_multiplier >= 1.0
        and pretrain_steps >= 0
        and backbone_learning_rate > 0.0
        and overcomplete_backbone_learning_rate >= 0.0
        and backbone_ramp_steps >= 0
        and args.head_initialization_noise >= 0.0
        and args.dpp_strength >= 0.0
        and args.dpp_bandwidth > 0.0
        and args.dpp_jitter > 0.0
        and args.dpp_decay_steps >= 0
    ):
        raise ValueError("invalid Wishart overcomplete configuration")
    if pretrain_steps == 0 and (
        freeze_encoder
        or args.dpp_strength > 0.0
    ):
        raise ValueError("frozen/DPP protocol requires K=1 pretraining")

    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_root, args.dataset)
    if args.target_components != dataset.n_components:
        raise ValueError("target K must equal the dataset's true K")
    if args.exploration_nu < args.initial_components * dataset.n_marks:
        raise ValueError("exploration nu must be at least initial K*C")
    split = shuffled_split(dataset, seed=args.split_seed)
    device = torch.device(args.device)
    architecture_index = ("thp", "cotic").index(args.architecture)
    initialization_seed = (
        2026121100
        + args.seed * 1_000_000
        + architecture_index * 2_000
    )
    _set_seed(initialization_seed)
    rng = np.random.default_rng(initialization_seed + 301)
    common = {
        "horizon": max(sequence.horizon for sequence in dataset.sequences),
        "quadrature_order": 4,
        "initialization_seed": initialization_seed,
    }
    if args.architecture == "thp":
        backbone = ReferenceTHPOutputMixture(
            1 if pretrain_steps > 0 else args.initial_components,
            dataset.n_marks,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            **common,
        ).to(device)
    else:
        backbone = ReferenceCOTICOutputMixture(
            1 if pretrain_steps > 0 else args.initial_components,
            dataset.n_marks,
            input_channels=32,
            hidden_size=64,
            num_layers=7,
            kernel_size=3,
            dropout=0.1,
            dilation_factor=1.29,
            **common,
        ).to(device)
    if pretrain_steps > 0:
        _pretrain_k1(
            backbone,
            split.train,
            split.validation,
            steps=pretrain_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            learning_rate=args.neural_learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            rng=np.random.default_rng(initialization_seed + 101),
            outdir=args.outdir,
        )
        backbone.expand_components(
            args.initial_components,
            noise_scale=args.head_initialization_noise,
            initialization_seed=initialization_seed + 17,
        )
    if freeze_encoder:
        for parameter in _shared_encoder(backbone).parameters():
            parameter.requires_grad_(False)
    initial_nu = _nu(
        args.nu_multiplier, args.initial_components, dataset.n_marks
    )
    model = SignedWishartTPP(
        backbone,
        degrees_of_freedom=initial_nu,
        alpha_max=1.0,
        initial_alpha=args.initial_alpha,
        interaction_temperature=args.alpha_temperature,
        alpha_parameterization=args.alpha_parameterization,
        fixed_alpha=(
            0.0
            if args.alpha_schedule == "linear_0_to_1"
            else args.initial_alpha
            if args.alpha_schedule == "fixed"
            else None
        ),
        interaction_mode=args.interaction_mode,
        exploration_beta=args.initial_beta,
        exploration_degrees_of_freedom=args.exploration_nu,
        exploration_mode="convex",
    ).to(device)
    (
        optimizer,
        assignment_dual,
        dual_optimizer,
        optimized,
        parameter_groups,
    ) = _make_optimizers(
        model,
        neural_lr=args.neural_learning_rate,
        backbone_lr=(
            overcomplete_backbone_learning_rate
            if use_separate_backbone_lr
            else None
        ),
        omega_lr=args.omega_learning_rate,
        alpha_lr=args.alpha_learning_rate,
        weight_decay=args.weight_decay,
        dual_lr=args.dual_learning_rate,
    )
    sample_seed = initialization_seed + 701
    removals = args.initial_components - args.target_components
    prune_steps = tuple(
        args.warmup_steps + index * args.prune_interval
        for index in range(removals)
    )
    # With initial_components == target_components, the separate K=1
    # pretraining phase is followed immediately by target-K training.  There
    # is no artificial pre-pruning warm-up or pruning event.
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
    best_alpha = math.nan
    progress = tqdm(
        range(1, total_steps + 1),
        desc=f"{args.architecture.upper()} Wishart overcomplete",
    )

    for step in progress:
        if args.alpha_schedule == "linear_0_to_1":
            with torch.no_grad():
                model.fixed_interaction_strength.fill_(
                    _linear_alpha_at_step(
                        step,
                        schedule_steps=args.alpha_schedule_steps,
                    )
                )
        current_backbone_lr = overcomplete_backbone_learning_rate
        if scheduled_backbone_restore:
            current_backbone_lr = _backbone_learning_rate_at_step(
                step,
                final_prune_step=final_prune_step,
                overcomplete_learning_rate=(
                    overcomplete_backbone_learning_rate
                ),
                post_target_learning_rate=backbone_learning_rate,
                ramp_steps=backbone_ramp_steps,
            )
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
        beta = _beta_at_step(
            step,
            total_steps=total_steps,
            initial_beta=args.initial_beta,
            decay_rate=args.beta_decay_rate,
        )
        model.exploration_beta = beta
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
        losses = []
        dpp_penalties = []
        assignments_rows = []
        reference_marginal = None
        for microbatch, batch in enumerate(batches):
            current_sample_seed = (
                sample_seed + step * 10_007 + microbatch * 1_009
            )
            if dpp_weight > 0.0:
                trace = model.backbone.build_trace(batch)
                matrices = model.sample_matrices(
                    len(batch),
                    args.train_samples,
                    sample_seed=current_sample_seed,
                )
                component = model.component_scores_from_trace(
                    trace, matrices
                )
                gates = cluster_log_weights_from_matrices(
                    matrices,
                    n_components=model.n_components,
                    n_marks=model.n_marks,
                )
                dpp_penalty = _functional_dpp_penalty(
                    trace,
                    bandwidth=args.dpp_bandwidth,
                    jitter=args.dpp_jitter,
                )
            else:
                component, gates, _ = model.sampled_component_scores(
                    batch,
                    n_samples=args.train_samples,
                    sample_seed=current_sample_seed,
                )
                dpp_penalty = component.new_zeros(())
            scores = integrated_wishart_component_scores(component, gates)
            free_energy, assignments, reference_marginal = (
                unbalanced_ot_dual_free_energy(
                    scores,
                    assignment_dual,
                    temperature=1.0,
                    marginal_penalty=rho,
                )
            )
            hyperprior = model.mean_matrix_hyperprior_penalty(
                strength=args.mean_hyperprior_strength
            ) / len(split.train)
            loss = free_energy + hyperprior + dpp_weight * dpp_penalty
            (loss / args.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            dpp_penalties.append(dpp_penalty.detach())
            assignments_rows.append(assignments.detach())
        gradient_statistics = {
            name: _gradient_norm(parameters)
            for name, parameters in parameter_groups.items()
        }
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                optimized, args.gradient_clip
            ).detach().cpu()
        )
        optimizer.step()
        model.project_interaction_strength_()
        dual_optimizer.step()
        with torch.no_grad():
            assignment_dual.sub_(assignment_dual.mean())
        train_objective = float(torch.stack(losses).mean().cpu())
        train_dpp_penalty = float(
            torch.stack(dpp_penalties).mean().cpu()
        )
        batch_assignments = torch.cat(assignments_rows)

        pruning_event = None
        if step in prune_steps:
            # Physical-removal scoring creates a full candidate model on the
            # GPU. The final microbatch graph and parameter gradients are no
            # longer needed, so release them before allocating that copy.
            optimizer.zero_grad(set_to_none=True)
            dual_optimizer.zero_grad(set_to_none=True)
            del batches, batch, losses, dpp_penalties, assignments_rows
            del batch_assignments, component, gates, scores
            del free_energy, assignments, reference_marginal
            del hyperprior, loss, dpp_penalty
            if dpp_weight > 0.0:
                del trace, matrices
            if next(model.parameters()).is_cuda:
                torch.cuda.empty_cache()
            criterion_seed = sample_seed + 2_000_000 + step * 10_000
            full_nll, deltas, removed_nlls = _physical_removal_deltas(
                model,
                split.train,
                nu_multiplier=args.nu_multiplier,
                n_samples=args.pruning_samples,
                sample_seed=criterion_seed,
                batch_size=pruning_batch_size,
            )
            removed_local = int(torch.argmin(deltas))
            removed_lineage = lineage[removed_local]
            components_before = model.n_components
            nu_before = model.degrees_of_freedom
            omega_before = model.mean_matrix().detach().cpu()
            new_nu = _nu(
                args.nu_multiplier,
                components_before - 1,
                model.n_marks,
            )
            kept_local = model.prune_component(
                removed_local,
                degrees_of_freedom=new_nu,
            )
            lineage = [lineage[index] for index in kept_local]
            pruning_event = {
                "step": step,
                "components_before": components_before,
                "components_after": model.n_components,
                "removed_local_component": removed_local,
                "removed_initial_component": removed_lineage,
                "kept_initial_components": lineage.copy(),
                "nu_before": nu_before,
                "nu_after": model.degrees_of_freedom,
                "full_train_marginal_nll_per_path": float(full_nll),
                "removed_nlls_per_path": removed_nlls.tolist(),
                "removal_nll_deltas_per_path": deltas.tolist(),
                "selected_removal_delta_per_path": float(
                    deltas[removed_local]
                ),
                "omega_dimension_before": len(omega_before),
                "omega_dimension_after": model.dimension,
                "optimizer_reset": True,
                "ot_dual_reset": True,
                "uniform_target_after": 1.0 / model.n_components,
            }
            pruning_events.append(pruning_event)
            _write_json(
                args.outdir / "pruning_events.partial.json", pruning_events
            )
            torch.save(
                model.state_dict(),
                args.outdir
                / f"checkpoint_post_prune_step_{step}_K{model.n_components}.pt",
            )
            if (
                step == final_prune_step
                and freeze_encoder
            ):
                for parameter in _shared_encoder(
                    model.backbone
                ).parameters():
                    parameter.requires_grad_(True)
            (
                optimizer,
                assignment_dual,
                dual_optimizer,
                optimized,
                parameter_groups,
            ) = _make_optimizers(
                model,
                neural_lr=args.neural_learning_rate,
                backbone_lr=(
                    current_backbone_lr
                    if use_separate_backbone_lr
                    else None
                ),
                omega_lr=args.omega_learning_rate,
                alpha_lr=args.alpha_learning_rate,
                weight_decay=args.weight_decay,
                dual_lr=args.dual_learning_rate,
            )

        training_beta = model.exploration_beta
        model.exploration_beta = 0.0
        validation = evaluate_latent_wishart_nhp(
            model,
            split.validation,
            cutoff=None,
            n_samples=args.validation_samples,
            sample_seed=sample_seed + 100_000,
            batch_size=args.evaluation_batch_size,
            assignment_dual=assignment_dual.detach(),
            assignment_temperature=1.0,
        )
        model.exploration_beta = training_beta
        validation_purity = cluster_purity(
            split.validation_labels,
            validation.full_cluster_probabilities.argmax(1).numpy(),
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
            best_alpha = float(model.interaction_strength().detach().cpu())
        history.append({
            "step": step,
            "optimizer_steps": step,
            "n_components": model.n_components,
            "degrees_of_freedom": model.degrees_of_freedom,
            "rho": rho,
            "training_exploration_beta": beta,
            "dpp_weight": dpp_weight,
            "functional_dpp_penalty": train_dpp_penalty,
            "backbone_learning_rate": current_backbone_lr,
            "head_learning_rate": args.neural_learning_rate,
            "validation_exploration_beta": 0.0,
            "interaction_strength": float(
                model.interaction_strength().detach().cpu()
            ),
            "train_objective_per_path": train_objective,
            "gradient_norm": gradient_norm,
            "gradient_norm_neural": gradient_statistics["neural"],
            "gradient_norm_backbone": gradient_statistics["backbone"],
            "gradient_norm_head": gradient_statistics["head"],
            "gradient_norm_omega": gradient_statistics["omega"],
            "gradient_norm_alpha": gradient_statistics["alpha"],
            "gradient_clip_coefficient": min(
                1.0, args.gradient_clip / max(gradient_norm, 1e-30)
            ),
            "batch_assignment_stationarity_error": (
                float(
                    (
                        batch_assignments.mean(0)
                        - reference_marginal.detach()
                    ).abs().max().cpu()
                )
                if pruning_event is None
                else float("nan")
            ),
            "validation_purity": validation_purity,
            "validation_nll_per_exposure": validation_nll,
            "selection_eligible": selection_eligible,
            "is_best_checkpoint": is_better,
            "pruned_component": (
                pruning_event["removed_local_component"]
                if pruning_event is not None
                else -1
            ),
            "dual_max_abs": float(assignment_dual.detach().abs().max().cpu()),
        })
        if step % 25 == 0 or pruning_event is not None:
            pd.DataFrame(history).to_csv(
                args.outdir / "history.partial.csv", index=False
            )
            _write_json(args.outdir / "progress.json", {
                "step": step,
                "total_steps": total_steps,
                "n_components": model.n_components,
                "degrees_of_freedom": model.degrees_of_freedom,
                "rho": rho,
                "training_exploration_beta": beta,
                "dpp_weight": dpp_weight,
                "functional_dpp_penalty": train_dpp_penalty,
                "backbone_learning_rate": current_backbone_lr,
                "head_learning_rate": args.neural_learning_rate,
                "interaction_strength": float(
                    model.interaction_strength().detach().cpu()
                ),
                "validation_purity": validation_purity,
                "selection_eligible": selection_eligible,
                "best_step": best_step,
                "best_validation_purity": (
                    best_purity if best_state is not None else None
                ),
            })
        progress.set_postfix({
            "K": model.n_components,
            "nu": model.degrees_of_freedom,
            "beta": f"{beta:.3f}",
            "rho": f"{rho:.1f}",
            "alpha": f"{float(model.interaction_strength().detach()):.3f}",
            "val": f"{validation_purity:.3f}",
            "best": f"{best_purity:.3f}" if best_state is not None else "-",
        }, refresh=False)

    if best_state is None or best_dual is None:
        raise RuntimeError("no target-K checkpoint was eligible")
    model.load_state_dict(best_state)
    model.exploration_beta = 0.0
    assignment_dual = best_dual
    model.eval()
    train_evaluation = evaluate_latent_wishart_nhp(
        model,
        split.train,
        cutoff=None,
        n_samples=args.test_samples,
        sample_seed=sample_seed + 300_000,
        batch_size=args.evaluation_batch_size,
        assignment_dual=assignment_dual,
        assignment_temperature=1.0,
    )
    test_evaluation = evaluate_latent_wishart_nhp(
        model,
        split.test,
        cutoff=None,
        n_samples=args.test_samples,
        sample_seed=sample_seed + 400_000,
        batch_size=args.evaluation_batch_size,
        assignment_dual=assignment_dual,
        assignment_temperature=1.0,
    )
    cluster, predictions = clustering_row(
        split.test_labels,
        test_evaluation.full_cluster_probabilities,
        model=args.architecture,
        degrees_of_freedom=model.degrees_of_freedom,
        representation="wishart_overcomplete_backward_elimination",
    )
    train_probabilities = train_evaluation.full_cluster_probabilities
    train_soft_marginal = train_probabilities.mean(0)
    train_counts = torch.bincount(
        train_probabilities.argmax(1), minlength=args.target_components
    )
    target = torch.full_like(train_soft_marginal, 1.0 / args.target_components)
    reference = torch.softmax(
        torch.log(target) + assignment_dual / best_rho, dim=0
    )
    mean_matrix = model.mean_matrix().detach().cpu()
    eigenvalues = torch.linalg.eigvalsh(mean_matrix)
    test_exposure = sum(sequence.horizon for sequence in split.test)
    alpha_values = np.asarray(
        [row["interaction_strength"] for row in history], dtype=float
    )
    alpha_gradient_norms = np.asarray(
        [row["gradient_norm_alpha"] for row in history], dtype=float
    )
    clip_coefficients = np.asarray(
        [row["gradient_clip_coefficient"] for row in history], dtype=float
    )
    result = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "initialization_seed": initialization_seed,
        "architecture": args.architecture,
        "variant": (
            "signed_wishart_overcomplete_backward_elimination"
            if removals > 0
            else "signed_wishart_fixed_k_warmup"
        ),
        "initial_components": args.initial_components,
        "target_components": args.target_components,
        "final_component_lineage": lineage,
        "initial_nu": initial_nu,
        "final_nu": model.degrees_of_freedom,
        "nu_multiplier": args.nu_multiplier,
        "pretrain_steps": pretrain_steps,
        "encoder_frozen_until_target": freeze_encoder,
        "backbone_learning_rate": backbone_learning_rate,
        "overcomplete_backbone_learning_rate": (
            overcomplete_backbone_learning_rate
        ),
        "post_target_backbone_learning_rate": backbone_learning_rate,
        "backbone_restored_after_target": scheduled_backbone_restore,
        "separate_backbone_learning_rate": use_separate_backbone_lr,
        "backbone_ramp_steps": backbone_ramp_steps,
        "head_initialization_noise": args.head_initialization_noise,
        "dpp_initial_strength": args.dpp_strength,
        "dpp_bandwidth": args.dpp_bandwidth,
        "dpp_jitter": args.dpp_jitter,
        "dpp_decay_steps": args.dpp_decay_steps,
        "warmup_steps": args.warmup_steps,
        "pruning_batch_size": pruning_batch_size,
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
        "neural_learning_rate": args.neural_learning_rate,
        "head_learning_rate": args.neural_learning_rate,
        "omega_learning_rate": args.omega_learning_rate,
        "alpha_learning_rate": args.alpha_learning_rate,
        "alpha_temperature": args.alpha_temperature,
        "alpha_parameterization": args.alpha_parameterization,
        "initial_alpha": args.initial_alpha,
        "alpha_schedule": args.alpha_schedule,
        "alpha_schedule_steps": args.alpha_schedule_steps,
        "interaction_mode": args.interaction_mode,
        "best_alpha": best_alpha,
        "final_loaded_alpha": float(
            model.interaction_strength().detach().cpu()
        ),
        "alpha_min_during_training": float(alpha_values.min()),
        "alpha_max_during_training": float(alpha_values.max()),
        "alpha_total_range_during_training": float(
            alpha_values.max() - alpha_values.min()
        ),
        "alpha_gradient_norm_median": float(
            np.median(alpha_gradient_norms)
        ),
        "alpha_gradient_norm_maximum": float(
            alpha_gradient_norms.max()
        ),
        "mean_hyperprior_strength": args.mean_hyperprior_strength,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "test_samples": args.test_samples,
        "pruning_samples": args.pruning_samples,
        "initial_exploration_beta": args.initial_beta,
        "final_exploration_beta": 0.0,
        "exploration_mode": "convex",
        "exploration_nu": args.exploration_nu,
        "beta_schedule": "exponential_to_zero",
        "beta_decay_rate": args.beta_decay_rate,
        "gradient_clip": args.gradient_clip,
        "gradient_clipped_step_fraction": float(
            np.mean(clip_coefficients < 1.0 - 1e-12)
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
        "mean_abs_omega_minus_identity": float(
            (
                mean_matrix
                - torch.eye(model.dimension, dtype=mean_matrix.dtype)
            ).abs().mean()
        ),
        "minimum_omega_eigenvalue": float(eigenvalues.min()),
        "maximum_omega_eigenvalue": float(eigenvalues.max()),
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
    np.save(args.outdir / "mean_omega.npy", mean_matrix.numpy())
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
