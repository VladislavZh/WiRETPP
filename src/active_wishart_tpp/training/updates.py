"""Direct marginal and non-centered Bartlett M-steps with native Adam arithmetic."""

import math

import numpy as np
import torch

from active_wishart_tpp.inference.gradient_population import GradientPopulation
from active_wishart_tpp.inference.relative_posterior import RelativePosterior
from active_wishart_tpp.training.initialization import initial_omega


class PureUpdate:
    """Accumulate the ordinary mixture likelihood before one neural Adam update."""

    def __init__(self, fabric, decoder, config):
        self.fabric, self.decoder, self.config = fabric, decoder, config

    def setup(self, model):
        t = self.config.training
        optimizer = torch.optim.Adam(
            model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
        )
        return self.fabric.setup(model, optimizer)

    def step(self, model, optimizer, train, indices):
        effective = train.select(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        c = self.config.compute
        for start in range(0, len(indices), c.physical_batch):
            micro = train.select(indices[start : start + c.physical_batch])
            score = model.mixture_logits.new_zeros(())
            for trace_start in range(0, len(micro.sequences), c.trace_batch):
                component = self.decoder.base_component_scores(
                    model(micro.sequences[trace_start : trace_start + c.trace_batch])
                )
                logits = component + model.mixture_log_weights()[None]
                # Preserve the native beta=1 floating-point expression exactly.
                log_k = math.log(logits.shape[1])
                score = score + ((torch.logsumexp(logits, dim=1) - log_k) + log_k).sum()
            loss = -score / effective.exposure
            self.fabric.backward(loss)
            value += float(loss.detach().cpu())
        # Native direct Pure performed this algebraic identity at specialization=1.
        # Keep its FP32 rounding, but expose no head-coupling/warm-up control.
        bank = getattr(model, "module", model)
        for parameter in bank.component_head_parameters():
            if parameter.grad is not None:
                blocks = parameter.grad.reshape(bank.n_components, bank.n_marks, -1)
                mean = blocks.mean(0, keepdim=True)
                blocks.copy_(mean + (blocks - mean))
        self.fabric.clip_gradients(
            model, optimizer, max_norm=self.config.training.gradient_clip
        )
        optimizer.step()
        return value


class WishartUpdate:
    """Jointly update neural, mixture, Omega and alpha parameters at fixed relative q."""

    def __init__(self, fabric, decoder, config):
        self.fabric, self.decoder, self.config = fabric, decoder, config

    def setup(self, model):
        t = self.config.training
        heads = model.component_head_parameters()
        head_ids = {id(p) for p in heads}
        backbone = [
            p
            for name, p in model.named_parameters()
            if name != "mixture_logits" and id(p) not in head_ids
        ]
        optimizer = torch.optim.Adam(
            [
                dict(params=backbone, role="backbone", lr=t.learning_rate),
                dict(params=heads, role="component_heads"),
            ],
            lr=t.learning_rate,
            weight_decay=t.weight_decay,
        )
        model, optimizer = self.fabric.setup(model, optimizer)
        identity = torch.eye(
            model.n_marks, device=model.device, dtype=model.mixture_logits.dtype
        ).expand(model.n_components, -1, -1)
        self.population = GradientPopulation(identity, 0.1)
        for role, parameter, rate in (
            ("omega", self.population.raw_omega, t.omega_lr),
            ("alpha", self.population.alpha, t.alpha_lr),
            ("mixture", model.mixture_logits, t.learning_rate),
        ):
            optimizer.add_param_group(
                dict(
                    params=[parameter],
                    role=role,
                    lr=rate,
                    weight_decay=0.0,
                    betas=(0.9, 0.95) if role != "mixture" else (0.9, 0.999),
                )
            )
        self.physical_omega = initial_omega(
            model, t.omega_noise, self.config.runtime.seed
        )
        self.population.initialize(self.physical_omega, t.initial_alpha)
        return model, optimizer

    def microbatch_loss(
        self, model, batch, selected, gamma, relative, generator, exposure
    ):
        """Use one set of anchored Bartlett draws for each physical microbatch."""
        indices = torch.as_tensor(selected, device=gamma.device)
        weights = gamma.index_select(0, indices).to(model.device).detach()
        draws = relative.sample(
            selected, self.population.means(), self.config.training.m_samples, generator
        )
        likelihood = model.mixture_logits.new_zeros(())
        size = self.config.compute.trace_batch
        for start in range(0, len(batch.sequences), size):
            stop = min(start + size, len(batch.sequences))
            scores = self.decoder.component_scores(
                model(batch.sequences[start:stop]),
                draws[start:stop],
                self.population.alpha,
            )
            likelihood = (
                likelihood - (weights[start:stop] * scores.mean(2)).sum() / exposure
            )
        self.step_likelihood += float(likelihood.detach())
        return (
            likelihood
            + relative.regularizer(selected, weights, model.mixture_logits) / exposure
        )

    def step(self, model, optimizer, train, indices, gamma, relative, generator):
        """Clip joint gradients after full accumulation, then apply unrestricted Adam."""
        model.train()
        optimizer.zero_grad(set_to_none=True)
        self.step_likelihood, self.step_objective = 0.0, 0.0
        exposure = train.select(indices).exposure
        size = self.config.compute.physical_batch
        for start in range(0, len(indices), size):
            selected = indices[start : start + size]
            loss = self.microbatch_loss(
                model,
                train.select(selected),
                selected,
                gamma,
                relative,
                generator,
                exposure,
            )
            self.fabric.backward(loss)
            self.step_objective += float(loss.detach())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.config.training.gradient_clip, error_if_nonfinite=True
        )
        self.diagnostics = dict(
            gradient_norm=float(norm),
            alpha_gradient=float(self.population.alpha.grad.detach()),
            omega_gradient_norm=float(self.population.raw_omega.grad.detach().norm()),
            mixture_gradient_norm=float(model.mixture_logits.grad.detach().norm()),
        )
        optimizer.step()
        self.population.project_alpha()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Nonfinite joint M-step parameter")
        return self.step_likelihood

    def cycle(self, model, optimizer, train, posterior, gamma, cycle):
        """Hold responsibilities and relative coordinates fixed for eight neural updates."""
        t = self.config.training
        relative = RelativePosterior(
            posterior.means,
            posterior.degrees_of_freedom,
            self.population.means(),
            t.population_df,
        )
        seed = self.config.runtime.seed + cycle * 30_013
        random = np.random.default_rng(seed)
        generator = torch.Generator(device=model.device).manual_seed(seed + 101)
        for _ in range(t.updates_per_cycle):
            indices = random.choice(
                len(train.sequences),
                min(t.effective_batch, len(train.sequences)),
                replace=False,
            )
            value = self.step(
                model, optimizer, train, indices, gamma, relative, generator
            )
        with torch.no_grad():
            means = relative.transported_means(self.population.means()).to(
                posterior.means
            )
        self.diagnostics["negative_elbo_last_step"] = self.step_objective
        return value, means
