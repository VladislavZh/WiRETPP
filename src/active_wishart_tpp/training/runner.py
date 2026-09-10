"""Run shared10, Direct Pure and Bartlett Wishart without experimental branches."""

import gc
import math
import time

import numpy as np
import pandas as pd
import torch

from active_wishart_tpp.backbones.factory import create_bank
from active_wishart_tpp.data import EventDataModule
from active_wishart_tpp.model.active_block import ActiveBlockDecoder
from active_wishart_tpp.training.artifact_io import write_csv, write_json, write_torch
from active_wishart_tpp.training.checkpoint import (
    CheckpointStore,
    completed_result,
    data_digest,
    neural_digest,
    seal_completion,
    source_digest,
)
from active_wishart_tpp.training.evaluation import ModelEvaluator
from active_wishart_tpp.training.expectation import ExpectationStep, em_block_indices
from active_wishart_tpp.training.initialization import seed_neural_randomness
from active_wishart_tpp.training.state import clone_state_dict
from active_wishart_tpp.training.updates import PureUpdate, WishartUpdate
from active_wishart_tpp.training.validation_curves import ValidationCurveWriter


def finite_metrics(value):
    """Represent unavailable label-based metrics as JSON null rather than NaN."""
    if isinstance(value, dict):
        return {key: finite_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_metrics(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def release_gradients(optimizer):
    """Release transient buffers at the native E/M/validation phase boundaries."""
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ExperimentRunner:
    """Coordinate one paired experiment with validation-only model selection."""

    def __init__(self, config, fabric):
        self.config, self.fabric = config, fabric
        r, c = config.runtime, config.compute
        data = EventDataModule(
            r.data_root,
            r.split_seed,
            mixture_components=r.components,
            split_protocol=r.split_protocol,
            packed_filename=r.packed_filename,
        )
        prepared = data.prepare(r.dataset)
        self.dataset, self.split = prepared.dataset, prepared.split
        if any(
            sequence.regime_edges is not None for sequence in self.dataset.sequences
        ):
            raise ValueError(
                "Observed-regime experiments are not part of this protocol"
            )
        if config.training.population_df <= self.dataset.n_marks - 1:
            raise ValueError("Wishart df must exceed C-1")
        self.output = r.output_root / r.dataset / f"seed{r.seed}"
        self.decoder = ActiveBlockDecoder(1e-6, "constant", c.mc_draw_shard)
        self.evaluator = ModelEvaluator(self.decoder, c.trace_batch, c.path_shard)
        self.signature = dict(
            protocol="bartlett-all-gradient-fast-v-no-temperature-v1",
            config=config.as_dict(),
            data_sha256=data_digest(self.split),
            source_sha256=source_digest(),
        )

    def new_model(self, shared=None):
        """Load the common K1 state before deterministic independent-head expansion."""
        config = self.config
        model = create_bank(config.model, self.dataset.n_marks, config.runtime.seed)
        if shared is not None:
            model.load_state_dict(shared)
            model.expand_components(
                config.runtime.components,
                config.model.component_noise,
                2026082300 + config.runtime.seed,
            )
        return model

    def curve(self, method):
        return ValidationCurveWriter(
            self.output / method,
            dataset=self.dataset.name,
            seed=self.config.runtime.seed,
            method=method,
        )

    def shared(self):
        """Fit or exact-resume ten K1 updates and retain their complete validation curve."""
        update = PureUpdate(self.fabric, self.decoder, self.config)
        model, optimizer = update.setup(self.new_model())
        seed_neural_randomness(self.config.runtime.seed)
        random = np.random.default_rng(self.config.runtime.seed + 103)
        store = CheckpointStore(
            self.output / "shared" / "active.pt", dict(self.signature, method="shared")
        )
        saved = store.restore(model, optimizer, random)
        done, history = (saved["completed"], saved["history"]) if saved else (0, [])
        writer = self.curve("shared")
        if saved is None:
            initial = self.evaluator.pure(model, self.split.validation)
            writer.record(0, 0, initial, self.split.validation)
            store.save(model, optimizer, random, dict(completed=0, history=[]))
        for step in range(done + 1, self.config.training.shared_steps + 1):
            indices = random.choice(
                len(self.split.train.sequences),
                min(
                    self.config.training.effective_batch,
                    len(self.split.train.sequences),
                ),
                replace=False,
            )
            loss = update.step(model, optimizer, self.split.train, indices)
            score = self.evaluator.pure(model, self.split.validation)
            row = writer.record(
                step,
                step,
                score,
                self.split.validation,
                extras=dict(
                    train_nll=loss, learning_rate=self.config.training.learning_rate
                ),
            )
            history.append(row)
            store.save(model, optimizer, random, dict(completed=step, history=history))
            self.fabric.print(
                f"[shared] step={step}/{self.config.training.shared_steps} nll={score.nll_per_exposure:.7f}",
                flush=True,
            )
        state = clone_state_dict(model)
        write_torch(self.output / "shared" / "selected.pt", state)
        return state

    def evaluate(self, method, model, update, partition, *, selected=False):
        """Score the pure marginal or unadapted Wishart prior predictive likelihood."""
        if method == "pure":
            return self.evaluator.pure(model, partition)
        t = self.config.training
        return self.evaluator.active(
            model,
            partition,
            update.physical_omega,
            model.mixture_log_weights().detach(),
            population_df=t.population_df,
            alpha=float(update.population.alpha.detach()),
            samples=t.selected_samples if selected else t.validation_samples,
            repeats=t.selected_repeats if selected else 1,
            seed=self.config.runtime.seed
            + (
                (89_000_009 if partition is self.split.validation else 90_000_007)
                if selected
                else 40_009
            ),
        )

    def selected_state(self, model, update, cycle, score):
        """Capture all parameters of one validation-selected state together."""
        result = dict(
            cycle=cycle, nll=score.nll_per_exposure, model=clone_state_dict(model)
        )
        if isinstance(update, WishartUpdate):
            result.update(
                population={
                    k: v.detach().cpu().clone()
                    for k, v in update.population.state_dict().items()
                },
                omega=update.physical_omega.detach().cpu().clone(),
            )
        return result

    def train_cycle(self, method, model, optimizer, update, random, cycle, previous):
        """Run one unchanged direct-marginal or generalized variational EM cycle."""
        t, c, r = self.config.training, self.config.compute, self.config.runtime
        train = self.split.train
        if method == "pure":
            losses = []
            for _ in range(t.updates_per_cycle):
                indices = random.choice(
                    len(train.sequences),
                    min(t.effective_batch, len(train.sequences)),
                    replace=False,
                )
                losses.append(update.step(model, optimizer, train, indices))
            return dict(train_nll=float(np.mean(losses))), None
        epoch = block = count = 1
        if c.em_block < len(train.sequences):
            indices, epoch, block, count = em_block_indices(
                len(train.sequences), c.em_block, cycle, r.seed
            )
            train, previous = train.select(indices), None
        getattr(model, "module", model).integration_rule.reset_training_draws(
            r.seed + cycle * 1_000_003
        )
        started = time.perf_counter()
        posterior = ExpectationStep(self.fabric, self.decoder, self.config).fit(
            model,
            train,
            update.physical_omega,
            float(update.population.alpha.detach()),
            cycle,
            previous,
        )
        gamma = torch.softmax(
            model.mixture_log_weights().detach().to(posterior.free_energy.device)[None]
            - posterior.free_energy,
            dim=1,
        ).detach()
        e_seconds = time.perf_counter() - started
        release_gradients(None)
        started = time.perf_counter()
        loss, means = update.cycle(model, optimizer, train, posterior, gamma, cycle)
        update.physical_omega = update.population.means().detach().clone()
        row = dict(
            train_nll=loss,
            e_seconds=e_seconds,
            m_seconds=time.perf_counter() - started,
            em_epoch=epoch,
            em_block=block,
            em_block_count=count,
            em_paths=len(train.sequences),
            alpha=float(update.population.alpha.detach()),
            df=t.population_df,
            masses=gamma.sum(0).tolist(),
            **update.diagnostics,
        )
        release_gradients(optimizer)
        previous = (means, posterior.degrees_of_freedom) if count == 1 else None
        return row, previous

    def fit(self, method, shared, *, stop_after=None):
        """Persist every committed cycle and evaluate only the final selected checkpoint."""
        if method not in ("pure", "wishart"):
            raise ValueError("Only pure and wishart are supported")
        t, r = self.config.training, self.config.runtime
        directory = self.output / method
        signature = dict(
            self.signature, method=method, shared_sha256=neural_digest(shared)
        )
        completed = completed_result(directory, signature)
        if completed is not None:
            return completed
        update = (PureUpdate if method == "pure" else WishartUpdate)(
            self.fabric, self.decoder, self.config
        )
        model, optimizer = update.setup(self.new_model(shared))
        seed_neural_randomness(r.seed)
        random = np.random.default_rng(r.seed + 107)
        store = CheckpointStore(directory / "active.pt", signature)
        population = update.population if method == "wishart" else None
        saved = store.restore(model, optimizer, random, population)
        if saved:
            done, history, best = saved["completed"], saved["history"], saved["best"]
            previous = saved["previous"]
            if method == "wishart":
                update.physical_omega = saved["physical_omega"].to(model.device)
        else:
            done, history, previous = 0, [], None
            initial = self.evaluate(method, model, update, self.split.validation)
            # Native Wishart selection starts after its first EM cycle; Pure includes cycle0.
            best = (
                self.selected_state(model, update, 0, initial)
                if method == "pure"
                else None
            )
            self.curve(method).record(0, 0, initial, self.split.validation)
        started = time.perf_counter()
        for cycle in range(done + 1, min(t.cycles, stop_after or t.cycles) + 1):
            cycle_started = time.perf_counter()
            row, previous = self.train_cycle(
                method, model, optimizer, update, random, cycle, previous
            )
            score = self.evaluate(method, model, update, self.split.validation)
            if best is None or score.nll_per_exposure < best["nll"]:
                best = self.selected_state(model, update, cycle, score)
            row.update(
                cycle=cycle,
                updates=cycle * t.updates_per_cycle,
                cumulative_updates=t.shared_steps + cycle * t.updates_per_cycle,
                nll=score.nll_per_exposure,
                purity=score.purity,
                ari=score.ari,
                best_cycle=best["cycle"],
                lr=t.learning_rate,
                next_lr=t.learning_rate,
                weights=model.mixture_log_weights().detach().cpu().exp().tolist(),
            )
            if method == "wishart":
                backbone = self.evaluator.pure(model, self.split.validation)
                row.update(
                    backbone_nll=backbone.nll_per_exposure,
                    backbone_purity=backbone.purity,
                    backbone_ari=backbone.ari,
                )
            self.curve(method).record(
                cycle, row["updates"], score, self.split.validation
            )
            row["cycle_seconds"] = time.perf_counter() - cycle_started
            history.append(row)
            store.save(
                model,
                optimizer,
                random,
                dict(
                    completed=cycle,
                    history=history,
                    best=best,
                    previous=previous,
                    physical_omega=None if method == "pure" else update.physical_omega,
                ),
                population,
            )
            write_json(
                directory / "history.json", finite_metrics(dict(history=history))
            )
            elapsed = time.perf_counter() - started
            eta = elapsed / (cycle - done) * (t.cycles - cycle)
            self.fabric.print(
                f"[{method}] cycle={cycle}/{t.cycles} nll={score.nll_per_exposure:.7f} "
                f"purity={score.purity:.4f} ari={score.ari:.4f} eta_seconds={eta:.1f}",
                flush=True,
            )
        if stop_after is not None and stop_after < t.cycles:
            return history
        model.load_state_dict(best["model"])
        if method == "wishart":
            update.population.load_state_dict(best["population"])
            update.physical_omega = best["omega"].to(model.device)
        validation = self.evaluate(
            method, model, update, self.split.validation, selected=True
        )
        test = self.evaluate(method, model, update, self.split.test, selected=True)
        result = dict(
            method=method,
            dataset=r.dataset,
            seed=r.seed,
            config=self.config.as_dict(),
            best_cycle=best["cycle"],
            selection_validation_nll=best["nll"],
            selected_validation=validation.as_dict(),
            test=test.as_dict(),
            completed_cycle=t.cycles,
            shared_sha256=neural_digest(shared),
            selected_state_matches_checkpoint=True,
            test_read=True,
            selection="minimum ordinary validation NLL; robust evaluation never reselects",
            elapsed_training_seconds=sum(row["cycle_seconds"] for row in history),
            labels_available=bool(np.all(self.split.test.labels >= 0)),
        )
        if method == "wishart":
            result.update(
                alpha=float(update.population.alpha.detach()),
                df=t.population_df,
                omega=update.physical_omega.tolist(),
            )
        weights = model.mixture_log_weights().detach().cpu().exp()
        result.update(
            mixture_weights=weights.tolist(),
            effective_k=float(torch.exp(-torch.special.xlogy(weights, weights).sum())),
        )
        write_torch(directory / "selected.pt", best)
        write_csv(directory / "history.csv", pd.DataFrame(history))
        write_json(directory / "result.json", finite_metrics(result))
        seal_completion(directory, signature, t.cycles)
        return result
