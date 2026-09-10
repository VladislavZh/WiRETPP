"""Fit-scoped likelihood statistics for an E-step with a frozen neural bank."""

from dataclasses import dataclass
import torch
from active_wishart_tpp.model.active_block import _RepeatPathRows, _segment_sum
from active_wishart_tpp.model.wishart import matrix_square_root


@dataclass(frozen=True)
class FrozenLikelihood:
    """Cache only quantities independent of the local variational parameters."""

    trace: object
    amplitudes: torch.Tensor
    event_base: torch.Tensor
    mark_indicators: torch.Tensor
    moment: torch.Tensor
    direct: torch.Tensor
    floor: torch.Tensor
    decoder: object

    @classmethod
    @torch.no_grad()
    def prepare(cls, trace, decoder):
        """Integrate sqrt(h) outer products once, without retaining an autograd graph."""
        if (
            trace.event_base_rates.requires_grad
            or trace.integral_base_rates.requires_grad
        ):
            raise ValueError(
                "Frozen likelihood cannot be used for neural M-step gradients"
            )
        paths, components, marks = (trace.n_paths, trace.n_components, trace.n_marks)
        moment_size = components * marks * marks
        width = moment_size + components + 1
        integrated = trace.integral_base_rates.new_zeros(paths, width)
        row_size = max(1, decoder.maximum_intermediate_elements // (2 * width))
        for start in range(0, len(trace.integral_paths), row_size):
            stop = min(start + row_size, len(trace.integral_paths))
            base = trace.integral_base_rates[start:stop].clamp_min(0.0)
            amplitudes = base.sqrt()
            weights = trace.integral_weights[start:stop]
            outer = amplitudes[..., :, None] * amplitudes[..., None, :]
            packed = torch.cat(
                (
                    (weights[:, None, None, None] * outer).flatten(1),
                    weights[:, None] * base.sum(-1),
                    (weights * marks * decoder.intensity_floor)[:, None],
                ),
                dim=1,
            )
            integrated += _segment_sum(packed, trace.integral_paths[start:stop], paths)
        base = trace.event_base_rates.clamp_min(0.0)
        indicators = torch.nn.functional.one_hot(trace.event_marks, marks).to(base)
        return cls(
            trace,
            base.sqrt(),
            base,
            indicators,
            integrated[:, :moment_size].reshape(paths, components, marks, marks),
            integrated[:, moment_size : moment_size + components],
            integrated[:, -1],
            decoder,
        )

    def scores(self, draws, alpha, roots=None):
        """Evaluate the same sampled likelihood using precomputed frozen statistics."""
        if self.decoder.intensity_floor < 1e-12:
            return self.decoder.component_scores(self.trace, draws, alpha, roots)
        roots = matrix_square_root(draws) if roots is None else roots
        trace = self.trace
        score = draws.new_zeros(trace.n_paths, trace.n_components, draws.shape[2])
        alpha = torch.as_tensor(alpha, device=draws.device, dtype=draws.dtype)
        size = self.decoder._row_shard_size(roots)
        for start in range(0, len(trace.event_paths), size):
            stop = min(start + size, len(trace.event_paths))
            paths = trace.event_paths[start:stop]
            selected = _RepeatPathRows.apply(roots, paths)
            transformed = torch.einsum(
                "rksij,rkj->rksi", selected, self.amplitudes[start:stop]
            ).square()
            rates = (
                (1.0 - alpha) * self.event_base[start:stop, :, None, :]
                + alpha * transformed
                + self.decoder.intensity_floor
            ).clamp_min(1e-12)
            marked = (rates * self.mark_indicators[start:stop, None, None, :]).sum(-1)
            score = score + _segment_sum(marked.log(), paths, trace.n_paths)
        gram = roots.transpose(-1, -2) @ roots
        transformed = torch.einsum("nksij,nkij->nks", gram, self.moment)
        return (
            score
            - (1.0 - alpha) * self.direct[:, :, None]
            - alpha * transformed
            - self.floor[:, None, None]
        )


class FrozenDecoder:
    """Resolve likelihood statistics only for the trace objects of one local fit."""

    def __init__(self, decoder, cache):
        self.statistics = {
            id(batch.trace): FrozenLikelihood.prepare(batch.trace, decoder)
            for batch in cache
        }

    def component_scores(self, trace, draws, alpha, roots=None):
        return self.statistics[id(trace)].scores(draws, alpha, roots)
