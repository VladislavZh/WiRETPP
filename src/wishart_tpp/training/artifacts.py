"""Persistence for one configured training run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from wishart_tpp.training.state import ActiveCheckpoint


class RunArtifactWriter:
    """Write only the artifacts produced by the selected method."""

    @staticmethod
    def _write_shared(
        output: Path,
        state: dict[str, Tensor],
        history: list[dict[str, float]],
    ) -> None:
        torch.save(state, output / "shared_checkpoint.pt")
        pd.DataFrame(history).to_csv(output / "shared_history.csv", index=False)

    @staticmethod
    def _write_result(output: Path, result: dict[str, object]) -> None:
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    def write_shared(
        self,
        output: Path,
        state: dict[str, Tensor],
        history: list[dict[str, float]],
    ) -> None:
        """Persist the shared boundary before any method-specific training."""

        output.mkdir(parents=True, exist_ok=True)
        self._write_shared(output, state, history)

    def write_pure(
        self,
        output: Path,
        *,
        shared_state: dict[str, Tensor],
        pure_state: dict[str, Tensor],
        shared_history: list[dict[str, float]],
        pure_history: list[dict[str, float]],
        result: dict[str, object],
    ) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self._write_shared(output, shared_state, shared_history)
        torch.save(pure_state, output / "pure_checkpoint.pt")
        pd.DataFrame(pure_history).to_csv(output / "pure_history.csv", index=False)
        self._write_result(output, result)

    def write_wishart(
        self,
        output: Path,
        *,
        shared_state: dict[str, Tensor],
        wishart_state: dict[str, Tensor],
        checkpoint: ActiveCheckpoint,
        shared_history: list[dict[str, float]],
        wishart_history: list[dict[str, float]],
        result: dict[str, object],
    ) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self._write_shared(output, shared_state, shared_history)

        # Wishart checkpoint includes the population law and selected alpha.
        torch.save(
            {
                "model": wishart_state,
                "population_means": checkpoint.population_means,
                "log_weights": checkpoint.log_weights,
                "alpha": checkpoint.alpha,
                "population_df": checkpoint.population_df,
                "cycle": checkpoint.cycle,
            },
            output / "wishart_checkpoint.pt",
        )
        pd.DataFrame(wishart_history).to_csv(
            output / "wishart_history.csv", index=False
        )
        self._write_result(output, result)
