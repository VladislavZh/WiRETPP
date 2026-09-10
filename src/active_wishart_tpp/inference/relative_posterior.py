"""Transport a fixed local Wishart posterior with its global Cholesky factor."""

import torch
from active_wishart_tpp.model.wishart import sample_wishart, wishart_kl


class RelativePosterior:
    """Hold detached local coordinates while exposing the global Omega gradient."""

    def __init__(self, means, degrees, omega, prior_df):
        self.means = means.detach()
        self.degrees = degrees.detach()
        self.anchor = omega.detach().clone()
        self.anchor_root = torch.linalg.cholesky(self.anchor)
        with torch.no_grad():
            self.kl = wishart_kl(
                self.means, self.degrees, self.anchor.to(self.means), prior_df
            )

    def transport(self, omega):
        """Return chol(Omega) chol(anchor)^-1 without an explicit matrix inverse."""
        root = torch.linalg.cholesky(omega)
        return torch.linalg.solve_triangular(
            self.anchor_root.transpose(-1, -2), root.transpose(-1, -2), upper=True
        ).transpose(-1, -2)

    @staticmethod
    def congruence(matrices, transform):
        """Apply a batched congruence and remove numerical antisymmetry."""
        result = transform @ matrices @ transform.transpose(-1, -2)
        return 0.5 * (result + result.transpose(-1, -2))

    def sample(self, selected, omega, samples, generator):
        """Transport anchored Bartlett draws, preserving their original RNG and jitter."""
        indices = torch.as_tensor(selected, device=self.means.device)
        means = self.means.index_select(0, indices).to(omega)
        degrees = self.degrees.index_select(0, indices).to(omega)
        draws = sample_wishart(means, degrees, samples, generator).detach()
        return self.congruence(draws, self.transport(omega)[None, :, None])

    def transported_means(self, omega):
        """Return the current physical posterior means for the next E-step warm start."""
        return self.congruence(self.means.to(omega), self.transport(omega))

    def regularizer(self, selected, gamma, logits):
        """Return complete Wishart and categorical KL terms at fixed local coordinates."""
        indices = torch.as_tensor(selected, device=self.kl.device)
        kl = self.kl.index_select(0, indices).to(logits)
        weights = gamma.detach()
        return (
            weights * (kl - logits.log_softmax(-1)) + torch.xlogy(weights, weights)
        ).sum()
