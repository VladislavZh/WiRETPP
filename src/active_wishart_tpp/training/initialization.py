"""Seeded expansion and population initialization shared by the fixed protocol."""

import torch


def initial_omega(model, noise_scale, seed):
    """Perturb identity by the native symmetric trace-zero seeded noise."""
    count, dimension = model.n_components, model.n_marks
    identity = torch.eye(dimension, device=model.device, dtype=model.dtype)
    generator = torch.Generator(device=model.device).manual_seed(seed + 50_021)
    noise = torch.randn(
        (count, dimension, dimension),
        device=model.device,
        dtype=model.dtype,
        generator=generator,
    )
    noise = 0.5 * (noise + noise.transpose(-1, -2))
    noise = (
        noise
        - identity
        * (noise.diagonal(dim1=-2, dim2=-1).sum(-1) / dimension)[:, None, None]
    )
    scale = torch.linalg.eigvalsh(noise).abs().amax(-1).clamp_min(1e-12)
    means = identity + noise_scale * noise / scale[:, None, None]
    return 0.5 * (means + means.transpose(-1, -2))


def seed_neural_randomness(seed):
    """Reset stochastic neural operations to the native seed convention."""
    torch.manual_seed(2026082200 + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026082200 + seed)
