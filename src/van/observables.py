"""
Sample-based observable computation from trained VAN models.

Provides two entry points:
1. compute_observables(model, K, h) — from a live model
2. observables_from_checkpoint(W, b, K, h) — from saved parameters

All observables are computed from Monte Carlo samples drawn from the VAN
distribution q(s), using the fluctuation-dissipation relations.
"""

import torch
import numpy as np
from .model import OneLayerVAN
from .energy import energy
from .utils import DEFAULT_DTYPE


def compute_observables(model, K, h, n_samples=50000):
    """Draw samples from a trained VAN and compute all observables.

    Parameters
    ----------
    model : OneLayerVAN
        Trained model.
    K : float
        Dimensionless coupling beta*J.
    h : float
        Dimensionless field beta*H.
    n_samples : int
        Number of MC samples.

    Returns
    -------
    dict with keys:
        magnetization : float — <m>
        nn_correlation : float — <s_i s_{i+1}>
        chi_bare : float — N * Var(m) [fluctuation-dissipation]
        energy : float — <beta*E> / N (dimensionless energy per spin)
        specific_heat_bare : float — Var(beta*E) / N
        free_energy : float — <beta*E + log q> / N (dimensionless beta*f per spin)
    """
    N = model.N

    with torch.no_grad():
        samples = model.sample(n_samples)  # (n_samples, N)

        # Magnetization per sample: m_i = (1/N) sum_j s_j
        m_per_sample = samples.mean(dim=1)  # (n_samples,)
        magnetization = m_per_sample.mean().item()

        # NN correlation: (1/N) sum_i s_i * s_{i+1}
        nn_per_sample = (samples * torch.roll(samples, -1, dims=1)).mean(dim=1)
        nn_correlation = nn_per_sample.mean().item()

        # Susceptibility (bare): chi_bare = N * Var(m)
        chi_bare = (N * m_per_sample.var()).item()

        # Energy per spin: <beta*E> / N
        beta_E = energy(samples, K=K, h=h)  # (n_samples,)
        energy_per_spin = (beta_E / N).mean().item()

        # Specific heat (bare): Var(beta*E) / N
        specific_heat_bare = (beta_E.var() / N).item()

        # Free energy per spin: <beta*E + log q> / N
        log_q = model.log_prob(samples)  # (n_samples,)
        free_energy = ((beta_E + log_q) / N).mean().item()

    return {
        'magnetization': magnetization,
        'nn_correlation': nn_correlation,
        'chi_bare': chi_bare,
        'energy': energy_per_spin,
        'specific_heat_bare': specific_heat_bare,
        'free_energy': free_energy,
    }


def observables_from_checkpoint(W, b, K, h, z2=False, n_samples=50000):
    """Load model from saved W, b arrays and compute observables.

    This is the main entry point for post-training analysis.

    Parameters
    ----------
    W : ndarray, shape (N, N)
        Saved weight matrix.
    b : ndarray, shape (N,)
        Saved bias vector.
    K : float
        Dimensionless coupling.
    h : float
        Dimensionless field.
    z2 : bool
        Whether to use Z2 symmetry.
    n_samples : int
        Number of MC samples.

    Returns
    -------
    dict — same keys as compute_observables.
    """
    model = OneLayerVAN.from_parameters(W, b, z2=z2)
    return compute_observables(model, K, h, n_samples=n_samples)
