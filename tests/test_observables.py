"""Tests for VAN sample-based observables.

At N=2, the 1-layer VAN is exact (3 parameters = 3 independent Boltzmann
weights), so sample observables should match exact values to within
statistical noise.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.van.model import OneLayerVAN
from src.van.train import train, train_and_evaluate_exact, TrainConfig
from src.van.observables import compute_observables, observables_from_checkpoint
from src.exact import analytical_formulas as af


N2_TEST_POINTS = [
    (0.5, 0.0),
    (1.0, 0.5),
    (2.0, 1.0),
    (0.5, 2.0),
]


@pytest.mark.parametrize("K,h", N2_TEST_POINTS)
def test_n2_magnetization(K, h):
    """VAN magnetization at N=2 should match exact after training."""
    N = 2
    tc = TrainConfig(N=N, K=K, h=h, use_bias=True,
                     z2=(abs(h) < 1e-10),
                     batch_size=1000, lr=0.01, max_step=5000, seed=42)
    result = train_and_evaluate_exact(tc)

    model = OneLayerVAN.from_parameters(
        result.parameters['W'], result.parameters['b'],
        z2=(abs(h) < 1e-10),
    )
    obs = compute_observables(model, K, h, n_samples=100000)

    exact_m = af.magnetization(K, h, N)
    # Statistical tolerance: larger for MC estimates
    assert abs(obs['magnetization'] - exact_m) < 0.05, \
        f"M: VAN={obs['magnetization']:.4f}, exact={exact_m:.4f}"


@pytest.mark.parametrize("K,h", N2_TEST_POINTS)
def test_n2_nn_correlation(K, h):
    """VAN NN correlation at N=2 should match exact."""
    N = 2
    tc = TrainConfig(N=N, K=K, h=h, use_bias=True,
                     z2=(abs(h) < 1e-10),
                     batch_size=1000, lr=0.01, max_step=5000, seed=42)
    result = train_and_evaluate_exact(tc)

    model = OneLayerVAN.from_parameters(
        result.parameters['W'], result.parameters['b'],
        z2=(abs(h) < 1e-10),
    )
    obs = compute_observables(model, K, h, n_samples=100000)

    exact_nn = af.nn_correlation(K, h, N)
    assert abs(obs['nn_correlation'] - exact_nn) < 0.05, \
        f"<ss>: VAN={obs['nn_correlation']:.4f}, exact={exact_nn:.4f}"


@pytest.mark.parametrize("K,h", N2_TEST_POINTS)
def test_n2_free_energy(K, h):
    """VAN free energy at N=2 should match exact."""
    N = 2
    tc = TrainConfig(N=N, K=K, h=h, use_bias=True,
                     z2=(abs(h) < 1e-10),
                     batch_size=1000, lr=0.01, max_step=5000, seed=42)
    result = train_and_evaluate_exact(tc)

    model = OneLayerVAN.from_parameters(
        result.parameters['W'], result.parameters['b'],
        z2=(abs(h) < 1e-10),
    )
    obs = compute_observables(model, K, h, n_samples=100000)

    exact_f = af.free_energy_per_spin(K, h, N)
    assert abs(obs['free_energy'] - exact_f) < 0.02, \
        f"F: VAN={obs['free_energy']:.4f}, exact={exact_f:.4f}"


def test_from_parameters_roundtrip():
    """from_parameters should reconstruct identical model."""
    N = 4
    model = OneLayerVAN(N, use_bias=True, z2=False)
    params = model.get_parameters_dict()
    model2 = OneLayerVAN.from_parameters(params['W'], params['b'], z2=False)

    import torch
    sample = torch.ones(1, N, dtype=torch.float64)
    lp1 = model.log_prob(sample).item()
    lp2 = model2.log_prob(sample).item()
    assert abs(lp1 - lp2) < 1e-12, f"log_prob mismatch: {lp1} vs {lp2}"


def test_observables_from_checkpoint():
    """observables_from_checkpoint should work with saved arrays."""
    N = 2
    K, h = 1.0, 0.5
    tc = TrainConfig(N=N, K=K, h=h, use_bias=True, z2=False,
                     batch_size=1000, lr=0.01, max_step=3000, seed=0)
    result = train_and_evaluate_exact(tc)

    obs = observables_from_checkpoint(
        result.parameters['W'], result.parameters['b'],
        K, h, z2=False, n_samples=50000,
    )
    assert 'magnetization' in obs
    assert 'nn_correlation' in obs
    assert 'chi_bare' in obs
    assert 'energy' in obs
    assert 'specific_heat_bare' in obs
    assert 'free_energy' in obs
    # Free energy should be close to exact for N=2
    exact_f = af.free_energy_per_spin(K, h, N)
    assert abs(obs['free_energy'] - exact_f) < 0.05


class TestNMFObservables:
    """Test NMF analytical observable functions."""

    def test_nmf_nn_correlation_is_m_squared(self):
        """NMF <ss> should exactly equal m^2."""
        from src.nmf.mean_field import nn_correlation_nmf, solve as nmf_solve
        for K, h in [(1.0, 0.5), (2.0, 1.0), (0.5, 0.0)]:
            result = nmf_solve(K, h)
            nn = nn_correlation_nmf(K, h)
            assert abs(nn - result.magnetization**2) < 1e-12, \
                f"NMF <ss> != m^2 at K={K}, h={h}"

    def test_nmf_chi_diverges_at_Kc(self):
        """NMF susceptibility should diverge at K=0.5, h=0."""
        from src.nmf.mean_field import susceptibility_nmf
        chi = susceptibility_nmf(0.5, 0.0)
        assert chi == np.inf or chi > 1e10, \
            f"NMF chi should diverge at K_c=0.5, got {chi}"

    def test_nmf_chi_finite_away_from_Kc(self):
        """NMF susceptibility should be finite away from K_c."""
        from src.nmf.mean_field import susceptibility_nmf
        chi = susceptibility_nmf(1.0, 0.5)
        assert np.isfinite(chi) and chi > 0, \
            f"NMF chi should be finite and positive, got {chi}"

    def test_nmf_specific_heat_positive(self):
        """NMF specific heat should be non-negative."""
        from src.nmf.mean_field import specific_heat_nmf
        for K, h in [(1.0, 0.0), (2.0, 1.0), (0.3, 0.5)]:
            C = specific_heat_nmf(K, h)
            assert C >= -1e-8, f"NMF C/k_B should be >= 0, got {C} at K={K}, h={h}"
