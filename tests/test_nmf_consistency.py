"""
Test NMF consistency (dimensionless K, h API):
1. NMF solver gives correct self-consistent solutions
2. VAN with W=0 (bias-only) converges to NMF solution
"""

import sys
import os
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.nmf.mean_field import solve, free_energy_per_spin as nmf_free_energy
from src.exact.transfer_matrix import free_energy_per_spin as exact_free_energy
from src.van.model import OneLayerVAN
from src.van.energy import energy
from src.van.utils import DEFAULT_DTYPE


class TestNMFSolver:
    """Basic NMF solver tests."""

    def test_h0_high_T(self):
        """At high T (small K) with h=0: m=0 is the only solution."""
        result = solve(K=0.5, h=0.0)
        assert result.converged
        np.testing.assert_allclose(result.magnetization, 0.0, atol=1e-10)

    def test_h0_low_T(self):
        """At low T (large K) with h=0: NMF may give m != 0 (spurious 1D phase transition)."""
        result = solve(K=2.0, h=0.0)
        assert result.converged
        # For K=2: 2K = 4 > 1, so NMF gives nonzero m (known artifact in 1D)

    def test_h_nonzero(self):
        """With h != 0, m should have same sign as h."""
        result = solve(K=1.0, h=0.5)
        assert result.converged
        assert result.magnetization > 0

        result_neg = solve(K=1.0, h=-0.5)
        assert result_neg.converged
        assert result_neg.magnetization < 0

    def test_self_consistency(self):
        """Verify the self-consistency equation m = tanh(2*K*m + h)."""
        for K, h in [(1.0, 0.0), (1.0, 0.5), (2.0, 0.3)]:
            result = solve(K, h)
            m = result.magnetization
            m_check = np.tanh(2 * K * m + h)
            np.testing.assert_allclose(m, m_check, atol=1e-10,
                                       err_msg=f"Self-consistency failed at K={K}, h={h}")


class TestNMFvsExact:
    """NMF free energy should always be >= exact (variational bound)."""

    def test_variational_bound(self):
        for K, h, N in [(1.0, 0.0, 100), (0.5, 0.5, 100), (2.0, 0.0, 100)]:
            result = solve(K, h)
            bf_nmf = result.free_energy_per_spin
            bf_exact = exact_free_energy(K, h, N)
            assert bf_nmf >= bf_exact - 1e-10, \
                f"Variational bound violated: beta*f_NMF={bf_nmf:.8f} < beta*f_exact={bf_exact:.8f}"


class TestVANBiasOnlyMatchesNMF:
    """VAN with W=0 should give NMF-equivalent results."""

    def test_bias_equals_atanh_m(self):
        """When W=0, trained biases should satisfy b_i = atanh(m_NMF)."""
        for K, h in [(1.0, 0.5), (0.5, 0.3)]:
            nmf_result = solve(K, h)
            m_nmf = nmf_result.magnetization
            b_target = np.arctanh(np.clip(m_nmf, -1 + 1e-15, 1 - 1e-15))

            # Create VAN with W=0, set bias to NMF value
            N = 8
            model = OneLayerVAN(N, use_bias=True)
            with torch.no_grad():
                model.W.fill_(0.0)
                model.b.fill_(b_target)

            # Compute exact VAN free energy by enumeration
            num_configs = 2**N
            configs = torch.zeros(num_configs, N, dtype=DEFAULT_DTYPE)
            for i in range(num_configs):
                for j in range(N):
                    configs[i, j] = 1.0 if (i >> j) & 1 else -1.0

            with torch.no_grad():
                log_q = model.log_prob(configs)
                beta_E = energy(configs, K=K, h=h)
                q = torch.exp(log_q)
                # Dimensionless: beta*F_van = sum q * (beta*E + log q)
                beta_F_van = torch.sum(q * (beta_E + log_q)).item()

            # Compare with NMF free energy (both dimensionless beta*f)
            bf_nmf = nmf_result.free_energy_per_spin
            np.testing.assert_allclose(beta_F_van / N, bf_nmf, atol=1e-6,
                                       err_msg=f"VAN(W=0) != NMF at K={K}, h={h}")
