"""
Test that N=2 VAN achieves near-exact agreement with exact solution.

For N=2: VAN has 3 parameters (b1, b2, W21) matching 3 independent Boltzmann
weights, so it should be exact at all (K, h) given sufficient training.

All quantities are now dimensionless: K = beta*J, h = beta*H.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.van.train import train_and_evaluate_exact, TrainConfig
from src.exact.transfer_matrix import free_energy_per_spin


class TestN2Exactness:
    """N=2 VAN should match exact free energy to high precision."""

    @pytest.mark.parametrize("K,h", [
        (1.0, 0.0),    # moderate K, no field
        (2.0, 0.0),    # large K, no field
        (0.5, 0.0),    # small K, no field
        (1.0, 0.5),    # moderate K, moderate field
        (2.0, 1.0),    # large K, strong field
        (0.5, 0.3),    # small K, weak field
        (0.5, 0.5),    # weak coupling with field
    ])
    def test_exact_agreement(self, K, h):
        bf_exact = free_energy_per_spin(K, h, N=2)
        config = TrainConfig(
            N=2, K=K, h=h,
            batch_size=2000, lr=0.005, max_step=10000, seed=42,
            conv_tol=1e-8,
        )
        result = train_and_evaluate_exact(config)
        assert abs(result.final_free_energy - bf_exact) < 1e-4, \
            f"N=2 not exact at K={K}, h={h}: " \
            f"beta*F_VAN={result.final_free_energy:.10f}, beta*F_exact={bf_exact:.10f}, " \
            f"diff={abs(result.final_free_energy - bf_exact):.2e}"
