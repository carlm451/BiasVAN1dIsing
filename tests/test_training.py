"""Tests for VAN training loop (dimensionless K, h API)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.van.train import train, TrainConfig
from src.exact.transfer_matrix import free_energy_per_spin


class TestTrainingSmoke:
    """Basic training smoke tests."""

    def test_loss_decreases(self):
        """Train 200 steps, loss should decrease."""
        config = TrainConfig(N=4, K=1.0, h=0.0,
                             batch_size=500, lr=0.01, max_step=200, seed=42)
        result = train(config)
        # Compare first 20 vs last 20 steps
        early = sum(result.free_energy_history[:20]) / 20
        late = sum(result.free_energy_history[-20:]) / 20
        assert late < early, f"Free energy should decrease: early={early:.4f}, late={late:.4f}"

    def test_variational_bound(self):
        """beta*F_VAN >= beta*F_exact (variational bound)."""
        test_points = [
            (1.0, 0.0),   # moderate K, no field
            (2.0, 0.5),   # large K, with field
            (0.5, 0.0),   # small K, no field
        ]
        for K, h in test_points:
            N = 4
            bf_exact = free_energy_per_spin(K, h, N)
            config = TrainConfig(N=N, K=K, h=h,
                                 batch_size=1000, lr=0.01, max_step=2000, seed=42)
            result = train(config)
            # VAN free energy should be >= exact (variational bound)
            # Allow small tolerance for MC noise
            assert result.final_free_energy > bf_exact - 0.05, \
                f"Variational bound violated at K={K}, h={h}: " \
                f"beta*F_VAN={result.final_free_energy:.6f} < beta*F_exact={bf_exact:.6f}"

    def test_convergence_high_T(self):
        """N=4 at high T (small K) should converge within 1000 steps."""
        config = TrainConfig(N=4, K=0.5, h=0.0,
                             batch_size=1000, lr=0.01, max_step=1000,
                             seed=42, conv_tol=1e-4)
        result = train(config)
        bf_exact = free_energy_per_spin(0.5, 0.0, 4)
        assert abs(result.final_free_energy - bf_exact) < 0.1
