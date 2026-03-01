"""Tests for the 1D periodic Ising energy function (dimensionless)."""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.van.energy import energy


class TestEnergyBasic:
    """Basic energy calculations."""

    def test_all_up_N4(self):
        """All spins +1, N=4: beta*E = -4K - 4h."""
        K, h = 1.0, 0.5
        sample = torch.ones(1, 4, dtype=torch.float64)
        E = energy(sample, K=K, h=h)
        expected = -4.0 * K - 4.0 * h
        torch.testing.assert_close(E, torch.tensor([expected], dtype=torch.float64))

    def test_all_down_N4(self):
        """All spins -1, N=4: beta*E = -4K + 4h."""
        K, h = 1.0, 0.5
        sample = -torch.ones(1, 4, dtype=torch.float64)
        E = energy(sample, K=K, h=h)
        expected = -4.0 * K + 4.0 * h
        torch.testing.assert_close(E, torch.tensor([expected], dtype=torch.float64))

    def test_alternating_N4(self):
        """Alternating +-1, N=4: [+1,-1,+1,-1]. nn sum = -4, field sum = 0. beta*E = +4K."""
        K, h = 1.0, 0.0
        sample = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
        E = energy(sample, K=K, h=h)
        # nn pairs: (1)(-1) + (-1)(1) + (1)(-1) + (-1)(1) = -1 -1 -1 -1 = -4
        expected = -K * (-4.0)
        torch.testing.assert_close(E, torch.tensor([expected], dtype=torch.float64))

    def test_pbc_included(self):
        """Verify s_N connects to s_1 (periodic boundary)."""
        K, h = 1.0, 0.0
        # [+1, +1, +1, -1]: bonds (+)(+)=1, (+)(+)=1, (+)(-)=-1, (-)(+)=-1 => nn_sum = 0
        sample = torch.tensor([[1.0, 1.0, 1.0, -1.0]], dtype=torch.float64)
        E = energy(sample, K=K, h=h)
        expected = -K * 0.0
        torch.testing.assert_close(E, torch.tensor([expected], dtype=torch.float64))


class TestEnergyBatch:
    """Batch shape checks."""

    def test_batch_shape(self):
        sample = torch.ones(10, 8, dtype=torch.float64)
        E = energy(sample, K=1.0, h=0.0)
        assert E.shape == (10,)

    def test_batch_values(self):
        """All-up batch should give identical energies."""
        sample = torch.ones(5, 4, dtype=torch.float64)
        E = energy(sample, K=1.0, h=0.5)
        expected = -4.0 - 2.0  # -4K - 4h
        for i in range(5):
            assert abs(E[i].item() - expected) < 1e-12


class TestEnergyZeroField:
    """h=0 cases."""

    def test_h_zero_all_up(self):
        sample = torch.ones(1, 6, dtype=torch.float64)
        E = energy(sample, K=1.0, h=0.0)
        torch.testing.assert_close(E, torch.tensor([-6.0], dtype=torch.float64))
