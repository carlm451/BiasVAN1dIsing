"""Tests for the OneLayerVAN model."""

import sys
import os
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.van.model import OneLayerVAN


class TestMask:
    """Mask should be strictly lower triangular."""

    def test_mask_lower_triangular(self):
        model = OneLayerVAN(5)
        mask = model.mask.numpy()
        # Diagonal and above should be zero
        for i in range(5):
            for j in range(i, 5):
                assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"
        # Below diagonal should be 1
        for i in range(1, 5):
            for j in range(i):
                assert mask[i, j] == 1.0, f"mask[{i},{j}] should be 1"


class TestNormalization:
    """Sum of probabilities over all configs should be 1."""

    def _enumerate_configs(self, N):
        """Generate all 2^N spin configurations."""
        configs = torch.zeros(2**N, N, dtype=torch.float64)
        for i in range(2**N):
            for j in range(N):
                configs[i, j] = 1.0 if (i >> j) & 1 else -1.0
        return configs

    def test_normalization_N2(self):
        torch.manual_seed(0)
        model = OneLayerVAN(2)
        configs = self._enumerate_configs(2)
        with torch.no_grad():
            log_q = model.log_prob(configs)
            total = torch.exp(log_q).sum().item()
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_normalization_N3(self):
        torch.manual_seed(0)
        model = OneLayerVAN(3)
        configs = self._enumerate_configs(3)
        with torch.no_grad():
            log_q = model.log_prob(configs)
            total = torch.exp(log_q).sum().item()
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_normalization_N4(self):
        torch.manual_seed(0)
        model = OneLayerVAN(4)
        configs = self._enumerate_configs(4)
        with torch.no_grad():
            log_q = model.log_prob(configs)
            total = torch.exp(log_q).sum().item()
        np.testing.assert_allclose(total, 1.0, atol=1e-10)


class TestWZero:
    """With W=0, mu_i = tanh(b_i) independent of other spins."""

    def test_independent_conditionals(self):
        model = OneLayerVAN(4, use_bias=True)
        # Set W to zero, biases to known values
        with torch.no_grad():
            model.W.fill_(0.0)
            model.b.copy_(torch.tensor([0.1, -0.2, 0.5, -0.3], dtype=torch.float64))

        # Two different samples should give same mu
        s1 = torch.tensor([[1., 1., 1., 1.]], dtype=torch.float64)
        s2 = torch.tensor([[-1., -1., -1., -1.]], dtype=torch.float64)

        with torch.no_grad():
            mu1 = model.conditional_magnetization(s1)
            mu2 = model.conditional_magnetization(s2)

        torch.testing.assert_close(mu1, mu2)
        # Should equal tanh(b)
        expected = torch.tanh(model.b).unsqueeze(0)
        torch.testing.assert_close(mu1, expected)


class TestSampleValues:
    """Samples should be in {-1, +1}."""

    def test_sample_values(self):
        torch.manual_seed(42)
        model = OneLayerVAN(8)
        samples = model.sample(100)
        unique = torch.unique(samples)
        assert set(unique.tolist()).issubset({-1.0, 1.0})

    def test_sample_shape(self):
        model = OneLayerVAN(6)
        samples = model.sample(50)
        assert samples.shape == (50, 6)


class TestGradients:
    """Gradients should flow through log_prob."""

    def test_grad_flow(self):
        torch.manual_seed(0)
        model = OneLayerVAN(4, use_bias=True)
        sample = torch.tensor([[1., -1., 1., 1.]], dtype=torch.float64)
        log_q = model.log_prob(sample)
        log_q.sum().backward()

        assert model.W.grad is not None
        assert model.b.grad is not None
        assert model.W.grad.abs().sum() > 0
