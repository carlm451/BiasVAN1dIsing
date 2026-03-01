"""Cross-validation: analytical_formulas.py (pure numpy) vs transfer_matrix.py (SymPy)."""

import pytest
import numpy as np

from src.exact import analytical_formulas as af
from src.exact import transfer_matrix as tm


# Diverse (K, h, N) spanning high-T/low-T, zero/nonzero field, small/large N
TEST_POINTS = [
    # (K, h, N)
    (0.1, 0.0, 4),       # high T, zero field, small N
    (0.5, 0.5, 8),       # moderate T, moderate field
    (1.0, 0.0, 16),      # moderate T, zero field
    (1.0, 1.0, 16),      # moderate T, moderate field
    (2.0, 0.1, 32),      # low T, weak field
    (2.0, 2.0, 4),       # low T, strong field, small N
    (5.0, 0.5, 64),      # very low T, large N
    (0.3, -1.5, 10),     # high T, negative field, odd N
    (3.0, 0.0, 100),     # low T, zero field, large N
    (0.01, 3.0, 6),      # very high T, strong field
]

RTOL = 1e-10
ATOL = 1e-14  # near-zero values (e.g. χ~1e-8 at K=5,N=64) differ by machine epsilon


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_eigenvalues(K, h, N):
    lp_af, lm_af = af.eigenvalues(K, h)
    lp_tm, lm_tm = tm.eigenvalues(K, h)
    np.testing.assert_allclose(lp_af, lp_tm, rtol=RTOL)
    np.testing.assert_allclose(lm_af, lm_tm, rtol=RTOL)


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_free_energy(K, h, N):
    np.testing.assert_allclose(
        af.free_energy_per_spin(K, h, N),
        tm.free_energy_per_spin(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_magnetization(K, h, N):
    np.testing.assert_allclose(
        af.magnetization(K, h, N),
        tm.magnetization(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_nn_correlation(K, h, N):
    np.testing.assert_allclose(
        af.nn_correlation(K, h, N),
        tm.nn_correlation(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_susceptibility(K, h, N):
    np.testing.assert_allclose(
        af.susceptibility(K, h, N),
        tm.susceptibility(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_specific_heat(K, h, N):
    np.testing.assert_allclose(
        af.specific_heat(K, h, N),
        tm.specific_heat(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_energy(K, h, N):
    np.testing.assert_allclose(
        af.energy_per_spin(K, h, N),
        tm.energy_per_spin(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_entropy(K, h, N):
    np.testing.assert_allclose(
        af.entropy_per_spin(K, h, N),
        tm.entropy_per_spin(K, h, N),
        rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("K,h,N", TEST_POINTS)
def test_partition_function(K, h, N):
    np.testing.assert_allclose(
        af.partition_function(K, h, N),
        tm.partition_function(K, h, N),
        rtol=RTOL, atol=ATOL,
    )
