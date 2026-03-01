"""Tests for the exact transfer matrix solution (dimensionless K, h API)."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.exact.transfer_matrix import (
    eigenvalues,
    partition_function,
    free_energy_per_spin,
    magnetization,
    magnetization_thermo_limit,
    nn_correlation,
    energy_per_spin,
    entropy_per_spin,
    susceptibility,
    specific_heat,
    compute_all,
    _log_partition_function,
)


class TestPartitionFunctionN2H0:
    """N=2, h=0: Z = 2*exp(2K) + 2*exp(-2K) = 4*cosh(2K)."""

    def test_K_1(self):
        K, h, N = 1.0, 0.0, 2
        Z = partition_function(K, h, N)
        expected = 4.0 * np.cosh(2.0)
        np.testing.assert_allclose(Z, expected, rtol=1e-12)

    def test_K_05(self):
        K, h, N = 0.5, 0.0, 2
        Z = partition_function(K, h, N)
        expected = 4.0 * np.cosh(1.0)
        np.testing.assert_allclose(Z, expected, rtol=1e-12)

    def test_free_energy(self):
        # N=2, h=0, K=1: beta*f = -(1/2)*ln(4*cosh(2))
        K, h, N = 1.0, 0.0, 2
        bf = free_energy_per_spin(K, h, N)
        expected = -(1.0 / 2) * np.log(4.0 * np.cosh(2.0))
        np.testing.assert_allclose(bf, expected, rtol=1e-12)


class TestPartitionFunctionN3H0:
    """N=3, h=0, K=0.5: Z = sum over 8 configs."""

    def test_Z_by_enumeration(self):
        K, h, N = 0.5, 0.0, 3
        # Enumerate all 2^3 = 8 configs for periodic ring
        # E = -J*(s0*s1 + s1*s2 + s2*s0), with K=beta*J=0.5
        Z_enum = 0.0
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    # beta*E = -K*(s0*s1 + s1*s2 + s2*s0)
                    beta_E = -K * (s0*s1 + s1*s2 + s2*s0)
                    Z_enum += np.exp(-beta_E)
        Z = partition_function(K, h, N)
        np.testing.assert_allclose(Z, Z_enum, rtol=1e-12)


class TestPartitionFunctionN2WithH:
    """N=2, h!=0: enumerate all 4 states."""

    def test_K_05_h_03(self):
        K, h, N = 0.5, 0.3, 2
        # N=2 periodic ring: beta*E = -K*(s0*s1 + s1*s0) - h*(s0 + s1) = -2K*s0*s1 - h*(s0+s1)
        Z_enum = 0.0
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                beta_E = -K * (s0*s1 + s1*s0) - h * (s0 + s1)
                Z_enum += np.exp(-beta_E)
        Z = partition_function(K, h, N)
        np.testing.assert_allclose(Z, Z_enum, rtol=1e-12)

    def test_free_energy(self):
        K, h, N = 0.5, 0.3, 2
        Z_enum = 0.0
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                beta_E = -K * (s0*s1 + s1*s0) - h * (s0 + s1)
                Z_enum += np.exp(-beta_E)
        bf = free_energy_per_spin(K, h, N)
        expected = -np.log(Z_enum) / N
        np.testing.assert_allclose(bf, expected, rtol=1e-12)


class TestHighTemperature:
    """K -> 0 (beta -> 0): beta*f -> -ln(2)."""

    def test_high_T_limit(self):
        K = 0.01  # small K ~ high T
        h, N = 0.0, 100
        bf = free_energy_per_spin(K, h, N)
        expected = -np.log(2.0)
        np.testing.assert_allclose(bf, expected, rtol=1e-1)

    def test_very_high_T(self):
        K = 0.001
        h, N = 0.0, 100
        bf = free_energy_per_spin(K, h, N)
        expected = -np.log(2.0)
        np.testing.assert_allclose(bf, expected, rtol=1e-2)


class TestLowTemperature:
    """K -> inf (beta -> inf), h > 0: beta*f -> -(K + h) (all spins aligned)."""

    def test_low_T_h_positive(self):
        K = 100.0
        h = 50.0  # h = beta*H, also large when beta is large
        N = 64
        bf = free_energy_per_spin(K, h, N)
        expected = -(K + h)
        np.testing.assert_allclose(bf, expected, atol=1e-6)


class TestThermoLimit:
    """magnetization(N=1000) matches magnetization_thermo_limit."""

    def test_finite_vs_thermo(self):
        K, h = 1.0, 0.5
        m_finite = magnetization(K, h, N=1000)
        m_thermo = magnetization_thermo_limit(K, h)
        np.testing.assert_allclose(m_finite, m_thermo, atol=1e-6)

    def test_various_points(self):
        for K, h in [(0.5, 0.1), (2.0, 1.0), (0.5, 0.5)]:
            m_finite = magnetization(K, h, N=1000)
            m_thermo = magnetization_thermo_limit(K, h)
            np.testing.assert_allclose(m_finite, m_thermo, atol=1e-5,
                                       err_msg=f"Failed at K={K}, h={h}")


class TestSymmetry:
    """h=0: <m> = 0 for all K, N."""

    def test_m_zero_at_h0(self):
        for N in [2, 4, 8, 16]:
            for K in [0.5, 1.0, 2.0]:
                m = magnetization(K, 0.0, N)
                np.testing.assert_allclose(m, 0.0, atol=1e-10,
                                           err_msg=f"N={N}, K={K}")

    def test_thermo_limit_m_zero_at_h0(self):
        for K in [0.5, 1.0, 2.0, 10.0]:
            m = magnetization_thermo_limit(K, 0.0)
            np.testing.assert_allclose(m, 0.0, atol=1e-15)


class TestDerivativeConsistency:
    """Thermodynamic identities from derivatives of free energy."""

    def test_nn_correlation_from_free_energy(self):
        """<s_i s_{i+1}> = (1/N) d(ln Z)/dK."""
        K, h, N = 1.0, 0.3, 8
        corr = nn_correlation(K, h, N)
        # Cross-check: compute from explicit numerical d(ln Z)/dK
        dK = 1e-7
        lnZ_p = _log_partition_function(K + dK, h, N)
        lnZ_m = _log_partition_function(K - dK, h, N)
        corr_check = (lnZ_p - lnZ_m) / (2.0 * dK * N)
        np.testing.assert_allclose(corr, corr_check, rtol=1e-5)

    def test_magnetization_from_free_energy(self):
        """<m> = (1/N) d(ln Z)/dh."""
        K, h, N = 1.0, 0.5, 8
        m = magnetization(K, h, N)
        # Cross-check with numerical derivative
        dh = 1e-7
        lnZ_p = _log_partition_function(K, h + dh, N)
        lnZ_m = _log_partition_function(K, h - dh, N)
        m_check = (lnZ_p - lnZ_m) / (2.0 * dh * N)
        np.testing.assert_allclose(m, m_check, rtol=1e-5)

    def test_energy_consistency(self):
        """beta*<E>/N should equal -(K*<s_i s_{i+1}> + h*<m>)."""
        K, h, N = 1.0, 0.3, 16
        E = energy_per_spin(K, h, N)
        corr = nn_correlation(K, h, N)
        m = magnetization(K, h, N)
        E_check = -(K * corr + h * m)
        np.testing.assert_allclose(E, E_check, rtol=1e-10)

    def test_susceptibility_from_free_energy(self):
        """chi = (1/N) d^2(ln Z)/dh^2, check vs numerical."""
        K, h, N = 1.0, 0.3, 8
        chi = susceptibility(K, h, N)
        dh = 1e-5
        lnZ_p = _log_partition_function(K, h + dh, N)
        lnZ_0 = _log_partition_function(K, h, N)
        lnZ_m = _log_partition_function(K, h - dh, N)
        chi_check = (lnZ_p - 2.0 * lnZ_0 + lnZ_m) / (dh**2 * N)
        np.testing.assert_allclose(chi, chi_check, rtol=1e-4)

    def test_specific_heat_from_free_energy(self):
        """c_v/k_B check vs numerical second derivative of ln Z w.r.t. beta at fixed J, H.

        Use K=1.0, h=0.5 (representing beta=1, J=1, H=0.5).
        c_v = -beta^2 * d^2(beta*f)/d(beta)^2 where beta*f = -(1/N)*ln Z.
        We check by varying beta (i.e., scaling K and h together).
        """
        # Choose a physical point: beta=1, J=1, H=0.5 => K=1, h=0.5
        beta_0 = 1.0
        J_phys = 1.0
        H_phys = 0.5
        N = 8
        K0 = beta_0 * J_phys
        h0 = beta_0 * H_phys

        cv = specific_heat(K0, h0, N)

        # Numerical check: vary beta
        dbeta = 1e-6
        def beta_f(b):
            return free_energy_per_spin(b * J_phys, b * H_phys, N)

        bf_p = beta_f(beta_0 + dbeta)
        bf_0 = beta_f(beta_0)
        bf_m = beta_f(beta_0 - dbeta)
        d2 = (bf_p - 2.0 * bf_0 + bf_m) / dbeta**2
        cv_check = -beta_0**2 * d2

        np.testing.assert_allclose(cv, cv_check, rtol=1e-3)


class TestEntropy:
    """Entropy should be non-negative and consistent."""

    def test_entropy_nonnegative(self):
        for K in [0.5, 1.0, 2.0]:
            for h in [0.0, 0.5]:
                s = entropy_per_spin(K, h, N=16)
                assert s >= -1e-10, f"Negative entropy at K={K}, h={h}: s={s}"

    def test_entropy_high_T(self):
        """At high T (small K, small h): s -> ln(2)."""
        s = entropy_per_spin(0.001, 0.0, N=100)
        np.testing.assert_allclose(s, np.log(2.0), rtol=1e-2)


class TestComputeAll:
    """Smoke test for compute_all."""

    def test_returns_all_fields(self):
        result = compute_all(1.0, 0.5, 16)
        assert hasattr(result, 'free_energy')
        assert hasattr(result, 'magnetization')
        assert hasattr(result, 'magnetization_thermo')
        assert hasattr(result, 'nn_correlation')
        assert hasattr(result, 'susceptibility')
        assert hasattr(result, 'specific_heat')
        assert hasattr(result, 'energy')
        assert hasattr(result, 'entropy')
        assert hasattr(result, 'partition_function')
