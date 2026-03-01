"""
Analytical solution of the 1D periodic Ising model — pure numpy implementation.

All formulas are taken directly from the boxed results in
latex/1disingtransferderivations.tex.  No SymPy dependency.

Dimensionless parameterization: K = βJ (coupling), h = βH (field).

Notation:
    c  = cosh(h)
    s  = sinh(h)
    w  = exp(-2K)
    w2 = exp(-4K) = w²
    D  = sqrt(s² + w²)

Log-eigenvalues:
    A ≡ ln λ₊ = K + ln(c + D)
    B ≡ ln λ₋ = K + ln(c - D)

Finite-size weight:
    σ_N = r^N / (1 + r^N)  with  r = λ₋/λ₊ = exp(B - A)

Master formulas for (1/N) ∂Ψ/∂x and (1/N) ∂²Ψ/∂x∂y:
    Ψ_x   = A_x  + σ_N (B_x - A_x)
    Ψ_xy  = A_xy + σ_N (B_xy - A_xy) + N σ_N (1-σ_N) (B_x - A_x)(B_y - A_y)
"""

import numpy as np


# ===================================================================
# Numerically stable sigmoid (identical to transfer_matrix.py)
# ===================================================================

def _sigmoid(x):
    """Numerically stable sigmoid: 1/(1 + exp(-x))."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ===================================================================
# Layer 1 — Building blocks
# ===================================================================

def _build_intermediates(K, h):
    """Compute reusable intermediates from K, h."""
    c = np.cosh(h)
    s = np.sinh(h)
    w = np.exp(-2.0 * K)
    w2 = w * w                       # exp(-4K)
    s2 = s * s
    D = np.sqrt(s2 + w2)
    # Stable form: c - D = (c² - D²)/(c + D) = (1 - w²)/(c + D)
    # since c² - s² = 1, so c² - D² = 1 - w².
    cpD = c + D                      # c + D (always > 0)
    cmD = (1.0 - w2) / cpD           # c - D (stable)
    return dict(c=c, s=s, w=w, w2=w2, s2=s2, D=D, cpD=cpD, cmD=cmD)


def _sigma_N(K, h, N):
    """σ_N = r^N / (1 + r^N) via sigmoid on ρ = N·ln(r).

    ρ = N·(B - A) = N·ln((c - D)/(c + D)) ≤ 0.
    """
    ib = _build_intermediates(K, h)
    # ρ = N * [ln(c - D) - ln(c + D)]
    rho = N * (np.log(ib['cmD']) - np.log(ib['cpD']))
    return _sigmoid(rho)


# --- First derivatives of ln λ± ---

def _A_h(ib):
    """∂_h ln λ₊ = s / D."""
    return ib['s'] / ib['D']


def _B_h(ib):
    """∂_h ln λ₋ = -s / D."""
    return -ib['s'] / ib['D']


def _A_K(ib):
    """∂_K ln λ₊ = (cD + s² - w²) / [D(c + D)]."""
    c, s2, w2, D, cpD = ib['c'], ib['s2'], ib['w2'], ib['D'], ib['cpD']
    return (c * D + s2 - w2) / (D * cpD)


def _B_K(ib):
    """∂_K ln λ₋ = (cD - s² + w²) / [D(c - D)].

    Rewrite denominator as D·cmD for stability.
    """
    c, s2, w2, D, cmD = ib['c'], ib['s2'], ib['w2'], ib['D'], ib['cmD']
    return (c * D - s2 + w2) / (D * cmD)


# --- Second derivatives of ln λ± ---

def _A_hh(ib):
    """∂²_h ln λ₊ = c w² / D³."""
    return ib['c'] * ib['w2'] / ib['D']**3


def _B_hh(ib):
    """∂²_h ln λ₋ = -c w² / D³."""
    return -ib['c'] * ib['w2'] / ib['D']**3


def _A_Kh(ib):
    """∂²_Kh ln λ₊ = 2 s w² / D³."""
    return 2.0 * ib['s'] * ib['w2'] / ib['D']**3


def _B_Kh(ib):
    """∂²_Kh ln λ₋ = -2 s w² / D³."""
    return -2.0 * ib['s'] * ib['w2'] / ib['D']**3


def _A_KK(ib):
    """∂²_K ln λ₊ = 4 w² [c(2s²+w²) + 2s²D] / [D³(c+D)²]."""
    c, s2, w2, D, cpD = ib['c'], ib['s2'], ib['w2'], ib['D'], ib['cpD']
    num = 4.0 * w2 * (c * (2.0 * s2 + w2) + 2.0 * s2 * D)
    den = D**3 * cpD**2
    return num / den


def _B_KK(ib):
    """∂²_K ln λ₋ = -4 w² [c(2s²+w²) - 2s²D] / [D³(c-D)²]."""
    c, s2, w2, D, cmD = ib['c'], ib['s2'], ib['w2'], ib['D'], ib['cmD']
    num = -4.0 * w2 * (c * (2.0 * s2 + w2) - 2.0 * s2 * D)
    den = D**3 * cmD**2
    return num / den


# ===================================================================
# Layer 2 — Master sigmoid-weighted formulas
# ===================================================================

def _first_deriv(A_x, B_x, sigma):
    """Ψ_x = A_x + σ (B_x - A_x)."""
    return A_x + sigma * (B_x - A_x)


def _second_deriv(A_xy, B_xy, A_x, B_x, A_y, B_y, sigma, N):
    """Ψ_xy = A_xy + σ(B_xy - A_xy) + N σ(1-σ)(B_x - A_x)(B_y - A_y)."""
    return (A_xy + sigma * (B_xy - A_xy)
            + N * sigma * (1.0 - sigma) * (B_x - A_x) * (B_y - A_y))


# ===================================================================
# Layer 3 — Public API (matching transfer_matrix.py signatures)
# ===================================================================

def eigenvalues(K, h):
    """Return (λ₊, λ₋) of the transfer matrix."""
    ib = _build_intermediates(K, h)
    lam_plus = np.exp(K) * ib['cpD']
    lam_minus = np.exp(K) * ib['cmD']
    return lam_plus, lam_minus


def free_energy_per_spin(K, h, N):
    """Dimensionless free energy per spin: βf = -(1/N) ln Z."""
    ib = _build_intermediates(K, h)
    ln_lam_plus = K + np.log(ib['cpD'])
    rho = np.log(ib['cmD']) - np.log(ib['cpD'])
    log_Z = N * ln_lam_plus + np.log1p(np.exp(N * rho))
    return -log_Z / N


def partition_function(K, h, N):
    """Partition function Z for N-site periodic ring."""
    return np.exp(-N * free_energy_per_spin(K, h, N))


def magnetization(K, h, N):
    """⟨m⟩ = (s/D)(1 - 2σ_N).

    Equivalently: Ψ_h = A_h + σ_N(B_h - A_h) since B_h - A_h = -2s/D.
    """
    ib = _build_intermediates(K, h)
    sigma = _sigma_N(K, h, N)
    return _first_deriv(_A_h(ib), _B_h(ib), sigma)


def magnetization_thermo_limit(K, h):
    """⟨m⟩ in N→∞ limit: sinh(h) / sqrt(sinh²(h) + exp(-4K))."""
    ib = _build_intermediates(K, h)
    return _A_h(ib)


def nn_correlation(K, h, N):
    """⟨s_i s_{i+1}⟩ = Ψ_K = A_K + σ_N(B_K - A_K)."""
    ib = _build_intermediates(K, h)
    sigma = _sigma_N(K, h, N)
    return _first_deriv(_A_K(ib), _B_K(ib), sigma)


def susceptibility(K, h, N):
    """χ = (1/N) ∂²ln Z/∂h² (no β prefactor)."""
    ib = _build_intermediates(K, h)
    sigma = _sigma_N(K, h, N)
    return _second_deriv(_A_hh(ib), _B_hh(ib),
                         _A_h(ib), _B_h(ib),
                         _A_h(ib), _B_h(ib),
                         sigma, N)


def specific_heat(K, h, N):
    """C/k_B = K² Ψ_KK + 2Kh Ψ_Kh + h² Ψ_hh."""
    ib = _build_intermediates(K, h)
    sigma = _sigma_N(K, h, N)

    Ah = _A_h(ib);   Bh = _B_h(ib)
    AK = _A_K(ib);   BK = _B_K(ib)

    psi_KK = _second_deriv(_A_KK(ib), _B_KK(ib), AK, BK, AK, BK, sigma, N)
    psi_Kh = _second_deriv(_A_Kh(ib), _B_Kh(ib), AK, BK, Ah, Bh, sigma, N)
    psi_hh = _second_deriv(_A_hh(ib), _B_hh(ib), Ah, Bh, Ah, Bh, sigma, N)

    return K**2 * psi_KK + 2.0 * K * h * psi_Kh + h**2 * psi_hh


def energy_per_spin(K, h, N):
    """Dimensionless energy per spin: βu = -(K⟨ss⟩ + h⟨m⟩)."""
    return -(K * nn_correlation(K, h, N) + h * magnetization(K, h, N))


def entropy_per_spin(K, h, N):
    """Entropy per spin: S/(Nk_B) = Ψ/N + βu = -βf + βu."""
    return -free_energy_per_spin(K, h, N) + energy_per_spin(K, h, N)


# ===================================================================
# Collect all quantities (matching transfer_matrix.py)
# ===================================================================

from dataclasses import dataclass


@dataclass
class IsingExact:
    free_energy: float
    magnetization: float
    magnetization_thermo: float
    nn_correlation: float
    susceptibility: float
    specific_heat: float
    energy: float
    entropy: float
    partition_function: float


def compute_all(K, h, N):
    """Compute all thermodynamic quantities and return IsingExact dataclass."""
    return IsingExact(
        free_energy=free_energy_per_spin(K, h, N),
        magnetization=magnetization(K, h, N),
        magnetization_thermo=magnetization_thermo_limit(K, h),
        nn_correlation=nn_correlation(K, h, N),
        susceptibility=susceptibility(K, h, N),
        specific_heat=specific_heat(K, h, N),
        energy=energy_per_spin(K, h, N),
        entropy=entropy_per_spin(K, h, N),
        partition_function=partition_function(K, h, N),
    )
