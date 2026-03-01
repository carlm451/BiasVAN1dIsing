"""
Naive Mean Field (NMF) self-consistency for the 1D periodic Ising model.

Dimensionless parameterization: K = beta*J, h = beta*H.

For coordination z=2 (1D ring), the self-consistency equation is:
    m = tanh(2*K*m + h)

The dimensionless NMF free energy per spin is:
    beta*f = -K*m^2 - h*m - S(m)
where S(m) is the binary entropy in the magnetization variable:
    S(m) = -[(1+m)/2 * ln((1+m)/2) + (1-m)/2 * ln((1-m)/2)]
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class NMFResult:
    magnetization: float
    free_energy_per_spin: float   # dimensionless beta*f
    bias: float  # atanh(m)
    converged: bool


def _binary_entropy(m):
    """Binary entropy in magnetization variable: S(m) = -[(1+m)/2 ln((1+m)/2) + (1-m)/2 ln((1-m)/2)]."""
    m = np.clip(m, -1.0 + 1e-15, 1.0 - 1e-15)
    p = (1.0 + m) / 2.0
    q = (1.0 - m) / 2.0
    S = 0.0
    if p > 0:
        S -= p * np.log(p)
    if q > 0:
        S -= q * np.log(q)
    return S


def free_energy_per_spin(m, K, h):
    """Dimensionless NMF free energy per spin: beta*f = -K*m^2 - h*m - S(m)."""
    S = _binary_entropy(m)
    return -K * m**2 - h * m - S


def solve(K, h, max_iter=1000, tol=1e-12, alpha=0.5):
    """
    Solve NMF self-consistency equation m = tanh(2*K*m + h)
    using damped fixed-point iteration.

    Parameters
    ----------
    K : float
        Dimensionless coupling beta*J.
    h : float
        Dimensionless field beta*H.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on |m_new - m_old|.
    alpha : float
        Damping factor (0 < alpha <= 1). m_{n+1} = alpha*tanh(...) + (1-alpha)*m_n.

    Returns
    -------
    NMFResult
    """
    results = []

    # Try multiple initial conditions
    if h == 0.0:
        inits = [0.0, 0.9, -0.9]
    else:
        inits = [0.0, np.sign(h) * 0.9]

    for m0 in inits:
        m = m0
        converged = False
        for _ in range(max_iter):
            m_new = np.tanh(2.0 * K * m + h)
            m_update = alpha * m_new + (1.0 - alpha) * m
            if abs(m_update - m) < tol:
                converged = True
                m = m_update
                break
            m = m_update

        f = free_energy_per_spin(m, K, h)
        b = np.arctanh(np.clip(m, -1.0 + 1e-15, 1.0 - 1e-15))
        results.append(NMFResult(
            magnetization=m,
            free_energy_per_spin=f,
            bias=b,
            converged=converged,
        ))

    # Return result with lowest free energy
    best = min(results, key=lambda r: r.free_energy_per_spin)
    return best


def nn_correlation_nmf(K, h):
    """NMF nearest-neighbor correlation: <s_i s_{i+1}> = m^2.

    Mean-field factorization: <s_i s_{i+1}> = <s_i><s_{i+1}> = m^2.
    """
    result = solve(K, h)
    return result.magnetization ** 2


def susceptibility_nmf(K, h):
    """NMF bare susceptibility: chi_bare = dm/dh.

    From the self-consistency m = tanh(2Km + h), differentiating:
        dm/dh = (1 - m^2) / (1 - 2K(1 - m^2))

    Diverges at K_c = 0.5 when h=0 (spurious NMF critical point).
    """
    result = solve(K, h)
    m = result.magnetization
    m2 = m ** 2
    denom = 1.0 - 2.0 * K * (1.0 - m2)
    if abs(denom) < 1e-15:
        return np.inf
    return (1.0 - m2) / denom


def specific_heat_nmf(K, h, dK=1e-5):
    """NMF specific heat C/k_B via numerical second derivative.

    C/k_B = -d^2(beta*f) / d(beta)^2, computed numerically.
    Using chain rule: d/d(beta) = J d/dK + H d/dh, and since
    beta*f depends on K and h which are beta*J and beta*H:

    C/k_B = K^2 d^2(beta*f)/dK^2 + 2*K*h d^2(beta*f)/dKdh + h^2 d^2(beta*f)/dh^2

    But for NMF, beta*f = -K*m^2 - h*m - S(m), where m=m(K,h).
    We use numerical derivatives for simplicity.
    """
    dh = dK  # same step size for h derivatives

    def bf(Kv, hv):
        r = solve(Kv, hv)
        return r.free_energy_per_spin

    # d^2/dK^2
    f_pp = bf(K + dK, h)
    f_mm = bf(K - dK, h)
    f_00 = bf(K, h)
    d2f_dK2 = (f_pp - 2.0 * f_00 + f_mm) / dK**2

    # d^2/dh^2
    f_hp = bf(K, h + dh)
    f_hm = bf(K, h - dh)
    d2f_dh2 = (f_hp - 2.0 * f_00 + f_hm) / dh**2

    # d^2/dKdh (mixed)
    f_KpHp = bf(K + dK, h + dh)
    f_KpHm = bf(K + dK, h - dh)
    f_KmHp = bf(K - dK, h + dh)
    f_KmHm = bf(K - dK, h - dh)
    d2f_dKdh = (f_KpHp - f_KpHm - f_KmHp + f_KmHm) / (4.0 * dK * dh)

    # C/k_B = K^2 * d2(-bf)/dK^2 + 2Kh * d2(-bf)/dKdh + h^2 * d2(-bf)/dh^2
    # Since bf = beta*f, and C/k_B = -d^2(beta*f)/d(beta)^2 using chain rule:
    return K**2 * (-d2f_dK2) + 2.0 * K * h * (-d2f_dKdh) + h**2 * (-d2f_dh2)


def energy_per_spin_nmf(K, h):
    """NMF energy per spin: beta*u = -K*m^2 - h*m (from variational ansatz)."""
    result = solve(K, h)
    m = result.magnetization
    return -K * m**2 - h * m


def entropy_per_spin_nmf(K, h):
    """NMF entropy per spin: S/(Nk_B) = -beta*f + beta*u = S(m)."""
    result = solve(K, h)
    m = result.magnetization
    return _binary_entropy(m)
