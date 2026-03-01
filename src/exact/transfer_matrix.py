"""
Exact solution of the 1D periodic Ising model via the transfer matrix method.

Dimensionless parameterization: K = beta*J (coupling), h = beta*H (field).

Transfer matrix for H = -J sum_i s_i s_{i+1} - H sum_i s_i  (PBC):

    P = [[exp(K + h), exp(-K)     ],
         [exp(-K),    exp(K - h)  ]]

Eigenvalues:
    lambda_pm = exp(K) * [cosh(h) +/- sqrt(sinh^2(h) + exp(-4K))]

Log-eigenvalues (the fundamental building blocks):
    ln lam_+ = K + ln(cosh(h) + sqrt(sinh^2(h) + exp(-4K)))
    ln lam_- = K + ln(cosh(h) - sqrt(sinh^2(h) + exp(-4K)))

All thermodynamic observables are expressed analytically via SymPy-derived
derivatives of ln(lam_pm) w.r.t. K and h, assembled with numerically stable
sigmoid weighting for finite N.

All functions use float64 for numerical precision.
"""

from dataclasses import dataclass
import numpy as np
import sympy as sp

# ===================================================================
# SymPy symbolic setup (runs once at import time)
# ===================================================================

_K_sym, _h_sym = sp.symbols('K h', real=True)

# Discriminant
_disc_sym = sp.sqrt(sp.sinh(_h_sym)**2 + sp.exp(-4 * _K_sym))

# Log-eigenvalues
_ln_lam_plus_sym = _K_sym + sp.log(sp.cosh(_h_sym) + _disc_sym)
_ln_lam_minus_sym = _K_sym + sp.log(sp.cosh(_h_sym) - _disc_sym)

# --- 12 derivatives (building blocks) ---
# First derivatives w.r.t. h
_A1_h_sym = sp.diff(_ln_lam_plus_sym, _h_sym)
_B1_h_sym = sp.diff(_ln_lam_minus_sym, _h_sym)

# First derivatives w.r.t. K
_A1_K_sym = sp.diff(_ln_lam_plus_sym, _K_sym)
_B1_K_sym = sp.diff(_ln_lam_minus_sym, _K_sym)

# Second derivatives w.r.t. h
_A2_hh_sym = sp.diff(_ln_lam_plus_sym, _h_sym, 2)
_B2_hh_sym = sp.diff(_ln_lam_minus_sym, _h_sym, 2)

# Second derivatives w.r.t. K
_A2_KK_sym = sp.diff(_ln_lam_plus_sym, _K_sym, 2)
_B2_KK_sym = sp.diff(_ln_lam_minus_sym, _K_sym, 2)

# Mixed second derivatives
_A2_Kh_sym = sp.diff(_ln_lam_plus_sym, _K_sym, _h_sym)
_B2_Kh_sym = sp.diff(_ln_lam_minus_sym, _K_sym, _h_sym)

# --- Lambdify all 14 functions (2 log-eigenvalues + 12 derivatives) ---
_args = (_K_sym, _h_sym)
_modules = ['numpy']

_ln_lam_plus = sp.lambdify(_args, _ln_lam_plus_sym, modules=_modules)
_ln_lam_minus = sp.lambdify(_args, _ln_lam_minus_sym, modules=_modules)

_A1_h_fn = sp.lambdify(_args, _A1_h_sym, modules=_modules)
_B1_h_fn = sp.lambdify(_args, _B1_h_sym, modules=_modules)

_A1_K_fn = sp.lambdify(_args, _A1_K_sym, modules=_modules)
_B1_K_fn = sp.lambdify(_args, _B1_K_sym, modules=_modules)

_A2_hh_fn = sp.lambdify(_args, _A2_hh_sym, modules=_modules)
_B2_hh_fn = sp.lambdify(_args, _B2_hh_sym, modules=_modules)

_A2_KK_fn = sp.lambdify(_args, _A2_KK_sym, modules=_modules)
_B2_KK_fn = sp.lambdify(_args, _B2_KK_sym, modules=_modules)

_A2_Kh_fn = sp.lambdify(_args, _A2_Kh_sym, modules=_modules)
_B2_Kh_fn = sp.lambdify(_args, _B2_Kh_sym, modules=_modules)


# ===================================================================
# Numerically stable helpers
# ===================================================================

def _sigmoid(x):
    """Numerically stable sigmoid: 1/(1 + exp(-x))."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _compute_rho(K, h):
    """rho = ln(lam_-) - ln(lam_+) <= 0."""
    return _ln_lam_minus(K, h) - _ln_lam_plus(K, h)


def _compute_S(K, h, N):
    """S = sigmoid(N * rho) where rho = ln(lam_-/lam_+) <= 0.

    S -> 0 as N -> inf (thermodynamic limit).
    S = fraction of Psi coming from lam_- eigenvalue.
    """
    rho = _compute_rho(K, h)
    return _sigmoid(N * rho)


# ===================================================================
# Public API
# ===================================================================

def eigenvalues(K, h):
    """Return (lambda_plus, lambda_minus) of the transfer matrix.

    Parameters
    ----------
    K : float
        Dimensionless coupling beta*J.
    h : float
        Dimensionless field beta*H.
    """
    return np.exp(_ln_lam_plus(K, h)), np.exp(_ln_lam_minus(K, h))


def _log_partition_function(K, h, N):
    """ln Z for N-site periodic ring.

    Uses log-sum-exp: ln Z = N*ln(lam_+) + ln(1 + exp(N*rho))
    where rho = ln(lam_-) - ln(lam_+) <= 0.
    """
    ln_lp = N * _ln_lam_plus(K, h)
    rho = _compute_rho(K, h)
    return ln_lp + np.log1p(np.exp(N * rho))


def partition_function(K, h, N):
    """Partition function Z for N-site periodic ring."""
    return np.exp(_log_partition_function(K, h, N))


def free_energy_per_spin(K, h, N):
    """Dimensionless free energy per spin: beta*f = -(1/N)*ln Z."""
    return -_log_partition_function(K, h, N) / N


def magnetization(K, h, N):
    """<m> = (1/N) * d(ln Z)/dh, computed analytically.

    m = A1_h + S*(B1_h - A1_h) + N*S*(1-S)*(B1_h - A1_h) * drho/dh ... NO.
    Actually Psi = ln Z = N*ln(lam_+) + ln(1 + r^N) where r = lam_-/lam_+.
    dPsi/dh = N*A1_h + N*S*(B1_h - A1_h)
    so m = (1/N)*dPsi/dh = A1_h + S*(B1_h - A1_h).
    """
    A1_h = _A1_h_fn(K, h)
    B1_h = _B1_h_fn(K, h)
    S = _compute_S(K, h, N)
    return A1_h + S * (B1_h - A1_h)


def magnetization_thermo_limit(K, h):
    """<m> in the N -> infinity limit (S -> 0).

    m = d(ln lam_+)/dh = sinh(h) / sqrt(sinh^2(h) + exp(-4K)).
    """
    return _A1_h_fn(K, h)


def nn_correlation(K, h, N):
    """<s_i s_{i+1}> = (1/N) * d(ln Z)/dK, computed analytically.

    Same sigmoid-weighted pattern as magnetization but w.r.t. K.
    """
    A1_K = _A1_K_fn(K, h)
    B1_K = _B1_K_fn(K, h)
    S = _compute_S(K, h, N)
    return A1_K + S * (B1_K - A1_K)


def susceptibility(K, h, N):
    """chi = (1/N) * d^2(ln Z)/dh^2, computed analytically.

    chi = A2_hh + S*(B2_hh - A2_hh) + N*S*(1-S)*(B1_h - A1_h)^2
    """
    A1_h = _A1_h_fn(K, h)
    B1_h = _B1_h_fn(K, h)
    A2_hh = _A2_hh_fn(K, h)
    B2_hh = _B2_hh_fn(K, h)
    S = _compute_S(K, h, N)
    delta_h = B1_h - A1_h
    return A2_hh + S * (B2_hh - A2_hh) + N * S * (1.0 - S) * delta_h**2


def specific_heat(K, h, N):
    """Dimensionless specific heat c_v / k_B, computed analytically.

    Since K = beta*J and h = beta*H, temperature derivatives at fixed J, H give:
    c_v/k_B = K^2 * (1/N)*d2Psi/dK2 + 2*K*h * (1/N)*d2Psi/dKdh + h^2 * (1/N)*d2Psi/dh2

    Each (1/N)*d2Psi/dx dy uses the same sigmoid-weighted pattern:
    (1/N)*d2Psi/dxdy = A2_xy + S*(B2_xy - A2_xy) + N*S*(1-S)*(B1_x - A1_x)*(B1_y - A1_y)
    """
    S = _compute_S(K, h, N)

    A1_K = _A1_K_fn(K, h)
    B1_K = _B1_K_fn(K, h)
    A1_h = _A1_h_fn(K, h)
    B1_h = _B1_h_fn(K, h)

    A2_KK = _A2_KK_fn(K, h)
    B2_KK = _B2_KK_fn(K, h)
    A2_hh = _A2_hh_fn(K, h)
    B2_hh = _B2_hh_fn(K, h)
    A2_Kh = _A2_Kh_fn(K, h)
    B2_Kh = _B2_Kh_fn(K, h)

    delta_K = B1_K - A1_K
    delta_h = B1_h - A1_h

    # (1/N) * d2 Psi / dK2
    psi_KK = A2_KK + S * (B2_KK - A2_KK) + N * S * (1.0 - S) * delta_K**2
    # (1/N) * d2 Psi / dh2
    psi_hh = A2_hh + S * (B2_hh - A2_hh) + N * S * (1.0 - S) * delta_h**2
    # (1/N) * d2 Psi / dKdh
    psi_Kh = A2_Kh + S * (B2_Kh - A2_Kh) + N * S * (1.0 - S) * delta_K * delta_h

    return K**2 * psi_KK + 2.0 * K * h * psi_Kh + h**2 * psi_hh


def energy_per_spin(K, h, N):
    """Dimensionless energy per spin: beta*u = -(K*<ss> + h*<m>)."""
    ss = nn_correlation(K, h, N)
    m = magnetization(K, h, N)
    return -(K * ss + h * m)


def entropy_per_spin(K, h, N):
    """Dimensionless entropy per spin: s/k_B = Psi/N + beta*u.

    Where Psi = ln Z and beta*u = energy_per_spin(K, h, N).
    """
    psi_per_spin = -free_energy_per_spin(K, h, N)  # Psi/N = ln(Z)/N
    beta_u = energy_per_spin(K, h, N)
    return psi_per_spin + beta_u


# ===================================================================
# Collect all quantities
# ===================================================================

@dataclass
class IsingExact:
    free_energy: float        # beta*f = -(1/N)*ln Z (dimensionless)
    magnetization: float
    magnetization_thermo: float
    nn_correlation: float
    susceptibility: float
    specific_heat: float
    energy: float             # beta*u (dimensionless)
    entropy: float            # s/k_B (dimensionless)
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


# ===================================================================
# Vectorized grid computation
# ===================================================================

def compute_grid(Ks, hs, N):
    """
    Compute exact quantities on a (K, h) grid.

    Parameters
    ----------
    Ks : 1D array of dimensionless couplings (beta*J)
    hs : 1D array of dimensionless fields (beta*H)
    N : system size

    Returns
    -------
    dict of 2D arrays with shape (len(Ks), len(hs)):
        'free_energy', 'magnetization', 'magnetization_thermo',
        'nn_correlation', 'susceptibility', 'specific_heat',
        'energy', 'entropy', 'partition_function'
    """
    Ks = np.asarray(Ks, dtype=np.float64)
    hs = np.asarray(hs, dtype=np.float64)
    nK, nh = len(Ks), len(hs)

    results = {
        'free_energy': np.empty((nK, nh)),
        'magnetization': np.empty((nK, nh)),
        'magnetization_thermo': np.empty((nK, nh)),
        'nn_correlation': np.empty((nK, nh)),
        'susceptibility': np.empty((nK, nh)),
        'specific_heat': np.empty((nK, nh)),
        'energy': np.empty((nK, nh)),
        'entropy': np.empty((nK, nh)),
        'partition_function': np.empty((nK, nh)),
    }

    for i, K in enumerate(Ks):
        for j, hval in enumerate(hs):
            r = compute_all(K, hval, N)
            results['free_energy'][i, j] = r.free_energy
            results['magnetization'][i, j] = r.magnetization
            results['magnetization_thermo'][i, j] = r.magnetization_thermo
            results['nn_correlation'][i, j] = r.nn_correlation
            results['susceptibility'][i, j] = r.susceptibility
            results['specific_heat'][i, j] = r.specific_heat
            results['energy'][i, j] = r.energy
            results['entropy'][i, j] = r.entropy
            results['partition_function'][i, j] = r.partition_function

    return results
