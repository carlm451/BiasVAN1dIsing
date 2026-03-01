"""Figure A5: Finite-size scaling collapse at K=2.0.

2x2 multi-panel: M, chi/N, (C/k_B)/N, S/(Nk_B) vs scaled field x_h = h*N.
The 1D Ising model has T_c=0 with nu=1, beta_crit=0, gamma=1.
The field scaling variable is x_h = h*N (since beta_crit*delta/nu = 1).
At K=2.0, the correlation length xi ~ 1/ln(coth(2)) ~ 27.3.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig
from src.exact import analytical_formulas as af

K = 2.0
N_VALUES = [4, 8, 16, 32, 64, 128]
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
XH_RANGE = (-20.0, 20.0)
N_POINTS = 401


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    # --- Panel (a): Magnetization ---
    ax = axes[0]
    for ni, N in enumerate(N_VALUES):
        h_arr = np.linspace(XH_RANGE[0] / N, XH_RANGE[1] / N, N_POINTS)
        xh = h_arr * N
        y = np.array([af.magnetization(K, h, N) for h in h_arr])
        ax.plot(xh, y, color=COLORS[ni], label=f'$N={N}$')

    # tanh(hN) reference — the superparamagnetic (N << xi) limit
    xh_ref = np.linspace(XH_RANGE[0], XH_RANGE[1], N_POINTS)
    ax.plot(xh_ref, np.tanh(xh_ref), 'k--', lw=1.0, label=r'$\tanh(hN)$')

    ax.set_xlabel(r'$x_h = hN$')
    ax.set_ylabel(r'$\langle m \rangle$')
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # --- Panel (b): Susceptibility / N ---
    ax = axes[1]
    for ni, N in enumerate(N_VALUES):
        h_arr = np.linspace(XH_RANGE[0] / N, XH_RANGE[1] / N, N_POINTS)
        xh = h_arr * N
        # af.susceptibility returns chi_bare (no beta prefactor)
        # chi = beta * chi_bare; here we plot chi/N = beta * chi_bare / N
        # but beta = 1/(k_B T) and K = beta J, so beta = K/J
        # For scaling collapse we just plot chi_bare / N (the bare second deriv / N)
        # which is the natural dimensionless quantity
        y = np.array([af.susceptibility(K, h, N) / N for h in h_arr])
        ax.plot(xh, y, color=COLORS[ni], label=f'$N={N}$')

    ax.set_xlabel(r'$x_h = hN$')
    ax.set_ylabel(r'$\chi_{\mathrm{bare}} / N$')
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # --- Panel (c): Heat capacity / N ---
    ax = axes[2]
    for ni, N in enumerate(N_VALUES):
        h_arr = np.linspace(XH_RANGE[0] / N, XH_RANGE[1] / N, N_POINTS)
        xh = h_arr * N
        y = np.array([af.specific_heat(K, h, N) / N for h in h_arr])
        ax.plot(xh, y, color=COLORS[ni], label=f'$N={N}$')

    ax.set_xlabel(r'$x_h = hN$')
    ax.set_ylabel(r'$(C/k_B) / N$')
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # --- Panel (d): Entropy per spin ---
    ax = axes[3]
    for ni, N in enumerate(N_VALUES):
        h_arr = np.linspace(XH_RANGE[0] / N, XH_RANGE[1] / N, N_POINTS)
        xh = h_arr * N
        y = np.array([af.entropy_per_spin(K, h, N) for h in h_arr])
        ax.plot(xh, y, color=COLORS[ni], label=f'$N={N}$')

    ax.set_xlabel(r'$x_h = hN$')
    ax.set_ylabel(r'$S / (N k_B)$')
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    fig.suptitle(
        r'Finite-size scaling collapse  ($K = 2.0$,  $\xi \approx 27$)',
        fontsize=14, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'figA5_scaling_collapse')
    plt.close(fig)


if __name__ == "__main__":
    main()
