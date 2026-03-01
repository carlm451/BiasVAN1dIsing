"""Figure A3: h=0 special case vs physical temperature T.

2x2 multi-panel: ⟨ss⟩, χ/β, C/k_B, S/(Nk_B) at h=0 vs T = J/(k_B K).
Natural units: J/k_B = 1, so T = 1/K.
Curves for N = 4, 8, 16, 64, 128 and N→∞.  Dashed reference lines for known limits.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig
from src.exact import analytical_formulas as af

N_VALUES = [4, 8, 16, 64, 128]
N_INF = 10000
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
T_arr = np.linspace(0.1, 5.0, 500)
K_arr = 1.0 / T_arr  # K = 1/T in natural units
h = 0.0


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # ---- Panel (0,0): NN correlation ⟨ss⟩ ----
    ax = axes[0, 0]
    for ni, N in enumerate(N_VALUES):
        y = np.array([af.nn_correlation(K, h, N) for K in K_arr])
        ax.plot(T_arr, y, color=COLORS[ni], label=f'$N={N}$')
    y_inf = np.array([af.nn_correlation(K, h, N_INF) for K in K_arr])
    ax.plot(T_arr, y_inf, 'k--', lw=1.2, label=r'$N\to\infty$')
    # Reference: tanh(1/T) = tanh(K)
    ax.plot(T_arr, np.tanh(K_arr), ':', color='gray', lw=1.0, label=r'$\tanh(1/T)$')
    ax.axhline(1.0, color='silver', lw=0.8, ls='--')
    ax.set_ylabel(r'$\langle s_i s_{i+1} \rangle$')
    ax.legend(fontsize=8)

    # ---- Panel (0,1): Susceptibility χ/β (log scale) ----
    ax = axes[0, 1]
    for ni, N in enumerate(N_VALUES):
        y = np.array([af.susceptibility(K, h, N) for K in K_arr])
        ax.plot(T_arr, y, color=COLORS[ni], label=f'$N={N}$')
    y_inf = np.array([af.susceptibility(K, h, N_INF) for K in K_arr])
    ax.plot(T_arr, y_inf, 'k--', lw=1.2, label=r'$N\to\infty$')
    # Reference: thermo limit χ(h=0) = exp(2K)/cosh(0)*w^2/D^3 → exp(2/T)
    ax.plot(T_arr, np.exp(2.0 / T_arr), ':', color='gray', lw=1.0,
            label=r'$e^{2/T}$')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\chi / \beta$')
    ax.legend(fontsize=8)

    # ---- Panel (1,0): Heat capacity C/k_B ----
    ax = axes[1, 0]
    for ni, N in enumerate(N_VALUES):
        y = np.array([af.specific_heat(K, h, N) for K in K_arr])
        ax.plot(T_arr, y, color=COLORS[ni], label=f'$N={N}$')
    y_inf = np.array([af.specific_heat(K, h, N_INF) for K in K_arr])
    ax.plot(T_arr, y_inf, 'k--', lw=1.2, label=r'$N\to\infty$')
    # Reference: (1/T)^2 sech^2(1/T) = K^2 sech^2(K)
    schottky = K_arr**2 / np.cosh(K_arr)**2
    ax.plot(T_arr, schottky, ':', color='gray', lw=1.0,
            label=r'$(1/T)^2 \mathrm{sech}^2(1/T)$')
    ax.set_ylabel(r'$C / k_B$')
    ax.legend(fontsize=8)

    # ---- Panel (1,1): Entropy S/(Nk_B) ----
    ax = axes[1, 1]
    for ni, N in enumerate(N_VALUES):
        y = np.array([af.entropy_per_spin(K, h, N) for K in K_arr])
        ax.plot(T_arr, y, color=COLORS[ni], label=f'$N={N}$')
    y_inf = np.array([af.entropy_per_spin(K, h, N_INF) for K in K_arr])
    ax.plot(T_arr, y_inf, 'k--', lw=1.2, label=r'$N\to\infty$')
    ax.axhline(np.log(2), color='silver', lw=0.8, ls='--', label=r'$\ln 2$')
    ax.set_ylabel(r'$S / (N k_B)$')
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel(r'$T = J / (k_B K)$')

    fig.suptitle(r'$h = 0$ observables vs temperature $T$', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'figA3_analytical_h0')
    plt.close(fig)


if __name__ == "__main__":
    main()
