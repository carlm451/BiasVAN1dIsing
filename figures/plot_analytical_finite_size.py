"""Figure A2: Finite-size effects at K=2.0.

2x2 multi-panel: M, χ, C/k_B, S/(Nk_B) for N = 4, 8, 16, 32, 64, 128 and N→∞.
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
N_INF = 10000  # approximate N→∞
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
h_arr = np.linspace(-3.0, 3.0, 301)

PANELS = [
    (r'$\langle m \rangle$',     af.magnetization),
    (r'$\chi$',                   af.susceptibility),
    (r'$C / k_B$',               af.specific_heat),
    (r'$S / (N k_B)$',           af.entropy_per_spin),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for idx, (ylabel, fn) in enumerate(PANELS):
        ax = axes[idx]

        for ni, N in enumerate(N_VALUES):
            y = np.array([fn(K, h, N) for h in h_arr])
            ax.plot(h_arr, y, color=COLORS[ni], label=f'$N={N}$')

        # N→∞ limit (dashed)
        y_inf = np.array([fn(K, h, N_INF) for h in h_arr])
        ax.plot(h_arr, y_inf, 'k--', lw=1.2, label=r'$N\to\infty$')

        ax.set_xlabel(r'$h = \beta H$')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')

    fig.suptitle(f'Finite-size effects  ($K = {K}$)', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'figA2_analytical_finite_size')
    plt.close(fig)


if __name__ == "__main__":
    main()
