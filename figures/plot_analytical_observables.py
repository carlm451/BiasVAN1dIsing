"""Figure A1: All observables vs h at fixed N=16, multiple K values.

3x2 multi-panel: M, βu, χ, C/k_B, S/(Nk_B), ⟨ss⟩.
Smooth curves from analytical_formulas, sparse markers from transfer_matrix.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig
from src.exact import analytical_formulas as af
from src.exact import transfer_matrix as tm

N = 16
K_VALUES = [0.5, 1.0, 2.0, 5.0]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
h_dense = np.linspace(-3.0, 3.0, 301)
h_sparse = np.linspace(-3.0, 3.0, 15)

# Observable functions and labels
PANELS = [
    ('magnetization',   r'$\langle m \rangle$',     af.magnetization,   tm.magnetization),
    ('energy',          r'$\beta u$',                af.energy_per_spin, tm.energy_per_spin),
    ('susceptibility',  r'$\chi$',                   af.susceptibility,  tm.susceptibility),
    ('specific_heat',   r'$C / k_B$',               af.specific_heat,   tm.specific_heat),
    ('entropy',         r'$S / (N k_B)$',           af.entropy_per_spin,tm.entropy_per_spin),
    ('nn_correlation',  r'$\langle s_i s_{i+1} \rangle$', af.nn_correlation, tm.nn_correlation),
]


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.ravel()

    max_errors = {}

    for idx, (name, ylabel, af_fn, tm_fn) in enumerate(PANELS):
        ax = axes[idx]
        panel_max_err = 0.0

        for ki, K in enumerate(K_VALUES):
            # Dense analytical curves
            y_dense = np.array([af_fn(K, h, N) for h in h_dense])
            ax.plot(h_dense, y_dense, color=COLORS[ki], label=f'$K={K}$')

            # Sparse SymPy markers
            y_sparse = np.array([tm_fn(K, h, N) for h in h_sparse])
            ax.plot(h_sparse, y_sparse, 'o', color=COLORS[ki],
                    markersize=4, markerfacecolor='none', markeredgewidth=1.0)

            # Track max relative error
            y_af_check = np.array([af_fn(K, h, N) for h in h_sparse])
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(y_af_check - y_sparse) / (np.abs(y_sparse) + 1e-30)
            panel_max_err = max(panel_max_err, np.nanmax(rel_err))

        max_errors[name] = panel_max_err
        ax.set_xlabel(r'$h = \beta H$')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')

    fig.suptitle(f'Analytical observables vs $h$  ($N = {N}$)', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'figA1_analytical_observables')
    plt.close(fig)

    print("\nMax relative errors (analytical vs SymPy) per panel:")
    for name, err in max_errors.items():
        print(f"  {name:20s}: {err:.2e}")


if __name__ == "__main__":
    main()
