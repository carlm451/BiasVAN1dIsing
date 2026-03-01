"""Figure A4: Exact observable heatmaps over the (K, h) plane.

3x2 multi-panel pcolormesh: M, βu, χ, C/k_B, S/(Nk_B), ⟨ss⟩.
Matches the style of fig2_exact_phase_diagram (log K axis, h on x).
Uses the analytical_formulas module (pure numpy, no SymPy).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from figures.style import savefig
from src.exact import analytical_formulas as af

N = 16
NK = 120
NH = 151
K_grid = np.logspace(-2, np.log10(10.0), NK)
h_grid = np.linspace(-3.0, 3.0, NH)


def compute_grid(fn):
    """Evaluate fn(K, h, N) on the 2D grid."""
    Z = np.empty((NK, NH))
    for i, K in enumerate(K_grid):
        for j, h in enumerate(h_grid):
            Z[i, j] = fn(K, h, N)
    return Z


PANELS = [
    ('magnetization',  r'$\langle m \rangle$',              af.magnetization,   'RdBu_r',  dict(vmin=-1, vmax=1)),
    ('energy',         r'$\beta u$',                        af.energy_per_spin, 'viridis_r', dict()),
    ('susceptibility', r'$\chi$',                           af.susceptibility,  'inferno',  dict(norm=LogNorm(vmin=0.1, vmax=1e3))),
    ('specific_heat',  r'$C / k_B$',                       af.specific_heat,   'magma',    dict(vmin=0)),
    ('entropy',        r'$S / (N k_B)$',                   af.entropy_per_spin,'cividis',  dict(vmin=0, vmax=np.log(2))),
    ('nn_correlation', r'$\langle s_i s_{i+1} \rangle$',   af.nn_correlation,  'RdBu_r',   dict(vmin=-1, vmax=1)),
]


def main():
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    axes = axes.ravel()

    for idx, (name, label, fn, cmap, kw) in enumerate(PANELS):
        ax = axes[idx]
        Z = compute_grid(fn)

        # Clip tiny negatives for log-norm panels
        if 'norm' in kw and isinstance(kw['norm'], LogNorm):
            Z = np.clip(Z, kw['norm'].vmin, None)

        im = ax.pcolormesh(h_grid, K_grid, Z, cmap=cmap, shading='auto', **kw)
        ax.set_yscale('log')
        ax.set_xlabel(r'$h = \beta H$')
        ax.set_ylabel(r'$K = \beta J$')
        plt.colorbar(im, ax=ax, label=label)

    fig.suptitle(f'Exact observables in the $(K, h)$ plane  ($N = {N}$)',
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'figA4_analytical_heatmaps')
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
