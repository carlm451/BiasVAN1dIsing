"""Figure 2: Exact <m>(K, h) heatmap from transfer matrix."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR


def plot_from_data(data_path):
    data = np.load(data_path)
    K_grid = data['K_grid']
    h_grid = data['h_grid']
    exact_m = data['exact_m']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(h_grid, K_grid, exact_m,
                       cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    ax.set_xlabel('$h = \\beta H$')
    ax.set_ylabel('$K = \\beta J$')
    ax.set_title('Exact magnetization $\\langle m \\rangle (K, h)$')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='$\\langle m \\rangle$')
    savefig(fig, 'fig2_exact_phase_diagram')
    plt.close(fig)


def plot_from_computation():
    """Compute directly if no sweep data available."""
    from src.exact.transfer_matrix import magnetization_thermo_limit

    K_grid = np.logspace(-2, 1, 120)  # [0.01, 10.0], 3 decades
    h_grid = np.linspace(-2.0, 2.0, 101)

    m = np.zeros((len(K_grid), len(h_grid)))
    for i, K in enumerate(K_grid):
        for j, h in enumerate(h_grid):
            m[i, j] = magnetization_thermo_limit(K, h)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(h_grid, K_grid, m,
                       cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    ax.set_xlabel('$h = \\beta H$')
    ax.set_ylabel('$K = \\beta J$')
    ax.set_title('Exact magnetization $\\langle m \\rangle (K, h)$ [thermo limit]')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='$\\langle m \\rangle$')
    savefig(fig, 'fig2_exact_phase_diagram')
    plt.close(fig)


if __name__ == "__main__":
    # Try loading from sweep data, fallback to direct computation
    sweep_path = os.path.join(RESULTS_DIR, "sweep_Kh_N16.npz")
    if os.path.exists(sweep_path):
        plot_from_data(sweep_path)
    else:
        plot_from_computation()
