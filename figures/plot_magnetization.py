"""Figure 6: m vs h at K=2.0, comparing exact/VAN/NMF."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR


def plot(N=16, K_target=2.0):
    data_path = os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run experiments/sweep_TH.py first.")
        return

    data = np.load(data_path)
    K_grid = data['K_grid']
    h_grid = data['h_grid']

    # Find nearest K index
    K_idx = np.argmin(np.abs(K_grid - K_target))
    K_actual = K_grid[K_idx]

    exact_m = data['exact_m'][K_idx, :]
    nmf_m = data['nmf_m'][K_idx, :]

    # Zoom into h in [-0.5, 0.5] to show the crossover region clearly
    mask = (h_grid >= -0.5) & (h_grid <= 0.5)
    h_zoom = h_grid[mask]
    exact_zoom = exact_m[mask]
    nmf_zoom = nmf_m[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h_zoom, exact_zoom, 'k-', linewidth=2, label='Exact')
    ax.plot(h_zoom, exact_zoom, 'ko', markersize=5)
    ax.plot(h_zoom, nmf_zoom, 'r--', linewidth=1.5, label='NMF')
    ax.plot(h_zoom, nmf_zoom, 'rs', markersize=5)
    ax.set_xlabel('$h = \\beta H$')
    ax.set_ylabel('$\\langle m \\rangle$')
    ax.set_title(f'Magnetization vs $h$ at $K={K_actual:.2f}$, N={N}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    savefig(fig, 'fig6_magnetization')
    plt.close(fig)


if __name__ == "__main__":
    plot()
