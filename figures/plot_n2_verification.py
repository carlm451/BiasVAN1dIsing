"""Figure 8: log10|deltaF| heatmap for N=2 verification."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR


def plot():
    data_path = os.path.join(RESULTS_DIR, "n2_verification.npz")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run experiments/n2_exact.py first.")
        return

    data = np.load(data_path)
    K_grid = data['K_grid']
    h_grid = data['h_grid']
    delta_f = data['delta_f']

    log_delta = np.log10(np.abs(delta_f) + 1e-16)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(h_grid, K_grid, log_delta,
                       cmap='viridis', shading='auto')
    ax.set_xlabel('$h = \\beta H$')
    ax.set_ylabel('$K = \\beta J$')
    ax.set_title('N=2 VAN: $\\log_{10} |\\Delta F|$')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label='$\\log_{10} |\\Delta F|$')

    savefig(fig, 'fig8_n2_verification')
    plt.close(fig)


if __name__ == "__main__":
    plot()
