"""Figure 4: Side-by-side no-bias vs with-bias VAN deltaF heatmap, log10 scale."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR


def plot(N=16):
    data_path = os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run experiments/sweep_TH.py first.")
        return

    data = np.load(data_path)
    K_grid = data['K_grid']
    h_grid = data['h_grid']
    delta_f_nobias = np.abs(data['delta_f_van_nobias'])
    delta_f_bias = np.abs(data['delta_f_van_bias'])

    # Use log10 for color mapping; floor at 1e-16 to avoid log(0)
    log_nobias = np.log10(np.maximum(delta_f_nobias, 1e-16))
    log_bias = np.log10(np.maximum(delta_f_bias, 1e-16))

    # Shared color range across both panels
    vmin = min(log_nobias.min(), log_bias.min())
    vmax = max(log_nobias.max(), log_bias.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    im1 = ax1.pcolormesh(h_grid, K_grid, log_nobias,
                         cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_xlabel('$h = \\beta H$')
    ax1.set_ylabel('$K = \\beta J$')
    ax1.set_title('No bias: $\\log_{10}|\\Delta F|$')
    ax1.set_yscale('log')
    plt.colorbar(im1, ax=ax1, label='$\\log_{10}|\\Delta F|$')

    im2 = ax2.pcolormesh(h_grid, K_grid, log_bias,
                         cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_xlabel('$h = \\beta H$')
    ax2.set_title('With bias: $\\log_{10}|\\Delta F|$')
    ax2.set_yscale('log')
    plt.colorbar(im2, ax=ax2, label='$\\log_{10}|\\Delta F|$')

    fig.suptitle(f'Bias ablation: VAN without vs with bias (N={N})', fontsize=14)
    plt.tight_layout()
    savefig(fig, 'fig4_bias_ablation_heatmap')
    plt.close(fig)


if __name__ == "__main__":
    plot()
