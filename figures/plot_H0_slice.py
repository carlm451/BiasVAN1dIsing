"""Figure 5: deltaF/N vs K at h=0 for multiple N."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR


def plot(system_sizes=None):
    if system_sizes is None:
        # Auto-detect available sweep files
        import glob
        pattern = os.path.join(RESULTS_DIR, "sweep_Kh_N*.npz")
        files = glob.glob(pattern)
        system_sizes = sorted(
            int(os.path.basename(f).replace("sweep_Kh_N", "").replace(".npz", ""))
            for f in files
        )
        if not system_sizes:
            print("No sweep_Kh_N*.npz files found in results/")
            return

    fig, ax = plt.subplots(figsize=(8, 5))

    for N in system_sizes:
        data_path = os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz")
        if not os.path.exists(data_path):
            print(f"Skipping N={N}: {data_path} not found")
            continue

        data = np.load(data_path)
        K_grid = data['K_grid']
        h_grid = data['h_grid']

        # Find h=0 index
        h0_idx = np.argmin(np.abs(h_grid))
        delta_f = data['delta_f_van_bias'][:, h0_idx]

        ax.semilogy(K_grid, np.abs(delta_f), 'o-', label=f'N={N}', markersize=3)

    ax.set_xlabel('$K = \\beta J$')
    ax.set_ylabel('$|\\Delta F / N|$')
    ax.set_title('VAN free energy error at $h=0$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(fig, 'fig5_H0_slice')
    plt.close(fig)


if __name__ == "__main__":
    plot()
