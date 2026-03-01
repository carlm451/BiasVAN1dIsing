"""Figure 7: deltaF/N vs 1/N log-log plot.

Supports extended N range (N=2..64+) and adds NMF as horizontal
dashed reference lines.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR
from src.nmf.mean_field import solve as nmf_solve
from src.exact.transfer_matrix import free_energy_per_spin


def plot():
    data_path = os.path.join(RESULTS_DIR, "finite_size.npz")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run experiments/finite_size.py first.")
        return

    data = np.load(data_path)
    system_sizes = data['system_sizes']
    rep_points = data['representative_points']

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = plt.cm.tab10(np.linspace(0, 1, len(rep_points)))

    for idx, (K, h) in enumerate(rep_points):
        key = f"K{K}_h{h}"
        delta_f_mean = data[f"{key}_delta_f_mean"]
        delta_f_std = data[f"{key}_delta_f_std"]

        # Only plot positive delta_f
        mask = delta_f_mean > 0
        if np.any(mask):
            ax.errorbar(1.0 / system_sizes[mask], delta_f_mean[mask],
                        yerr=delta_f_std[mask],
                        marker=markers[idx % len(markers)],
                        color=colors[idx],
                        label=f'K={K}, h={h}', capsize=3)

        # NMF horizontal reference: deltaF/N is N-independent for NMF
        nmf_result = nmf_solve(K, h)
        # Use a large N for exact reference (NMF is N-independent)
        exact_f = free_energy_per_spin(K, h, 1000)
        nmf_delta = nmf_result.free_energy_per_spin - exact_f
        if nmf_delta > 0:
            ax.axhline(nmf_delta, color=colors[idx], linestyle='--',
                       alpha=0.3, lw=0.8)

    # Reference line: 1/N scaling (match data range)
    x_ref = np.array([1.0 / system_sizes.max(), 1.0 / system_sizes.min()])
    ax.plot(x_ref, 0.1 * x_ref, 'k--', alpha=0.5, label='$\\propto 1/N$')

    # Legend entry for NMF reference
    ax.plot([], [], 'k--', alpha=0.3, lw=0.8, label='NMF $\\Delta F/N$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('1/N')
    ax.set_ylabel('$\\Delta F / N$')
    ax.set_title('Finite-size scaling of VAN error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    savefig(fig, 'fig7_finite_size')
    plt.close(fig)


if __name__ == "__main__":
    plot()
