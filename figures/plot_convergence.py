"""Figure 10: beta*F_q vs training step (convergence curves).

Shows VAN convergence with exact and NMF free energy reference lines.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig
from src.van.train import train, TrainConfig
from src.exact.transfer_matrix import free_energy_per_spin
from src.nmf.mean_field import solve as nmf_solve


def plot(N=16, test_points=None):
    if test_points is None:
        test_points = [
            (1.0, 0.0, 'K=1, h=0'),
            (1.0, 0.5, 'K=1, h=0.5'),
            (0.5, 0.0, 'K=0.5, h=0'),
            (2.0, 0.0, 'K=2, h=0'),
        ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_points)))

    for idx, (K, h, label) in enumerate(test_points):
        config = TrainConfig(N=N, K=K, h=h,
                             batch_size=1000, lr=0.01, max_step=5000, seed=42)
        result = train(config)
        bf_exact = free_energy_per_spin(K, h, N)

        # NMF reference
        nmf_result = nmf_solve(K, h)
        bf_nmf = nmf_result.free_energy_per_spin

        ax.plot(result.free_energy_history, color=colors[idx],
                label=label, alpha=0.8)
        ax.axhline(bf_exact, color=colors[idx], linestyle='-',
                   alpha=0.3, lw=0.8)
        ax.axhline(bf_nmf, color=colors[idx], linestyle='--',
                   alpha=0.3, lw=0.8)

    # Legend entries for reference lines
    ax.plot([], [], 'k-', alpha=0.3, lw=0.8, label='Exact $\\beta f$')
    ax.plot([], [], 'k--', alpha=0.3, lw=0.8, label='NMF $\\beta f$')

    ax.set_xlabel('Training step')
    ax.set_ylabel('$\\beta F_q / N$')
    ax.set_title(f'VAN convergence (N={N})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    savefig(fig, 'fig10_convergence')
    plt.close(fig)


if __name__ == "__main__":
    plot()
