"""Figure 9: Converged b_i, W_ij vs site index."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig
from src.van.train import train, TrainConfig


def plot(N=16, K=1.0, h_values=None):
    if h_values is None:
        h_values = [0.0, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for h in h_values:
        config = TrainConfig(N=N, K=K, h=h,
                             batch_size=1000, lr=0.01, max_step=5000, seed=42)
        result = train(config)
        params = result.parameters

        # Bias plot
        axes[0].plot(range(N), params['b'], 'o-', label=f'h={h}', markersize=4)

        # Weight matrix: plot first sub-diagonal
        W = params['W']
        sub_diag = [W[i, i-1] for i in range(1, N)]
        axes[1].plot(range(1, N), sub_diag, 'o-', label=f'h={h}', markersize=4)

    axes[0].set_xlabel('Site index $i$')
    axes[0].set_ylabel('$b_i$')
    axes[0].set_title(f'Converged biases (N={N}, K={K:.1f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linewidth=0.5)

    axes[1].set_xlabel('Site index $i$')
    axes[1].set_ylabel('$W_{i,i-1}$')
    axes[1].set_title(f'Converged nearest-neighbor weights (N={N}, K={K:.1f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, 'fig9_parameters')
    plt.close(fig)


if __name__ == "__main__":
    plot()
