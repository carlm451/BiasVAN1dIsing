"""
h=0 training sweep for Fig 6.

Trains VAN at h=0 with Z2 symmetry across a dense K grid.
Saves checkpoints (W, b) and free energies; observable computation
happens separately via h0_inference.py.

Output: results/h0_checkpoints_N{N}.npz
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from src.exact.analytical_formulas import free_energy_per_spin as exact_f
from src.nmf.mean_field import solve as nmf_solve
from src.van.train import train, TrainConfig


def run_h0_train(N=16, n_K=60, n_seeds=5, cfg=None):
    if cfg is None:
        cfg = ExperimentConfig()

    K_grid = np.logspace(-2, 1, n_K)  # [0.01, 10.0]
    h = 0.0

    # Allocate arrays
    van_W = np.zeros((n_K, n_seeds, N, N))
    van_b = np.zeros((n_K, n_seeds, N))
    van_f = np.zeros((n_K, n_seeds))
    exact_free = np.zeros(n_K)
    nmf_free = np.zeros(n_K)
    nmf_mag = np.zeros(n_K)

    for i, K in enumerate(K_grid):
        print(f"[{i+1}/{n_K}] K={K:.4f}")

        # Exact (cheap)
        exact_free[i] = exact_f(K, h, N)

        # NMF (cheap)
        nmf_result = nmf_solve(K, h)
        nmf_free[i] = nmf_result.free_energy_per_spin
        nmf_mag[i] = nmf_result.magnetization

        # VAN with Z2 symmetry
        for seed in range(n_seeds):
            tc = TrainConfig(
                N=N, K=K, h=h,
                use_bias=True,
                z2=True,
                batch_size=cfg.batch_size, lr=cfg.lr,
                max_step=cfg.max_step, seed=seed,
                conv_tol=cfg.conv_tol, conv_window=cfg.conv_window,
            )
            result = train(tc)
            van_W[i, seed] = result.parameters['W']
            van_b[i, seed] = result.parameters['b']
            van_f[i, seed] = result.final_free_energy

    # Save
    os.makedirs(cfg.results_dir, exist_ok=True)
    outpath = os.path.join(cfg.results_dir, f"h0_checkpoints_N{N}.npz")
    np.savez(outpath,
             K_grid=K_grid,
             N=N,
             van_W=van_W,
             van_b=van_b,
             van_f=van_f,
             exact_f=exact_free,
             nmf_f=nmf_free,
             nmf_m=nmf_mag)
    print(f"\nSaved to {outpath}")
    return outpath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--n_K", type=int, default=60)
    parser.add_argument("--n_seeds", type=int, default=5)
    args = parser.parse_args()
    run_h0_train(N=args.N, n_K=args.n_K, n_seeds=args.n_seeds)
