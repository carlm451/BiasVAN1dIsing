"""
Multi-K observable training sweep for Fig 5.

For K in {0.5, 1.0, 2.0, 5.0}, trains VAN across h grid.
Saves checkpoints (W, b) and free energies; observable computation
happens separately via obs_inference.py.

Output: results/obs_checkpoints_K{K}_N{N}.npz (one file per K)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from src.exact.analytical_formulas import free_energy_per_spin as exact_f
from src.nmf.mean_field import solve as nmf_solve
from src.van.train import train, TrainConfig


K_VALUES = [0.5, 1.0, 2.0, 5.0]


def run_obs_train(K_values=None, N=16, n_h=51, h_max=3.0, n_seeds=5, cfg=None):
    if cfg is None:
        cfg = ExperimentConfig()
    if K_values is None:
        K_values = K_VALUES

    h_grid = np.linspace(0, h_max, n_h)  # h >= 0 only (mirror later)

    for K in K_values:
        print(f"\n{'='*50}")
        print(f"K = {K}")
        print(f"{'='*50}")

        nh = len(h_grid)
        van_W = np.zeros((nh, n_seeds, N, N))
        van_b = np.zeros((nh, n_seeds, N))
        van_f = np.zeros((nh, n_seeds))
        exact_free = np.zeros(nh)
        nmf_free = np.zeros(nh)
        nmf_mag = np.zeros(nh)

        for j, h in enumerate(h_grid):
            print(f"  [{j+1}/{nh}] h={h:.3f}")

            # Exact
            exact_free[j] = exact_f(K, h, N)

            # NMF
            nmf_result = nmf_solve(K, h)
            nmf_free[j] = nmf_result.free_energy_per_spin
            nmf_mag[j] = nmf_result.magnetization

            # VAN
            for seed in range(n_seeds):
                tc = TrainConfig(
                    N=N, K=K, h=h,
                    use_bias=True,
                    z2=(abs(h) < 1e-10),
                    batch_size=cfg.batch_size, lr=cfg.lr,
                    max_step=cfg.max_step, seed=seed,
                    conv_tol=cfg.conv_tol, conv_window=cfg.conv_window,
                )
                result = train(tc)
                van_W[j, seed] = result.parameters['W']
                van_b[j, seed] = result.parameters['b']
                van_f[j, seed] = result.final_free_energy

        # Save per-K checkpoint file
        os.makedirs(cfg.results_dir, exist_ok=True)
        outpath = os.path.join(cfg.results_dir, f"obs_checkpoints_K{K}_N{N}.npz")
        np.savez(outpath,
                 K=K,
                 N=N,
                 h_grid=h_grid,
                 van_W=van_W,
                 van_b=van_b,
                 van_f=van_f,
                 exact_f=exact_free,
                 nmf_f=nmf_free,
                 nmf_m=nmf_mag)
        print(f"  Saved to {outpath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--n_h", type=int, default=51)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--K", type=float, nargs='+', default=None,
                        help="K values to run (default: 0.5 1.0 2.0 5.0)")
    args = parser.parse_args()
    run_obs_train(K_values=args.K, N=args.N, n_h=args.n_h, n_seeds=args.n_seeds)
