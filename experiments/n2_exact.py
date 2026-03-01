"""
N=2 verification across (K, h) grid.

Compute exact VAN free energy by enumerating all 4 configs (no MC noise).
Expected: |deltaF| < 1e-10 everywhere after sufficient training.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from src.exact.transfer_matrix import free_energy_per_spin
from src.van.train import train_and_evaluate_exact, TrainConfig


def run_n2_verification(cfg=None):
    if cfg is None:
        cfg = ExperimentConfig()

    # Use a coarser grid for N=2 verification: 20 K points x full h grid
    K_grid = np.logspace(np.log10(cfg.K_min), np.log10(cfg.K_max), 20)
    h_grid = cfg.h_grid
    nK, nh = len(K_grid), len(h_grid)

    exact_f = np.zeros((nK, nh))
    van_f = np.zeros((nK, nh))
    delta_f = np.zeros((nK, nh))

    total = nK * nh
    count = 0

    for i, K in enumerate(K_grid):
        for j, h in enumerate(h_grid):
            count += 1
            print(f"[{count}/{total}] K={K:.3f}, h={h:.3f}", end=" ... ")

            bf_ex = free_energy_per_spin(K, h, N=2)
            exact_f[i, j] = bf_ex

            config = TrainConfig(
                N=2, K=K, h=h,
                z2=(abs(h) < 1e-10),
                batch_size=2000, lr=0.005, max_step=10000, seed=42,
                conv_tol=1e-8,
            )
            result = train_and_evaluate_exact(config)
            van_f[i, j] = result.final_free_energy
            delta_f[i, j] = result.final_free_energy - bf_ex

            print(f"deltaF = {delta_f[i,j]:.2e}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    outpath = os.path.join(cfg.results_dir, "n2_verification.npz")
    np.savez(outpath,
             K_grid=K_grid,
             h_grid=h_grid,
             exact_f=exact_f,
             van_f=van_f,
             delta_f=delta_f)
    print(f"\nSaved to {outpath}")
    print(f"Max |deltaF| = {np.max(np.abs(delta_f)):.2e}")

    return outpath


if __name__ == "__main__":
    run_n2_verification()
