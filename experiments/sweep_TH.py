"""
Main (K, h) sweep experiment.

For each (K, h, N):
1. Exact solution via transfer matrix
2. NMF solution
3. VAN with bias x n_seeds
4. VAN without bias x n_seeds

Saves per-N results to results/sweep_Kh_N{N}.npz.
Also saves VAN checkpoints (W, b per seed) to results/checkpoints_N{N}.npz.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from src.exact.transfer_matrix import free_energy_per_spin, magnetization as exact_mag
from src.nmf.mean_field import solve as nmf_solve
from src.van.train import train, TrainConfig


def run_sweep(N, cfg=None):
    if cfg is None:
        cfg = ExperimentConfig()

    K_grid = cfg.K_grid
    h_grid = cfg.h_grid_positive  # Use symmetry
    nK, nh = len(K_grid), len(h_grid)
    n_seeds = cfg.n_seeds

    # Allocate result arrays
    exact_f = np.zeros((nK, nh))
    exact_m = np.zeros((nK, nh))
    nmf_f = np.zeros((nK, nh))
    nmf_m = np.zeros((nK, nh))
    van_bias_f_mean = np.zeros((nK, nh))
    van_bias_f_std = np.zeros((nK, nh))
    van_nobias_f_mean = np.zeros((nK, nh))
    van_nobias_f_std = np.zeros((nK, nh))

    # Checkpoint arrays — W, b per seed
    van_bias_W = np.zeros((nK, nh, n_seeds, N, N))
    van_bias_b = np.zeros((nK, nh, n_seeds, N))
    van_nobias_W = np.zeros((nK, nh, n_seeds, N, N))
    van_nobias_b = np.zeros((nK, nh, n_seeds, N))

    total = nK * nh
    count = 0

    for i, K in enumerate(K_grid):
        for j, h in enumerate(h_grid):
            count += 1
            print(f"N={N} [{count}/{total}] K={K:.3f}, h={h:.3f}")

            # Exact
            exact_f[i, j] = free_energy_per_spin(K, h, N)
            exact_m[i, j] = exact_mag(K, h, N)

            # NMF
            nmf_result = nmf_solve(K, h)
            nmf_f[i, j] = nmf_result.free_energy_per_spin
            nmf_m[i, j] = nmf_result.magnetization

            # VAN with bias
            f_bias = []
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
                f_bias.append(result.final_free_energy)
                van_bias_W[i, j, seed] = result.parameters['W']
                van_bias_b[i, j, seed] = result.parameters['b']

            van_bias_f_mean[i, j] = np.mean(f_bias)
            van_bias_f_std[i, j] = np.std(f_bias)

            # VAN without bias
            f_nobias = []
            for seed in range(n_seeds):
                tc = TrainConfig(
                    N=N, K=K, h=h,
                    use_bias=False,
                    z2=(abs(h) < 1e-10),
                    batch_size=cfg.batch_size, lr=cfg.lr,
                    max_step=cfg.max_step, seed=seed,
                    conv_tol=cfg.conv_tol, conv_window=cfg.conv_window,
                )
                result = train(tc)
                f_nobias.append(result.final_free_energy)
                van_nobias_W[i, j, seed] = result.parameters['W']
                van_nobias_b[i, j, seed] = result.parameters['b']

            van_nobias_f_mean[i, j] = np.mean(f_nobias)
            van_nobias_f_std[i, j] = np.std(f_nobias)

    # Mirror h < 0 results from h > 0 using symmetry
    h_full = cfg.h_grid
    nh_full = len(h_full)

    def mirror_array(arr):
        """Mirror h>=0 results to full h grid. arr shape: (nK, nh_pos)."""
        full = np.zeros((nK, nh_full))
        # h >= 0 indices
        mid = nh_full // 2  # index of h=0
        full[:, mid:] = arr
        # h < 0: f(K, -h) = f(K, h) for free energy
        full[:, :mid] = arr[:, -1:0:-1]
        return full

    def mirror_array_mag(arr):
        """Mirror magnetization: m(K, -h) = -m(K, h)."""
        full = np.zeros((nK, nh_full))
        mid = nh_full // 2
        full[:, mid:] = arr
        full[:, :mid] = -arr[:, -1:0:-1]
        return full

    # Save sweep results
    os.makedirs(cfg.results_dir, exist_ok=True)
    outpath = os.path.join(cfg.results_dir, f"sweep_Kh_N{N}.npz")
    np.savez(outpath,
             K_grid=K_grid,
             h_grid=h_full,
             exact_f=mirror_array(exact_f),
             exact_m=mirror_array_mag(exact_m),
             nmf_f=mirror_array(nmf_f),
             nmf_m=mirror_array_mag(nmf_m),
             van_bias_f_mean=mirror_array(van_bias_f_mean),
             van_bias_f_std=mirror_array(van_bias_f_std),
             van_nobias_f_mean=mirror_array(van_nobias_f_mean),
             van_nobias_f_std=mirror_array(van_nobias_f_std),
             delta_f_nmf=mirror_array(nmf_f - exact_f),
             delta_f_van_bias=mirror_array(van_bias_f_mean - exact_f),
             delta_f_van_nobias=mirror_array(van_nobias_f_mean - exact_f))
    print(f"\nSaved sweep to {outpath}")

    # Save checkpoints (W, b per seed) separately
    ckpt_path = os.path.join(cfg.results_dir, f"checkpoints_N{N}.npz")
    np.savez(ckpt_path,
             K_grid=K_grid,
             h_grid=h_grid,  # h >= 0 only (as trained)
             van_bias_W=van_bias_W,
             van_bias_b=van_bias_b,
             van_nobias_W=van_nobias_W,
             van_nobias_b=van_nobias_b)
    print(f"Saved checkpoints to {ckpt_path}")

    return outpath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--n_seeds", type=int, default=None,
                        help="Override number of seeds (default: from config, 5)")
    args = parser.parse_args()
    cfg = ExperimentConfig()
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    run_sweep(args.N, cfg=cfg)
