"""
Sweep inference: load checkpoints from the main (K, h) sweep,
compute all observables over the full grid.

Loads results/checkpoints_N{N}.npz, calls observables_from_checkpoint()
per (K, h, seed), saves to results/sweep_observables_N{N}.npz.

Useful for generating heatmap-style figures of any observable.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.van.observables import observables_from_checkpoint
from src.exact import analytical_formulas as af


def run_sweep_inference(N=16, n_samples=50000, results_dir="results"):
    ckpt_path = os.path.join(results_dir, f"checkpoints_N{N}.npz")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file not found: {ckpt_path}")
        print("Run experiments/sweep_TH.py first.")
        return

    data = np.load(ckpt_path)
    K_grid = data['K_grid']
    h_grid = data['h_grid']   # h >= 0 only
    van_bias_W = data['van_bias_W']   # (nK, nh, n_seeds, N, N)
    van_bias_b = data['van_bias_b']   # (nK, nh, n_seeds, N)
    nK, nh = len(K_grid), len(h_grid)
    n_seeds = van_bias_W.shape[2]

    obs_keys = ['magnetization', 'nn_correlation', 'chi_bare',
                'energy', 'specific_heat_bare', 'free_energy']

    # Use seed-averaged observables
    van_obs_mean = {k: np.zeros((nK, nh)) for k in obs_keys}
    van_obs_std = {k: np.zeros((nK, nh)) for k in obs_keys}

    total = nK * nh
    count = 0

    for i, K in enumerate(K_grid):
        for j, h in enumerate(h_grid):
            count += 1
            print(f"[{count}/{total}] K={K:.3f}, h={h:.3f}")

            seed_obs = {k: [] for k in obs_keys}
            for seed in range(n_seeds):
                z2 = abs(h) < 1e-10
                obs = observables_from_checkpoint(
                    van_bias_W[i, j, seed], van_bias_b[i, j, seed],
                    K, h, z2=z2, n_samples=n_samples,
                )
                for k in obs_keys:
                    seed_obs[k].append(obs[k])

            for k in obs_keys:
                van_obs_mean[k][i, j] = np.mean(seed_obs[k])
                van_obs_std[k][i, j] = np.std(seed_obs[k])

    # Save
    outpath = os.path.join(results_dir, f"sweep_observables_N{N}.npz")
    save_dict = {'K_grid': K_grid, 'h_grid': h_grid, 'N': N, 'n_seeds': n_seeds}
    for k in obs_keys:
        save_dict[f'van_{k}_mean'] = van_obs_mean[k]
        save_dict[f'van_{k}_std'] = van_obs_std[k]

    np.savez(outpath, **save_dict)
    print(f"\nSaved to {outpath}")
    return outpath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=50000)
    args = parser.parse_args()
    run_sweep_inference(N=args.N, n_samples=args.n_samples)
