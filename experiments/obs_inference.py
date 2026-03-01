"""
Multi-K observable inference: load checkpoints, compute all observables.

Loads results/obs_checkpoints_K{K}_N{N}.npz for each K, calls
observables_from_checkpoint() per (h, seed), computes exact + NMF
observables, saves to results/observables_K{K}_N{N}.npz.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.van.observables import observables_from_checkpoint
from src.exact import analytical_formulas as af
from src.nmf.mean_field import (
    nn_correlation_nmf, susceptibility_nmf, specific_heat_nmf,
    energy_per_spin_nmf, solve as nmf_solve,
)


K_VALUES = [0.5, 1.0, 2.0, 5.0]


def run_obs_inference(K_values=None, N=16, n_samples=50000, results_dir="results"):
    if K_values is None:
        K_values = K_VALUES

    for K in K_values:
        ckpt_path = os.path.join(results_dir, f"obs_checkpoints_K{K}_N{N}.npz")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint file not found: {ckpt_path}")
            print(f"Run experiments/obs_train.py --K {K} first.")
            continue

        print(f"\n{'='*50}")
        print(f"K = {K}")
        print(f"{'='*50}")

        data = np.load(ckpt_path)
        h_grid = data['h_grid']
        van_W = data['van_W']   # (nh, n_seeds, N, N)
        van_b = data['van_b']   # (nh, n_seeds, N)
        nh = len(h_grid)
        n_seeds = van_W.shape[1]

        obs_keys = ['magnetization', 'nn_correlation', 'chi_bare',
                    'energy', 'specific_heat_bare', 'free_energy']
        van_obs = {k: np.zeros((nh, n_seeds)) for k in obs_keys}

        # Exact + NMF
        exact_m = np.zeros(nh)
        exact_nn = np.zeros(nh)
        exact_chi = np.zeros(nh)
        exact_C = np.zeros(nh)
        exact_f = np.zeros(nh)
        nmf_m = np.zeros(nh)
        nmf_nn = np.zeros(nh)
        nmf_chi = np.zeros(nh)
        nmf_C = np.zeros(nh)
        nmf_f = np.zeros(nh)

        for j, h in enumerate(h_grid):
            print(f"  [{j+1}/{nh}] h={h:.3f}")

            exact_m[j] = af.magnetization(K, h, N)
            exact_nn[j] = af.nn_correlation(K, h, N)
            exact_chi[j] = af.susceptibility(K, h, N)
            exact_C[j] = af.specific_heat(K, h, N)
            exact_f[j] = af.free_energy_per_spin(K, h, N)

            nmf_result = nmf_solve(K, h)
            nmf_m[j] = nmf_result.magnetization
            nmf_f[j] = nmf_result.free_energy_per_spin
            nmf_nn[j] = nn_correlation_nmf(K, h)
            nmf_chi[j] = susceptibility_nmf(K, h)
            nmf_C[j] = specific_heat_nmf(K, h)

            for seed in range(n_seeds):
                z2 = abs(h) < 1e-10
                obs = observables_from_checkpoint(
                    van_W[j, seed], van_b[j, seed],
                    K, h, z2=z2, n_samples=n_samples,
                )
                for k in obs_keys:
                    van_obs[k][j, seed] = obs[k]

        # Save
        outpath = os.path.join(results_dir, f"observables_K{K}_N{N}.npz")
        save_dict = {
            'K': K, 'N': N, 'h_grid': h_grid, 'n_seeds': n_seeds,
            'exact_m': exact_m, 'exact_nn': exact_nn,
            'exact_chi': exact_chi, 'exact_C': exact_C, 'exact_f': exact_f,
            'nmf_m': nmf_m, 'nmf_nn': nmf_nn,
            'nmf_chi': nmf_chi, 'nmf_C': nmf_C, 'nmf_f': nmf_f,
        }
        for k in obs_keys:
            save_dict[f'van_{k}_mean'] = van_obs[k].mean(axis=1)
            save_dict[f'van_{k}_std'] = van_obs[k].std(axis=1)

        np.savez(outpath, **save_dict)
        print(f"  Saved to {outpath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--K", type=float, nargs='+', default=None)
    args = parser.parse_args()
    run_obs_inference(K_values=args.K, N=args.N, n_samples=args.n_samples)
