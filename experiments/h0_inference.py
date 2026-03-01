"""
h=0 inference: load checkpoints, compute all observables.

Loads results/h0_checkpoints_N{N}.npz, calls observables_from_checkpoint()
for each (K, seed), computes exact + NMF observables, saves to
results/h0_observables_N{N}.npz.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.van.observables import observables_from_checkpoint
from src.exact import analytical_formulas as af
from src.nmf.mean_field import (
    nn_correlation_nmf, susceptibility_nmf, specific_heat_nmf,
    energy_per_spin_nmf, entropy_per_spin_nmf, solve as nmf_solve,
)


def run_h0_inference(N=16, n_samples=50000, results_dir="results"):
    ckpt_path = os.path.join(results_dir, f"h0_checkpoints_N{N}.npz")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file not found: {ckpt_path}")
        print("Run experiments/h0_train.py first.")
        return

    data = np.load(ckpt_path)
    K_grid = data['K_grid']
    van_W = data['van_W']    # (nK, n_seeds, N, N)
    van_b = data['van_b']    # (nK, n_seeds, N)
    nK = len(K_grid)
    n_seeds = van_W.shape[1]
    h = 0.0

    # VAN observables (averaged over seeds)
    obs_keys = ['magnetization', 'nn_correlation', 'chi_bare',
                'energy', 'specific_heat_bare', 'free_energy']
    van_obs = {k: np.zeros((nK, n_seeds)) for k in obs_keys}

    # Exact observables
    exact_m = np.zeros(nK)
    exact_nn = np.zeros(nK)
    exact_chi = np.zeros(nK)
    exact_C = np.zeros(nK)
    exact_f = np.zeros(nK)
    exact_S = np.zeros(nK)
    exact_u = np.zeros(nK)

    # NMF observables
    nmf_m = np.zeros(nK)
    nmf_nn = np.zeros(nK)
    nmf_chi = np.zeros(nK)
    nmf_C = np.zeros(nK)
    nmf_f = np.zeros(nK)
    nmf_u = np.zeros(nK)
    nmf_S = np.zeros(nK)

    for i, K in enumerate(K_grid):
        print(f"[{i+1}/{nK}] K={K:.4f}")

        # Exact
        exact_m[i] = af.magnetization(K, h, N)
        exact_nn[i] = af.nn_correlation(K, h, N)
        exact_chi[i] = af.susceptibility(K, h, N)
        exact_C[i] = af.specific_heat(K, h, N)
        exact_f[i] = af.free_energy_per_spin(K, h, N)
        exact_u[i] = af.energy_per_spin(K, h, N)
        exact_S[i] = af.entropy_per_spin(K, h, N)

        # NMF
        nmf_result = nmf_solve(K, h)
        nmf_m[i] = nmf_result.magnetization
        nmf_f[i] = nmf_result.free_energy_per_spin
        nmf_nn[i] = nn_correlation_nmf(K, h)
        nmf_chi[i] = susceptibility_nmf(K, h)
        nmf_C[i] = specific_heat_nmf(K, h)
        nmf_u[i] = energy_per_spin_nmf(K, h)
        nmf_S[i] = entropy_per_spin_nmf(K, h)

        # VAN (from checkpoints)
        for seed in range(n_seeds):
            obs = observables_from_checkpoint(
                van_W[i, seed], van_b[i, seed],
                K, h, z2=True, n_samples=n_samples,
            )
            for k in obs_keys:
                van_obs[k][i, seed] = obs[k]

    # Save
    outpath = os.path.join(results_dir, f"h0_observables_N{N}.npz")
    save_dict = {
        'K_grid': K_grid, 'N': N, 'n_seeds': n_seeds,
        # Exact
        'exact_m': exact_m, 'exact_nn': exact_nn, 'exact_chi': exact_chi,
        'exact_C': exact_C, 'exact_f': exact_f, 'exact_u': exact_u,
        'exact_S': exact_S,
        # NMF
        'nmf_m': nmf_m, 'nmf_nn': nmf_nn, 'nmf_chi': nmf_chi,
        'nmf_C': nmf_C, 'nmf_f': nmf_f, 'nmf_u': nmf_u, 'nmf_S': nmf_S,
    }
    # VAN: save mean and std over seeds
    for k in obs_keys:
        save_dict[f'van_{k}_mean'] = van_obs[k].mean(axis=1)
        save_dict[f'van_{k}_std'] = van_obs[k].std(axis=1)

    np.savez(outpath, **save_dict)
    print(f"\nSaved to {outpath}")
    return outpath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=50000)
    args = parser.parse_args()
    run_h0_inference(N=args.N, n_samples=args.n_samples)
