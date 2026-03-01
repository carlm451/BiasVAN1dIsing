"""
Merge chunked sweep results into a single sweep_Kh_N{N}.npz file.

Usage:
    python experiments/merge_chunks.py --N 32
    python experiments/merge_chunks.py --N 32 --results_dir results

Combines sweep_Kh_N{N}_chunk*.npz and checkpoints_N{N}_chunk*.npz files
into the standard sweep_Kh_N{N}.npz and checkpoints_N{N}.npz format.
Output is identical to non-chunked sweep_TH.py — all downstream scripts
work unchanged.
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig


def merge_sweep_chunks(N, results_dir="results"):
    """Merge chunk files into a single sweep file with mirror symmetry."""
    pattern = os.path.join(results_dir, f"sweep_Kh_N{N}_chunk*.npz")
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        print(f"No chunk files found matching {pattern}")
        return None

    print(f"Found {len(chunk_files)} chunk files for N={N}")

    # Load first chunk to get grid info
    first = np.load(chunk_files[0])
    K_grid = first['K_grid']
    h_grid = first['h_grid']  # h >= 0 only
    nK, nh = len(K_grid), len(h_grid)

    # Allocate combined arrays (h >= 0 only, pre-mirror)
    fields = ['exact_f', 'exact_m', 'nmf_f', 'nmf_m',
              'van_bias_f_mean', 'van_bias_f_std',
              'van_nobias_f_mean', 'van_nobias_f_std']
    combined = {f: np.zeros((nK, nh)) for f in fields}

    # Track which points have been filled
    filled = np.zeros((nK, nh), dtype=bool)

    for path in chunk_files:
        data = np.load(path)
        chunk_id = int(data['chunk'])
        idx_i = data['chunk_indices_i']
        idx_j = data['chunk_indices_j']

        print(f"  Chunk {chunk_id}: {len(idx_i)} points")

        for k in range(len(idx_i)):
            i, j = int(idx_i[k]), int(idx_j[k])
            for f in fields:
                combined[f][i, j] = data[f][i, j]
            filled[i, j] = True

        data.close()

    # Check completeness
    n_filled = filled.sum()
    n_total = nK * nh
    if n_filled < n_total:
        missing = n_total - n_filled
        print(f"  WARNING: {missing}/{n_total} points missing "
              f"({100*missing/n_total:.1f}%)")
        # Report which (i,j) are missing
        missing_ij = np.argwhere(~filled)
        if len(missing_ij) <= 10:
            for mi, mj in missing_ij:
                print(f"    Missing: K={K_grid[mi]:.3f}, h={h_grid[mj]:.3f}")
        else:
            print(f"    (showing first 10)")
            for mi, mj in missing_ij[:10]:
                print(f"    Missing: K={K_grid[mi]:.3f}, h={h_grid[mj]:.3f}")
    else:
        print(f"  All {n_total} points present")

    # Apply mirror symmetry for h < 0
    cfg = ExperimentConfig()
    h_full = cfg.h_grid
    nh_full = len(h_full)
    mid = nh_full // 2

    def mirror_array(arr):
        full = np.zeros((nK, nh_full))
        full[:, mid:] = arr
        full[:, :mid] = arr[:, -1:0:-1]
        return full

    def mirror_array_mag(arr):
        full = np.zeros((nK, nh_full))
        full[:, mid:] = arr
        full[:, :mid] = -arr[:, -1:0:-1]
        return full

    # Save merged sweep
    outpath = os.path.join(results_dir, f"sweep_Kh_N{N}.npz")
    np.savez(outpath,
             K_grid=K_grid,
             h_grid=h_full,
             exact_f=mirror_array(combined['exact_f']),
             exact_m=mirror_array_mag(combined['exact_m']),
             nmf_f=mirror_array(combined['nmf_f']),
             nmf_m=mirror_array_mag(combined['nmf_m']),
             van_bias_f_mean=mirror_array(combined['van_bias_f_mean']),
             van_bias_f_std=mirror_array(combined['van_bias_f_std']),
             van_nobias_f_mean=mirror_array(combined['van_nobias_f_mean']),
             van_nobias_f_std=mirror_array(combined['van_nobias_f_std']),
             delta_f_nmf=mirror_array(combined['nmf_f'] - combined['exact_f']),
             delta_f_van_bias=mirror_array(combined['van_bias_f_mean'] - combined['exact_f']),
             delta_f_van_nobias=mirror_array(combined['van_nobias_f_mean'] - combined['exact_f']))
    print(f"\nSaved merged sweep to {outpath}")

    return outpath


def merge_checkpoint_chunks(N, results_dir="results"):
    """Merge chunked checkpoint files into a single checkpoints file."""
    pattern = os.path.join(results_dir, f"checkpoints_N{N}_chunk*.npz")
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        print(f"No checkpoint chunk files found matching {pattern}")
        return None

    print(f"\nFound {len(chunk_files)} checkpoint chunk files for N={N}")

    # Load first chunk to get dimensions
    first = np.load(chunk_files[0])
    K_grid = first['K_grid']
    h_grid = first['h_grid']
    nK, nh = len(K_grid), len(h_grid)

    # Infer n_seeds and N from array shapes
    n_seeds = first['van_bias_W'].shape[2]
    first.close()

    # Allocate
    van_bias_W = np.zeros((nK, nh, n_seeds, N, N))
    van_bias_b = np.zeros((nK, nh, n_seeds, N))
    van_nobias_W = np.zeros((nK, nh, n_seeds, N, N))
    van_nobias_b = np.zeros((nK, nh, n_seeds, N))

    for path in chunk_files:
        data = np.load(path)
        chunk_id = int(data['chunk'])
        idx_i = data['chunk_indices_i']
        idx_j = data['chunk_indices_j']

        print(f"  Checkpoint chunk {chunk_id}: {len(idx_i)} points")

        for k in range(len(idx_i)):
            i, j = int(idx_i[k]), int(idx_j[k])
            van_bias_W[i, j] = data['van_bias_W'][i, j]
            van_bias_b[i, j] = data['van_bias_b'][i, j]
            van_nobias_W[i, j] = data['van_nobias_W'][i, j]
            van_nobias_b[i, j] = data['van_nobias_b'][i, j]

        data.close()

    ckpt_path = os.path.join(results_dir, f"checkpoints_N{N}.npz")
    np.savez(ckpt_path,
             K_grid=K_grid,
             h_grid=h_grid,  # h >= 0 only (as trained)
             van_bias_W=van_bias_W,
             van_bias_b=van_bias_b,
             van_nobias_W=van_nobias_W,
             van_nobias_b=van_nobias_b)
    print(f"Saved merged checkpoints to {ckpt_path}")

    return ckpt_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge chunked sweep results")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    merge_sweep_chunks(args.N, results_dir=args.results_dir)
    merge_checkpoint_chunks(args.N, results_dir=args.results_dir)
