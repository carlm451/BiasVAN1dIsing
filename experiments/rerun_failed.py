"""
Targeted re-run of (K, h) points where the VAN did not converge.

Identifies points where |delta_f_van_bias| > threshold from existing
sweep results, re-trains with a higher budget, and patches the results.

Usage:
    # Identify and re-run (interactive, single process)
    python experiments/rerun_failed.py --N 64 --max_step 50000 --batch_size 8000

    # Multi-GPU with chunking
    python experiments/rerun_failed.py --N 64 --chunk 0 --n_chunks 4 --device cuda \\
        --max_step 50000 --batch_size 8000

    # Just identify bad points (dry run)
    python experiments/rerun_failed.py --N 64 --dry-run

    # Apply patches after chunked re-runs
    python experiments/rerun_failed.py --N 64 --merge-only
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from src.exact.transfer_matrix import free_energy_per_spin, magnetization as exact_mag
from src.nmf.mean_field import solve as nmf_solve
from src.van.train import train, TrainConfig


def identify_failed_points(N, threshold=0.05, results_dir="results"):
    """Find (K_idx, h_pos_idx) pairs where with-bias VAN error exceeds threshold.

    Works on the h >= 0 half of the grid (what was actually trained).
    Returns list of (i, j) indices into (K_grid, h_grid_positive).
    """
    sweep_path = os.path.join(results_dir, f"sweep_Kh_N{N}.npz")
    if not os.path.exists(sweep_path):
        print(f"Sweep file not found: {sweep_path}")
        return [], None, None

    data = np.load(sweep_path)
    K_grid = data['K_grid']
    h_full = data['h_grid']
    delta_f = np.abs(data['delta_f_van_bias'])

    # h >= 0 half: indices mid..end in the full grid
    mid = len(h_full) // 2
    cfg = ExperimentConfig()
    h_pos = cfg.h_grid_positive
    nh_pos = len(h_pos)

    failed = []
    for i in range(len(K_grid)):
        for j in range(nh_pos):
            j_full = mid + j  # index into full h grid
            if delta_f[i, j_full] > threshold:
                failed.append((i, j))

    return failed, K_grid, h_pos


def rerun_points(N, failed_points, K_grid, h_grid, cfg,
                 chunk=None, n_chunks=None, device=None,
                 results_dir="results"):
    """Re-train the specified (K, h) points with higher budget."""
    n_seeds = cfg.n_seeds

    if chunk is not None and n_chunks is not None:
        my_points = failed_points[chunk::n_chunks]
        tag = f"chunk{chunk}"
        print(f"Chunk {chunk}/{n_chunks}: {len(my_points)} of {len(failed_points)} points")
    else:
        my_points = failed_points
        tag = "patch"

    nK, nh = len(K_grid), len(h_grid)

    # Allocate arrays (full grid size, only filling our points)
    exact_f = np.zeros((nK, nh))
    exact_m = np.zeros((nK, nh))
    nmf_f = np.zeros((nK, nh))
    nmf_m = np.zeros((nK, nh))
    van_bias_f_mean = np.zeros((nK, nh))
    van_bias_f_std = np.zeros((nK, nh))
    van_nobias_f_mean = np.zeros((nK, nh))
    van_nobias_f_std = np.zeros((nK, nh))

    van_bias_W = np.zeros((nK, nh, n_seeds, N, N))
    van_bias_b = np.zeros((nK, nh, n_seeds, N))
    van_nobias_W = np.zeros((nK, nh, n_seeds, N, N))
    van_nobias_b = np.zeros((nK, nh, n_seeds, N))

    total = len(my_points)
    for count, (i, j) in enumerate(my_points, 1):
        K = K_grid[i]
        h = h_grid[j]
        prefix = f"[{tag}] " if chunk is not None else ""
        print(f"{prefix}N={N} [{count}/{total}] K={K:.3f}, h={h:.3f} "
              f"(max_step={cfg.max_step}, batch={cfg.batch_size})")

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
                device=device,
            )
            result = train(tc)
            f_bias.append(result.final_free_energy)
            van_bias_W[i, j, seed] = result.parameters['W']
            van_bias_b[i, j, seed] = result.parameters['b']
            converged_str = "converged" if result.converged else f"max_step ({result.final_step})"
            print(f"    bias seed={seed}: f={result.final_free_energy:.8f} "
                  f"(exact={exact_f[i,j]:.8f}, err={result.final_free_energy - exact_f[i,j]:+.2e}) "
                  f"[{converged_str}]")

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
                device=device,
            )
            result = train(tc)
            f_nobias.append(result.final_free_energy)
            van_nobias_W[i, j, seed] = result.parameters['W']
            van_nobias_b[i, j, seed] = result.parameters['b']

        van_nobias_f_mean[i, j] = np.mean(f_nobias)
        van_nobias_f_std[i, j] = np.std(f_nobias)

    # Save patch file
    os.makedirs(results_dir, exist_ok=True)
    patch_indices = my_points
    outpath = os.path.join(results_dir, f"rerun_N{N}_{tag}.npz")
    np.savez(outpath,
             K_grid=K_grid,
             h_grid=h_grid,
             patch_indices_i=np.array([p[0] for p in patch_indices]),
             patch_indices_j=np.array([p[1] for p in patch_indices]),
             exact_f=exact_f,
             exact_m=exact_m,
             nmf_f=nmf_f,
             nmf_m=nmf_m,
             van_bias_f_mean=van_bias_f_mean,
             van_bias_f_std=van_bias_f_std,
             van_nobias_f_mean=van_nobias_f_mean,
             van_nobias_f_std=van_nobias_f_std,
             van_bias_W=van_bias_W,
             van_bias_b=van_bias_b,
             van_nobias_W=van_nobias_W,
             van_nobias_b=van_nobias_b)
    print(f"\nSaved patch to {outpath}")
    return outpath


def merge_patches(N, results_dir="results"):
    """Apply rerun patches to existing sweep and checkpoint files."""
    # Find all patch files
    pattern = os.path.join(results_dir, f"rerun_N{N}_*.npz")
    patch_files = sorted(glob.glob(pattern))
    if not patch_files:
        print(f"No patch files found matching {pattern}")
        return

    print(f"Found {len(patch_files)} patch file(s) for N={N}")

    # Load existing sweep
    sweep_path = os.path.join(results_dir, f"sweep_Kh_N{N}.npz")
    sweep = dict(np.load(sweep_path))
    K_grid = sweep['K_grid']
    h_full = sweep['h_grid']
    nK = len(K_grid)
    mid = len(h_full) // 2

    # Load existing checkpoints
    ckpt_path = os.path.join(results_dir, f"checkpoints_N{N}.npz")
    ckpt = dict(np.load(ckpt_path))

    n_patched = 0
    for path in patch_files:
        patch = np.load(path)
        idx_i = patch['patch_indices_i']
        idx_j = patch['patch_indices_j']
        print(f"  Applying {os.path.basename(path)}: {len(idx_i)} points")

        for k in range(len(idx_i)):
            i, j = int(idx_i[k]), int(idx_j[k])
            j_full = mid + j  # h >= 0 index in full grid

            # Check if new result is actually better
            old_err = abs(sweep['delta_f_van_bias'][i, j_full])
            new_van_f = patch['van_bias_f_mean'][i, j]
            new_exact_f = patch['exact_f'][i, j]
            new_err = abs(new_van_f - new_exact_f)

            if new_err < old_err:
                # Patch sweep arrays (h >= 0 side)
                for key in ['exact_f', 'exact_m', 'nmf_f', 'nmf_m',
                            'van_bias_f_mean', 'van_bias_f_std',
                            'van_nobias_f_mean', 'van_nobias_f_std']:
                    if key in patch:
                        sweep[key][i, j_full] = patch[key][i, j]

                # Mirror to h < 0 side
                if j > 0:  # don't mirror h=0
                    j_neg = mid - j
                    for key in ['exact_f', 'nmf_f',
                                'van_bias_f_mean', 'van_bias_f_std',
                                'van_nobias_f_mean', 'van_nobias_f_std']:
                        if key in patch:
                            sweep[key][i, j_neg] = patch[key][i, j]
                    # Magnetization: odd symmetry
                    for key in ['exact_m', 'nmf_m']:
                        if key in patch:
                            sweep[key][i, j_neg] = -patch[key][i, j]

                # Patch checkpoints
                for key in ['van_bias_W', 'van_bias_b',
                            'van_nobias_W', 'van_nobias_b']:
                    if key in patch and key in ckpt:
                        ckpt[key][i, j] = patch[key][i, j]

                n_patched += 1
                print(f"    K={K_grid[i]:.3f}, h={h_full[j_full]:+.2f}: "
                      f"err {old_err:.2e} -> {new_err:.2e}")
            else:
                print(f"    K={K_grid[i]:.3f}, h={h_full[j_full]:+.2f}: "
                      f"no improvement ({old_err:.2e} -> {new_err:.2e}), skipping")

        patch.close()

    # Recompute delta_f arrays
    sweep['delta_f_nmf'] = sweep['nmf_f'] - sweep['exact_f']
    sweep['delta_f_van_bias'] = sweep['van_bias_f_mean'] - sweep['exact_f']
    sweep['delta_f_van_nobias'] = sweep['van_nobias_f_mean'] - sweep['exact_f']

    # Save updated files (backup originals first)
    import shutil
    backup_sweep = sweep_path.replace('.npz', '_pre_rerun.npz')
    backup_ckpt = ckpt_path.replace('.npz', '_pre_rerun.npz')
    if not os.path.exists(backup_sweep):
        shutil.copy2(sweep_path, backup_sweep)
        print(f"\nBacked up {sweep_path} -> {backup_sweep}")
    if not os.path.exists(backup_ckpt):
        shutil.copy2(ckpt_path, backup_ckpt)
        print(f"Backed up {ckpt_path} -> {backup_ckpt}")

    np.savez(sweep_path, **sweep)
    print(f"Updated {sweep_path} ({n_patched} points patched)")

    np.savez(ckpt_path, **ckpt)
    print(f"Updated {ckpt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Re-run failed (K, h) points with higher training budget")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Error threshold for identifying failed points (default: 0.01)")
    parser.add_argument("--max_step", type=int, default=50000,
                        help="Training steps for re-run (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=8000,
                        help="Batch size for re-run (default: 8000)")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds (default: 1)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Chunk index for multi-GPU")
    parser.add_argument("--n_chunks", type=int, default=None,
                        help="Total number of chunks")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g. 'cuda', 'cpu')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just identify bad points, don't re-train")
    parser.add_argument("--merge-only", action="store_true",
                        help="Just merge existing patches, don't re-train")
    args = parser.parse_args()

    if args.merge_only:
        merge_patches(args.N)
        sys.exit(0)

    # Identify failed points
    failed, K_grid, h_grid = identify_failed_points(
        args.N, threshold=args.threshold)

    if not failed:
        print(f"No points with error > {args.threshold} for N={args.N}")
        sys.exit(0)

    print(f"\nFound {len(failed)} points with |delta_f| > {args.threshold} "
          f"for N={args.N}")
    print(f"Re-run config: max_step={args.max_step}, batch_size={args.batch_size}, "
          f"n_seeds={args.n_seeds}, lr={args.lr}")
    print()

    if args.dry_run:
        print("Points to re-run:")
        for i, j in failed:
            print(f"  K={K_grid[i]:.3f}, h={h_grid[j]:.3f}")
        sys.exit(0)

    # Set up config
    cfg = ExperimentConfig()
    cfg.max_step = args.max_step
    cfg.batch_size = args.batch_size
    cfg.n_seeds = args.n_seeds
    cfg.lr = args.lr

    # Re-run
    rerun_points(args.N, failed, K_grid, h_grid, cfg,
                 chunk=args.chunk, n_chunks=args.n_chunks,
                 device=args.device)

    # If not chunked, merge immediately
    if args.chunk is None:
        print("\nMerging patches...")
        merge_patches(args.N)
