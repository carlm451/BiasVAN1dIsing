"""
Finite-size scaling experiment.

Selected (K, h) points x N in {2, 4, 8, 16, 32, 64, 128}, 10 seeds each.
Confirm deltaF/N ~ 1/N scaling.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from src.exact.transfer_matrix import free_energy_per_spin
from src.van.train import train, TrainConfig


# Representative (K, h) points
REPRESENTATIVE_POINTS = [
    (0.5, 0.0),
    (2.0, 0.0),
    (2.0, 0.5),
    (2.0, 2.0),
    (1.0, 0.1),
    (0.2, 0.0),
]

SYSTEM_SIZES = [2, 4, 8, 16, 32, 64, 128]


def run_finite_size(cfg=None, n_seeds=10):
    if cfg is None:
        cfg = ExperimentConfig()

    results = {}

    for K, h in REPRESENTATIVE_POINTS:
        key = f"K{K}_h{h}"
        print(f"\n=== {key} ===")

        delta_f_mean = []
        delta_f_std = []

        for N in SYSTEM_SIZES:
            print(f"  N={N}", end=" ... ")
            bf_exact = free_energy_per_spin(K, h, N)

            f_van_list = []
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
                f_van_list.append(result.final_free_energy)

            df = np.array(f_van_list) - bf_exact
            delta_f_mean.append(np.mean(df))
            delta_f_std.append(np.std(df))
            print(f"deltaF/N = {np.mean(df):.6f} +/- {np.std(df):.6f}")

        results[f"{key}_delta_f_mean"] = np.array(delta_f_mean)
        results[f"{key}_delta_f_std"] = np.array(delta_f_std)

    # Save
    os.makedirs(cfg.results_dir, exist_ok=True)
    outpath = os.path.join(cfg.results_dir, "finite_size.npz")
    np.savez(outpath,
             system_sizes=np.array(SYSTEM_SIZES),
             representative_points=np.array(REPRESENTATIVE_POINTS),
             **results)
    print(f"\nSaved to {outpath}")
    return outpath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--max_N", type=int, default=None,
                        help="Cap max system size (e.g. 16 to skip N=32,64,128)")
    args = parser.parse_args()
    if args.max_N is not None:
        SYSTEM_SIZES[:] = [n for n in SYSTEM_SIZES if n <= args.max_N]
    run_finite_size(n_seeds=args.n_seeds)
