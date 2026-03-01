"""
Run N=16 sweep only (with N<=8 exact enumeration threshold).
N=8 and N=2 results already saved from previous run.

Usage:
    source venv/bin/activate
    python -u run_n16_sweep.py 2>&1 | tee results/n16_sweep.log
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from experiments.config import ExperimentConfig
from experiments.sweep_TH import run_sweep


def main():
    cfg = ExperimentConfig(
        n_K=40,
        n_h=21,
        n_seeds=5,
        max_step=5000,
        batch_size=1000,
    )

    print("N=16 sweep (MC estimation, no 2^16 enumeration)")
    print(f"  K grid: {cfg.n_K} points, h grid: {cfg.n_h} points")
    print(f"  Seeds: {cfg.n_seeds}, max steps: {cfg.max_step}")
    print()

    t0 = time.time()
    run_sweep(16, cfg)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Regenerate figures that use N=16 data
    print("\nGenerating figures...")
    from figures.plot_exact_phase_diagram import plot_from_data
    from figures.plot_delta_F_heatmap import plot as plot_df
    from figures.plot_bias_ablation_heatmap import plot as plot_ba
    from figures.plot_H0_slice import plot as plot_h0
    from figures.plot_magnetization import plot as plot_mag
    from figures.plot_parameters import plot as plot_params
    from figures.plot_convergence import plot as plot_conv
    from figures.plot_n2_verification import plot as plot_n2
    from figures.plot_finite_size import plot as plot_fs
    from figures.style import RESULTS_DIR

    figs = [
        ("Fig 2: Phase diagram",    lambda: plot_from_data(os.path.join(RESULTS_DIR, "sweep_Kh_N16.npz"))),
        ("Fig 3: Delta-F heatmap",  lambda: plot_df(N=16)),
        ("Fig 4: Bias ablation",    lambda: plot_ba(N=16)),
        ("Fig 5: h=0 slice",        lambda: plot_h0()),
        ("Fig 6: Magnetization",    lambda: plot_mag(N=16)),
        ("Fig 7: Finite-size",      lambda: plot_fs()),
        ("Fig 8: N=2 verification", lambda: plot_n2()),
        ("Fig 9: Parameters",       lambda: plot_params()),
        ("Fig 10: Convergence",     lambda: plot_conv()),
    ]
    for name, fn in figs:
        try:
            fn()
            print(f"  {name} - OK")
        except Exception as e:
            print(f"  {name} - FAILED: {e}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
