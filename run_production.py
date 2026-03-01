"""
Production run: full grid, Z2 symmetry, up to N=64.

Usage:
    source venv/bin/activate
    python run_production.py 2>&1 | tee results/production_run.log
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from experiments.config import ExperimentConfig
from experiments.n2_exact import run_n2_verification
from experiments.sweep_TH import run_sweep
from experiments import finite_size as fs_mod
from experiments.finite_size import run_finite_size


def build_production_config():
    return ExperimentConfig(
        n_K=40,
        n_h=21,
        n_seeds=5,
        max_step=5000,
        batch_size=1000,
    )


def run_experiments(cfg):
    # Stage 1: N=2 verification (full grid)
    print("=" * 60)
    print("STAGE 1/5: N=2 exact verification (full grid)")
    print("=" * 60)
    t0 = time.time()
    run_n2_verification(cfg)
    print(f"  Done in {time.time() - t0:.0f}s\n")

    # Stage 2: Sweep N=8
    print("=" * 60)
    print("STAGE 2/5: Sweep N=8")
    print("=" * 60)
    t0 = time.time()
    run_sweep(8, cfg)
    print(f"  Done in {time.time() - t0:.0f}s\n")

    # Stage 3: Sweep N=16
    print("=" * 60)
    print("STAGE 3/5: Sweep N=16")
    print("=" * 60)
    t0 = time.time()
    run_sweep(16, cfg)
    print(f"  Done in {time.time() - t0:.0f}s\n")

    # Stage 4: Sweep N=32
    print("=" * 60)
    print("STAGE 4/5: Sweep N=32")
    print("=" * 60)
    t0 = time.time()
    run_sweep(32, cfg)
    print(f"  Done in {time.time() - t0:.0f}s\n")

    # Stage 5: Finite-size scaling up to N=64, 10 seeds
    print("=" * 60)
    print("STAGE 5/5: Finite-size scaling (N=2..64, 10 seeds)")
    print("=" * 60)
    t0 = time.time()
    fs_mod.SYSTEM_SIZES = [2, 4, 8, 16, 32, 64]
    run_finite_size(cfg, n_seeds=10)
    print(f"  Done in {time.time() - t0:.0f}s\n")


def run_figures():
    print("=" * 60)
    print("GENERATING ALL FIGURES")
    print("=" * 60)

    from figures.plot_exact_phase_diagram import plot_from_data
    from figures.plot_delta_F_heatmap import plot as plot_df
    from figures.plot_bias_ablation_heatmap import plot as plot_ba
    from figures.plot_H0_slice import plot as plot_h0
    from figures.plot_magnetization import plot as plot_mag
    from figures.plot_finite_size import plot as plot_fs
    from figures.plot_n2_verification import plot as plot_n2
    from figures.plot_parameters import plot as plot_params
    from figures.plot_convergence import plot as plot_conv
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


def main():
    total_start = time.time()

    cfg = build_production_config()
    print("Production run configuration:")
    print(f"  K grid: {cfg.n_K} points [{cfg.K_min}, {cfg.K_max}]")
    print(f"  h grid: {cfg.n_h} points [{cfg.h_min}, {cfg.h_max}]")
    print(f"  Seeds: {cfg.n_seeds}")
    print(f"  Max training steps: {cfg.max_step}")
    print(f"  Batch size: {cfg.batch_size}")
    print()

    run_experiments(cfg)
    run_figures()

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE in {total:.0f}s ({total / 60:.1f} min, {total / 3600:.1f} hr)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
