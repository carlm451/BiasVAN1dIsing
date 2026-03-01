"""
Run all experiments (reduced grid) and generate all figures.

Reduced configuration for first-pass results:
  - 20 K points x 11 h points (6 positive, mirrored)
  - 3 seeds per VAN variant
  - 3000 max training steps
  - Finite-size: N in [2, 4, 8, 16], 5 seeds

Total: ~1560 training runs + N=2 verification + figure generation.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.config import ExperimentConfig
from experiments.n2_exact import run_n2_verification
from experiments.sweep_TH import run_sweep
from experiments import finite_size as finite_size_module
from experiments.finite_size import run_finite_size


def build_reduced_config():
    """Build a reduced ExperimentConfig for faster first-pass runs."""
    return ExperimentConfig(
        n_K=20,
        n_h=11,
        n_seeds=3,
        max_step=3000,
        batch_size=1000,
    )


def run_all_experiments(cfg):
    """Run all four experiment stages."""

    print("=" * 60)
    print("STAGE 1: N=2 exact verification")
    print("=" * 60)
    t0 = time.time()
    n2_path = run_n2_verification(cfg)
    print(f"  Completed in {time.time() - t0:.1f}s\n")

    print("=" * 60)
    print("STAGE 2: Sweep N=8")
    print("=" * 60)
    t0 = time.time()
    n8_path = run_sweep(8, cfg)
    print(f"  Completed in {time.time() - t0:.1f}s\n")

    print("=" * 60)
    print("STAGE 3: Sweep N=16")
    print("=" * 60)
    t0 = time.time()
    n16_path = run_sweep(16, cfg)
    print(f"  Completed in {time.time() - t0:.1f}s\n")

    print("=" * 60)
    print("STAGE 4: Finite-size scaling (N=2,4,8,16)")
    print("=" * 60)
    t0 = time.time()
    # Override module-level SYSTEM_SIZES to limit to N<=16
    finite_size_module.SYSTEM_SIZES = [2, 4, 8, 16]
    fs_path = run_finite_size(cfg, n_seeds=5)
    print(f"  Completed in {time.time() - t0:.1f}s\n")

    return n2_path, n8_path, n16_path, fs_path


def run_all_figures():
    """Generate all 9 figures."""
    from figures.plot_exact_phase_diagram import plot_from_data, plot_from_computation
    from figures.plot_delta_F_heatmap import plot as plot_delta_F
    from figures.plot_bias_ablation_heatmap import plot as plot_bias_ablation
    from figures.plot_H0_slice import plot as plot_H0
    from figures.plot_magnetization import plot as plot_mag
    from figures.plot_finite_size import plot as plot_fs
    from figures.plot_n2_verification import plot as plot_n2
    from figures.plot_parameters import plot as plot_params
    from figures.plot_convergence import plot as plot_conv
    from figures.style import RESULTS_DIR

    figure_runners = [
        ("Fig 2: Exact phase diagram", lambda: (
            plot_from_data(os.path.join(RESULTS_DIR, "sweep_Kh_N16.npz"))
            if os.path.exists(os.path.join(RESULTS_DIR, "sweep_Kh_N16.npz"))
            else plot_from_computation()
        )),
        ("Fig 3: Delta-F heatmap (NMF vs VAN)", lambda: plot_delta_F(N=16)),
        ("Fig 4: Bias ablation heatmap", lambda: plot_bias_ablation(N=16)),
        ("Fig 5: h=0 slice", lambda: plot_H0()),
        ("Fig 6: Magnetization vs h", lambda: plot_mag(N=16)),
        ("Fig 7: Finite-size scaling", lambda: plot_fs()),
        ("Fig 8: N=2 verification", lambda: plot_n2()),
        ("Fig 9: Converged parameters", lambda: plot_params()),
        ("Fig 10: Convergence curves", lambda: plot_conv()),
    ]

    for name, runner in figure_runners:
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            runner()
            print(f"  Done ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")


def main():
    total_start = time.time()

    cfg = build_reduced_config()
    print("Reduced experiment configuration:")
    print(f"  K grid: {cfg.n_K} points [{cfg.K_min}, {cfg.K_max}]")
    print(f"  h grid: {cfg.n_h} points [{cfg.h_min}, {cfg.h_max}]")
    print(f"  Seeds: {cfg.n_seeds}")
    print(f"  Max training steps: {cfg.max_step}")
    print(f"  Batch size: {cfg.batch_size}")
    print()

    # Run experiments
    run_all_experiments(cfg)

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    run_all_figures()

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
