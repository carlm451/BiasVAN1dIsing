# BiasVAN1dIsing

Variational Autoregressive Networks for the 1D Ising model with an external magnetic field, benchmarked against exact transfer-matrix solutions.

## Overview

This repository accompanies **[paper title TBD]** and provides:

- **Exact analytical solutions** for all thermodynamic observables of the 1D Ising model with periodic boundary conditions and an external field, derived from the transfer-matrix eigenvalues in closed form.
- **A minimal Variational Autoregressive Network (VAN)** — a single masked-linear layer with `tanh` activation — trained via variational free-energy minimization.
- **Systematic comparison** of VAN (with and without bias) against exact results across the full dimensionless `(K = betaJ, h = betaH)` phase space, including finite-size scaling up to `N = 64`.
- **Naive mean-field baseline** demonstrating that VAN with `W = 0` reduces to a uniform mean-field approximation.

## Repository Structure

```
src/
  exact/
    analytical_formulas.py   # Closed-form expressions (numpy, fast)
    transfer_matrix.py       # SymPy-based transfer matrix (cross-validation)
  van/
    model.py                 # 1-layer autoregressive network
    train.py                 # Variational training loop
    energy.py                # Energy computation for VAN samples
  nmf/
    mean_field.py            # Naive mean-field baseline
experiments/
    config.py                # Central experiment configuration
    run_all.py               # Run all experiments (reduced grid)
    sweep_TH.py              # (K, h) parameter sweep
    finite_size.py           # Finite-size scaling experiments
figures/
    plot_*.py                # All plotting scripts
    style.py                 # Shared plot styling
docs/                        # Internal documentation
tests/                       # Pytest suite (50+ tests)
run_production.py            # Full production run script
```

## Quick Start

```bash
git clone https://github.com/<your-username>/BiasVAN1dIsing.git
cd BiasVAN1dIsing

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Verify everything works
pytest tests/ -v
```

Requires Python 3.10+.

## Running Experiments

**Reduced grid** (fast first-pass, ~1500 training runs):

```bash
python -m experiments.run_all
```

**Full production** (40 K-points x 21 h-points, 5 seeds, N up to 64):

```bash
python run_production.py 2>&1 | tee results/production_run.log
```

Results are saved to `results/` as `.npz` files. See `docs/experiments.md` and `docs/results_format.md` for details.

## Generating Figures

After experiments complete, regenerate all paper figures:

```bash
python figures/plot_analytical_observables.py    # Fig A1: field dependence
python figures/plot_analytical_finite_size.py     # Fig A2: finite-size effects
python figures/plot_analytical_h0.py             # Fig A3: h=0 limit checks
python figures/plot_analytical_heatmaps.py       # Fig A4: (K, h) heatmaps
python figures/plot_observable_comparison.py      # VAN vs exact comparison
python figures/plot_bias_ablation_heatmap.py     # Bias ablation study
```

Output PNGs are written to `figures/output/`.

## Key Results

- **Bias terms are essential:** VAN without bias fails to capture the magnetization and related observables when `h != 0`, while the bias-enabled VAN matches exact solutions across the full `(K, h)` plane.
- **Exact benchmarks verified:** All six thermodynamic observables (M, energy, susceptibility, heat capacity, entropy, nearest-neighbor correlation) agree with the SymPy transfer-matrix reference to relative tolerance `1e-10`.
- **Finite-size convergence:** VAN accuracy improves systematically with increasing system size N, with the largest deviations confined to the low-temperature, strong-field regime.

## Citation

```bibtex
@article{biasvan1dising2026,
  title   = {TBD},
  author  = {TBD},
  year    = {2026},
  journal = {TBD}
}
```

## License

TBD
