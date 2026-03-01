# Project: VAN 1D Ising Model with External Field

Variational Autoregressive Network (VAN) for the 1D Ising model in an external magnetic field, with exact analytical benchmarks.

## Project Structure

```
src/exact/analytical_formulas.py   — Pure-numpy closed-form expressions (main analytical engine)
src/exact/transfer_matrix.py       — SymPy-based transfer matrix (cross-validation reference)
src/van/model.py                   — VAN model (1-layer autoregressive, Z2 symmetry at h=0)
src/van/train.py                   — Training loop
src/van/energy.py                  — Energy computation for VAN samples
src/nmf/mean_field.py              — Naive mean field baseline
latex/1disingtransferderivations.tex — Derivation document (Appendix A)
figures/plot_analytical_*.py       — Plotting scripts for Appendix A figures
figures/output/                    — Generated figure PNGs
tests/                             — Pytest suite (52+ tests)
```

## Analytical Calculations — Key Facts

### Parameterization
- Dimensionless couplings: `K = betaJ` (coupling), `h = betaH` (field)
- Shorthand: `c = cosh(h)`, `s = sinh(h)`, `w = exp(-2K)`, `D = sqrt(s^2 + w^2)`
- Eigenvalues: `lambda_{1,2} = e^K (c +/- D)`
- Eigenvalue ratio: `r = lambda_2 / lambda_1`, finite-size weight: `sigma_N = r^N / (1 + r^N)`

### Hamiltonian Sign Convention
`-beta E = beta(J sum s_i s_{i+1} + H sum s_i)` with periodic BCs (`s_{N+1} = s_1`).
The transfer matrix is:
```
T = [[e^{K+h}, e^{-K}],
     [e^{-K},  e^{K-h}]]
```

### Master Formulas (verified correct)
All six observables derive from two master formulas:
- **First derivative:** `Psi_x = A_x + sigma_N (B_x - A_x)`
- **Second derivative:** `Psi_xy = A_xy + sigma_N (B_xy - A_xy) + N sigma_N (1-sigma_N)(B_x - A_x)(B_y - A_y)`

where `A_x = d/dx ln(lambda_1)`, `B_x = d/dx ln(lambda_2)`, etc.

### Critical: Energy and Heat Capacity Require Both K and h Derivatives
When `H != 0`, beta-derivatives pick up contributions from both K and h via the chain rule:
```
d/d(beta) = J d/dK + H d/dh
```
- **Energy:** `beta u = -K * Psi_K - h * Psi_h`
- **Heat capacity:** `C/k_B = K^2 Psi_KK + 2Kh Psi_Kh + h^2 Psi_hh` (three-term structure)
- **Susceptibility:** `chi = beta * Psi_hh` (note: `chi` has a beta prefactor from `dM/dH = beta dM/dh`)

### Susceptibility Convention (Caption Pitfall)
- `analytical_formulas.py` returns `chi_bare = (1/N) d^2 ln Z / dh^2` (no beta prefactor)
- The LaTeX and plots use `chi = beta * chi_bare`
- **Fig A3 (h=0)** plots `chi/beta` on the y-axis (correct, matches the plotting script)
- **Figs A1, A2, A4** plot `chi` (not `chi/beta`) — captions were corrected to match

### Building Blocks (all verified against SymPy to rtol=1e-10)
```
A_h  =  s/D          B_h  = -s/D
A_K  = (cD+s^2-w^2)/[D(c+D)]    B_K = (cD-s^2+w^2)/[D(c-D)]
A_hh =  cw^2/D^3     B_hh = -cw^2/D^3
A_Kh =  2sw^2/D^3    B_Kh = -2sw^2/D^3
A_KK =  4w^2[c(2s^2+w^2)+2s^2D]/[D^3(c+D)^2]
B_KK = -4w^2[c(2s^2+w^2)-2s^2D]/[D^3(c-D)^2]
```

### h=0 Limit Checks
- `A_K -> tanh(K)`, `B_K -> coth(K)`
- `A_KK -> sech^2(K)`, `B_KK -> -csch^2(K)`
- `M = 0` (Z2 symmetry), `chi -> beta e^{2K}`

### Numerical Stability
- `c - D` computed as `(1 - w^2)/(c + D)` to avoid catastrophic cancellation
- `sigma_N` computed via sigmoid on `rho = N ln(r)` to avoid overflow

## LaTeX Document
- **File:** `latex/1disingtransferderivations.tex`
- **Title:** "Appendix A: Exact Solutions for 1d Ising Model In an External Magnetic Field"
- Intended as supplementary material / appendix for a paper on VAN benchmarking
- Uses `float` package for figure placement; compiles on Overleaf
- Figures reference PNGs in `figures/` subdirectory (relative path from latex/)
- Bibliography: single entry for Goldenfeld (1992), transfer matrix formulation

### Figure-Caption Mapping
| Figure | File | Panels |
|--------|------|--------|
| A1 (field dependence) | `figA1_analytical_observables.png` | (a) M, (b) beta u, (c) chi, (d) C/k_B, (e) S/(Nk_B), (f) <s_i s_{i+1}> |
| A2 (finite-size) | `figA2_analytical_finite_size.png` | (a) M, (b) chi, (c) C/k_B, (d) S/(Nk_B) |
| A3 (h=0 limit) | `figA3_analytical_h0.png` | (a) <s_i s_{i+1}>, (b) chi/beta, (c) C/k_B, (d) S/(Nk_B) |
| A4 (heatmaps) | `figA4_analytical_heatmaps.png` | (a) M, (b) beta u, (c) chi, (d) C/k_B, (e) S/(Nk_B), (f) <s_i s_{i+1}> |

## Testing
```bash
cd /Users/carlmerrigan/DeckerCode/VAN1dIsingWithHField
source venv/bin/activate
pytest tests/ -v
```
- `test_analytical_vs_sympy.py` cross-validates all 8 quantities across 10 diverse (K, h, N) points to rtol=1e-10
- Python 3.13 venv with numpy, scipy, sympy, matplotlib, pytorch
