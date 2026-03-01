# Exact Transfer Matrix Solutions

## Dimensionless Parameterization

All computations use the natural dimensionless variables of the transfer matrix:

| Symbol | Definition | Meaning |
|--------|-----------|---------|
| $K$ | $\beta J$ | Dimensionless coupling |
| $h$ | $\beta H$ | Dimensionless field |
| $N$ | — | Number of spins (periodic ring) |

These are the **only** combinations of $(\beta, J, H)$ that enter the transfer matrix, so they fully specify the thermodynamic state. Large $K$ = low temperature (strong coupling); small $K$ = high temperature.

## Transfer Matrix

The Hamiltonian is $\mathcal{H} = -J \sum_i s_i s_{i+1} - H \sum_i s_i$ with periodic boundary conditions ($s_{N+1} = s_1$). The $2 \times 2$ transfer matrix is:

$$\mathbf{P} = \begin{pmatrix} e^{K + h} & e^{-K} \\ e^{-K} & e^{K - h} \end{pmatrix}$$

## Eigenvalues

$$\lambda_\pm = e^K \left[ \cosh(h) \pm \sqrt{\sinh^2(h) + e^{-4K}} \right]$$

The log-eigenvalues, which are the fundamental building blocks:

$$\ln \lambda_+ = K + \ln\!\left(\cosh(h) + \sqrt{\sinh^2(h) + e^{-4K}}\right)$$
$$\ln \lambda_- = K + \ln\!\left(\cosh(h) - \sqrt{\sinh^2(h) + e^{-4K}}\right)$$

Note: $\lambda_+ \geq |\lambda_-|$ always, so $\ln\lambda_+$ is well-defined and $\rho = \ln\lambda_- - \ln\lambda_+ \leq 0$.

## Partition Function

$$Z_N = \lambda_+^N + \lambda_-^N$$

Computed via log-sum-exp for numerical stability:

$$\ln Z = N \ln\lambda_+ + \ln(1 + e^{N\rho})$$

## SymPy Analytical Derivatives

All observables are expressed in terms of **12 building-block functions**: the first and second derivatives of $\ln\lambda_\pm$ with respect to $K$ and $h$. These are derived symbolically using SymPy at import time and lambdified to fast NumPy functions.

| Notation | Expression |
|----------|-----------|
| $A_h^{(1)}$ | $\partial(\ln\lambda_+)/\partial h$ |
| $B_h^{(1)}$ | $\partial(\ln\lambda_-)/\partial h$ |
| $A_K^{(1)}$ | $\partial(\ln\lambda_+)/\partial K$ |
| $B_K^{(1)}$ | $\partial(\ln\lambda_-)/\partial K$ |
| $A_{hh}^{(2)}$ | $\partial^2(\ln\lambda_+)/\partial h^2$ |
| $B_{hh}^{(2)}$ | $\partial^2(\ln\lambda_-)/\partial h^2$ |
| $A_{KK}^{(2)}$ | $\partial^2(\ln\lambda_+)/\partial K^2$ |
| $B_{KK}^{(2)}$ | $\partial^2(\ln\lambda_-)/\partial K^2$ |
| $A_{Kh}^{(2)}$ | $\partial^2(\ln\lambda_+)/\partial K\,\partial h$ |
| $B_{Kh}^{(2)}$ | $\partial^2(\ln\lambda_-)/\partial K\,\partial h$ |

These are closed-form expressions of $(K, h)$ only — no $N$ dependence, no overflow risk.

## Sigmoid Weighting

For finite $N$, the contribution of $\lambda_-$ is controlled by:

$$S = \sigma(N\rho) = \frac{1}{1 + e^{-N\rho}}$$

where $\rho = \ln\lambda_- - \ln\lambda_+ \leq 0$. Key properties:
- $S \to 0$ as $N \to \infty$ (thermodynamic limit: only $\lambda_+$ matters)
- $S = 1/2$ when $\rho = 0$ (degenerate eigenvalues, e.g., $h = 0$ with $K \to \infty$)
- Numerically stable via the standard sigmoid trick

## Observable Formulas

All observables follow the pattern: **$\lambda_+$ contribution + sigmoid-weighted correction from $\lambda_-$**.

### Free energy per spin (dimensionless)

$$\beta f = -\frac{1}{N}\ln Z$$

### Magnetization

$$\langle m \rangle = \frac{1}{N}\frac{\partial \ln Z}{\partial h} = A_h^{(1)} + S\,(B_h^{(1)} - A_h^{(1)})$$

In the thermodynamic limit ($S \to 0$):

$$\langle m \rangle_\infty = A_h^{(1)} = \frac{\sinh(h)}{\sqrt{\sinh^2(h) + e^{-4K}}}$$

### Nearest-neighbor correlation

$$\langle s_i s_{i+1} \rangle = \frac{1}{N}\frac{\partial \ln Z}{\partial K} = A_K^{(1)} + S\,(B_K^{(1)} - A_K^{(1)})$$

### Susceptibility

$$\chi = \frac{1}{N}\frac{\partial^2 \ln Z}{\partial h^2} = A_{hh}^{(2)} + S\,(B_{hh}^{(2)} - A_{hh}^{(2)}) + N\,S(1-S)\,(B_h^{(1)} - A_h^{(1)})^2$$

The last term comes from differentiating $S$ itself, which depends on $h$ through $\rho$.

### Specific heat

Since $K = \beta J$ and $h = \beta H$, temperature derivatives at fixed $J, H$ give:

$$\frac{c_v}{k_B} = K^2 \cdot \frac{1}{N}\frac{\partial^2\Psi}{\partial K^2} + 2Kh \cdot \frac{1}{N}\frac{\partial^2\Psi}{\partial K\,\partial h} + h^2 \cdot \frac{1}{N}\frac{\partial^2\Psi}{\partial h^2}$$

where $\Psi = \ln Z$ and each second derivative uses the same sigmoid-weighted pattern as $\chi$.

### Energy per spin (dimensionless)

$$\beta u = -(K\langle s_i s_{i+1}\rangle + h\langle m\rangle)$$

### Entropy per spin (dimensionless)

$$\frac{s}{k_B} = \frac{\Psi}{N} + \beta u$$

## Advantages Over Finite Differences

The previous implementation used finite differences for magnetization, susceptibility, specific heat, energy, and nearest-neighbor correlation. The SymPy analytical approach:

1. **Eliminates truncation error** — exact derivatives instead of $(f(x+\delta) - f(x-\delta))/2\delta$
2. **No step-size tuning** — finite differences require careful $\delta$ choice (too large = truncation error, too small = cancellation error)
3. **Stable at all $(K, h)$** — no numerical noise, especially for second derivatives ($\chi$, $c_v$) at large $K$ (low temperature)
4. **Faster** — one evaluation instead of 2-5 evaluations per observable

## API

All functions in `src/exact/transfer_matrix.py` take `(K, h, N)`:

```python
from src.exact.transfer_matrix import (
    eigenvalues,          # (K, h) -> (lam_plus, lam_minus)
    partition_function,   # (K, h, N) -> Z
    free_energy_per_spin, # (K, h, N) -> beta*f
    magnetization,        # (K, h, N) -> <m>
    magnetization_thermo_limit,  # (K, h) -> <m> as N->inf
    nn_correlation,       # (K, h, N) -> <s_i s_{i+1}>
    susceptibility,       # (K, h, N) -> chi
    specific_heat,        # (K, h, N) -> c_v/k_B
    energy_per_spin,      # (K, h, N) -> beta*u
    entropy_per_spin,     # (K, h, N) -> s/k_B
    compute_all,          # (K, h, N) -> IsingExact dataclass
    compute_grid,         # (Ks, hs, N) -> dict of 2D arrays
)
```

All returned quantities are dimensionless.
