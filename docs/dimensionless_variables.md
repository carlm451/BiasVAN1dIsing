# Dimensionless Variables: From (T, H) to (K, h)

## Motivation

The 1D Ising Hamiltonian is:

$$\mathcal{H} = -J \sum_i s_i s_{i+1} - H \sum_i s_i$$

The partition function depends on $(\beta, J, H, N)$ through only two dimensionless combinations:

$$K = \beta J, \quad h = \beta H$$

These are the natural variables of the transfer matrix — $K$ and $h$ appear directly in the matrix elements $e^{K \pm h}$ and $e^{-K}$. The system size $N$ is the only other parameter.

## Why Switch?

### Before: (T, H) with J

The old parameterization used physical variables $(T, H)$ with a coupling constant $J$:

```python
# Old API
f = free_energy_per_spin(beta=1.0, J=1.0, h=0.5, N=16)
result = TrainConfig(N=8, beta=2.0, J=1.0, h=0.3)
```

Problems:
1. **Redundancy** — $\beta$ and $J$ always appear as the product $\beta J$. Setting $J=1$ and varying $\beta$ is equivalent to setting $\beta=1$ and varying $J$.
2. **Intermediate conversions** — every function had to compute `bJ = beta * J` and `bh = beta * h` internally.
3. **Training loop complexity** — loss required explicit $\beta$ multiplication: `loss = log_q + beta * E`.

### After: (K, h)

```python
# New API
f = free_energy_per_spin(K=1.0, h=0.5, N=16)
result = TrainConfig(N=8, K=2.0, h=0.3)
```

Benefits:
1. **Minimal parameterization** — two variables instead of three.
2. **Direct physical meaning** — $K$ and $h$ are the actual parameters that enter the Boltzmann weight $e^{-\beta\mathcal{H}}$.
3. **Simpler training loop** — energy function returns $\beta E$ directly, loss is just `log_q + beta_E`.
4. **Natural phase space** — the $(K, h)$ plane is the true parameter space of the model.

## Mapping Between Parameterizations

| Old | New | Relation |
|-----|-----|----------|
| $\beta$ | — | Absorbed into $K$ and $h$ |
| $J$ | — | Absorbed into $K$ |
| $H$ | — | Absorbed into $h$ |
| — | $K$ | $K = \beta J$ |
| — | $h$ | $h = \beta H$ |

To convert back: choose a $J$ (or equivalently set $J = 1$), then $T = J/K$ and $H = h \cdot T = h \cdot J / K$.

## What Changes Are Dimensionless

All quantities returned by the exact module are now dimensionless:

| Quantity | Old return value | New return value | Relation |
|----------|-----------------|------------------|----------|
| Free energy | $f$ (energy units) | $\beta f$ (dimensionless) | Multiply by $\beta$ |
| Energy | $\langle E \rangle / N$ | $\beta \langle E \rangle / N$ | Multiply by $\beta$ |
| Entropy | — (not computed) | $s / k_B$ (dimensionless) | New |
| Magnetization | $\langle m \rangle$ | $\langle m \rangle$ | Unchanged (already dimensionless) |
| nn-correlation | $\langle s_i s_{i+1} \rangle$ | $\langle s_i s_{i+1} \rangle$ | Unchanged |
| Susceptibility | $\partial m / \partial H$ | $\partial^2 \ln Z / (N \, \partial h^2)$ | Different normalization |
| Specific heat | $C_v / N$ | $c_v / k_B$ | Dimensionless |

## VAN Training Changes

The training loop simplifies because the energy function now returns $\beta E$:

```python
# Old
E = energy(sample, J=config.J, h=config.h)          # physical E
loss_per_sample = log_q + config.beta * E             # need beta
f_per_sample = (E + log_q / config.beta) / config.N   # need 1/beta

# New
beta_E = energy(sample, K=config.K, h=config.h)       # dimensionless beta*E
loss_per_sample = log_q + beta_E                       # no beta needed
beta_f_per_sample = (beta_E + log_q) / config.N        # no beta needed
```

The `TrainConfig` dataclass drops `beta` and `J`, replacing them with `K`:

```python
# Old
TrainConfig(N=8, beta=2.0, J=1.0, h=0.3)

# New
TrainConfig(N=8, K=2.0, h=0.3)
```

## NMF Changes

The self-consistency equation simplifies:

```python
# Old: m = tanh(beta * (2*J*m + h))
# New: m = tanh(2*K*m + h)
```

And the free energy becomes:

```python
# Old: f = -J*m^2 - h*m - (1/beta)*S(m)
# New: beta*f = -K*m^2 - h*m - S(m)
```

## Experiment Grid Changes

| Old | New | Notes |
|-----|-----|-------|
| `T_min = 0.1`, `T_max = 5.0` | `K_min = 0.2`, `K_max = 10.0` | $K = 1/T$ when $J=1$ |
| `T_grid` (log-spaced) | `K_grid` (log-spaced) | Both capture wide dynamic range |
| `H_min = -2`, `H_max = 2` | `h_min = -2`, `h_max = 2` | Same range, different meaning |
| `H_grid` (linear) | `h_grid` (linear) | Symmetry: odd count includes $h=0$ |

Note: $K = 0.2$ corresponds to $T/J = 5$ (high temperature), and $K = 10$ corresponds to $T/J = 0.1$ (low temperature). The K axis is "inverted" relative to T.
