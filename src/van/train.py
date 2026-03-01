"""
REINFORCE training loop for the VAN (dimensionless formulation).

Minimizes the dimensionless variational free energy:
    beta * F_q = <beta*E>_q + <log q>_q

using the REINFORCE gradient estimator (adapted from Wu et al. stat-mech-van).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
import numpy as np
from .model import OneLayerVAN
from .energy import energy
from .utils import set_seed, get_device, DEFAULT_DTYPE


@dataclass
class TrainConfig:
    N: int = 8
    K: float = 1.0        # dimensionless coupling beta*J
    h: float = 0.0        # dimensionless field beta*H
    use_bias: bool = True
    z2: bool = False       # Z2 spin-flip symmetry (use only when h=0)
    batch_size: int = 1000
    lr: float = 0.01
    max_step: int = 5000
    seed: int = 42
    conv_window: int = 100
    conv_tol: float = 1e-6
    device: str = None


@dataclass
class TrainResult:
    final_free_energy: float   # dimensionless beta*f per spin
    free_energy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    final_step: int = 0
    parameters: Optional[Dict] = None


def train(config: TrainConfig) -> TrainResult:
    """
    Train a OneLayerVAN using REINFORCE.

    Parameters
    ----------
    config : TrainConfig

    Returns
    -------
    TrainResult with dimensionless beta*f per spin
    """
    set_seed(config.seed)
    device = get_device(config.device)

    model = OneLayerVAN(config.N, use_bias=config.use_bias, z2=config.z2, dtype=DEFAULT_DTYPE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    f_history = []
    loss_history = []

    for step in range(config.max_step):
        optimizer.zero_grad()

        # Sample (no grad for samples themselves)
        sample = model.sample(config.batch_size)

        # Dimensionless energy beta*E (no grad)
        with torch.no_grad():
            beta_E = energy(sample, K=config.K, h=config.h)

        # Log probability (has grad)
        log_q = model.log_prob(sample)

        # Dimensionless variational free energy per sample: beta*F_q = log_q + beta*E
        with torch.no_grad():
            loss_per_sample = log_q.detach() + beta_E
            beta_f_per_sample = (beta_E + log_q.detach()) / config.N

        # REINFORCE loss (variance-reduced)
        baseline = loss_per_sample.mean()
        loss_reinforce = ((loss_per_sample - baseline) * log_q).mean()
        loss_reinforce.backward()
        optimizer.step()

        # Record (dimensionless beta*f per spin)
        f_mean = beta_f_per_sample.mean().item()
        f_history.append(f_mean)
        loss_history.append(baseline.item())

        # Convergence check
        if step >= config.conv_window:
            recent = f_history[-config.conv_window:]
            if np.std(recent) < config.conv_tol:
                result = TrainResult(
                    final_free_energy=np.mean(recent),
                    free_energy_history=f_history,
                    loss_history=loss_history,
                    converged=True,
                    final_step=step,
                    parameters=model.get_parameters_dict(),
                )
                return result

    # Did not converge within max_step
    recent = f_history[-config.conv_window:] if len(f_history) >= config.conv_window else f_history
    return TrainResult(
        final_free_energy=np.mean(recent),
        free_energy_history=f_history,
        loss_history=loss_history,
        converged=False,
        final_step=config.max_step,
        parameters=model.get_parameters_dict(),
    )


def train_and_evaluate_exact(config: TrainConfig):
    """
    Train VAN and compute exact free energy by enumerating all 2^N configs.
    Only practical for N <= ~16.

    Returns
    -------
    TrainResult with exact dimensionless beta*f per spin (no MC noise)
    """
    set_seed(config.seed)
    device = get_device(config.device)

    model = OneLayerVAN(config.N, use_bias=config.use_bias, z2=config.z2, dtype=DEFAULT_DTYPE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    f_history = []
    loss_history = []

    for step in range(config.max_step):
        optimizer.zero_grad()
        sample = model.sample(config.batch_size)
        with torch.no_grad():
            beta_E = energy(sample, K=config.K, h=config.h)
        log_q = model.log_prob(sample)
        with torch.no_grad():
            loss_per_sample = log_q.detach() + beta_E
            beta_f_per_sample = (beta_E + log_q.detach()) / config.N
        baseline = loss_per_sample.mean()
        loss_reinforce = ((loss_per_sample - baseline) * log_q).mean()
        loss_reinforce.backward()
        optimizer.step()

        f_mean = beta_f_per_sample.mean().item()
        f_history.append(f_mean)
        loss_history.append(baseline.item())

        if step >= config.conv_window:
            recent = f_history[-config.conv_window:]
            if np.std(recent) < config.conv_tol:
                break

    # Compute exact VAN free energy per spin by enumeration
    exact_f = _exact_van_free_energy(model, config)

    return TrainResult(
        final_free_energy=exact_f,
        free_energy_history=f_history,
        loss_history=loss_history,
        converged=True,
        final_step=len(f_history),
        parameters=model.get_parameters_dict(),
    )


def _exact_van_free_energy(model, config):
    """
    Compute exact dimensionless VAN free energy by summing over all 2^N configs.

    beta*f_VAN = (1/N) * sum_s q(s) * [beta*E(s) + log q(s)]
    """
    N = config.N
    assert N <= 20, f"Exact enumeration impractical for N={N}"

    device = next(model.parameters()).device

    # Generate all 2^N configurations
    num_configs = 2**N
    configs = torch.zeros(num_configs, N, dtype=DEFAULT_DTYPE, device=device)
    for i in range(num_configs):
        for j in range(N):
            configs[i, j] = 1.0 if (i >> j) & 1 else -1.0

    with torch.no_grad():
        log_q = model.log_prob(configs)
        beta_E = energy(configs, K=config.K, h=config.h)
        q = torch.exp(log_q)
        # Total dimensionless free energy, then divide by N for per-spin
        beta_F_total = torch.sum(q * (beta_E + log_q)).item()

    return beta_F_total / N
