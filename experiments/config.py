"""Central configuration for experiments (dimensionless K, h phase space)."""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ExperimentConfig:
    # System sizes
    system_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])

    # Dimensionless coupling grid: K = beta*J, log-spaced
    K_min: float = 0.01      # corresponds to T/J = 100 (3 decades: 0.01 to 10)
    K_max: float = 10.0      # corresponds to T/J = 0.1
    n_K: int = 50

    # Dimensionless field grid: h = beta*H, linear
    h_min: float = -2.0
    h_max: float = 2.0
    n_h: int = 21            # odd so h=0 included

    # Training parameters
    n_seeds: int = 5
    batch_size: int = 1000
    lr: float = 0.01
    max_step: int = 5000
    conv_tol: float = 1e-6
    conv_window: int = 100

    # Results directory
    results_dir: str = "results"

    @property
    def K_grid(self):
        return np.logspace(np.log10(self.K_min), np.log10(self.K_max), self.n_K)

    @property
    def h_grid(self):
        return np.linspace(self.h_min, self.h_max, self.n_h)

    @property
    def h_grid_positive(self):
        """h >= 0 only (exploit symmetry, mirror later)."""
        return np.linspace(0.0, self.h_max, (self.n_h + 1) // 2)
