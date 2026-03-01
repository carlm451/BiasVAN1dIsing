"""Utility functions for VAN training."""

import torch
import numpy as np

DEFAULT_DTYPE = torch.float64


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(device=None):
    """Return the best available device, or the specified one."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('cpu')  # Use CPU for float64 compatibility
    return torch.device('cpu')
