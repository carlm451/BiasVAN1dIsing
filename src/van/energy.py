"""
1D periodic Ising model energy function in PyTorch (dimensionless).

beta * E = -K * sum_i s_i * s_{i+1} - h * sum_i s_i   (PBC)

where K = beta*J (dimensionless coupling) and h = beta*H (dimensionless field).
"""

import torch


def energy(sample, K=1.0, h=0.0):
    """
    Compute the dimensionless energy beta*E of spin configurations on a 1D periodic ring.

    Parameters
    ----------
    sample : Tensor, shape (batch, N)
        Spin configurations with values in {-1, +1}.
    K : float
        Dimensionless coupling beta*J.
    h : float
        Dimensionless external field beta*H.

    Returns
    -------
    Tensor, shape (batch,)
        Dimensionless energy beta*E of each configuration.
    """
    nn_term = torch.sum(sample * torch.roll(sample, -1, dims=1), dim=1)
    field_term = torch.sum(sample, dim=1)
    return -K * nn_term - h * field_term
