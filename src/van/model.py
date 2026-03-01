"""
One-layer Variational Autoregressive Network (VAN) with tanh+bias parameterization.

The model uses a single masked linear layer with tanh activation:
    mu_i = tanh(b_i + sum_{j<i} W_{ij} * s_j)     [magnetization form]
    q(s_i=+1 | s_{<i}) = (1 + mu_i) / 2            [probability form]

Parameters:
    W: (N, N) weight matrix, masked to strictly lower-triangular
    b: (N,) bias vector (optional via use_bias flag)

This connects to NMF when W=0: mu_i = tanh(b_i), a uniform mean-field.
"""

import torch
import torch.nn as nn
import numpy as np
from .utils import DEFAULT_DTYPE


class OneLayerVAN(nn.Module):
    def __init__(self, N, use_bias=True, z2=False, dtype=None):
        """
        Parameters
        ----------
        N : int
            Number of spins (system size).
        use_bias : bool
            Whether to include bias parameters.
        z2 : bool
            Whether to enforce Z2 spin-flip symmetry.
            Use only when h=0 (the Hamiltonian has Z2 symmetry).
        dtype : torch.dtype
            Data type. Defaults to float64.
        """
        super().__init__()
        self.N = N
        self.use_bias = use_bias
        self.z2 = z2
        self.dtype = dtype or DEFAULT_DTYPE

        # Weight matrix W (N, N) — will be masked to strictly lower triangular
        self.W = nn.Parameter(torch.randn(N, N, dtype=self.dtype) * 0.01)

        # Bias vector b (N,)
        if use_bias:
            self.b = nn.Parameter(torch.zeros(N, dtype=self.dtype))
        else:
            self.register_buffer('b', torch.zeros(N, dtype=self.dtype))

        # Strictly lower triangular mask (1 below diagonal, 0 on and above)
        mask = torch.tril(torch.ones(N, N, dtype=self.dtype), diagonal=-1)
        self.register_buffer('mask', mask)

    def conditional_magnetization(self, sample):
        """
        Compute conditional magnetizations mu_i for each site.

        Parameters
        ----------
        sample : Tensor, shape (batch, N) with values in {-1, +1}

        Returns
        -------
        mu : Tensor, shape (batch, N) with values in (-1, 1)
        """
        # W_masked: strictly lower triangular
        W_masked = self.W * self.mask
        # linear: (batch, N) @ (N, N)^T -> (batch, N)
        linear = torch.matmul(sample, W_masked.t()) + self.b
        return torch.tanh(linear)

    def logits(self, sample):
        """
        Compute pre-activation logits a_i for each site.

        Parameters
        ----------
        sample : Tensor, shape (batch, N) with values in {-1, +1}

        Returns
        -------
        a : Tensor, shape (batch, N)
            Pre-activations such that mu_i = tanh(a_i).
            Note: p(s_i=+1) = (1+tanh(a_i))/2 = sigmoid(2*a_i),
            so the Bernoulli logit is 2*a_i.
        """
        W_masked = self.W * self.mask
        return torch.matmul(sample, W_masked.t()) + self.b

    def log_prob(self, sample):
        """
        Compute log probability log q(s) = sum_i log p(s_i | s_{<i}).

        Uses the identity log((1 + tanh(a)*s) / 2) = -softplus(-2*s*a)
        to work entirely in logit space, avoiding numerical issues
        from explicit probability computation.

        Since p(s_i=+1) = sigmoid(2*a_i), we have
        log p(s_i | s_{<i}) = log sigmoid(2*s_i*a_i) = -softplus(-2*s_i*a_i).

        If z2=True, symmetrizes: log q_sym(s) = logsumexp(log q(s), log q(-s)) - log(2).

        Parameters
        ----------
        sample : Tensor, shape (batch, N) with values in {-1, +1}

        Returns
        -------
        log_q : Tensor, shape (batch,)
        """
        a = self.logits(sample)
        # Bernoulli logit = 2*a_i, so log p(s_i) = -softplus(-s_i * 2*a_i)
        log_q = torch.sum(-torch.nn.functional.softplus(-2.0 * sample * a), dim=1)
        if self.z2:
            # Evaluate log prob of the spin-flipped configuration
            a_flip = self.logits(-sample)
            # For flipped config: log q(-s) uses -s_i with a_flip_i
            # log p(-s_i | ...) = -softplus(-2*(-s_i)*a_flip_i) = -softplus(2*s_i*a_flip_i)
            log_q_flip = torch.sum(-torch.nn.functional.softplus(2.0 * sample * a_flip), dim=1)
            # Symmetrize: log[(q(s) + q(-s))/2]
            log_q = torch.logsumexp(torch.stack([log_q, log_q_flip], dim=0), dim=0) - np.log(2)
        return log_q

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Autoregressive sampling.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        samples : Tensor, shape (batch_size, N) with values in {-1, +1}
        """
        samples = torch.zeros(batch_size, self.N, dtype=self.dtype,
                               device=self.W.device)
        W_masked = self.W * self.mask

        for i in range(self.N):
            # pre-activation a_i = b_i + sum_{j<i} W_{ij} * s_j
            if i == 0:
                a_i = self.b[i].expand(batch_size)
            else:
                a_i = torch.matmul(samples[:, :i], W_masked[i, :i]) + self.b[i]
            # p(s_i=+1) = sigmoid(2*a_i), so Bernoulli logit = 2*a_i
            bits = torch.distributions.Bernoulli(logits=2.0 * a_i).sample()
            samples[:, i] = 2.0 * bits - 1.0
        if self.z2:
            # Randomly flip entire sample with 50% probability to symmetrize
            flip = torch.randint(2, (batch_size, 1), dtype=self.dtype,
                                 device=self.W.device) * 2 - 1
            samples = samples * flip
        return samples

    @classmethod
    def from_parameters(cls, W, b, z2=False):
        """Reconstruct a OneLayerVAN from saved numpy W, b arrays.

        Parameters
        ----------
        W : ndarray, shape (N, N)
            Weight matrix (already masked to lower-triangular).
        b : ndarray, shape (N,)
            Bias vector.
        z2 : bool
            Whether to enforce Z2 symmetry.

        Returns
        -------
        OneLayerVAN with loaded parameters.
        """
        N = len(b)
        model = cls(N, use_bias=True, z2=z2)
        with torch.no_grad():
            model.W.copy_(torch.from_numpy(W))
            model.b.copy_(torch.from_numpy(b))
        return model

    def get_parameters_dict(self):
        """Return numpy dict of converged W, b for analysis."""
        W_masked = (self.W * self.mask).detach().cpu().numpy()
        b = self.b.detach().cpu().numpy()
        return {'W': W_masked, 'b': b}
