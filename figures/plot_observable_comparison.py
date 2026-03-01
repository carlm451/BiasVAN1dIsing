"""Figure 5: Multi-K observable comparison vs h.

2x2 panels with 4 K values (K=0.5, 1.0, 2.0, 5.0), N=16.
Three-way comparison: Exact (solid), VAN (circles), NMF (dashed).

Panels:
  (a) Magnetization M vs h
  (b) Susceptibility chi vs h
  (c) NN correlation <s_i s_{i+1}> vs h
  (d) Free energy error |DeltaF/N| vs h

Data: results/observables_K{K}_N16.npz from obs_inference.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR


K_VALUES = [0.5, 1.0, 2.0, 5.0]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
K_LABELS = [f'$K={K}$' for K in K_VALUES]
N = 16


def plot():
    # Load data for all K values
    datasets = {}
    for K in K_VALUES:
        path = os.path.join(RESULTS_DIR, f"observables_K{K}_N{N}.npz")
        if not os.path.exists(path):
            print(f"Data not found: {path}")
            print("Run experiments/obs_train.py then experiments/obs_inference.py first.")
            return
        datasets[K] = np.load(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Magnetization
    ax = axes[0, 0]
    for ki, K in enumerate(K_VALUES):
        d = datasets[K]
        h_grid = d['h_grid']
        # Mirror to negative h: M(-h) = -M(h)
        # For exact and VAN, M is continuous through h=0 (M=0), so single h=0 is fine.
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        exact_m_full = np.concatenate([-d['exact_m'][-1:0:-1], d['exact_m']])
        van_m_full = np.concatenate([-d['van_magnetization_mean'][-1:0:-1],
                                     d['van_magnetization_mean']])

        # For NMF below T_c: m(h=0) = +m* (spontaneous), so approaching from h<0
        # gives -m* and from h>0 gives +m*. Include h=0 in BOTH branches with a
        # NaN gap between them so the discontinuity is visible as a break, not a
        # diagonal connecting line.
        nmf_m = d['nmf_m']
        nmf_m_neg = -nmf_m[::-1]   # h<0 branch: includes h=0 → -m*
        nmf_m_pos = nmf_m           # h>0 branch: includes h=0 → +m*
        h_neg = -h_grid[::-1]       # [..., -0.06, 0.0]
        h_pos = h_grid              # [0.0, 0.06, ...]
        # Insert NaN gap between the two branches
        nmf_h_full = np.concatenate([h_neg, [np.nan], h_pos])
        nmf_m_full = np.concatenate([nmf_m_neg, [np.nan], nmf_m_pos])

        ax.plot(h_full, exact_m_full, color=COLORS[ki], label=K_LABELS[ki])
        ax.plot(h_full, van_m_full, 'o', color=COLORS[ki], markersize=3, alpha=0.7)
        ax.plot(nmf_h_full, nmf_m_full, '--', color=COLORS[ki], alpha=0.6)

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\langle m \rangle$')
    ax.set_title('(a) Magnetization')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # (b) Susceptibility chi = beta * chi_bare
    ax = axes[0, 1]
    K_c_nmf = 0.5  # NMF spurious critical coupling
    for ki, K in enumerate(K_VALUES):
        d = datasets[K]
        h_grid = d['h_grid']
        # chi is symmetric in h
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        exact_chi_full = np.concatenate([d['exact_chi'][-1:0:-1], d['exact_chi']])
        van_chi_full = np.concatenate([d['van_chi_bare_mean'][-1:0:-1],
                                       d['van_chi_bare_mean']])
        nmf_chi = d['nmf_chi'].copy()
        # For K > K_c, NMF has a first-order jump at h=0, so chi is a delta
        # function there — the intra-branch formula is misleading. Exclude h=0.
        if K > K_c_nmf:
            nmf_chi[0] = np.nan
        nmf_chi_full = np.concatenate([nmf_chi[-1:0:-1], nmf_chi])

        ax.plot(h_full, exact_chi_full, color=COLORS[ki], label=K_LABELS[ki])
        ax.plot(h_full, van_chi_full, 'o', color=COLORS[ki], markersize=3, alpha=0.7)
        ax.plot(h_full, nmf_chi_full, '--', color=COLORS[ki], alpha=0.6)

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\chi_{\mathrm{bare}}$')
    ax.set_title(r'(b) Susceptibility')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # (c) NN correlation
    ax = axes[1, 0]
    for ki, K in enumerate(K_VALUES):
        d = datasets[K]
        h_grid = d['h_grid']
        # <ss> is symmetric in h
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        exact_nn_full = np.concatenate([d['exact_nn'][-1:0:-1], d['exact_nn']])
        van_nn_full = np.concatenate([d['van_nn_correlation_mean'][-1:0:-1],
                                      d['van_nn_correlation_mean']])
        nmf_nn_full = np.concatenate([d['nmf_nn'][-1:0:-1], d['nmf_nn']])

        ax.plot(h_full, exact_nn_full, color=COLORS[ki], label=K_LABELS[ki])
        ax.plot(h_full, van_nn_full, 'o', color=COLORS[ki], markersize=3, alpha=0.7)
        ax.plot(h_full, nmf_nn_full, '--', color=COLORS[ki], alpha=0.6)

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\langle s_i s_{i+1} \rangle$')
    ax.set_title('(c) NN correlation')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # (d) Free energy error |DeltaF/N|
    ax = axes[1, 1]
    for ki, K in enumerate(K_VALUES):
        d = datasets[K]
        h_grid = d['h_grid']
        # Error is symmetric in h
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        van_df = np.abs(d['van_free_energy_mean'] - d['exact_f'])
        nmf_df = np.abs(d['nmf_f'] - d['exact_f'])
        van_df_full = np.concatenate([van_df[-1:0:-1], van_df])
        nmf_df_full = np.concatenate([nmf_df[-1:0:-1], nmf_df])

        ax.plot(h_full, nmf_df_full, '--', color=COLORS[ki],
                alpha=0.6, label=f'NMF {K_LABELS[ki]}')
        ax.plot(h_full, van_df_full, 'o-', color=COLORS[ki],
                markersize=3, alpha=0.7, label=f'VAN {K_LABELS[ki]}')

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$|\Delta F / N|$')
    ax.set_title('(d) Free energy error')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)

    # Restrict h range to [-1, 1] to better show crossover differences
    for ax in axes.ravel():
        ax.set_xlim(-1, 1)

    # Add legend note for line styles
    fig.text(0.5, 0.01,
             'Solid: Exact | Circles: VAN | Dashed: NMF',
             ha='center', fontsize=10, style='italic')

    fig.suptitle(f'Observable comparison vs $h$ ($N={N}$)', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    savefig(fig, 'fig5_observable_comparison')
    plt.close(fig)


if __name__ == "__main__":
    plot()
