"""Figure 6b: h=0 observables vs temperature T = 1/K (log-scale T axis).

Same data and layout as Fig 6 but with log-scale T axis to better resolve
the low-temperature regime where VAN and NMF differences are most pronounced.

Data: results/h0_observables_N16.npz from h0_inference.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from figures.style import savefig, RESULTS_DIR

N = 16


def plot():
    data_path = os.path.join(RESULTS_DIR, f"h0_observables_N{N}.npz")
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        print("Run experiments/h0_train.py then experiments/h0_inference.py first.")
        return

    d = np.load(data_path)
    K_grid = d['K_grid']
    T_grid = 1.0 / K_grid  # T = 1/K in natural units (J/k_B = 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) NN correlation
    ax = axes[0, 0]
    ax.plot(T_grid, d['exact_nn'], 'k-', lw=1.5, label='Exact')
    ax.plot(T_grid, d['van_nn_correlation_mean'], 'o', color='#1f77b4',
            markersize=4, label='VAN')
    ax.plot(T_grid, d['nmf_nn'], '--', color='#d62728', lw=1.5, label='NMF')
    ax.plot(T_grid, np.tanh(K_grid), ':', color='gray', lw=1.0,
            label=r'$\tanh(1/T)$')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\langle s_i s_{i+1} \rangle$')
    ax.set_title(r'(a) NN correlation')
    ax.legend(fontsize=9)

    # (b) Susceptibility chi/beta (= chi_bare) — log-log
    ax = axes[0, 1]
    ax.plot(T_grid, d['exact_chi'], 'k-', lw=1.5, label='Exact')
    ax.plot(T_grid, d['van_chi_bare_mean'], 'o', color='#1f77b4',
            markersize=4, label='VAN')
    nmf_chi = d['nmf_chi'].copy()
    nmf_chi_finite = np.where(np.isfinite(nmf_chi), nmf_chi, np.nan)
    ax.plot(T_grid, nmf_chi_finite, '--', color='#d62728', lw=1.5, label='NMF')
    ax.plot(T_grid, np.exp(2.0 / T_grid), ':', color='gray', lw=1.0,
            label=r'$e^{2/T}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\chi / \beta$')
    ax.set_title(r'(b) Susceptibility')
    ax.legend(fontsize=9)

    # (c) Heat capacity C/k_B
    ax = axes[1, 0]
    ax.plot(T_grid, d['exact_C'], 'k-', lw=1.5, label='Exact')
    ax.plot(T_grid, d['van_specific_heat_bare_mean'], 'o', color='#1f77b4',
            markersize=4, label='VAN')
    ax.plot(T_grid, d['nmf_C'], '--', color='#d62728', lw=1.5, label='NMF')
    schottky = K_grid**2 / np.cosh(K_grid)**2
    ax.plot(T_grid, schottky, ':', color='gray', lw=1.0,
            label=r'$K^2 \mathrm{sech}^2(K)$')
    ax.set_xscale('log')
    ax.set_ylabel(r'$C / k_B$')
    ax.set_title(r'(c) Heat capacity')
    ax.legend(fontsize=9)

    # (d) Free energy error |DeltaF/N|
    ax = axes[1, 1]
    van_df = np.abs(d['van_free_energy_mean'] - d['exact_f'])
    nmf_df = np.abs(d['nmf_f'] - d['exact_f'])
    ax.plot(T_grid, nmf_df, '--', color='#d62728', lw=1.5, label='NMF')
    ax.plot(T_grid, van_df, 'o-', color='#1f77b4', markersize=4, label='VAN')
    ax.axhline(np.log(2), color='silver', lw=0.8, ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$|\Delta F / N|$')
    ax.set_title(r'(d) Free energy error')
    ax.legend(fontsize=9)

    # Add reference vertical lines to all panels
    T_nmf = 2.0    # NMF spurious critical temperature
    T_s = 0.83     # Exact Schottky anomaly peak
    for ax in axes.ravel():
        ax.axvline(T_nmf, color='#d62728', lw=1.0, ls='--', alpha=0.5)
        ax.axvline(T_s, color='#1f77b4', lw=1.0, ls='--', alpha=0.5)
        ax.set_xlabel(r'$T = J / (k_B K)$')
        ax.grid(True, alpha=0.2)
    # Label the reference lines in one panel each (avoid clutter)
    axes[0, 1].text(T_nmf * 1.15, 1e2,
                    r'$T_c^{\mathrm{NMF}}$', fontsize=9, color='#d62728')
    axes[1, 0].text(T_s * 1.15, axes[1, 0].get_ylim()[1] * 0.85,
                    r'$T_s$', fontsize=9, color='#1f77b4')

    fig.suptitle(f'$h=0$ observables vs temperature ($N={N}$)', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'fig6b_h0_comparison_logT')
    plt.close(fig)


if __name__ == "__main__":
    plot()
