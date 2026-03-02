"""
Generate all figures for a given N from sweep data and observables.

Usage:
    source venv/bin/activate
    python figures/generate_N32_figures.py --N 32
    python figures/generate_N32_figures.py --N 64 --skip-training
    python figures/generate_N32_figures.py --N 128 --skip-training

Generates from sweep_Kh_N{N}.npz:
  - Exact phase diagram heatmap
  - Delta F heatmap (NMF vs VAN)
  - Bias ablation heatmap
  - H0 slice (multi-N, auto-detects all available)
  - Magnetization vs h

Generates from sweep_observables_N{N}.npz:
  - 6-panel observable heatmaps (VAN vs Exact)
  - VAN observable error heatmaps

Generates from live training (unless --skip-training):
  - Convergence curves
  - Parameter visualization
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures.style import ensure_output_dir, OUTPUT_DIR, RESULTS_DIR

N = 32  # default, overridden by --N CLI arg

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# Colors
C_EXACT = 'black'
C_VAN = '#1f77b4'
C_NMF = '#d62728'


def save(fig, name):
    """Save figure as PDF and PNG with N32 suffix."""
    ensure_output_dir()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_N{N}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_N{N}.png"))
    print(f"  Saved {name}_N{N}.pdf/.png")


# ============================================================
# 1. Exact phase diagram heatmap
# ============================================================
def fig_exact_phase_diagram():
    print("Generating: exact phase diagram...")
    data = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz"))
    K_grid, h_grid = data['K_grid'], data['h_grid']
    exact_m = data['exact_m']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(h_grid, K_grid, exact_m,
                       cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$K = \beta J$')
    ax.set_title(f'Exact magnetization $\\langle m \\rangle$ ($N={N}$)')
    ax.set_yscale('log')
    plt.colorbar(im, ax=ax, label=r'$\langle m \rangle$')
    save(fig, 'fig2_exact_phase_diagram')
    plt.close(fig)


# ============================================================
# 2. Delta F heatmap (NMF vs VAN)
# ============================================================
def fig_delta_F_heatmap():
    print("Generating: delta F heatmap (NMF vs VAN)...")
    data = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz"))
    K_grid, h_grid = data['K_grid'], data['h_grid']
    log_nmf = np.log10(np.maximum(np.abs(data['delta_f_nmf']), 1e-16))
    log_van = np.log10(np.maximum(np.abs(data['delta_f_van_bias']), 1e-16))

    vmin = min(log_nmf.min(), log_van.min())
    vmax = max(log_nmf.max(), log_van.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    im1 = ax1.pcolormesh(h_grid, K_grid, log_nmf,
                         cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_xlabel(r'$h = \beta H$')
    ax1.set_ylabel(r'$K = \beta J$')
    ax1.set_title(r'NMF: $\log_{10}|\Delta F|$')
    ax1.set_yscale('log')
    plt.colorbar(im1, ax=ax1, label=r'$\log_{10}|\Delta F|$')

    im2 = ax2.pcolormesh(h_grid, K_grid, log_van,
                         cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_xlabel(r'$h = \beta H$')
    ax2.set_title(f'VAN ($N={N}$): $\\log_{{10}}|\\Delta F|$')
    ax2.set_yscale('log')
    plt.colorbar(im2, ax=ax2, label=r'$\log_{10}|\Delta F|$')

    fig.suptitle(f'Free energy error: NMF vs VAN ($N={N}$)', fontsize=14)
    plt.tight_layout()
    save(fig, 'fig3_delta_F_heatmap')
    plt.close(fig)


# ============================================================
# 3. Bias ablation heatmap
# ============================================================
def fig_bias_ablation():
    print("Generating: bias ablation heatmap...")
    data = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz"))
    K_grid, h_grid = data['K_grid'], data['h_grid']
    log_nobias = np.log10(np.maximum(np.abs(data['delta_f_van_nobias']), 1e-16))
    log_bias = np.log10(np.maximum(np.abs(data['delta_f_van_bias']), 1e-16))

    vmin = min(log_nobias.min(), log_bias.min())
    vmax = max(log_nobias.max(), log_bias.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    im1 = ax1.pcolormesh(h_grid, K_grid, log_nobias,
                         cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_xlabel(r'$h = \beta H$')
    ax1.set_ylabel(r'$K = \beta J$')
    ax1.set_title(r'No bias: $\log_{10}|\Delta F|$')
    ax1.set_yscale('log')
    plt.colorbar(im1, ax=ax1, label=r'$\log_{10}|\Delta F|$')

    im2 = ax2.pcolormesh(h_grid, K_grid, log_bias,
                         cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
    ax2.set_xlabel(r'$h = \beta H$')
    ax2.set_title(r'With bias: $\log_{10}|\Delta F|$')
    ax2.set_yscale('log')
    plt.colorbar(im2, ax=ax2, label=r'$\log_{10}|\Delta F|$')

    fig.suptitle(f'Bias ablation: VAN without vs with bias ($N={N}$)', fontsize=14)
    plt.tight_layout()
    save(fig, 'fig4_bias_ablation')
    plt.close(fig)


# ============================================================
# 4. H=0 slice (multi-N)
# ============================================================
def fig_h0_slice():
    print("Generating: H0 slice (multi-N)...")
    import glob
    pattern = os.path.join(RESULTS_DIR, "sweep_Kh_N*.npz")
    files = glob.glob(pattern)
    # Filter out chunk files
    files = [f for f in files if '_chunk' not in f]
    system_sizes = sorted(
        int(os.path.basename(f).replace("sweep_Kh_N", "").replace(".npz", ""))
        for f in files
    )
    if not system_sizes:
        print("  No sweep files found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for sz in system_sizes:
        data = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{sz}.npz"))
        K_grid, h_grid = data['K_grid'], data['h_grid']
        h0_idx = np.argmin(np.abs(h_grid))
        delta_f = data['delta_f_van_bias'][:, h0_idx]
        ax.semilogy(K_grid, np.abs(delta_f), 'o-', label=f'N={sz}', markersize=3)

    ax.set_xlabel(r'$K = \beta J$')
    ax.set_ylabel(r'$|\Delta F / N|$')
    ax.set_title('VAN free energy error at $h=0$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, 'fig5_H0_slice')
    plt.close(fig)


# ============================================================
# 5. Magnetization vs h
# ============================================================
def fig_magnetization():
    print("Generating: magnetization vs h...")
    data = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz"))
    K_grid, h_grid = data['K_grid'], data['h_grid']

    K_values = [0.5, 1.0, 2.0, 5.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(8, 5))
    for K_target, color in zip(K_values, colors):
        K_idx = np.argmin(np.abs(K_grid - K_target))
        K_actual = K_grid[K_idx]
        exact_m = data['exact_m'][K_idx, :]
        nmf_m = data['nmf_m'][K_idx, :]

        mask = (h_grid >= -0.5) & (h_grid <= 0.5)
        h_zoom = h_grid[mask]
        ax.plot(h_zoom, exact_m[mask], '-', color=color, lw=2,
                label=f'Exact $K={K_actual:.2f}$')
        ax.plot(h_zoom, nmf_m[mask], '--', color=color, lw=1.5,
                label=f'NMF $K={K_actual:.2f}$')

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\langle m \rangle$')
    ax.set_title(f'Magnetization vs $h$ ($N={N}$)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    save(fig, 'fig6_magnetization')
    plt.close(fig)


# ============================================================
# 6. Observable heatmaps (VAN vs Exact, 6 panels)
# ============================================================
def fig_observable_heatmaps():
    print("Generating: observable heatmaps (VAN vs Exact)...")
    obs_path = os.path.join(RESULTS_DIR, f"sweep_observables_N{N}.npz")
    sweep_path = os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz")
    if not os.path.exists(obs_path):
        print("  sweep_observables not found, skipping.")
        return

    obs = np.load(obs_path)
    sweep = np.load(sweep_path)
    K_grid = obs['K_grid']
    h_grid_pos = obs['h_grid']  # h >= 0 only

    from src.exact import analytical_formulas as af

    # Compute exact observables over h >= 0 grid
    nK, nh = len(K_grid), len(h_grid_pos)
    exact = {
        'm': np.zeros((nK, nh)),
        'nn': np.zeros((nK, nh)),
        'chi': np.zeros((nK, nh)),
        'energy': np.zeros((nK, nh)),
        'C': np.zeros((nK, nh)),
        'f': np.zeros((nK, nh)),
    }
    for i, K in enumerate(K_grid):
        for j, h in enumerate(h_grid_pos):
            exact['m'][i, j] = af.magnetization(K, h, N)
            exact['nn'][i, j] = af.nn_correlation(K, h, N)
            exact['chi'][i, j] = af.susceptibility(K, h, N)
            exact['energy'][i, j] = af.energy_per_spin(K, h, N)
            exact['C'][i, j] = af.specific_heat(K, h, N)
            exact['f'][i, j] = af.free_energy_per_spin(K, h, N)

    # --- Panel A: VAN observables ---
    obs_panels = [
        ('van_magnetization_mean', exact['m'], r'$\langle m \rangle$', 'Magnetization'),
        ('van_energy_mean', exact['energy'], r'$\beta u$', 'Energy per spin'),
        ('van_chi_bare_mean', exact['chi'], r'$\chi_{\mathrm{bare}}$', 'Susceptibility'),
        ('van_specific_heat_bare_mean', exact['C'], r'$C/k_B$', 'Heat capacity'),
        ('van_free_energy_mean', exact['f'], r'$\beta f / N$', 'Free energy'),
        ('van_nn_correlation_mean', exact['nn'], r'$\langle s_i s_{i+1} \rangle$', 'NN correlation'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for idx, (van_key, exact_arr, cbar_label, title) in enumerate(obs_panels):
        ax = axes.ravel()[idx]
        van_arr = obs[van_key]
        im = ax.pcolormesh(h_grid_pos, K_grid, van_arr,
                           cmap='viridis', shading='auto')
        ax.set_xlabel(r'$h = \beta H$')
        if idx % 3 == 0:
            ax.set_ylabel(r'$K = \beta J$')
        ax.set_yscale('log')
        ax.set_title(f'{panel_labels[idx]} {title}')
        plt.colorbar(im, ax=ax, label=cbar_label)

    fig.suptitle(f'VAN observables ($N={N}$, $h \\geq 0$)', fontsize=14)
    plt.tight_layout()
    save(fig, 'fig_van_observables')
    plt.close(fig)

    # --- Panel B: Observable errors |VAN - Exact| ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (van_key, exact_arr, cbar_label, title) in enumerate(obs_panels):
        ax = axes.ravel()[idx]
        van_arr = obs[van_key]
        error = np.abs(van_arr - exact_arr)
        log_err = np.log10(np.maximum(error, 1e-16))

        im = ax.pcolormesh(h_grid_pos, K_grid, log_err,
                           cmap='inferno', shading='auto')
        ax.set_xlabel(r'$h = \beta H$')
        if idx % 3 == 0:
            ax.set_ylabel(r'$K = \beta J$')
        ax.set_yscale('log')
        ax.set_title(f'{panel_labels[idx]} {title} error')
        plt.colorbar(im, ax=ax, label=r'$\log_{10}|$error$|$')

    fig.suptitle(f'VAN observable errors vs Exact ($N={N}$)', fontsize=14)
    plt.tight_layout()
    save(fig, 'fig_observable_errors')
    plt.close(fig)


# ============================================================
# 7. Observable slices at fixed K (from sweep data)
# ============================================================
def fig_observable_slices():
    """Plot observables vs h at fixed K values, comparing VAN/Exact/NMF."""
    print("Generating: observable slices at fixed K...")
    obs_path = os.path.join(RESULTS_DIR, f"sweep_observables_N{N}.npz")
    if not os.path.exists(obs_path):
        print("  sweep_observables not found, skipping.")
        return

    obs = np.load(obs_path)
    sweep = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz"))
    K_grid = obs['K_grid']
    h_grid = obs['h_grid']  # h >= 0

    from src.exact import analytical_formulas as af
    from src.nmf.mean_field import solve as nmf_solve

    K_values = [0.5, 1.0, 2.0, 5.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Compute exact and NMF for these K values at the h grid
    exact_data = {}
    nmf_data = {}
    for K in K_values:
        exact_data[K] = {
            'm': np.array([af.magnetization(K, h, N) for h in h_grid]),
            'nn': np.array([af.nn_correlation(K, h, N) for h in h_grid]),
            'chi': np.array([af.susceptibility(K, h, N) for h in h_grid]),
            'f': np.array([af.free_energy_per_spin(K, h, N) for h in h_grid]),
        }
        nmf_m, nmf_f = [], []
        for h in h_grid:
            r = nmf_solve(K, h)
            nmf_m.append(r.magnetization)
            nmf_f.append(r.free_energy_per_spin)
        nmf_data[K] = {'m': np.array(nmf_m), 'f': np.array(nmf_f)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Magnetization
    ax = axes[0, 0]
    for ki, K in enumerate(K_values):
        K_idx = np.argmin(np.abs(K_grid - K))
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        # Exact
        em = exact_data[K]['m']
        em_full = np.concatenate([-em[-1:0:-1], em])
        ax.plot(h_full, em_full, color=colors[ki], label=f'$K={K}$')
        # VAN
        vm = obs['van_magnetization_mean'][K_idx]
        vm_full = np.concatenate([-vm[-1:0:-1], vm])
        ax.plot(h_full, vm_full, 'o', color=colors[ki], markersize=3, alpha=0.7)
        # NMF
        nm = nmf_data[K]['m']
        # Handle NMF discontinuity at h=0
        nm_neg = -nm[::-1]
        nm_pos = nm
        h_neg = -h_grid[::-1]
        h_pos = h_grid
        nmf_h_full = np.concatenate([h_neg, [np.nan], h_pos])
        nmf_m_full = np.concatenate([nm_neg, [np.nan], nm_pos])
        ax.plot(nmf_h_full, nmf_m_full, '--', color=colors[ki], alpha=0.6)

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\langle m \rangle$')
    ax.set_title('(a) Magnetization')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # (b) Susceptibility chi_bare
    ax = axes[0, 1]
    for ki, K in enumerate(K_values):
        K_idx = np.argmin(np.abs(K_grid - K))
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        ec = exact_data[K]['chi']
        ec_full = np.concatenate([ec[-1:0:-1], ec])
        ax.plot(h_full, ec_full, color=colors[ki], label=f'$K={K}$')
        vc = obs['van_chi_bare_mean'][K_idx]
        vc_full = np.concatenate([vc[-1:0:-1], vc])
        ax.plot(h_full, vc_full, 'o', color=colors[ki], markersize=3, alpha=0.7)

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\chi_{\mathrm{bare}}$')
    ax.set_title(r'(b) Susceptibility')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # (c) NN correlation
    ax = axes[1, 0]
    for ki, K in enumerate(K_values):
        K_idx = np.argmin(np.abs(K_grid - K))
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        en = exact_data[K]['nn']
        en_full = np.concatenate([en[-1:0:-1], en])
        ax.plot(h_full, en_full, color=colors[ki], label=f'$K={K}$')
        vn = obs['van_nn_correlation_mean'][K_idx]
        vn_full = np.concatenate([vn[-1:0:-1], vn])
        ax.plot(h_full, vn_full, 'o', color=colors[ki], markersize=3, alpha=0.7)

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$\langle s_i s_{i+1} \rangle$')
    ax.set_title('(c) NN correlation')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # (d) Free energy error
    ax = axes[1, 1]
    for ki, K in enumerate(K_values):
        K_idx = np.argmin(np.abs(K_grid - K))
        h_full = np.concatenate([-h_grid[-1:0:-1], h_grid])
        vf = obs['van_free_energy_mean'][K_idx]
        ef = exact_data[K]['f']
        van_df = np.abs(vf - ef)
        van_df_full = np.concatenate([van_df[-1:0:-1], van_df])
        nmf_df = np.abs(nmf_data[K]['f'] - ef)
        nmf_df_full = np.concatenate([nmf_df[-1:0:-1], nmf_df])

        ax.plot(h_full, nmf_df_full, '--', color=colors[ki], alpha=0.6,
                label=f'NMF $K={K}$')
        ax.plot(h_full, van_df_full, 'o-', color=colors[ki], markersize=3,
                alpha=0.7, label=f'VAN $K={K}$')

    ax.set_xlabel(r'$h = \beta H$')
    ax.set_ylabel(r'$|\Delta F / N|$')
    ax.set_title('(d) Free energy error')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)

    for ax in axes.ravel():
        ax.set_xlim(-2, 2)

    fig.text(0.5, 0.01,
             'Solid: Exact | Circles: VAN | Dashed: NMF',
             ha='center', fontsize=10, style='italic')
    fig.suptitle(f'Observable comparison vs $h$ ($N={N}$)', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save(fig, 'fig_observable_slices')
    plt.close(fig)


# ============================================================
# 8. Convergence curves (trains live)
# ============================================================
def fig_convergence():
    print("Generating: convergence curves (training live)...")
    from src.van.train import train, TrainConfig
    from src.exact.transfer_matrix import free_energy_per_spin
    from src.nmf.mean_field import solve as nmf_solve

    test_points = [
        (1.0, 0.0, 'K=1, h=0'),
        (1.0, 0.5, 'K=1, h=0.5'),
        (0.5, 0.0, 'K=0.5, h=0'),
        (2.0, 0.0, 'K=2, h=0'),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_points)))

    for idx, (K, h, label) in enumerate(test_points):
        print(f"  Training {label}...")
        config = TrainConfig(N=N, K=K, h=h,
                             batch_size=1000, lr=0.01, max_step=5000, seed=42)
        result = train(config)
        bf_exact = free_energy_per_spin(K, h, N)
        nmf_result = nmf_solve(K, h)
        bf_nmf = nmf_result.free_energy_per_spin

        ax.plot(result.free_energy_history, color=colors[idx],
                label=label, alpha=0.8)
        ax.axhline(bf_exact, color=colors[idx], linestyle='-',
                   alpha=0.3, lw=0.8)
        ax.axhline(bf_nmf, color=colors[idx], linestyle='--',
                   alpha=0.3, lw=0.8)

    ax.plot([], [], 'k-', alpha=0.3, lw=0.8, label=r'Exact $\beta f$')
    ax.plot([], [], 'k--', alpha=0.3, lw=0.8, label=r'NMF $\beta f$')

    ax.set_xlabel('Training step')
    ax.set_ylabel(r'$\beta F_q / N$')
    ax.set_title(f'VAN convergence ($N={N}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, 'fig10_convergence')
    plt.close(fig)


# ============================================================
# 9. Parameter visualization (trains live)
# ============================================================
def fig_parameters():
    print("Generating: parameter visualization (training live)...")
    from src.van.train import train, TrainConfig

    h_values = [0.0, 0.5, 1.0]
    K = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for h in h_values:
        print(f"  Training K={K}, h={h}...")
        config = TrainConfig(N=N, K=K, h=h,
                             batch_size=1000, lr=0.01, max_step=5000, seed=42)
        result = train(config)
        params = result.parameters

        axes[0].plot(range(N), params['b'], 'o-', label=f'h={h}', markersize=3)
        W = params['W']
        sub_diag = [W[i, i-1] for i in range(1, N)]
        axes[1].plot(range(1, N), sub_diag, 'o-', label=f'h={h}', markersize=3)

    axes[0].set_xlabel('Site index $i$')
    axes[0].set_ylabel('$b_i$')
    axes[0].set_title(f'Converged biases ($N={N}$, $K={K}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linewidth=0.5)

    axes[1].set_xlabel('Site index $i$')
    axes[1].set_ylabel('$W_{i,i-1}$')
    axes[1].set_title(f'Converged NN weights ($N={N}$, $K={K}$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, 'fig9_parameters')
    plt.close(fig)


# ============================================================
# 10. Free energy h-slices from sweep data
# ============================================================
def fig_free_energy_h_slices():
    """Free energy vs h at fixed K values, using sweep data."""
    print("Generating: free energy h-slices...")
    sweep = np.load(os.path.join(RESULTS_DIR, f"sweep_Kh_N{N}.npz"))
    K_grid = sweep['K_grid']
    h_grid = sweep['h_grid']  # full grid
    exact_f_grid = sweep['exact_f']
    van_f_grid = sweep['van_bias_f_mean']
    nmf_f_grid = sweep['nmf_f']

    K_VALUES = [0.5, 1.0, 2.0, 5.0]
    PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    for idx, K in enumerate(K_VALUES):
        ax = axes.ravel()[idx]
        K_idx = np.argmin(np.abs(K_grid - K))

        exact_f = exact_f_grid[K_idx]
        van_f = van_f_grid[K_idx]
        nmf_f = nmf_f_grid[K_idx]

        # Dimensional free energy: F/N = (beta*f) / K
        FN_exact = exact_f / K
        FN_van = van_f / K
        FN_nmf = nmf_f / K

        ax.plot(h_grid, FN_exact, '-', color=C_EXACT, lw=2, label='Exact')
        ax.plot(h_grid, FN_van, 'o', color=C_VAN, ms=3, label='VAN', zorder=5)
        ax.plot(h_grid, FN_nmf, '--', color=C_NMF, lw=1.5, label='NMF')

        ax.set_xlabel(r'$h = \beta H$')
        ax.set_ylabel(r'$F/N$  (units of $J$)')
        ax.set_title(f'{PANEL_LABELS[idx]} $K = {K_grid[K_idx]:.2f}$')
        ax.axvline(0, color='gray', lw=0.5, ls='--')

        if idx == 0:
            ax.legend(loc='upper center', fontsize=9)

        # Inset: relative error
        mask_h = h_grid >= 0
        h_pos = h_grid[mask_h]
        rel_err_van = np.abs(van_f[mask_h] - exact_f[mask_h]) / np.abs(exact_f[mask_h])
        rel_err_nmf = np.abs(nmf_f[mask_h] - exact_f[mask_h]) / np.abs(exact_f[mask_h])
        rel_err_van = np.clip(rel_err_van, 1e-16, None)
        rel_err_nmf = np.clip(rel_err_nmf, 1e-16, None)

        ax_inset = inset_axes(ax, width="40%", height="35%", loc='lower center')
        ax_inset.semilogy(h_pos, rel_err_nmf, '--', color=C_NMF, lw=1.0)
        ax_inset.semilogy(h_pos, rel_err_van, '-', color=C_VAN, lw=1.0)
        ax_inset.set_xlabel(r'$h$', fontsize=7)
        ax_inset.set_ylabel('Rel. err.', fontsize=7)
        ax_inset.tick_params(labelsize=6)

    fig.text(0.5, 0.01,
             'Solid: Exact | Circles: VAN | Dashed: NMF',
             ha='center', fontsize=10, style='italic')
    fig.suptitle(f'Free energy vs field ($N={N}$)', fontsize=14, y=0.99)
    fig.subplots_adjust(hspace=0.35, wspace=0.3, bottom=0.06, top=0.93)
    save(fig, 'fig12_free_energy_h_slices')
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate figures for a given N")
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip convergence/parameter plots (which train live)")
    args = parser.parse_args()
    N = args.N

    print(f"Generating all N={N} figures...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # From sweep data (fast)
    fig_exact_phase_diagram()
    fig_delta_F_heatmap()
    fig_bias_ablation()
    fig_h0_slice()
    fig_magnetization()
    fig_free_energy_h_slices()

    # From sweep_observables (fast, data already computed)
    fig_observable_heatmaps()
    fig_observable_slices()

    # From live training (slower — trains 7 models)
    if not args.skip_training:
        fig_convergence()
        fig_parameters()
    else:
        print("\nSkipping live training plots (--skip-training)")

    print(f"\nDone! All figures saved to {OUTPUT_DIR}/")
