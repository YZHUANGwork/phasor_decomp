"""
plot_scan_sig2_quality.py
=========================
Read scan results and plot sig2 reconstruction quality as 2D heatmaps.

Usage
-----
python plot_scan_sig2_quality.py <scan_folder>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SIM_DIR     = 'sim_data'
FIGURES_DIR = 'figures'

def load_scan(scan_dir):
    metadata            = pd.read_csv(os.path.join(scan_dir, 'metadata.csv')).set_index('param')
    df_SNR              = pd.read_csv(os.path.join(scan_dir, 'SNR.csv'),           index_col=0)
    df_power_testA      = pd.read_csv(os.path.join(scan_dir, 'power_testA.csv'),   index_col=0)
    df_power_testB      = pd.read_csv(os.path.join(scan_dir, 'power_testB.csv'),   index_col=0)
    df_bias_mod1_counts = pd.read_csv(os.path.join(scan_dir, 'bias_mod1_counts.csv'),     index_col=0)
    df_bias_phase1      = pd.read_csv(os.path.join(scan_dir, 'bias_phase1_deg.csv'), index_col=0)
    return metadata, df_SNR, df_power_testA, df_power_testB, df_bias_mod1_counts, df_bias_phase1


def plot_sig2_quality(scan_folder):
    scan_dir = os.path.join(SIM_DIR, scan_folder)
    metadata, df_SNR, df_power_testA, df_power_testB, df_bias_mod1_counts, df_bias_phase1 = load_scan(scan_dir)

    # read metadata
    Ad1        = float(metadata.loc['Ad1',         'value'])
    N1_in_bin  = float(metadata.loc['N1_in_bin',   'value'])
    Phase1_rad = float(metadata.loc['Phase1_rad',  'value'])
    Rs1        = float(metadata.loc['Rs1',         'value'])
    N2_in_bin  = float(metadata.loc['N2_in_bin',   'value'])# use Ad2_ratio to control relative modulation 2 counts, keep N the same
    D          = float(metadata.loc['D',           'value'])
    T          = float(metadata.loc['T',           'value'])
    dt         = float(metadata.loc['dt',          'value'])
    n_toys     = int(float(metadata.loc['n_toys',  'value']))
    sig        = float(metadata.loc['significance','value'])
    bias_tol   = float(metadata.loc['bias_tolerance', 'value'])

    # x-axis: mod2_true_counts
    Ad2_ratios       = np.array([float(c.split('=')[1]) for c in df_SNR.columns])
    mod2_true_counts = Ad2_ratios * Ad1 * N2_in_bin

    # y-axis: use dphi directly so it runs 0→350, no wrap ambiguity
    dphi_vals      = np.array([float(r.split('=')[1].replace('deg','')) for r in df_SNR.index])
    Phase2_true_deg = dphi_vals

    SNR                 = df_SNR.values.astype(float)
    power_testA         = df_power_testA.values.astype(float)
    power_testB         = df_power_testB.values.astype(float)
    bias_mod1_counts    = df_bias_mod1_counts.values.astype(float)
    bias_phase1         = df_bias_phase1.values.astype(float)

    # budget for bias_mod1 contour
    mod1_true_counts = Ad1 * N1_in_bin
    budget = bias_tol * mod1_true_counts

    info = (f'Rs1={Rs1} 1/(t·yr)  Ad1={Ad1}  Phase1={Phase1_rad:.3f} rad  '
            f'D={D} t  T={T} yr  dt={dt} d  n_toys={n_toys}')

    df_median_mod2_fit       = pd.read_csv(os.path.join(scan_dir, 'median_mod2_fit_counts.csv'),       index_col=0)
    df_mean_Phase2_fit       = pd.read_csv(os.path.join(scan_dir, 'mean_Phase2_fit_rad.csv'),           index_col=0)
    df_std_Phase2_fit        = pd.read_csv(os.path.join(scan_dir, 'std_Phase2_fit_rad.csv'),            index_col=0)
    df_median_mod2_fit_floor = pd.read_csv(os.path.join(scan_dir, 'median_mod2_fit_counts_floor.csv'), index_col=0)
    df_mean_Phase2_fit_floor = pd.read_csv(os.path.join(scan_dir, 'mean_Phase2_fit_rad_floor.csv'),    index_col=0)
    df_std_Phase2_fit_floor  = pd.read_csv(os.path.join(scan_dir, 'std_Phase2_fit_rad_floor.csv'),     index_col=0)

    # select up to 4 Ad2/Ad1 column slices evenly spaced
    n_ratio = len(Ad2_ratios)
    col_indices = np.linspace(0, n_ratio - 1, min(4, n_ratio), dtype=int)

    # select up to 4 dphi slices evenly spaced
    n_dphi = len(dphi_vals)
    slice_indices = np.linspace(0, n_dphi - 1, min(4, n_dphi), dtype=int)

    # ── 3×3 layout ───────────────────────────────────────────────────────────
    # Col 0: blank | median mod2 fit (sig1+sig2) | circular mean Phase2_fit
    # Col 1: SNR   | power_testA                 | power_testB
    # Col 2: blank | bias_mod1_counts            | bias_phase1_deg
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))


    # share x across rows 0-2 within col 1 and col 2
    for col in (1, 2):
        axes[1, col].sharex(axes[0, col])
        axes[2, col].sharex(axes[0, col])
    # sharex suppresses tick labels on non-primary axes by default — re-enable all
    for row in range(3):
        for col in (1, 2):
            axes[row, col].tick_params(axis='x', labelbottom=True)

    # centers for contour; edges for pcolormesh
    Xc, Yc = np.meshgrid(mod2_true_counts, Phase2_true_deg)

    def _edges(centers, log=False):
        if log:
            log_c = np.log10(centers)
            log_e = np.concatenate([[log_c[0] - (log_c[1]-log_c[0])/2],
                                     (log_c[:-1] + log_c[1:]) / 2,
                                     [log_c[-1] + (log_c[-1]-log_c[-2])/2]])
            return 10**log_e
        else:
            return np.concatenate([[centers[0] - (centers[1]-centers[0])/2],
                                    (centers[:-1] + centers[1:]) / 2,
                                    [centers[-1] + (centers[-1]-centers[-2])/2]])

    x_edges = _edges(mod2_true_counts, log=True)
    y_edges = _edges(Phase2_true_deg,  log=False)
    Xe, Ye = np.meshgrid(x_edges, y_edges)

    def _set_axes(ax):
        ax.set_xscale('log')
        ax.set_xlim(mod2_true_counts.min(), mod2_true_counts.max())
        ax.set_ylim(Phase2_true_deg.min(), Phase2_true_deg.max())
        ax.set_yticks(np.arange(0, 361, 30))
        ax.set_ylabel(r'$\Delta\phi$ [deg]')
        ax.set_xlabel('mod2_true_counts [counts]')

    # ── Row 0, Col 0: run-parameter info box ─────────────────────────────────
    ax_info = axes[0, 0]
    ax_info.axis('off')
    mod1_true_counts_display = Ad1 * N1_in_bin
    info_lines = [
        r'$\bf{Run\ parameters}$',
        f'Rs1        = {Rs1:.3g} t⁻¹yr⁻¹',
        f'Ad1        = {Ad1:.5g}',
        f'Phase1     = {Phase1_rad:.4f} rad  ({np.rad2deg(Phase1_rad):.2f}°)',
        f'N1_in_bin  = {N1_in_bin:.4g} counts',
        f'mod1_true  = {mod1_true_counts_display:.4g} counts',
        f'N2_in_bin  = {N2_in_bin:.4g} counts',
        f'D          = {D:.3g} t',
        f'T          = {T:.3g} yr',
        f'dt         = {dt:.3g} d',
        f'n_toys     = {n_toys}',
        f'bias_tol   = {bias_tol:.3g}  (budget = ±{budget:.4g})',
        f'significance = {sig:.3g}',
    ]
    ax_info.text(
        0.05, 0.97, '\n'.join(info_lines),
        transform=ax_info.transAxes, fontsize=13,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                  edgecolor='goldenrod', alpha=0.9),
    )

    # ── Row 0, Col 1: SNR ────────────────────────────────────────────────────
    ax = axes[0, 1]
    im = ax.pcolormesh(Xe, Ye, SNR, cmap='viridis',
                       norm=mcolors.LogNorm(vmin=max(SNR.min(), 1e-2), vmax=SNR.max()))
    fig.colorbar(im, ax=ax, label='SNR')
    cs1 = ax.contour(Xc, Yc, SNR, levels=[1.0], colors='white', linewidths=2)
    cs2 = ax.contour(Xc, Yc, SNR, levels=[2.0], colors='black', linewidths=2, linestyles='--')
    ax.clabel(cs1, fmt='SNR=1', fontsize=9)
    ax.clabel(cs2, fmt='SNR=2', fontsize=9)
    ax.set_title('SNR sig2 recovered after split', fontweight='bold')
    _set_axes(ax)
    ax_ann = axes[0, 2]
    ax_ann.axis('off')
    annotation_lines = [
        r'$\bf{Column\ 3:\ impact\ on\ signal\ 1}$',
        r'$\bf{when\ signal\ 2\ is\ neglected}$',
    ]
    ax_ann.text(
        0.05, 0.97, '\n'.join(annotation_lines),
        transform=ax_ann.transAxes, fontsize=13,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f4f8',
                  edgecolor='steelblue', alpha=0.9),
    )

    # ── Row 1, Col 0: median fitted sig2 modulation ──────────────────────────
    # pick dphi closest to 0° and closest to 180°
    idx_0   = int(np.argmin(np.abs(dphi_vals - 0)))
    idx_180 = int(np.argmin(np.abs(dphi_vals - 180)))
    mod2_slice_indices = [idx_0, idx_180]
    colors_line = plt.cm.tab10([0.0, 0.2])
    ax = axes[1, 0]
    for idx, color in zip(mod2_slice_indices, colors_line):
        dphi_label = f'dphi={dphi_vals[idx]:.0f}°'
        y_fit   = df_median_mod2_fit.iloc[idx].values.astype(float)
        y_floor = df_median_mod2_fit_floor.iloc[idx].values.astype(float)
        ax.plot(mod2_true_counts, y_fit,   'o-',  color=color, label=f'{dphi_label} (sig1+sig2)')
        ax.plot(mod2_true_counts, y_floor, 'o--', color=color, label=f'{dphi_label} (sig2 only)', alpha=0.7)
    ax.plot(mod2_true_counts, mod2_true_counts, 'k--', lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(mod2_true_counts.min(), mod2_true_counts.max())
    ax.set_xlabel('mod2_true_counts [counts]')
    ax.set_ylabel('median_mod2_fit_counts [counts]')
    ax.set_title('median fitted sig2 modulation', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, which='both')
    ax.set_aspect('equal')

    # ── Row 1, Col 1: power_testA ────────────────────────────────────────────
    power_norm = mcolors.Normalize(vmin=0, vmax=1)
    power_cmap = plt.get_cmap('plasma_r')

    ax = axes[1, 1]
    im_A = ax.pcolormesh(Xe, Ye, power_testA, cmap=power_cmap, norm=power_norm)
    fig.colorbar(im_A, ax=ax, label='power')
    cs = ax.contour(Xc, Yc, power_testA, levels=[1 - sig], colors='black', linewidths=2)
    ax.clabel(cs, fmt=f'={1-sig:.1f}', fontsize=9)
    ax.set_title('Is sig2 being detected?  want ↑', fontweight='bold')
    _set_axes(ax)

    # ── Row 2, Col 1: power_testB ────────────────────────────────────────────
    ax = axes[2, 1]
    im_B = ax.pcolormesh(Xe, Ye, power_testB, cmap=power_cmap, norm=power_norm)
    fig.colorbar(im_B, ax=ax, label='power')
    cs = ax.contour(Xc, Yc, power_testB, levels=[sig], colors='black', linewidths=2)
    ax.clabel(cs, fmt=f'={sig:.1f}', fontsize=9)
    ax.set_title('Is reconstructed sig2 close to truth? want ↓', fontweight='bold')
    _set_axes(ax)

    # ── Row 1, Col 2: bias_mod1 ──────────────────────────────────────────────
    ax = axes[1, 2]
    vmax = max(abs(bias_mod1_counts).max(), budget)
    im = ax.pcolormesh(Xe, Ye, bias_mod1_counts, cmap='RdBu_r',
                       norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    fig.colorbar(im, ax=ax, label='bias_mod1_counts [counts]')
    cs = ax.contour(Xc, Yc, np.abs(bias_mod1_counts), levels=[budget], colors='black', linewidths=2)
    ax.clabel(cs, fmt={level: f'budget={budget:.3f}' for level in cs.levels}, fontsize=9)
    ax.set_title('sig1 amplitude bias (ignoring sig2)', fontweight='bold')
    _set_axes(ax)

    # ── Row 2, Col 0: sig2 mean fitted phase ─────────────────────────────────
    # pick smallest and largest Ad2/Ad1
    phase_col_indices = [0, len(Ad2_ratios) - 1]
    colors_col = plt.cm.tab10([0.0, 0.2])
    ax = axes[2, 0]
    for idx, color in zip(phase_col_indices, colors_col):
        col_label   = df_mean_Phase2_fit.columns[idx]
        ratio_label = col_label
        mean_rad = df_mean_Phase2_fit.iloc[:, idx].values.astype(float)
        std_rad  = df_std_Phase2_fit.iloc[:,  idx].values.astype(float)
        mean_deg = np.rad2deg(mean_rad) % 360
        std_deg  = np.rad2deg(std_rad)
        mask_ok  = std_deg <= 90
        ax.errorbar(Phase2_true_deg[mask_ok], mean_deg[mask_ok], yerr=std_deg[mask_ok],
                    fmt='o-', color=color, capsize=3, label=ratio_label)
        ax.plot(Phase2_true_deg[~mask_ok], mean_deg[~mask_ok], 'x', color=color, ms=8, alpha=0.5)
    ax.plot([], [], 'kx', ms=8, label='std > 90°')
    ax.plot(Phase2_true_deg, Phase2_true_deg, 'k--', lw=1.5)
    ax.set_xlabel('Phase2_true')
    ax.set_ylabel('mean Phase2_fit')
    tick_deg    = np.arange(0, 361, 45)
    tick_labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
                   r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$']
    ax.set_xticks(tick_deg); ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_deg); ax.set_yticklabels(tick_labels)
    ax.set_xlim(0, 360); ax.set_ylim(0, 360)
    ax.set_aspect('equal')
    ax.set_title('sig2 mean fitted phase', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True)

    # ── Row 2, Col 2: bias_phase1_deg ────────────────────────────────────────
    ax = axes[2, 2]
    bp1 = bias_phase1.copy()
    finite_vals = bp1[np.isfinite(bp1)]
    vmax_ph = np.percentile(np.abs(finite_vals), 95) if finite_vals.size else 1.0
    im = ax.pcolormesh(Xe, Ye, np.ma.masked_invalid(bp1), cmap='RdBu_r',
                       vmin=-vmax_ph, vmax=vmax_ph)
    fig.colorbar(im, ax=ax, label='bias_phase1 [deg]')
    ax.set_title('sig1 phase bias (ignoring sig2)', fontweight='bold')
    _set_axes(ax)
 
    plt.tight_layout()
 
    save_path = os.path.join(FIGURES_DIR, scan_folder+'.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')
    return fig

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_scan_sig2_quality.py <scan_folder>')
        sys.exit(1)
    plot_sig2_quality(sys.argv[1])
    plt.show()