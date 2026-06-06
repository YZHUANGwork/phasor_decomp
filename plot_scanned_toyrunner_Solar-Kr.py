"""
plot_combined.py
================
Produces a single 3×3 figure:

    Row 0, col 0      : blank
    Row 0, col 1      : ER Xe spectrum  (plot_setup style, shaded Er window)
    Row 0, col 2      : legend for the ER spectrum
    Row 1, cols 0–2   : median mod2 fit  |  power_testA  |  Solar nu modulation bias
    Row 2, cols 0–2   : sig2 mean phase  |  power_testB  |  Solar nu phase bias

Usage
-----
    python plot_combined.py <scan_folder>

The scan_folder is looked up under SIM_DIR (default: 'sim_data').
Set DETECTOR to match your efficiency files.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import astropy.units as u

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from get_sumraw_spectrum    import get_sumraw_spectrum
from get_realistic_spectrum import get_realistic_spectrum

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

SIM_DIR     = 'sim_data'
FIGURES_DIR = 'figures'
DETECTOR    = 'Xe1t'

ER_SOURCES      = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17']
BKGD_SOURCES_ER = ['Kr85', 'Rn222', 'nubb']


# ---------------------------------------------------------------------------
# ER spectrum — draws onto ax_spec, puts legend onto ax_leg
# ---------------------------------------------------------------------------

def _draw_ER_spectrum(ax_spec, ax_leg, signal1, signal2, bkgd,
                      detector=DETECTOR, Er_min=None, Er_max=None):
    target, channel = 'Xe', 'ER'
    handles, labels = [], []

    from get_realistic_spectrum import _is_bkgd

    # ---- all individual components — thin grey, no legend ------------------
    # classify every source correctly regardless of which bucket it came from
    all_sources = list({s for s in (signal1 or []) + (signal2 or []) + (bkgd or [])
                        if isinstance(s, str)})

    for source in all_sources:
        try:
            if _is_bkgd(source, target):
                Er, _, _, rate = get_sumraw_spectrum(
                    target=target, channel=channel,
                    nu_sources=[], wimp_mass=None, bkgd_sources=[source])
            else:
                Er, rate, _, _ = get_sumraw_spectrum(
                    target=target, channel=channel,
                    nu_sources=[source], wimp_mass=None, bkgd_sources=[])
            if rate is None:
                continue
        except Exception as e:
            print(f'  [skip] {source}: {e}')
            continue
        mask = rate.value > 0
        if mask.any():
            ax_spec.loglog(Er[mask].to(u.keV).value, rate[mask].value,
                           color='grey', lw=0.8, alpha=0.6)

    # ---- realistic totals via get_realistic_spectrum -----------------------
    try:
        Er_r, rate_s1, rate_s2, rate_bkgd_r = get_realistic_spectrum(
            target=target, channel=channel,
            signal1=signal1 or ER_SOURCES,
            signal2=signal2 or None,
            bkgd=bkgd or None,
            mode='realistic', detector=detector)

        colors_totals = [('tab:blue', 'signal1'), ('tab:orange', 'signal2'), ('tab:red', 'bkgd')]
        for rate, (color, name) in zip([rate_s1, rate_s2, rate_bkgd_r], colors_totals):
            if rate is None:
                continue
            line, = ax_spec.loglog(Er_r.to(u.keV).value, rate.value,
                                       color=color, ls='-', lw=2)
            handles.append(line)
            labels.append(name)
    except Exception as e:
        print(f'  [realistic] {e}')

    # ---- Er window shading -------------------------------------------------
    if Er_min is not None and Er_max is not None:
        ax_spec.axvspan(Er_min, Er_max, alpha=0.15, color='magenta', zorder=0)
        ax_spec.annotate('Er window', xy=(Er_min, 1), xycoords=('data', 'axes fraction'),
                         xytext=(-4, -4), textcoords='offset points',
                         color='magenta', fontsize=9, va='top', ha='right')

    # ---- axis styling ------------------------------------------------------
    ax_spec.set_xlabel('Electron recoil kinetic energy $E_r$ [keV]', fontsize=13)
    ax_spec.set_ylabel(r'$\dfrac{d\mathcal{R}}{dE_r}$ [ton$^{-1}$ yr$^{-1}$ keV$^{-1}$]', fontsize=13)
    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.set_xlim(0.05, 3000)
    ax_spec.set_ylim(1e-5, 200)
    ax_spec.tick_params(labelsize=13)
    ax_spec.grid(True)

    # ---- legend panel (only if separate from ax_spec) ----------------------
    if ax_leg is not ax_spec:
        ax_leg.axis('off')
    ax_leg._spectrum_handles = handles
    ax_leg._spectrum_labels  = labels


# ---------------------------------------------------------------------------
# Scan data loader
# ---------------------------------------------------------------------------

def load_scan(scan_dir):
    metadata       = pd.read_csv(os.path.join(scan_dir, 'metadata.csv')).set_index('param')
    df_SNR         = pd.read_csv(os.path.join(scan_dir, 'SNR.csv'),             index_col=0)
    df_power_testA = pd.read_csv(os.path.join(scan_dir, 'power_testA.csv'),     index_col=0)
    df_power_testB = pd.read_csv(os.path.join(scan_dir, 'power_testB.csv'),     index_col=0)
    df_bias_mod1   = pd.read_csv(os.path.join(scan_dir, 'bias_mod1_counts.csv'), index_col=0)
    df_bias_phase1 = pd.read_csv(os.path.join(scan_dir, 'bias_phase1_deg.csv'), index_col=0)
    return metadata, df_SNR, df_power_testA, df_power_testB, df_bias_mod1, df_bias_phase1


# ---------------------------------------------------------------------------
# Main combined figure
# ---------------------------------------------------------------------------

def plot_combined(scan_folder, detector=DETECTOR):
    scan_dir = os.path.join(SIM_DIR, scan_folder)
    metadata, df_SNR, df_power_testA, df_power_testB, df_bias_mod1, df_bias_phase1 = load_scan(scan_dir)

    meta           = metadata['value'].to_dict()
    Ad1            = float(meta.get('Ad1',            1.0))
    N1_in_bin      = float(meta.get('N1_in_bin',      1.0))
    N2_in_bin      = float(meta.get('N2_in_bin',      meta.get('N1_in_bin', 1.0)))
    Phase1_rad     = float(meta.get('Phase1_rad',     0.0))
    P_days         = float(meta.get('P', 365.25))
    sig            = float(meta.get('significance',   0.1))
    bias_tol       = float(meta.get('bias_tolerance', 0.1))
    n_toys         = int(float(meta.get('n_toys',     1000)))
    Er_min         = float(meta['Er_window_i']) if 'Er_window_i' in meta else None
    Er_max         = float(meta['Er_window_f']) if 'Er_window_f' in meta else None

    # source lists — stored as comma-separated strings in metadata
    def _parse_sources(key):
        val = meta.get(key, '')
        return [s.strip() for s in val.split(',') if s.strip()] if val else None
    signal1_sources = _parse_sources('signal1 components') or ER_SOURCES
    signal2_sources = _parse_sources('signal2 components') or None
    bkgd_sources    = _parse_sources('bkgd components')    or None

    Ad2              = np.array([float(c.split('=')[1]) for c in df_SNR.columns])
    mod2_true_counts = Ad2 * N2_in_bin

    Phase2_true_deg  = np.array([float(r.split('=')[1].replace('deg','').replace('rad',''))
                                  for r in df_SNR.index])
    Phase2_true_days = Phase2_true_deg / 360.0 * P_days

    SNR         = df_SNR.values.astype(float)
    power_testA = df_power_testA.values.astype(float)
    power_testB = df_power_testB.values.astype(float)
    bias_mod1   = df_bias_mod1.values.astype(float)
    bias_phase1 = df_bias_phase1.values.astype(float)

    mod1_true_counts = Ad1 * N1_in_bin
    budget           = bias_tol * mod1_true_counts

    df_median_mod2_fit       = pd.read_csv(os.path.join(scan_dir, 'median_mod2_fit_counts.csv'),       index_col=0)
    df_mean_Phase2_fit       = pd.read_csv(os.path.join(scan_dir, 'mean_Phase2_fit_rad.csv'),           index_col=0)
    df_std_Phase2_fit        = pd.read_csv(os.path.join(scan_dir, 'std_Phase2_fit_rad.csv'),            index_col=0)
    df_median_mod2_fit_floor = pd.read_csv(os.path.join(scan_dir, 'median_mod2_fit_counts_floor.csv'), index_col=0)

    vline1 = N2_in_bin * 0.002
    vline2 = N2_in_bin * 0.008

    idx_0    = int(np.argmin(np.abs(Phase2_true_days - 0)))
    idx_half = int(np.argmin(np.abs(Phase2_true_days - P_days / 2)))
    mod2_slice_indices = [idx_0, idx_half]
    phase_col_indices  = [0, len(Ad2) - 1]

    X, Y = np.meshgrid(Ad2, Phase2_true_days)

    def _set_heatmap_axes(ax):
        ax.set_xlim(Ad2.min(), Ad2.max())
        ax.set_ylim(Phase2_true_days.min(), Phase2_true_days.max())
        ax.set_xticks(Ad2)
        ax.set_xticklabels([f'{v:.3g}' for v in Ad2], rotation=45, ha='right', fontsize=11)
        ax.set_yticks(np.linspace(Phase2_true_days.min(), Phase2_true_days.max(), 9))
        ax.set_xlabel(r'$A_{\mathrm{Kr}}$')
        ax.set_ylabel(r'$\phi_{\mathrm{Kr}}$ true [days]')
        ax.axvline(0.002, color='red',    lw=1.5, ls='--')
        ax.axvline(0.008, color='orange', lw=1.5, ls='--')

    # ── Build figure with GridSpec ────────────────────────────────────────────
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0, cols 0-1 — run parameters spanning two columns
    ax_info = fig.add_subplot(gs[0, 0:2])
    ax_info.axis('off')
    Nb_in_bin = float(meta.get('N_bkgd_in_bin', meta.get('Nb_in_bin', 0.0)))
    s1_str = ', '.join(signal1_sources) if signal1_sources else '—'
    s2_str = ', '.join(signal2_sources) if signal2_sources else '—'
    bk_str = ', '.join(bkgd_sources)    if bkgd_sources    else '—'
    info_lines = [
        r'$\bf{Run\ parameters}$',
        f'signal1    : {s1_str}',
        f'signal2    : {s2_str}',
        f'bkgd       : {bk_str}',
        f'detector   : {detector}',
        f'Ad1        = {Ad1:.4g}',
        f'Phase1     = {np.rad2deg(Phase1_rad):.2f}°',
        f't0_1       = {Phase1_rad / (2*np.pi) * P_days:.2f} days',
        f'N1_in_bin  = {N1_in_bin:.4g}',
        f'N2_in_bin  = {N2_in_bin:.4g}',
        f'Nb_in_bin  = {Nb_in_bin:.4g}',
        f'mod1_true  = {Ad1 * N1_in_bin:.4g} counts',
        f'P          = {P_days:.5g} days',
        f'n_toys     = {n_toys}',
    ]
    ax_info.text(
        0.05, 0.97, '\n'.join(info_lines),
        transform=ax_info.transAxes, fontsize=13,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                  edgecolor='goldenrod', alpha=0.9),
    )

    # Row 0, col 2 — ER spectrum with legend inside
    ax_leg  = None  # no separate legend panel
    ax_spec = fig.add_subplot(gs[0, 2])
    _draw_ER_spectrum(ax_spec, ax_spec, signal1=signal1_sources, signal2=signal2_sources,
                      bkgd=bkgd_sources, detector=detector, Er_min=Er_min, Er_max=Er_max)

    handles = ax_spec._spectrum_handles
    labels  = ax_spec._spectrum_labels

    ax_spec.legend(handles, labels, loc='upper right',
                   fontsize=10, frameon=True, framealpha=0.9,
                   edgecolor='steelblue', borderpad=0.8, labelspacing=0.4)

    # ── Row 1, Col 0: median fitted sig2 modulation ──────────────────────────
    colors_line = plt.cm.tab10([0.0, 0.2])
    ax = fig.add_subplot(gs[1, 0])
    for idx, color in zip(mod2_slice_indices, colors_line):
        phase_label = f'$\\phi_{{\\mathrm{{Kr}}}}$={Phase2_true_days[idx]:.1f} d'
        y_fit   = df_median_mod2_fit.iloc[idx].values.astype(float)
        y_floor = df_median_mod2_fit_floor.iloc[idx].values.astype(float)
        ax.plot(Ad2, y_fit / N2_in_bin, 'o-',  color=color, label=f'{phase_label}')
    ax.axvline(0.002, color='red',    lw=1.5, ls='--', label=r'$A_{\mathrm{Kr}}$=0.002')
    ax.axvline(0.008, color='orange', lw=1.5, ls='--', label=r'$A_{\mathrm{Kr}}$=0.008')
    ax.set_xlim(Ad2.min(), Ad2.max())
    ax.set_xticks(Ad2)
    ax.set_xticklabels([f'{v:.3g}' for v in Ad2], rotation=45, ha='right', fontsize=11)
    # draw diagonal over the full visible range after autoscale
    diag = [Ad2.min(), Ad2.max()]
    ax.plot(diag, diag, 'k--', lw=1.5)
    ax.set_xlabel(r'$A_{\mathrm{Kr}}$')
    ax.set_ylabel(r'median $A_{\mathrm{Kr}}$ fit')
    ax.set_title(r'median fitted $A_{\mathrm{Kr}}$', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True)

    # ── Row 1, Col 1: median fitted sig2 modulation heatmap ─────────────────
    # average median_mod2_fit over the two slice rows to get a full 2D map
    median_mod2_fit_2d = df_median_mod2_fit.values.astype(float) / N2_in_bin
    ax = fig.add_subplot(gs[1, 1])
    vmin_m = mod2_true_counts.min()
    vmax_m = mod2_true_counts.max()
    im = ax.pcolormesh(X, Y, median_mod2_fit_2d,
                       cmap='viridis',
                       vmin=median_mod2_fit_2d.min(), vmax=median_mod2_fit_2d.max())
    cb = fig.colorbar(im, ax=ax, label=r'$A_{\mathrm{Kr}}$ fit (median)')
    cb.formatter = plt.matplotlib.ticker.FormatStrFormatter('%.4f')
    cb.update_ticks()
    # diagonal reference contour: fit == truth
    cs = ax.contour(X, Y, median_mod2_fit_2d / X, levels=[1.0],
                    colors='white', linewidths=2, linestyles='--')
    ax.clabel(cs, fmt=lambda x: r'fit=truth', fontsize=11)
    ax.set_title(r'median fitted $A_{\mathrm{Kr}}$', fontweight='bold')
    _set_heatmap_axes(ax)

    # ── Row 1, Col 2: Solar nu modulation bias (as % of mod1_true) ─────────
    ax = fig.add_subplot(gs[1, 2])
    bias_mod1_pct = bias_mod1 / mod1_true_counts * 100
    vmax_pct = max(abs(bias_mod1_pct).max(), bias_tol * 100)
    im = ax.pcolormesh(X, Y, bias_mod1_pct, cmap='RdBu_r',
                       norm=mcolors.TwoSlopeNorm(vmin=-vmax_pct, vcenter=0, vmax=vmax_pct))
    cb = fig.colorbar(im, ax=ax, label='bias mod1 [% of mod1_true]')
    cb.formatter = plt.matplotlib.ticker.FormatStrFormatter('%.1f%%')
    cb.update_ticks()
    cs = ax.contour(X, Y, np.abs(bias_mod1_pct), levels=[bias_tol * 100],
                    colors='black', linewidths=2)
    ax.clabel(cs, fmt=lambda x: f'{bias_tol*100:.0f}%', fontsize=11)
    ax.set_title('Solar $\\nu$ modulation bias', fontweight='bold')
    _set_heatmap_axes(ax)

    # ── Row 2, Col 0: sig2 mean fitted phase — line plot (two Ad2 slices) ────
    colors_col = plt.cm.tab10([0.0, 0.2])
    ax = fig.add_subplot(gs[2, 0])
    for idx, color in zip(phase_col_indices, colors_col):
        _ad2_val  = df_mean_Phase2_fit.columns[idx].split('=')[1]
        col_label = f'$A_{{\\mathrm{{Kr}}}}$={_ad2_val}'
        mean_rad  = df_mean_Phase2_fit.iloc[:, idx].values.astype(float)
        std_rad   = df_std_Phase2_fit.iloc[:,  idx].values.astype(float)
        mean_days = np.rad2deg(mean_rad) % 360 / 360.0 * P_days
        std_days  = np.rad2deg(std_rad)  / 360.0 * P_days
        mask_ok   = std_days <= P_days / 4
        ax.errorbar(Phase2_true_days[mask_ok], mean_days[mask_ok], yerr=std_days[mask_ok],
                    fmt='o-', color=color, capsize=3, label=col_label)
        ax.plot(Phase2_true_days[~mask_ok], mean_days[~mask_ok], 'x', color=color, ms=8, alpha=0.5)
    ax.plot([], [], 'kx', ms=8, label='std > P/4 (masked)')
    ax.plot(Phase2_true_days, Phase2_true_days, 'k--', lw=1.5)
    ax.set_xlabel(r'$\phi_{\mathrm{Kr}}$ true [days]')
    ax.set_ylabel(r'mean $\phi_{\mathrm{Kr}}$ fit [days]')
    lim = (Phase2_true_days.min(), Phase2_true_days.max())
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect('equal')
    ax.set_title(r'mean fitted $\phi_{\mathrm{Kr}}$', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True)

    # ── Row 2, Col 1: mean fitted sig2 phase heatmap + hatch where std large
    mean_phase2_2d = df_mean_Phase2_fit.values.astype(float)
    std_phase2_2d  = df_std_Phase2_fit.values.astype(float)
    mean_phase2_days = np.rad2deg(mean_phase2_2d) % 360 / 360.0 * P_days
    std_phase2_days  = np.rad2deg(std_phase2_2d)        / 360.0 * P_days
    ax = fig.add_subplot(gs[2, 1])
    # full unmasked heatmap
    im = ax.pcolormesh(X, Y, mean_phase2_days,
                       cmap='twilight',
                       vmin=Phase2_true_days.min(), vmax=Phase2_true_days.max())
    fig.colorbar(im, ax=ax, label=r'mean $\phi_{\mathrm{Kr}}$ fit [days]')
    # hatch overlay where std > P/4
    hatch_mask = np.where(std_phase2_days > P_days / 4, 1.0, np.nan)
    ax.pcolormesh(X, Y, hatch_mask, cmap='Greys', alpha=0.0)  # invisible base
    ax.contourf(X, Y, (std_phase2_days > P_days / 4).astype(float),
                levels=[0.5, 1.5], hatches=['////'], colors='none',
                linewidths=0)
    # error contour where unmasked
    phase_err = np.where(std_phase2_days <= P_days / 4,
                         np.abs(mean_phase2_days - Y), np.nan)
    cs = ax.contour(X, Y, phase_err, levels=[P_days / 8],
                    colors='black', linewidths=2, linestyles='--')
    ax.clabel(cs, fmt=lambda x: f'err={P_days/8:.0f}d', fontsize=11)
    ax.set_title(r'mean $\phi_{\mathrm{Kr}}$ fit [days]  (hatched: std > P/4)', fontweight='bold')
    _set_heatmap_axes(ax)

    # ── Row 2, Col 2: Solar nu phase bias ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    bias_phase1_days = bias_phase1 / 360.0 * P_days
    vmax_ph = np.abs(bias_phase1_days).max()
    im = ax.pcolormesh(X, Y, bias_phase1_days, cmap='RdBu_r',
                       norm=mcolors.TwoSlopeNorm(vmin=-vmax_ph, vcenter=0, vmax=vmax_ph))
    fig.colorbar(im, ax=ax, label='Solar $\\nu$ phase bias [days]')
    ax.set_title('Solar $\\nu$ phase bias', fontweight='bold')
    _set_heatmap_axes(ax)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, scan_folder + '_combined.pdf')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved {save_path}')
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_combined.py <scan_folder>')
        sys.exit(1)
    plot_combined(sys.argv[1])
    plt.show()