"""
plot_toy_runner_singlecase.py
=============================
Plotting helpers for the single-case toy runner study.

Key addition: plot_combined_page()
    Assembles ALL panels onto one PDF page with a structured 4-row × 3-col
    GridSpec layout:

    Row 0  │ Time series (spans all 3 columns)
    ───────┼──────────────────────────────────────────────────────────────
    Row 1  │ [info text + sig2 scatter  │  sig2 fitted modulation  │  sig2 fitted phase
           │  (sig2 scatter spans       │                           │
    Row 2  │   rows 1 & 2, col 0)]     │  sig2 testA               │  sig2 testB
    ───────┼──────────────────────────────────────────────────────────────
    Row 3  │ sig1 scatter (col 0)       │  bias amplitude           │  bias phase

    The sig2 phasor scatter occupies the full height of rows 1-2 in column 0.
    Inside that tall axes, the top portion is reserved for the run-parameter
    information text (R_s, D, dt, T, phi, N, …).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.stats import chi2 as chi2_dist, norm as norm_dist, kstest
import decomp_calc as calc
import toy_runner as toys


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _phasor_scatter(ax, cos_vals, sin_vals, cos_true, sin_true,
                    delta_cos, delta_sin, label_sig='recovered', title=''):
    """Draw a phasor-scatter panel (reusable for sig1 and sig2)."""
    ax.scatter(cos_vals, sin_vals, s=4, alpha=0.22, color='steelblue')
    ax.scatter([cos_true], [sin_true], s=200, color='red', marker='*', zorder=6)
    ax.scatter([np.mean(cos_vals)], [np.mean(sin_vals)],
               s=80, color='orange', marker='D', zorder=7)

    cov_plot = np.cov(delta_cos, delta_sin)
    vals, vecs = np.linalg.eigh(cov_plot)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for cl, col_ell, lw, ls in [(0.68, 'steelblue', 1.5, '-'),
                                  (0.95, 'cornflowerblue', 1.0, '--')]:
        scale = np.sqrt(chi2_dist.ppf(cl, df=2))
        ell = Ellipse(
            xy=(np.mean(cos_vals), np.mean(sin_vals)),
            width=2 * scale * np.sqrt(vals[1]),
            height=2 * scale * np.sqrt(vals[0]),
            angle=angle, edgecolor=col_ell, fc='None', lw=lw, ls=ls)
        ax.add_patch(ell)
        ax.text(np.mean(cos_vals) + scale * np.sqrt(vals[1]),
                np.mean(sin_vals),
                f'{int(cl * 100)}%', color=col_ell, fontsize=7)

    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('para_cos [counts]', fontsize=12)
    ax.set_ylabel('para_sin [counts]', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMBINED-PAGE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_page(
    nl,                     # noiseless-check result dict  (for time-series row)
    res_split2,             # toys.run_toys result         (sig2 scatter + mod + phase + testA/B)
    res_ignore2,            # toys.run_bias_toys result    (sig1 bias panels)
    split_stats,            # calctoys.compute_sig2_recovery output
    bias_stats,             # calctoys.compute_bias_stats output
    # ---- run parameters to display in the info box ----
    Rs1, Rs2, Rb,
    D, dt, T,
    phi1, phi2,
    N1, N2, Nb,
    Ad1, Ad2,
    N_TOYS, NOISE_TYPE, SEED,
    significance_level=0.1,
    save_path=None,
):
    """
    Single-page combined figure with all analysis panels.

    Parameters
    ----------
    nl           : noiseless check dict returned by toys.run_noiseless_check
    res_split2   : full-decomp toy result dict
    res_ignore2  : bias (ignore-sig2) toy result dict
    split_stats  : stats dict from calctoys.compute_sig2_recovery
    bias_stats   : stats dict from calctoys.compute_bias_stats
    Rs1 … Ad2    : physical run parameters (plain floats or with units stripped)
    N_TOYS …     : meta-parameters
    significance_level : float, default 0.1
    save_path    : optional file path to save the figure
    """

    plt.rcParams.update({'font.size': 12})
    thr = chi2_dist.ppf(1 - significance_level, df=2)

    # ── Figure & GridSpec ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor('#f8f8f8')

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.52,
        wspace=0.38,
        height_ratios=[1, 1.1, 1.1, 1.1],
    )

    # ── Row 0: Time series (full width) ───────────────────────────────────
    ax_ts = fig.add_subplot(gs[0, :])
    _plot_timeseries(ax_ts, nl)

    # ── Col 0, Rows 1-2: sig2 phasor scatter + info text ──────────────────
    ax_s2sc = fig.add_subplot(gs[1:3, 0])
    _plot_sig2_scatter_with_info(
        ax_s2sc, res_split2, split_stats,
        Rs1, Rs2, Rb, D, dt, T, phi1, phi2,
        N1, N2, Nb, Ad1, Ad2, N_TOYS, NOISE_TYPE, SEED,
    )

    # ── Row 1, Col 1: sig2 fitted modulation ──────────────────────────────
    ax_mod2 = fig.add_subplot(gs[1, 1])
    _plot_sig2_modulation(ax_mod2, res_split2, split_stats)

    # ── Row 1, Col 2: sig2 fitted phase ───────────────────────────────────
    ax_ph2 = fig.add_subplot(gs[1, 2])
    _plot_sig2_phase(ax_ph2, res_split2)

    # ── Row 2, Col 1: sig2 testA ──────────────────────────────────────────
    ax_tA = fig.add_subplot(gs[2, 1])
    _plot_testA(ax_tA, split_stats, thr, significance_level)

    # ── Row 2, Col 2: sig2 testB ──────────────────────────────────────────
    ax_tB = fig.add_subplot(gs[2, 2])
    _plot_testB(ax_tB, split_stats, thr, significance_level)

    # ── Row 3, Col 0: sig1 phasor scatter ─────────────────────────────────
    ax_s1sc = fig.add_subplot(gs[3, 0])
    _phasor_scatter(
        ax_s1sc,
        res_ignore2['para1_cos'], res_ignore2['para1_sin'],
        res_ignore2['para1_cos_true'], res_ignore2['para1_sin_true'],
        res_ignore2['delta_para1_cos'], res_ignore2['delta_para1_sin'],
        label_sig='recovered sig1',
        title='sig1 fit treating superposition as sig1 only',
    )
    # draw tolerance budget circle
    budget_circle = plt.Circle(
        (res_ignore2['para1_cos_true'], res_ignore2['para1_sin_true']),
        bias_stats['budget_counts'], color='gold', alpha=0.25,
        label=f'budget ({bias_stats["tolerance"]*100:.0f}%)')
    ax_s1sc.add_patch(budget_circle)
    ax_s1sc.legend(fontsize=12, loc='upper right')

    # ── Row 3, Col 1: bias amplitude ──────────────────────────────────────
    ax_biasA = fig.add_subplot(gs[3, 1])
    _plot_bias_amplitude(ax_biasA, res_ignore2, bias_stats)

    # ── Row 3, Col 2: bias phase ───────────────────────────────────────────
    ax_biasph = fig.add_subplot(gs[3, 2])
    _plot_bias_phase(ax_biasph, res_ignore2, bias_stats)

    # ── Overall title ──────────────────────────────────────────────────────
    verdict = 'BIASED ✗' if bias_stats['is_biased'] else 'unbiased ✓'
    v_color = 'crimson' if bias_stats['is_biased'] else 'seagreen'
    fig.suptitle(
        f'Toy-runner single-case summary  |  '
        f'sig2 SNR = {split_stats["SNR"]:.2f}  '
        f'power_A = {split_stats["power_testA"]:.3f}  '
        f'power_B = {split_stats["power_testB"]:.3f}  |  '
        f'sig1 bias: {verdict}',
        fontsize=12, fontweight='bold', color=v_color, y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {save_path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PANEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_timeseries(ax, nl, Ad1, Phase1_true_rad, Ad2, Phase2_true_rad, N1_in_bin, N2_in_bin):
    """Row 0: exact reproduction of decomp_calc.plot_noiseless_check."""
    r          = nl
    t          = r['time_bin_centers']
    known_ts   = r['known_ts']
    unknown_ts = r['unknown_ts']
    total_ts   = r['total_ts']

    mod1_true_counts = N1_in_bin * Ad1
    mod2_true_counts = N2_in_bin * Ad2
    mod_total_true_counts = np.sqrt(
        mod1_true_counts**2 + mod2_true_counts**2 +
        2 * mod1_true_counts * mod2_true_counts * np.cos(Phase2_true_rad - Phase1_true_rad)
    )
    y_sum = mod1_true_counts * np.sin(Phase1_true_rad) + mod2_true_counts * np.sin(Phase2_true_rad)
    x_sum = mod1_true_counts * np.cos(Phase1_true_rad) + mod2_true_counts * np.cos(Phase2_true_rad)
    Phase_total_true_rad = np.arctan2(y_sum, x_sum)
    omega = 2 * np.pi / r['period_days']

    ax.axhline( mod_total_true_counts, color='grey', lw=1)
    ax.axhline(-mod_total_true_counts, color='grey', lw=1)
    ax.plot(t, known_ts   - known_ts.mean(),   marker='o', ms=6, label='input KNOWN signal')
    ax.plot(t, unknown_ts - unknown_ts.mean(), marker='o', ms=6, label='input UNKNOWN signal')
    ax.plot(t, total_ts   - total_ts.mean(),   marker='o', ms=6, color='black',
            label='input KNOWN + UNKNOWN signals')
    ax.plot(t, mod_total_true_counts * np.cos(omega * t - Phase_total_true_rad),
            color='red', ls='--', lw=2, label='input KNOWN + UNKNOWN signals')
    ax.plot(t, r['mod2_fit_counts'] * np.cos(omega * t - r['Phase2_fit_rad']),
            color='green', ls='--', lw=2, label='fitted UNKNOWN')

    ax.set_xlabel('day', fontsize=12)
    ax.set_ylabel('counts in bin [#]', fontsize=12)
    ax.set_title('Noiseless input time series', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, ncol=3, loc='upper right')
    ax.tick_params(labelsize=11)


def _plot_sig2_scatter_with_info(
    ax, res, split_stats,
    Rs1, Rs2, Rb, D, dt, T, phi1, phi2,
    N1, N2, Nb, Ad1, Ad2, N_TOYS, NOISE_TYPE, SEED,
):
    """
    Col 0, rows 1-2: sig2 phasor scatter in the lower ~60 % of the axes;
    run-parameter info block in the upper ~40 %.
    """
    # -- phasor scatter --
    ax.scatter(res['para2_cos'], res['para2_sin'], s=3, alpha=0.20,
               color='steelblue')
    ax.scatter([res['para2_cos_true']], [res['para2_sin_true']],
               s=200, color='red', marker='*', zorder=6, label='truth')
    ax.scatter([np.mean(res['para2_cos'])], [np.mean(res['para2_sin'])],
               s=80, color='orange', marker='D', zorder=7, label='mean fit')

    cov_plot = np.cov(res['delta_para2_cos'], res['delta_para2_sin'])
    vals, vecs = np.linalg.eigh(cov_plot)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for cl, col_ell, lw, ls in [(0.68, 'steelblue', 1.5, '-'),
                                  (0.95, 'cornflowerblue', 1.0, '--')]:
        scale = np.sqrt(chi2_dist.ppf(cl, df=2))
        ell = Ellipse(
            xy=(np.mean(res['para2_cos']), np.mean(res['para2_sin'])),
            width=2 * scale * np.sqrt(vals[1]),
            height=2 * scale * np.sqrt(vals[0]),
            angle=angle, edgecolor=col_ell, fc='None', lw=lw, ls=ls)
        ax.add_patch(ell)
        ax.text(np.mean(res['para2_cos']) + scale * np.sqrt(vals[1]),
                np.mean(res['para2_sin']),
                f'{int(cl * 100)}%', color=col_ell, fontsize=7)

    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('para_cos [counts]', fontsize=12)
    ax.set_ylabel('para_sin [counts]', fontsize=12)
    ax.set_title(f'sig2 recovered after split  SNR={split_stats["SNR"]:.2f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)


def _plot_sig2_modulation(ax, res, split_stats):
    """Row 1, col 1: sig2 fitted modulation histogram."""
    ax.hist(res['mod2_fit_counts'], bins=40, density=True,
            alpha=0.7, color='mediumpurple', label='fit mod2')
    ax.axvline(res['mod2_true_counts'], color='red', lw=2, ls='--',
               label=f'truth = {res["mod2_true_counts"]:.4f}')
    ax.axvline(split_stats['median_mod2_fit_counts'], color='blue', lw=2, ls=':',
               label=f'median fit = {split_stats["median_mod2_fit_counts"]:.4f}')
    ax.set_xlabel('modulation [counts]', fontsize=12)
    ax.set_title('sig2 fitted modulation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)


def _plot_sig2_phase(ax, res):
    """Row 1, col 2: sig2 fitted phase histogram."""
    phi_deg      = np.rad2deg(res['Phase2_fit_rad'])
    phi_true_deg = np.rad2deg(res['Phase2_true_rad']) % 360
    phi_centered = (phi_deg - phi_true_deg + 180) % 360 - 180 + phi_true_deg

    ax.hist(phi_centered, bins=40, density=True,
            alpha=0.7, color='tomato', label='fit phase2')
    ax.axvline(phi_true_deg, color='red', lw=2, ls='--',
               label=f'truth = {phi_true_deg:.2f}°')
    ax.axvline(np.mean(phi_centered), color='orange', lw=2, ls=':',
               label=f'mean = {np.mean(phi_centered):.2f}°')
    ax.set_xlabel('phase [deg]', fontsize=12)
    ax.set_title('sig2 fitted phase\n(uniform → buried; peaked → detectable)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)


def _plot_testA(ax, split_stats, thr, significance_level):
    """Row 2, col 1: sig2 testA — Mahalanobis distance from zero."""
    Z2z = split_stats['Z2_H1_testA']
    ax.hist(Z2z, bins=40, density=True, alpha=0.7, color='darkorange', label='Z2 under H1')
    z_x = np.linspace(0, max(Z2z.max(), thr * 2.5), 300)
    ax.plot(z_x, chi2_dist.pdf(z_x, df=2), 'k--', lw=2, label='Z2 under H0  chi2(dof=2)')
    ax.axvline(thr, color='red', lw=2, label=f'H0 rejection threshold  α={significance_level}')
    ax.set_xlabel('Z2 (squared Mahalanobis distance from zero)', fontsize=12)
    ax.set_title('is signal2 being detected? power = {:.3f}'.format(split_stats['power_testA']), fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)


def _plot_testB(ax, split_stats, thr, significance_level):
    """Row 2, col 2: sig2 testB — Mahalanobis distance from truth."""
    Z2t = split_stats['Z2_H1_testB']
    ax.hist(Z2t, bins=40, density=True, alpha=0.7, color='mediumseagreen', label='Z2 under H1')
    z_x2 = np.linspace(0, max(Z2t.max(), thr * 2.5), 300)
    ax.plot(z_x2, chi2_dist.pdf(z_x2, df=2), 'k--', lw=2, label='Z2 under H0  chi2(dof=2)')
    ax.axvline(thr, color='red', lw=2, label=f'H0 rejection threshold  α={significance_level}')
    ax.set_xlabel('Z2 (squared Mahalanobis distance from sig2 truth)', fontsize=12)
    ax.set_title('is reconstructed signal2 close to truth? power = {:.3f}'.format(split_stats['power_testB']), fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)


def _plot_bias_amplitude(ax, res, bias_stats):
    """Row 3, col 1: sig1 modulation bias."""
    ax.hist(res['mod1_fit_counts'], bins=40, density=True,
            alpha=0.7, color='steelblue', label='fit mod1')
    ax.axvline(res['mod1_true_counts'], color='red', lw=2, ls='--',
               label=f'truth = {res["mod1_true_counts"]:.4f}')
    ax.axvline(np.mean(res['mod1_fit_counts']), color='orange', lw=2, ls=':',
               label=f'mean = {np.mean(res["mod1_fit_counts"]):.4f}')
    pct = bias_stats['bias_mod1_counts'] / res['mod1_true_counts'] * 100
    ax.set_xlabel('signal 1 modulation [counts]', fontsize=12)
    ax.set_title(
        f'sig1 amplitude bias (ignoring sig2)\n'
        f'Δ = {bias_stats["bias_mod1_counts"]:+.4f}  ({pct:+.1f}%)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)


def _plot_bias_phase(ax, res, bias_stats):
    """Row 3, col 2: sig1 phase bias."""
    phi_fit  = np.rad2deg(res['Phase1_fit_rad'])
    phi_true = np.rad2deg(res['Phase1_true_rad']) % 360
    phi_centered = (phi_fit - phi_true + 180) % 360 - 180 + phi_true

    ax.hist(phi_centered, bins=40, density=True,
            alpha=0.7, color='mediumpurple', label='fit phase1')
    ax.axvline(phi_true, color='red', lw=2, ls='--',
               label=f'truth = {phi_true:.2f}°')
    ax.axvline(np.mean(phi_centered), color='orange', lw=2, ls=':',
               label=f'mean = {np.mean(phi_centered):.2f}°')
    ax.set_xlabel('signal 1 phase [deg]', fontsize=12)
    ax.set_title(
        f'sig1 phase bias (ignoring sig2)\n'
        f'Δ = {bias_stats["bias_phase1_deg"]:+.3f}°',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=11)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY INDIVIDUAL-FIGURE FUNCTIONS  (kept for backwards compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sig2_recovery(toy_results, split_stats,
                       significance_level=0.1,
                       save_path=None):
    plt.rcParams.update({'font.size': 15})
    """
    Four-panel figure for sig2 recovery quality.

    Panel 1 : recovered phasor scatter — sig2 truth (red star) vs zero (black +)
    Panel 2 : Z2_H1_testA — Test A (squared Mahalanobis distance from zero)
    Panel 3 : Z2_H1_testB — Test B (squared Mahalanobis distance from sig2 truth)
    Panel 4 : recovered mod2_fit_counts distribution vs truth
    """
    thr = chi2_dist.ppf(1 - significance_level, df=2)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('signal 2 reconstruction test', fontsize=12)

    ax = axes[0]
    ax.scatter(toy_results['para2_cos'], toy_results['para2_sin'], s=4, alpha=0.25, color='steelblue',
               label='recovered sig2')
    ax.scatter([toy_results['para2_cos_true']], [toy_results['para2_sin_true']], s=160, color='red', zorder=6,
               marker='*', label='sig2 truth ({:.3f},{:.3f})'.format(toy_results['para2_cos_true'], toy_results['para2_sin_true']))
    cov_plot = np.cov(toy_results['delta_para2_cos'], toy_results['delta_para2_sin'])
    vals, vecs = np.linalg.eigh(cov_plot)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for cl, color, lw, ls in [(0.68, 'steelblue', 1.5, '-'),
                               (0.95, 'cornflowerblue', 1.0, '--')]:
        scale = np.sqrt(chi2_dist.ppf(cl, df=2))
        ell = Ellipse(xy=(np.mean(toy_results['para2_cos']), np.mean(toy_results['para2_sin'])),
                      width=2*scale*np.sqrt(vals[1]),
                      height=2*scale*np.sqrt(vals[0]),
                      angle=angle, edgecolor=color, fc='None', lw=lw, ls=ls)
        ax.add_patch(ell)
        ax.text(np.mean(toy_results['para2_cos']) + scale*np.sqrt(vals[1]),
                np.mean(toy_results['para2_sin']),
                '{}%'.format(int(cl*100)), color=color, fontsize=8)
    ax.text(toy_results['para2_cos_true'], toy_results['para2_sin_true'],
            '  ({:.3f}, {:.3f})'.format(toy_results['para2_cos_true'], toy_results['para2_sin_true']),
            color='red', fontsize=8)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('para2_cos [counts]')
    ax.set_ylabel('para2_sin [counts]')
    ax.set_title('fitted signal 2, SNR = {:.2f}'.format(split_stats['SNR']))
    ax.set_aspect('equal')

    ax = axes[1]
    Z2z = split_stats['Z2_H1_testA']
    ax.hist(Z2z, bins=40, density=True, alpha=0.7, color='darkorange', label='Z2 under H1')
    z_x = np.linspace(0, max(Z2z.max(), thr * 2.5), 300)
    ax.plot(z_x, chi2_dist.pdf(z_x, df=2), 'k--', lw=2, label='Z2 under H0  chi2(dof=2)')
    ax.axvline(thr, color='red', lw=2, label=f'H0 rejection threshold  α={significance_level}')
    ax.set_xlabel('Z2 (squared Mahalanobis distance from zero)', fontsize=12)
    ax.set_title('is signal2 being detected? power = {:.3f}'.format(split_stats['power_testA']), fontsize=12)
    ax.legend(fontsize=10)

    ax = axes[2]
    Z2t = split_stats['Z2_H1_testB']
    ax.hist(Z2t, bins=40, density=True, alpha=0.7, color='mediumseagreen', label='Z2 under H1')
    z_x2 = np.linspace(0, max(Z2t.max(), thr * 2.5), 300)
    ax.plot(z_x2, chi2_dist.pdf(z_x2, df=2), 'k--', lw=2, label='Z2 under H0  chi2(dof=2)')
    ax.axvline(thr, color='red', lw=2, label=f'H0 rejection threshold  α={significance_level}')
    ax.set_xlabel('Z2 (squared Mahalanobis distance from sig2 truth)', fontsize=12)
    ax.set_title('is reconstructed signal2 close to truth? power = {:.3f}'.format(split_stats['power_testB']), fontsize=12)
    ax.legend(fontsize=10)

    ax = axes[3]
    ax.hist(toy_results['mod2_fit_counts'], bins=40, density=True, alpha=0.7, color='mediumpurple', label='reconstructed modulation')
    ax.axvline(toy_results['mod2_true_counts'], color='red', lw=2, ls='--',
               label='input modulation = {:.4f}'.format(toy_results['mod2_true_counts']))
    ax.axvline(split_stats['median_mod2_fit_counts'], color='blue', lw=2, ls=':',
               label='median fit = {:.4f}'.format(split_stats['median_mod2_fit_counts']))
    ax.set_xlabel('modulation [counts]')
    ax.set_title('signal 2 modulation recovery')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print('  Saved {}'.format(save_path))
    return fig


def plot_bias_single(toy_results, bias_stats, save_path=None):
    """
    Three-panel figure: how much does ignoring sig2 bias the sig1 recovery.
    """
    verdict = 'BIASED ✗' if bias_stats['is_biased'] else 'OK ✓'
    color   = 'red' if bias_stats['is_biased'] else 'green'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('how neglecting signal 2 brings bias to signal 1  →  {}'.format(verdict), fontsize=12, color=color)

    ax = axes[0]
    ax.scatter(toy_results['para1_cos'], toy_results['para1_sin'], s=4, alpha=0.20,
               color='steelblue', label='recovered sig1')
    ax.scatter([toy_results['para1_cos_true']], [toy_results['para1_sin_true']],
               s=200, color='red', marker='*', zorder=6,
               label='input ({:.3f},{:.3f})'.format(toy_results['para1_cos_true'], toy_results['para1_sin_true']))
    ax.scatter([np.mean(toy_results['para1_cos'])], [np.mean(toy_results['para1_sin'])],
               s=100, color='orange', marker='D', zorder=7, label='mean fit')
    budget_circle = plt.Circle((toy_results['para1_cos_true'], toy_results['para1_sin_true']),
                                bias_stats['budget_counts'], color='gold', alpha=0.3,
                                label='budget ({:.0f}%)'.format(bias_stats['tolerance']*100))
    ax.add_patch(budget_circle)
    cov_plot = np.cov(toy_results['delta_para1_cos'], toy_results['delta_para1_sin'])
    vals, vecs = np.linalg.eigh(cov_plot)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for cl, color_ell, lw, ls in [(0.68, 'steelblue', 1.5, '-'),
                                   (0.95, 'cornflowerblue', 1.0, '--')]:
        scale = np.sqrt(chi2_dist.ppf(cl, df=2))
        ell = Ellipse(xy=(np.mean(toy_results['para1_cos']), np.mean(toy_results['para1_sin'])),
                      width=2*scale*np.sqrt(vals[1]),
                      height=2*scale*np.sqrt(vals[0]),
                      angle=angle, edgecolor=color_ell, fc='None', lw=lw, ls=ls)
        ax.add_patch(ell)
        ax.text(np.mean(toy_results['para1_cos']) + scale*np.sqrt(vals[1]),
                np.mean(toy_results['para1_sin']),
                '{}%'.format(int(cl*100)), color=color_ell, fontsize=8)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('para1_cos [counts]')
    ax.set_ylabel('para1_sin [counts]')
    ax.set_title('sig1 phasor scatter')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')

    ax = axes[1]
    ax.hist(toy_results['mod1_fit_counts'], bins=40, density=True, alpha=0.7, color='steelblue',
            label='reconstructed mod1')
    ax.axvline(toy_results['mod1_true_counts'], color='red', lw=2, ls='--',
               label='input = {:.4f}'.format(toy_results['mod1_true_counts']))
    ax.axvline(np.mean(toy_results['mod1_fit_counts']), color='orange', lw=2, ls=':',
               label='mean fit = {:.4f}'.format(np.mean(toy_results['mod1_fit_counts'])))
    ax.set_xlabel('signal 1 modulation [counts]')
    ax.set_title('amplitude bias = {:+.4f} ({:+.1f}%) budget = ±{:.4f}'.format(
        bias_stats['bias_mod1_counts'],
        bias_stats['bias_mod1_counts'] / toy_results['mod1_true_counts'] * 100,
        bias_stats['budget_counts']))
    ax.legend(fontsize=8)

    ax = axes[2]
    phi_fit  = np.rad2deg(toy_results['Phase1_fit_rad'])
    phi_true = np.rad2deg(toy_results['Phase1_true_rad']) % 360
    phi_centered = (phi_fit - phi_true + 180) % 360 - 180 + phi_true
    ax.hist(phi_centered, bins=40, density=True, alpha=0.7, color='mediumpurple',
            label='reconstructed phase1')
    ax.axvline(phi_true, color='red', lw=2, ls='--',
               label='input = {:.2f}°'.format(phi_true))
    ax.axvline(np.mean(phi_centered), color='orange', lw=2, ls=':',
               label='mean fit = {:.2f}°'.format(np.mean(phi_centered)))
    ax.set_xlabel('signal 1 phase [deg]')
    ax.set_title('phase bias = {:+.3f}°'.format(bias_stats['bias_phase1_deg']))
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {save_path}')
    return fig


def plot_inflation_single(toy_results, inf_stats, save_path=None):
    """Three-panel: sig2 inflation from sig1 noise."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'signal 2 inflation with signal 1  | '
        f'phys_ratio = {inf_stats["phys_ratio"]:.4f}  '
        f'median inflation = {inf_stats["median_inflation"]:.1f}x  ',
        fontsize=11
    )

    ax = axes[0]
    ax.scatter(toy_results['para2_cos'], toy_results['para2_sin'], s=4, alpha=0.20,
               color='steelblue', label='recovered sig2')
    ax.scatter([toy_results['para2_cos_true']], [toy_results['para2_sin_true']],
               s=250, color='red', marker='*', zorder=6,
               label=f'truth ({toy_results["para2_cos_true"]:.4f}, {toy_results["para2_sin_true"]:.4f})')
    ax.scatter([np.mean(toy_results['para2_cos'])], [np.mean(toy_results['para2_sin'])],
               s=100, color='orange', marker='D', zorder=7, label='mean fit')
    cov_plot = np.cov(toy_results['delta_para2_cos'], toy_results['delta_para2_sin'])
    vals, vecs = np.linalg.eigh(cov_plot)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for cl, color_ell, lw, ls in [(0.68, 'steelblue', 1.5, '-'),
                                   (0.95, 'cornflowerblue', 1.0, '--')]:
        scale = np.sqrt(chi2_dist.ppf(cl, df=2))
        ell = Ellipse(xy=(np.mean(toy_results['para2_cos']), np.mean(toy_results['para2_sin'])),
                      width=2*scale*np.sqrt(vals[1]),
                      height=2*scale*np.sqrt(vals[0]),
                      angle=angle, edgecolor=color_ell, fc='None', lw=lw, ls=ls)
        ax.add_patch(ell)
        ax.text(np.mean(toy_results['para2_cos']) + scale*np.sqrt(vals[1]),
                np.mean(toy_results['para2_sin']),
                '{}%'.format(int(cl*100)), color=color_ell, fontsize=8)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('para2_cos [counts]')
    ax.set_ylabel('para2_sin [counts]')
    ax.set_title('sig2 phasor scatter (buried → cloud on origin)')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')

    ax = axes[1]
    ax.hist(inf_stats['inflation'], bins=40, density=True, alpha=0.7, color='steelblue')
    ax.axvline(1.0, color='red', lw=2, ls='--', label='perfect (1x)')
    ax.axvline(inf_stats['median_inflation'], color='orange', lw=2, ls=':',
               label=f'median = {inf_stats["median_inflation"]:.1f}x')
    ax.set_xlabel('mod2_fit_counts / mod2_true_counts')
    ax.set_title('enhanced signal 2 reconstruction')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(inf_stats['rel_sig1'], bins=40, density=True, alpha=0.7, color='tomato')
    ax.axvline(inf_stats['phys_ratio'], color='red', lw=2, ls='--',
               label=f'true ratio = {inf_stats["phys_ratio"]:.4f}')
    ax.axvline(inf_stats['median_rel_sig1'], color='orange', lw=2, ls=':',
               label=f'median fit = {inf_stats["median_rel_sig1"]:.4f}')
    ax.set_xlabel('mod2_fit_counts / mod1_true_counts')
    ax.set_title('enhanced sig2 relative to sig1')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {save_path}')
    return fig


def plot_floor_single(toy_results, floor_stats, save_path=None):
    """Three-panel: sig2 inflation from pure Poisson noise (no sig1)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'signal 2 inflation without signal 1  |  '
        f'inflation = {floor_stats["inflation"]:.2f}x   ',
        fontsize=15
    )

    ax = axes[0]
    ax.scatter(toy_results['para2_cos'], toy_results['para2_sin'], s=4, alpha=0.2, color='steelblue',
               label='recovered sig2')
    ax.scatter([toy_results['para2_cos_true']], [toy_results['para2_sin_true']], s=250, color='red', marker='*', zorder=6,
               label=f'truth ({toy_results["para2_cos_true"]:.4f}, {toy_results["para2_sin_true"]:.4f})')
    ax.scatter([np.mean(toy_results['para2_cos'])], [np.mean(toy_results['para2_sin'])],
               s=100, color='orange', marker='D', zorder=7, label='mean fit')
    cov_plot = np.cov(toy_results['delta_para2_cos'], toy_results['delta_para2_sin'])
    vals, vecs = np.linalg.eigh(cov_plot)
    angle = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for cl, color_ell, lw, ls in [(0.68, 'steelblue', 1.5, '-'),
                                   (0.95, 'cornflowerblue', 1.0, '--')]:
        scale = np.sqrt(chi2_dist.ppf(cl, df=2))
        ell = Ellipse(xy=(np.mean(toy_results['para2_cos']), np.mean(toy_results['para2_sin'])),
                      width=2*scale*np.sqrt(vals[1]),
                      height=2*scale*np.sqrt(vals[0]),
                      angle=angle, edgecolor=color_ell, fc='None', lw=lw, ls=ls)
        ax.add_patch(ell)
        ax.text(np.mean(toy_results['para2_cos']) + scale*np.sqrt(vals[1]),
                np.mean(toy_results['para2_sin']),
                '{}%'.format(int(cl*100)), color=color_ell, fontsize=8)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('para2_cos [counts]')
    ax.set_ylabel('para2_sin [counts]')
    ax.set_title('sig2 phasor scatter')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')

    ax = axes[1]
    ax.hist(toy_results['mod2_fit_counts'], bins=40, density=True, alpha=0.7, color='steelblue',
            label='reconstructed mod2')
    ax.axvline(toy_results['mod2_true_counts'], color='red', lw=2, ls='--',
               label=f'input = {toy_results["mod2_true_counts"]:.5f}')
    ax.axvline(floor_stats['median_mod2_fit_counts'], color='orange', lw=2, ls=':',
               label=f'median = {floor_stats["median_mod2_fit_counts"]:.5f}')
    ax.set_xlabel('signal 2 modulation [counts]')
    ax.set_title('signal 2 modulation')
    ax.legend(fontsize=8)

    ax = axes[2]
    phi_deg      = np.rad2deg(toy_results['Phase2_fit_rad'])
    phi_true_deg = np.rad2deg(toy_results['Phase2_true_rad']) % 360
    phi_centered = (phi_deg - phi_true_deg + 180) % 360 - 180 + phi_true_deg
    ax.hist(phi_centered, bins=40, density=True, alpha=0.7, color='mediumpurple',
            label='reconstructed phase2')
    ax.axvline(phi_true_deg, color='red', lw=2, ls='--',
               label=f'input = {phi_true_deg:.2f}°')
    ax.set_xlabel('signal 2 phase [deg]')
    ax.set_title('phase distribution (uniform = buried, peaked = detectable)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run as: python3 plot_toy_runner_singlecase.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/WIMP_decomp')
    import sim_toyrunner_singlecase as sim
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import chi2 as chi2_dist
    import os

    # all parameters and results come from sim
    nl         = sim.nl
    res_split2 = sim.res_split2
    res_ignore2= sim.res_ignore2
    split      = sim.split
    bst        = sim.bst

    Rs1  = sim.Rs1;  Rs2  = sim.Rs2;  Rb   = sim.Rb
    D    = sim.D;    dt   = sim.dt;   T    = sim.T
    phi1 = sim.phi1; phi2 = sim.phi2
    N1   = sim.N1;   N2   = sim.N2;   Nb   = sim.Nb
    Ad1  = sim.Ad1;  Ad2  = sim.Ad2
    N_TOYS     = sim.N_TOYS
    NOISE_TYPE = sim.NOISE_TYPE
    SEED       = sim.SEED
    SIGNIFICANCE = sim.SIGNIFICANCE

    import astropy.units as u

    plt.rcParams.update({'font.size': 12})
    thr = chi2_dist.ppf(1 - SIGNIFICANCE, df=2)

    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.20,
                           height_ratios=[1, 1, 1.1, 1.1])

    # Row 0: time series
    ax_ts = fig.add_subplot(gs[0, :])
    _plot_timeseries(ax_ts, nl, Ad1, phi1.value, Ad2, phi2.value, N1, N2)

    # Col 0, Row 1: info text only (axis off)
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    info_lines = [
        r'$\bf{Run\ parameters}$',
        f'Rs1 = {Rs1.value:.3g} t⁻¹yr⁻¹    Rs2 = {Rs2.value:.3g} t⁻¹yr⁻¹',
        f'Rb  = {Rb.value:.3g} t⁻¹yr⁻¹',
        f'D   = {D.value:.3g} t      dt = {dt.to(u.day).value:.3g} day      T = {T.to(u.yr).value:.3g} yr',
        f'phi1 = {phi1.value:.4f} rad      phi2 = {phi2.value:.4f} rad',
        f'N1  = {N1:.4f} counts in bin (unitless)',
        f'N2  = {N2:.4f} counts in bin (unitless)',
        f'Nb  = {Nb:.4f} counts in bin (unitless)',
        f'Ad1 = {Ad1:.5f}    Ad2 = {Ad2:.5f}',
        f'N_toys = {N_TOYS}    noise = {NOISE_TYPE}',
    ]
    ax_info.text(0.05, 0.95, '\n'.join(info_lines),
                 transform=ax_info.transAxes, fontsize=13,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                           edgecolor='goldenrod', alpha=0.9))

    # Col 0, Row 2: sig2 phasor scatter
    ax_s2sc = fig.add_subplot(gs[2, 0])
    _plot_sig2_scatter_with_info(
        ax_s2sc, res_split2, split,
        Rs1.value, Rs2.value, Rb.value,
        D.value, dt.to(u.day).value, T.to(u.yr).value,
        phi1.value, phi2.value, N1, N2, Nb, Ad1, Ad2,
        N_TOYS, NOISE_TYPE, SEED,
    )

    # Row 1, Col 1: sig2 fitted modulation
    ax_mod2 = fig.add_subplot(gs[1, 1])
    _plot_sig2_modulation(ax_mod2, res_split2, split)

    # Row 1, Col 2: sig2 fitted phase
    ax_ph2 = fig.add_subplot(gs[1, 2])
    _plot_sig2_phase(ax_ph2, res_split2)

    # Row 2, Col 1: sig2 testA
    ax_tA = fig.add_subplot(gs[2, 1])
    _plot_testA(ax_tA, split, thr, SIGNIFICANCE)

    # Row 2, Col 2: sig2 testB
    ax_tB = fig.add_subplot(gs[2, 2])
    _plot_testB(ax_tB, split, thr, SIGNIFICANCE)

    # Row 3, Col 0: sig1 phasor scatter
    ax_s1sc = fig.add_subplot(gs[3, 0])
    _phasor_scatter(
        ax_s1sc,
        res_ignore2['para1_cos'], res_ignore2['para1_sin'],
        res_ignore2['para1_cos_true'], res_ignore2['para1_sin_true'],
        res_ignore2['delta_para1_cos'], res_ignore2['delta_para1_sin'],
        label_sig='recovered sig1',
        title='sig1 fit treating superposition as sig1 only',
    )


    # Row 3, Col 1: bias amplitude
    ax_biasA = fig.add_subplot(gs[3, 1])
    _plot_bias_amplitude(ax_biasA, res_ignore2, bst)

    # Row 3, Col 2: bias phase
    ax_biasph = fig.add_subplot(gs[3, 2])
    _plot_bias_phase(ax_biasph, res_ignore2, bst)

    # ── Save ──────────────────────────────────────────────────────────────────
    FIGURES_folder = './figures'
    os.makedirs(FIGURES_folder, exist_ok=True)
    output_png = os.path.join(FIGURES_folder, 'toyrunner_singlecase.png')
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f'Saved {output_png}')
    '''
    output_pdf = os.path.join(FIGURES_folder, 'toyrunner_singlecase.pdf')
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close('all')
    print(f'Saved {output_pdf}')
    '''