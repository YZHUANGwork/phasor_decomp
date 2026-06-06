"""
plot_scanned_toyrunner_wimp_erwindow.py
=======================================
Scan data: single JSON produced by sim_scanned_toyrunner_Solar-WIMP.py
           (one record per WIMP mass)

Layout
------
  Row 0, col 0  : run-parameter info box
  Row 0, col 1  : NR spectrum (dN/dEr top, A_d bottom) + Er window shading
  Row 0, col 2  : spectrum legend

  Row 1, col 0  : mod2_input vs mod2_fit  (vs WIMP mass)
  Row 1, col 1  : fitted phi2             (vs WIMP mass)
  Row 1, col 2  : bias_mod1              (vs WIMP mass)

  Row 2, col 0  : bias_phase1            (vs WIMP mass)
  Row 2, col 1  : blank
  Row 2, col 2  : blank

Usage
-----
    python plot_scanned_toyrunner_wimp_erwindow.py <scan_file.json>
"""

import sys
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
import astropy.constants as const

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, '/home/echo/git2026_phasor_decomp')

from get_sumraw_spectrum    import get_sumraw_spectrum
from get_realistic_spectrum import get_realistic_spectrum

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SIM_DIR     = 'sim_data'
FIGURES_DIR = 'figures'
P_REF       = 365.25   # days

# 3 reference WIMP masses: grey theory curves, distinguished by line style
_WIMP_STYLES = {
    6:   dict(color='#555555', ls='-',  lw=0.8, alpha=1, label=r'WIMP 6 GeV (theory)'),
    10:  dict(color='#888888', ls='--', lw=0.8, alpha=1, label=r'WIMP 10 GeV (theory)'),
    100: dict(color='#bbbbbb', ls=':',  lw=0.8, alpha=1, label=r'WIMP 100 GeV (theory)'),
}


# ---------------------------------------------------------------------------
# NR spectrum panel  (row 0, col 1)
# ---------------------------------------------------------------------------

def _draw_NR_spectrum(ax_spec, ax_Ad, signal1_sources, detector,
                      Er_min=None, Er_max=None, er_color='tab:blue'):
    target, channel = 'Xe', 'NR'
    print(signal1_sources)
    # ---- individual solar-nu components: thin grey, raw ----------------------
    for source in signal1_sources:
        if not isinstance(source, str):
            continue
        try:
            Er, rate, _, _ = get_sumraw_spectrum(
                target=target, channel=channel,
                nu_sources=[source], wimp_mass=None, bkgd_sources=[])
            if rate is None:
                continue
            ax_spec.loglog(Er.to(u.keV).value, rate.value,
                           color='grey', lw=0.8, alpha=0.5)
        except Exception as e:
            print(f'  [skip {source}] {e}')

    # ---- realistic solar-nu total: black, bold --------------------------------
    try:
        Er_r, rate_s1, _, _ = get_realistic_spectrum(
            target=target, channel=channel,
            signal1=signal1_sources, signal2=None, bkgd=None,
            mode='realistic', detector=detector)
        ax_spec.loglog(Er_r.to(u.keV).value, rate_s1.value,
                       color='black', ls='-', lw=2.5,
                       label=r'Solar $\nu$')
    except Exception as e:
        print(f'  [realistic signal1] {e}')

    # ---- WIMP theory: transparent/thin grey; realistic: orange, bold ---------
    from DM_modulation_model import get_SHM_Ads
    A   = 131.293
    m_N = A * (const.m_n * const.c**2).to(u.GeV)

    for mass_val, style in _WIMP_STYLES.items():
        m_wimp = mass_val * u.GeV

        # raw theory spectrum
        try:
            Er_w, _, rate_w, _ = get_sumraw_spectrum(
                target=target, channel=channel,
                nu_sources=[], wimp_mass=m_wimp, bkgd_sources=[])
            if rate_w is not None:
                ax_spec.loglog(
                        Er_w.to(u.keV).value, rate_w.value,
                        color='tab:red', ls=style['ls'],
                        lw=style['lw'], alpha=style['alpha'])

                # A_d on lower panel — same transparent style
                Ad_arr = get_SHM_Ads(Er_w, m_wimp, m_N,
                                     vc=220*u.km/u.s, vesc=553*u.km/u.s, xp=0.89)
                mAd = Ad_arr != 0
                ax_Ad.semilogx(Er_w[mAd].to(u.keV).value, Ad_arr[mAd],
                                   color=style['color'], ls=style['ls'],
                                   lw=style['lw'], alpha=style['alpha'])
        except Exception as e:
            print(f'  [WIMP raw {mass_val} GeV] {e}')

        # realistic spectrum (with efficiency): red, bold
        try:
            Er_re, _, rate_we, _ = get_realistic_spectrum(
                target=target, channel=channel,
                signal1=None, signal2=[m_wimp], bkgd=None,
                mode='realistic', detector=detector)
            if rate_we is not None:
                ax_spec.loglog(
                    Er_re.to(u.keV).value, rate_we.value,
                    color='tab:red', ls=style['ls'], lw=2.5,
                    label=f'WIMP {mass_val} GeV')

                # A_d on lower panel — orange, bold
                Ad_re = get_SHM_Ads(Er_re, m_wimp, m_N,
                                    vc=220*u.km/u.s, vesc=553*u.km/u.s, xp=0.89)
                mAd = ~np.isnan(Ad_re)
                ax_Ad.semilogx(Er_re[mAd].to(u.keV).value, Ad_re[mAd],
                               color='tab:red', ls=style['ls'], lw=2.5)
        except Exception as e:
            print(f'  [WIMP eff {mass_val} GeV] {e}')

    # ---- Er window shading ---------------------------------------------------
    if Er_min is not None and Er_max is not None:
        ax_spec.axvspan(Er_min, Er_max, alpha=0.15, color=er_color, zorder=0)
        ax_Ad.axvspan(Er_min, Er_max, alpha=0.15, color=er_color, zorder=0)
        ax_spec.annotate('Er window', xy=(Er_min, 1), xycoords=('data', 'axes fraction'),
                         xytext=(-4, -4), textcoords='offset points',
                         color=er_color, fontsize=9, va='top', ha='right')

    # ---- legend on ax_spec ---------------------------------------------------
    ax_spec.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9,
                   edgecolor='steelblue', borderpad=0.8, labelspacing=0.4,
                   prop={'family': 'monospace', 'size': 10})
    ax_spec.set_ylabel(
        r'$\dfrac{d\mathcal{R}}{dE_r}$ [ton$^{-1}$ yr$^{-1}$ keV$^{-1}$]', fontsize=13)
    ax_spec.set_xlim(0.01, 1000)
    ax_spec.set_ylim(1e-5, 1e7)
    ax_spec.tick_params(labelsize=12)
    ax_spec.tick_params(axis='x', which='both', labelbottom=False)
    ax_spec.set_xlabel('')
    ax_spec.grid(True, which='both', ls='--', alpha=0.4)
    ax_spec.spines['top'].set_visible(False)
    ax_spec.spines['right'].set_visible(False)

    # ---- ax_Ad styling -------------------------------------------------------
    AD_ECC = 0.03342
    ax_Ad.axhline(y=AD_ECC, lw=1.5, ls=':', color='black')
    ax_Ad.text(0.12, AD_ECC, r'$A_{d,\nu}$', ha='left', va='bottom', fontsize=10)
    ax_Ad.axhspan(-0.05, 0, alpha=0.2, color='grey')
    ax_Ad.text(0.12, 0., 'phase reverse', color='grey', ha='left', va='top', fontsize=10)
    ax_Ad.set_xlabel(r'Nuclear recoil energy $E_r$ [keV]', fontsize=13)
    ax_Ad.set_ylabel(r'$A_{d,\mathrm{SHM}}$', fontsize=13)
    ax_Ad.set_xlim(0.01, 1000)
    ax_Ad.set_ylim(-0.05, 0.2)
    ax_Ad.grid(True, which='both', ls='--', alpha=0.4)
    ax_Ad.spines['top'].set_visible(False)
    ax_Ad.spines['right'].set_visible(False)




# ---------------------------------------------------------------------------
# bias_sig1only vs Er  (row 0, col 1)
# ---------------------------------------------------------------------------

def _draw_bias_sig1only(ax, datasets, colors=None):
    """Plot bias_mod1_counts_sig1only as horizontal line segments vs Er window.

    datasets      : list of (meta, df) tuples  — or a single (meta, df) for one file.
    colors        : list of colours matching datasets; defaults to tab10.
    """
    # accept a single (meta, df) pair as well
    if isinstance(datasets, tuple) and len(datasets) == 2 and not isinstance(datasets[0], tuple):
        datasets = [datasets]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))

    for (meta, _df), color in zip(datasets, colors):
        Er_i = float(meta['Er_window_i'])
        Er_f = float(meta['Er_window_f'])
        val  = float(meta.get('bias_mod1_counts_sig1only', np.nan))
        if np.isnan(val):
            continue
        ax.hlines(val, Er_i, Er_f, colors=color, lw=2.5,
                  label=f'Er=[{Er_i:.2g},{Er_f:.2g}] keV')
        ax.plot([Er_i, Er_f], [val, val], '|', color=color, ms=8, mew=1.5)

    ax.axhline(0, color='grey', lw=0.8, ls='-')
    ax.set_xlabel(r'Nuclear recoil energy $E_r$ [keV]', fontsize=13)
    ax.set_ylabel(r'Bias modulation $\nu$ [counts]', fontsize=13)

    ax.grid(True, which='major', ls='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_bias_phi_sig1only(ax, datasets, colors=None):
    """Plot bias_phase1_deg_sig1only as horizontal line segments vs Er window (in days)."""
    if isinstance(datasets, tuple) and len(datasets) == 2 and not isinstance(datasets[0], tuple):
        datasets = [datasets]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))
    for (meta, _df), color in zip(datasets, colors):
        Er_i = float(meta['Er_window_i'])
        Er_f = float(meta['Er_window_f'])
        val_deg = float(meta.get('bias_phase1_deg_sig1only', np.nan))
        if np.isnan(val_deg):
            continue
        P   = float(meta.get('P_day', 365.25))
        val = val_deg / 360.0 * P  # convert to days
        ax.hlines(val, Er_i, Er_f, colors=color, lw=2.5,
                  label=f'Er=[{Er_i:.2g},{Er_f:.2g}] keV')
        ax.plot([Er_i, Er_f], [val, val], '|', color=color, ms=8, mew=1.5)
    ax.axhline(0, color='grey', lw=0.8, ls='-')
    ax.set_xlabel(r'Nuclear recoil energy $E_r$ [keV]', fontsize=13)
    ax.set_ylabel(r'Bias $\phi_{\nu}$ [days]', fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_wimp_erwindow(scan_file):
    fpath = scan_file if os.path.isabs(scan_file) \
            else os.path.join(SIM_DIR, scan_file)
    with open(fpath) as f:
        raw = json.load(f)

    meta = raw['meta']
    df   = pd.DataFrame(raw['data']).sort_values('m_chi_GeV').reset_index(drop=True)

    m                 = df['m_chi_GeV'].values
    AdR2              = df['mod2_true_rate'].values
#    N2                = df['N2_wimp_in_bin'].values
    N1                = df['N1_nu_in_bin'].values
    Ad1               = df['Ad1_nu'].iloc[0]
    mod2_input        = df['mod2_true_counts']
    mod2_fit          = df['median_mod2_fit_counts'].values
    phase2_fit_rad    = df['mean_Phase2_fit_rad'].values
    phase2_std_rad    = df['std_Phase2_fit_rad'].values
    bias_mod1_counts  = df['bias_mod1_counts'].values
    bias_phase1_deg   = df['bias_phase1_deg'].values
    P        = float(meta['P_day'])
    t0_day   = float(meta['t01_day'])
    t0_SHM   = float(meta['t02_day'])
    detector = meta['detector']
    signal1  = meta['signal1_sources']
    Er_min   = float(meta['Er_window_i'])
    Er_max   = float(meta['Er_window_f'])

    phase2_fit_days = phase2_fit_rad / (2 * np.pi) * P % P
    phase2_std_days = phase2_std_rad / (2 * np.pi) * P
    phi1_days       = t0_day % P
    phi2_days       = t0_SHM % P
    bias_mod1_rel   = bias_mod1_counts / (Ad1 * N1)
    bias_phase1_days = bias_phase1_deg / 360.0 * P

    def _spine(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 11})
    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.35,
                            height_ratios=[3, 2, 2])

    # ── Row 0, col 0 : dN/dEr + Ad stacked in a nested grid ─────────────────
    gs_left0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                                hspace=0.05, height_ratios=[3, 2])
    ax_spec = fig.add_subplot(gs_left0[0])
    ax_Ad   = fig.add_subplot(gs_left0[1], sharex=ax_spec)

    _draw_NR_spectrum(ax_spec, ax_Ad,
                      signal1_sources=signal1, detector=detector,
                      Er_min=Er_min, Er_max=Er_max,
                      er_color='tab:blue')

    # ── Row 1, col 0 : bias phi vs Er (sig1 only) ────────────────────────────
    ax_bph_er = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    _draw_bias_phi_sig1only(ax_bph_er, (meta, df))

    # ── Row 2, col 0 : bias modulation ν vs Er (sig1 only) ───────────────────
    ax_bias_er = fig.add_subplot(gs[2, 0], sharex=ax_spec)
    # ── Row 0, col 1 : run parameter info box ────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    info_lines = [
        r'$\bf{Run\ parameters}$',
        f'signal1    : {", ".join(signal1)}',
        f'target     : {meta["target"]}',
        f'detector   : {detector}',
        f'channel    : {meta["channel"]}',
        f'D          = {meta["D_tonne"]} t',
        f'dt         = {meta["dt_day"]} days',
        f'T          = {meta["T_yr"]} yr',
        f'Ad1        = {Ad1:.4g}',
        f't01        = {t0_day:.2f} days',
        f't02        = {t0_SHM:.2f} days',
        f'n_toys     = {meta["N_toys"]}',
    ]
    ax_info.text(0.05, 0.98, '\n'.join(info_lines),
                 transform=ax_info.transAxes, fontsize=12,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                           edgecolor='goldenrod', alpha=0.9))

    from matplotlib.ticker import FixedLocator, NullLocator

    # ── Row 1, col 1 : bias phase days vs mass ───────────────────────────────
    ax_bph1 = fig.add_subplot(gs[1, 1])
    ln_ph, = ax_bph1.semilogx(m, bias_phase1_days, 'o-', color='tab:red',
                               label=f'Er=[{Er_min:.2g},{Er_max:.2g}] keV')
    ax_bph1.axhline(0, color='grey', lw=0.8, ls='-')
    ax_bph1.set_xlabel(r'WIMP mass [GeV]')
    ax_bph1.set_ylabel(r'Bias $\phi_{\nu}$ [days]')
    ax_bph1.set_yscale('symlog', linthresh=5)
    from matplotlib.ticker import FixedLocator, AutoMinorLocator
    _phi_lin  = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    _phi_ymax = max(abs(v) for v in ax_bph1.get_ylim()) if ax_bph1.get_ylim()[1] != 1.0 else 50
    _phi_log  = [s * n * 10**e
                 for s in [1, -1]
                 for e in range(1, int(np.ceil(np.log10(max(_phi_ymax, 6)))) + 1)
                 for n in range(1, 10)
                 if n * 10**e > 5]
    _phi_ticks = sorted(set(_phi_lin + _phi_log))
    _phi_labels = ['' if abs(t) == 40 else (str(int(t)) if t == int(t) else str(t)) for t in _phi_ticks]
    ax_bph1.yaxis.set_major_locator(FixedLocator(_phi_ticks))
    from matplotlib.ticker import FixedFormatter
    ax_bph1.yaxis.set_major_formatter(FixedFormatter(_phi_labels))
    ax_bph1.yaxis.set_minor_locator(FixedLocator(
        [s * n * 10**e for s in [1,-1] for e in range(1,4)
         for n in range(1,10) if n*10**e > 5 and n != 1]))
    ax_bph1.grid(True, which='both', ls='--', alpha=0.4)
    _spine(ax_bph1)
    ax_bph1.text(0.98, 0.95, r'Solar $\nu$ + WIMP SHM',
              transform=ax_bph1.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='top')
    # ── copy exact y-axis from ax_bph1 onto ax_bph_er ────────────────────────
    ax_bph_er.set_yscale('symlog', linthresh=5)
    ax_bph_er.yaxis.set_major_locator(FixedLocator(_phi_ticks))
    ax_bph_er.yaxis.set_major_formatter(FixedFormatter(_phi_labels))
    ax_bph_er.yaxis.set_minor_locator(FixedLocator(
        [s * n * 10**e for s in [1,-1] for e in range(1,4)
         for n in range(1,10) if n*10**e > 5 and n != 1]))
    ax_bph_er.set_ylim(ax_bph1.get_ylim())
    ax_bph_er.set_ylabel(r'Bias $\phi_{{\nu}}$ [days]', fontsize=13)
    ax_bph_er.grid(True, which='both', ls='--', alpha=0.4)
    _spine(ax_bph_er)
    ax_bph_er.text(0.98, 0.95, r'Solar $\nu$ only',
              transform=ax_bph_er.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='top')


    # ── Row 2, col 1 : bias modulation counts vs mass ────────────────────────
    ax_bmod1 = fig.add_subplot(gs[2, 1])
    ln_mod, = ax_bmod1.semilogx(m, bias_mod1_counts, 'o-', color='tab:red',
                                 label=f'Er=[{Er_min:.2g},{Er_max:.2g}] keV')
    ax_bmod1.axhline(0, color='grey', lw=0.8, ls='-')
    ax_bmod1.set_xlabel(r'WIMP mass [GeV]')
    ax_bmod1.set_ylabel(r'Bias modulation $\nu$ [counts]')
    ax_bmod1.set_yscale('symlog', linthresh=0.1)
    _ydata_max = max(np.nanmax(np.abs(bias_mod1_counts)), 0.2)
    _n_decades = int(np.ceil(np.log10(_ydata_max / 0.1))) + 1
    # linear zone: -0.1 to +0.1 in steps of 0.025
    _lin_ticks = list(np.arange(-0.1, 0.1 + 1e-9, 0.025))
    # positive log zone: dense 2-9 subdivisions per decade
    _pos_log = []
    for e in range(-1, -1 + _n_decades):
        for n in range(1, 10):
            v = n * 10**e
            if v > 0.1:
                _pos_log.append(v)
    # negative log zone: sparse — only 0.2, 0.5 x 10^e and decade boundaries
    _neg_log = []
    for e in range(-1, -1 + _n_decades):
        for n in [1, 2, 5]:
            v = n * 10**e
            if v > 0.1:
                _neg_log.append(-v)
    _fixed_ticks = sorted(set([round(t, 10) for t in _lin_ticks + _pos_log + _neg_log]))
    def _fmt(v):
        if v == 0:
            return '0'
        a = abs(v)
        if a < 0.1 + 1e-9:
            return f'{v:.3g}'
        if a >= 1:
            return f'{v:.3g}'
        return f'{v:.2g}'
    _labels = [_fmt(t) for t in _fixed_ticks]
    from matplotlib.ticker import FixedFormatter
    ax_bmod1.yaxis.set_major_locator(FixedLocator(_fixed_ticks))
    ax_bmod1.yaxis.set_major_formatter(FixedFormatter(_labels))
    ax_bmod1.yaxis.set_minor_locator(NullLocator())
    ax_bmod1.grid(True, which='major', ls='--', alpha=0.4)
    _spine(ax_bmod1)
    ax_bmod1.text(0.98, 0.05, r'Solar $\nu$ + WIMP SHM',
              transform=ax_bmod1.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='bottom')
    # ── draw bias_sig1only data then copy y-axis exactly from ax_bmod1 ────────
    _draw_bias_sig1only(ax_bias_er, (meta, df))
    ax_bias_er.set_yscale('symlog', linthresh=0.1)
    ax_bias_er.yaxis.set_major_locator(FixedLocator(_fixed_ticks))
    ax_bias_er.yaxis.set_major_formatter(FixedFormatter(_labels))
    ax_bias_er.yaxis.set_minor_locator(NullLocator())
    ax_bias_er.set_ylabel(r'Bias modulation $\nu$ [counts]')
    ax_bias_er.set_ylim(ax_bmod1.get_ylim())
    ax_bias_er.grid(True, which='major', ls='--', alpha=0.4)
    _spine(ax_bias_er)
    ax_bias_er.text(0.98, 0.05, r'Solar $\nu$ only',
              transform=ax_bias_er.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='bottom')

    # Er window colour legend in the info box panel
    _er_patch = [plt.Line2D([0],[0], color='tab:red', lw=2,
                             label=f'Er=[{Er_min:.2g},{Er_max:.2g}] keV')]
    ax_info.legend(handles=_er_patch, loc='lower left',
                   bbox_to_anchor=(0, 0), fontsize=12,
                   frameon=True, framealpha=0.9, edgecolor='grey')

    os.makedirs(FIGURES_DIR, exist_ok=True)
    stem     = os.path.basename(scan_file).replace('.json', '')
    out_path = os.path.join(FIGURES_DIR, stem + '_combined.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out_path}')
    return fig


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def select_files(signal1_sources, sim_dir=SIM_DIR):
    """Return all JSON files in sim_dir whose signal1_sources matches the list."""
    files = sorted(glob.glob(os.path.join(sim_dir, 'Solar-WIMP_scan_*_results.json')))
    selected = []
    for f in files:
        with open(f) as fh:
            meta = json.load(fh)['meta']
        if meta['signal1_sources'] == signal1_sources:
            selected.append(f)
    return selected


# ---------------------------------------------------------------------------
# Multi-file plot  (same signal1, different Er windows)
# ---------------------------------------------------------------------------

def plot_multi(signal1_sources, sim_dir=SIM_DIR):
    files = select_files(signal1_sources, sim_dir)
    if not files:
        print('No matching files found.')
        return

    print(f'Found {len(files)} file(s) for signal1={signal1_sources}:')
    for f in files:
        with open(f) as fh:
            meta = json.load(fh)['meta']
        print(f'  {os.path.basename(f)}  Er=[{meta["Er_window_i"]:.3g}, {meta["Er_window_f"]:.3g}] keV')

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(files)))

    # load all data
    datasets = []
    for f in files:
        with open(f) as fh:
            raw = json.load(fh)
        meta = raw['meta']
        df   = pd.DataFrame(raw['data']).sort_values('m_chi_GeV').reset_index(drop=True)
        datasets.append((meta, df))

    # use first file for shared metadata
    meta0    = datasets[0][0]
    detector = meta0['detector']
    signal1  = meta0['signal1_sources']
    P        = float(meta0['P_day'])
    t0_day   = float(meta0['t01_day'])
    t0_SHM   = float(meta0['t02_day'])
    Ad1      = datasets[0][1]['Ad1_nu'].iloc[0]
    N1_0     = datasets[0][1]['N1_nu_in_bin'].values

    phi1_days = t0_day % P
    phi2_days = t0_SHM % P

    def _spine(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 11})
    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.35,
                            height_ratios=[3, 2, 2])

    # ── Row 0, col 0 : dN/dEr + Ad stacked in a nested grid ─────────────────
    Er_min0 = float(meta0['Er_window_i'])
    Er_max0 = float(meta0['Er_window_f'])
    gs_left0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                                hspace=0.05, height_ratios=[3, 2])
    ax_spec = fig.add_subplot(gs_left0[0])
    ax_Ad   = fig.add_subplot(gs_left0[1], sharex=ax_spec)

    _draw_NR_spectrum(ax_spec, ax_Ad,
                      signal1_sources=signal1, detector=detector,
                      Er_min=Er_min0, Er_max=Er_max0,
                      er_color=colors[0])

    # ── Row 1, col 0 : bias phi vs Er (sig1 only) ────────────────────────────
    ax_bph_er = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    _draw_bias_phi_sig1only(ax_bph_er, datasets, colors)

    # ── Row 2, col 0 : bias modulation ν vs Er (drawn after ticks computed) ──
    ax_bias_er = fig.add_subplot(gs[2, 0], sharex=ax_spec)

    # ── Row 0, col 1 : run parameter info box ────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    info_lines = [
        r'$\bf{Run\ parameters}$',
        f'signal1    : {", ".join(signal1)}',
        f'target     : {meta0["target"]}',
        f'detector   : {detector}',
        f'channel    : {meta0["channel"]}',
        f'D          = {meta0["D_tonne"]} t',
        f'dt         = {meta0["dt_day"]} days',
        f'T          = {meta0["T_yr"]} yr',
        f'Ad1        = {Ad1:.4g}',
        f't01        = {t0_day:.2f} days',
        f't02        = {t0_SHM:.2f} days',
        f'n_toys     = {meta0["N_toys"]}',
    ]
    ax_info.text(0.05, 0.98, '\n'.join(info_lines),
                 transform=ax_info.transAxes, fontsize=12,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                           edgecolor='goldenrod', alpha=0.9))

    # ── Row 1, col 1 : bias phase days vs mass ───────────────────────────────
    ax_bph1  = fig.add_subplot(gs[1, 1])

    # ── Row 2, col 1 : bias modulation counts vs mass ────────────────────────
    ax_bmod1 = fig.add_subplot(gs[2, 1])

    for (meta_i, df_i), color in zip(datasets, colors):
        m      = df_i['m_chi_GeV'].values
        N1     = df_i['N1_nu_in_bin'].values
        lbl    = f'Er=[{meta_i["Er_window_i"]:.2g},{meta_i["Er_window_f"]:.2g}] keV'

        bias_mod1_counts_i = df_i['bias_mod1_counts'].values
        bias_phase1_days   = df_i['bias_phase1_deg'].values / 360.0 * P

        ax_bph1.semilogx(m, bias_phase1_days, 'o-', color=color, label=lbl)
        ax_bmod1.semilogx(m, bias_mod1_counts_i, 'o-', color=color, label=lbl)

    from matplotlib.ticker import FixedLocator, NullLocator

    ax_bph1.axhline(0, color='grey', lw=0.8, ls='-')
    ax_bph1.set_xlabel(r'WIMP mass [GeV]')
    ax_bph1.set_ylabel(r'Bias $\phi_{\nu}$ [days]')
    ax_bph1.set_yscale('symlog', linthresh=5)
    from matplotlib.ticker import FixedLocator, AutoMinorLocator
    _phi_lin  = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    _phi_ymax = max(abs(v) for v in ax_bph1.get_ylim()) if ax_bph1.get_ylim()[1] != 1.0 else 50
    _phi_log  = [s * n * 10**e
                 for s in [1, -1]
                 for e in range(1, int(np.ceil(np.log10(max(_phi_ymax, 6)))) + 1)
                 for n in range(1, 10)
                 if n * 10**e > 5]
    _phi_ticks = sorted(set(_phi_lin + _phi_log))
    _phi_labels = ['' if abs(t) == 40 else (str(int(t)) if t == int(t) else str(t)) for t in _phi_ticks]
    ax_bph1.yaxis.set_major_locator(FixedLocator(_phi_ticks))
    from matplotlib.ticker import FixedFormatter
    ax_bph1.yaxis.set_major_formatter(FixedFormatter(_phi_labels))
    ax_bph1.yaxis.set_minor_locator(FixedLocator(
        [s * n * 10**e for s in [1,-1] for e in range(1,4)
         for n in range(1,10) if n*10**e > 5 and n != 1]))
    ax_bph1.grid(True, which='both', ls='--', alpha=0.4)
    _spine(ax_bph1)
    ax_bph1.text(0.98, 0.95, r'Solar $\nu$ + WIMP SHM',
              transform=ax_bph1.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='top')
    # ── copy exact y-axis from ax_bph1 onto ax_bph_er ────────────────────────
    ax_bph_er.set_yscale('symlog', linthresh=5)
    ax_bph_er.yaxis.set_major_locator(FixedLocator(_phi_ticks))
    ax_bph_er.yaxis.set_major_formatter(FixedFormatter(_phi_labels))
    ax_bph_er.yaxis.set_minor_locator(FixedLocator(
        [s * n * 10**e for s in [1,-1] for e in range(1,4)
         for n in range(1,10) if n*10**e > 5 and n != 1]))
    ax_bph_er.set_ylim(ax_bph1.get_ylim())
    ax_bph_er.set_ylabel(r'Bias $\phi_{{\nu}}$ [days]', fontsize=13)
    ax_bph_er.grid(True, which='both', ls='--', alpha=0.4)
    _spine(ax_bph_er)
    ax_bph_er.text(0.98, 0.95, r'Solar $\nu$ only',
              transform=ax_bph_er.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='top')


    # collect all bias_mod1 values across datasets for tick range
    _all_mod1 = np.concatenate([df_i['bias_mod1_counts'].values for _, df_i in datasets])
    _ydata_max = max(np.nanmax(np.abs(_all_mod1)), 0.2)
    _n_decades = int(np.ceil(np.log10(_ydata_max / 0.1))) + 1
    _lin_ticks = list(np.arange(-0.1, 0.1 + 1e-9, 0.025))
    _pos_log = []
    for e in range(-1, -1 + _n_decades):
        for n in range(1, 10):
            v = n * 10**e
            if v > 0.1:
                _pos_log.append(v)
    _neg_log = []
    for e in range(-1, -1 + _n_decades):
        for n in [1, 2, 5]:
            v = n * 10**e
            if v > 0.1:
                _neg_log.append(-v)
    _fixed_ticks = sorted(set([round(t, 10) for t in _lin_ticks + _pos_log + _neg_log]))

    ax_bmod1.axhline(0, color='grey', lw=0.8, ls='-')
    ax_bmod1.set_xlabel(r'WIMP mass [GeV]')
    ax_bmod1.set_ylabel(r'Bias modulation $\nu$ [counts]')
    ax_bmod1.set_yscale('symlog', linthresh=0.1)
    def _fmt(v):
        if v == 0:
            return '0'
        a = abs(v)
        if a < 0.1 + 1e-9:
            return f'{v:.3g}'
        if a >= 1:
            return f'{v:.3g}'
        return f'{v:.2g}'
    _labels = [_fmt(t) for t in _fixed_ticks]
    from matplotlib.ticker import FixedFormatter
    ax_bmod1.yaxis.set_major_locator(FixedLocator(_fixed_ticks))
    ax_bmod1.yaxis.set_major_formatter(FixedFormatter(_labels))
    ax_bmod1.yaxis.set_minor_locator(NullLocator())
    ax_bmod1.grid(True, which='major', ls='--', alpha=0.4)
    _spine(ax_bmod1)
    ax_bmod1.text(0.98, 0.05, r'Solar $\nu$ + WIMP SHM',
              transform=ax_bmod1.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='bottom')
    # ── draw bias_sig1only data then copy y-axis exactly from ax_bmod1 ────────
    _draw_bias_sig1only(ax_bias_er, datasets, colors)
    ax_bias_er.set_yscale('symlog', linthresh=0.1)
    ax_bias_er.yaxis.set_major_locator(FixedLocator(_fixed_ticks))
    ax_bias_er.yaxis.set_major_formatter(FixedFormatter(_labels))
    ax_bias_er.yaxis.set_minor_locator(NullLocator())
    ax_bias_er.set_ylabel(r'Bias modulation $\nu$ [counts]')
    ax_bias_er.set_ylim(ax_bmod1.get_ylim())
    ax_bias_er.grid(True, which='major', ls='--', alpha=0.4)
    _spine(ax_bias_er)
    ax_bias_er.text(0.98, 0.05, r'Solar $\nu$ only',
              transform=ax_bias_er.transAxes, fontsize=13, fontweight='bold',
              ha='right', va='bottom')



    # Er window colour legend in the info box panel
    _er_handles = [plt.Line2D([0],[0], color=c, lw=2.5,
                               label=f'Er=[{mi["Er_window_i"]:.2g},{mi["Er_window_f"]:.2g}] keV')
                   for (mi, _), c in zip(datasets, colors)]
    ax_info.legend(handles=_er_handles, loc='lower left',
                   bbox_to_anchor=(0, 0), fontsize=12, ncol=2,
                   frameon=True, framealpha=0.9, edgecolor='grey')



    os.makedirs(FIGURES_DIR, exist_ok=True)
    src_tag  = '_'.join(signal1)
    out_path = os.path.join(FIGURES_DIR, f'Solar-WIMP_{src_tag}_multiEr_combined.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out_path}')
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_scanned_toyrunner_Solar-WIMP.py \'8B\'')
        print('       python plot_scanned_toyrunner_Solar-WIMP.py \'"pp","Be7_384","8B"\'')
        sys.exit(1)
    import re
    raw_arg = ' '.join(sys.argv[1:])
    signal1 = re.findall(r'[\w]+', raw_arg)
    # re.findall splits on punctuation — rejoin sources that were split
    # actually parse quoted strings if present, else treat each word as a source
    quoted = re.findall(r'"([^"]+)"', raw_arg)
    if quoted:
        signal1 = quoted
    print(f'Parsed signal1: {signal1}')
    plot_multi(signal1)
    plt.show()