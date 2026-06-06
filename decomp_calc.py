"""
decomp_calc.py
==============
Pure calculation layer — no astropy units, no noise, no I/O.
Everything operates on plain numpy arrays of counts (floats).

This file only cares about numbers:
  - build noiseless time series from count-level parameters
  - extract LS phasor at a fixed period
  - decompose known component out of total phasor
  - compute Cartesian residuals and statistics across an ensemble

Called by toy_runner.py (which handles noise + random trials)
and by convert_params.py (which handles unit conversion).

── Notation glossary ──────────────────────────────────────────────────────────
Ad              : fractional modulation amplitude, dimensionless number between
                  0 and 1.  Ad=0 means no modulation; Ad=1 means 100% peak-to-
                  mean variation.  Ad alone has no physical scale.

N_in_bin        : number of events in one time bin [#], unitless integer-valued
                  float.  Encodes detector mass × run rate × bin width.
                  N_in_bin already has physical scale baked in.

mod_true_counts : Ad * N_in_bin  [#], unitless.  The actual modulation amplitude
                  in counts — the peak-to-mean variation of the signal expressed
                  as a number of events.  This is the physically meaningful
                  quantity; Ad and N_in_bin individually are not.

mod_counts_ts   : mod_true_counts * time_variation(t)  [#], unitless.
                  A time series of the modulation — how many events above/below
                  the flat mean in each bin.  Zero-mean by construction.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy.optimize import root
from scipy.stats import chi2 as chi2_dist, norm as norm_dist, kstest
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# support function circular mean std for phases
# ─────────────────────────────────────────────────────────────────────────────
def circular_mean_deg(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360 

def circular_std_deg(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.cos(angles_rad))**2 + np.mean(np.sin(angles_rad))**2)
    return np.rad2deg(np.sqrt(-2 * np.log(R))) # in degrees


# ─────────────────────────────────────────────────────────────────────────────
# TIME SERIES  (all inputs in days / counts)
# ─────────────────────────────────────────────────────────────────────────────

def make_time_bins(T_days, dt_days):
    """
    Parameters
    ----------
    T_days  : float  total run time [days]
    dt_days : float  bin width [days]

    Returns
    -------
    time_bins        : ndarray [days]
    time_bin_centers : ndarray [days]
    """
    time_bins        = np.arange(0, T_days + dt_days, dt_days)
    time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    return time_bins, time_bin_centers


def make_signal_timeseries(Ad, Phase_rad, N_in_bin, Nbkg_in_bin,
                           period_days, time_bins, time_bin_centers, dt_days,
                           fluctuation_type='integral'):
    """
    Noiseless counts time series for one periodic signal component.
    Mirrors get_numperton_H0HA (integral mode) from generate_time_series.py.

    Parameters
    ----------
    Ad            : float  fractional modulation amplitude
    Phase_rad     : float  phase [rad, dimensionless]   peak at t = Phase_rad / (2π) * period
    N_in_bin     : float  number of events in one bin [#], unitless  (flat part)
    Nbkg_in_bin : float  number of background events in one bin [#], unitless  (flat, shared)
    period_days   : float  [days]
    time_bins     : ndarray [days]
    time_bin_centers : ndarray [days]
    dt_days       : float  [days]
    fluctuation_type : 'integral' (default) or 'discrete'

    Returns
    -------
    counts : ndarray  (noiseless number of events in each bin [#], unitless)
    """
    omega = 2 * np.pi / period_days          # rad/day [/time]
    t0    = Phase_rad / (2 * np.pi) * period_days   # peak day [time]

    if fluctuation_type == 'discrete':
        time_var = np.cos(omega * (time_bin_centers - t0)) * dt_days #[time]
    elif fluctuation_type == 'integral':
        time_var = (np.sin(omega * (time_bins[1:] - t0)) -
                    np.sin(omega * (time_bins[:-1] - t0))) / omega #[time]

    # fluctuation amplitude in counts = Ad * N_in_bin * (time_var / dt)
    mod_true_counts = Ad * N_in_bin  # modulation amplitude [#], unitless
    modulation_ts = mod_true_counts * time_var / dt_days  # [#], unitless
    total_counts_ts = (N_in_bin + Nbkg_in_bin) * np.ones(len(time_bin_centers)) + modulation_ts  # [#], unitless
    return total_counts_ts


def make_superposed_timeseries(Ad1, Phase1_rad, N1_in_bin,
                               Ad2, Phase2_rad, N2_in_bin,
                               Nbkg_in_bin,
                               period_days, time_bins, time_bin_centers, dt_days,
                               fluctuation_type='integral'):
    """
    Noiseless superposition of KNOWN + UNKNOWN + background.
    Background is flat; each signal contributes its own modulation on top.

    Returns
    -------
    total_ts   : ndarray  total noiseless number of events in each bin [#]
    known_ts   : ndarray  known signal: number of events in each bin [#], modulation only
    unknown_ts : ndarray  unknown signal: number of events in each bin [#], modulation only
    bkg_ts     : ndarray  flat background: number of events in each bin [#]
    """
    omega = 2 * np.pi / period_days
    t1    = Phase1_rad / (2 * np.pi) * period_days
    t2    = Phase2_rad / (2 * np.pi) * period_days

    if fluctuation_type == 'integral':
        tv1 = (np.sin(omega * (time_bins[1:] - t1)) -
               np.sin(omega * (time_bins[:-1] - t1))) / omega / dt_days
        tv2 = (np.sin(omega * (time_bins[1:] - t2)) -
               np.sin(omega * (time_bins[:-1] - t2))) / omega / dt_days
    else:
        tv1 = np.cos(omega * (time_bin_centers - t1))
        tv2 = np.cos(omega * (time_bin_centers - t2))

    mod1_true_counts = Ad1 * N1_in_bin  # modulation counts of signal 1 [#], unitless
    mod2_true_counts = Ad2 * N2_in_bin  # modulation counts of signal 2 [#], unitless #for wimp is 1 * mod2_true_counts

    mod1_counts_ts = mod1_true_counts * tv1
    mod2_counts_ts = mod2_true_counts * tv2
    bkg_ts  = Nbkg_in_bin * np.ones(len(time_bin_centers))

    known_ts        = mod1_counts_ts
    unknown_ts      = mod2_counts_ts
    total_counts_ts = (N1_in_bin + N2_in_bin) * np.ones(len(time_bin_centers)) + bkg_ts + mod1_counts_ts + mod2_counts_ts
    return total_counts_ts, known_ts, unknown_ts, bkg_ts


# ─────────────────────────────────────────────────────────────────────────────
# LS PHASOR EXTRACTION  (mirrors get_LS_results)
# ─────────────────────────────────────────────────────────────────────────────

def get_LS_phasor(period_days, time_bin_centers_days, input_counts_ts):
    """
    LombScargle at one fixed frequency → (para_cos, para_sin, Phase_fit_rad).
    Follows same convention as your original get_LS_results.

    Parameters
    ----------
    period_days           : float
    time_bin_centers_days : ndarray [days]
    input_counts_ts       : ndarray  input counts time series [#], plain floats

    Returns
    -------
    para_cos      : float  [#], unitless
    para_sin      : float  [#], unitless
    Phase_fit_rad : float  fitted phase [rad, dimensionless]
    """
    from astropy.timeseries import LombScargle
    import astropy.units as u

    freq = (1.0 / period_days) / u.day #[1/time]
    t    = time_bin_centers_days * u.day #[time]

    LS = LombScargle(t, input_counts_ts, normalization='psd')
    N_t = len(input_counts_ts)
    para_sin, para_cos = LS.model_parameters(freq)[1:]

    Phase_fit_rad = np.arctan2(para_sin, para_cos) #[rad, unitless]

    # unit check: para_cos, para_sin should be dimensionless counts [#]
    if hasattr(para_cos, 'unit'):
        if para_cos.unit != u.dimensionless_unscaled:
            import warnings
            warnings.warn(
                f'para_cos has unit {para_cos.unit}, expected dimensionless counts [#]. '
                f'Check that input_counts_ts to get_LS_phasor is plain [#] with no astropy unit.',
                UserWarning, stacklevel=2
            )
        para_cos = para_cos.value
        para_sin = para_sin.value

    return para_cos, para_sin, Phase_fit_rad


def fit_total_signal(input_counts_ts, period_days, time_bin_centers):
    """
    Treat the total observed counts as a single signal.
    Calls get_LS_phasor and returns the total amplitude and phase.
 
    Parameters
    ----------
    input_counts_ts   : ndarray  input counts time series [#], unitless
    period_days       : float    signal period [days]
    time_bin_centers  : ndarray  bin center times [days]
 
    Returns
    -------
    mod_total_fit_counts : float  total fitted modulation amplitude [#], unitless
    Phase_fit_rad        : float  fitted total phase [rad, dimensionless]
    para_cos             : float  cos component from LS [#], unitless
    para_sin             : float  sin component from LS [#], unitless
    """
    para_cos, para_sin, Phase_fit_rad = get_LS_phasor(period_days, time_bin_centers, input_counts_ts)
    
    mod_total_fit_counts = np.sqrt(para_cos**2 + para_sin**2)
    #print(mod_total_fit_counts)
    
    return mod_total_fit_counts, Phase_fit_rad, para_cos, para_sin



def split_signal_Ads_Phases(mod1_true_counts, Phase1_true_rad,
                             para_cos, para_sin,
                             dt_days, period_days):
    """
    Recover (mod2_fit_counts, Phase2_fit_rad) given known component and total phasor.

    Closed-form vector subtraction:
        para_cos = mod1_true_counts*cos(φ1) + mod2_fit_counts*cos(φ2)
        para_sin = mod1_true_counts*sin(φ1) + mod2_fit_counts*sin(φ2)
    → subtract known phasor, read off unknown directly.
    No root finder, no initial guess, no rounding from solver.
    Residual is purely floating point (~1e-15).

    Parameters
    ----------
    mod1_true_counts : float  known signal modulation amplitude [#], unitless
    Phase1_true_rad  : float  known signal true phase [rad, dimensionless]
    para_cos         : float  total phasor cos component from LS
    para_sin         : float  total phasor sin component from LS

    Returns
    -------
    mod2_fit_counts  : float  fitted unknown modulation amplitude [#], unitless
    Phase2_fit_rad   : float  fitted unknown phase [rad, dimensionless]
    para_cos2, para_sin2 : ndarray  UNKNOWN phasor components after subtracting KNOWN
    # delta_angle_deg, round_digit, initial_guess : removed — no longer needed (analytical method)
    """
    # subtract known phasor from total
    attenuation = np.sinc(dt_days / period_days)# claude said this shit can fix reduce error under noiseless case
    para_cos2 = para_cos - mod1_true_counts * np.cos(Phase1_true_rad) * attenuation
    para_sin2 = para_sin - mod1_true_counts * np.sin(Phase1_true_rad) * attenuation

    # unit check: para_cos2, para_sin2 should be dimensionless counts [#]
    if hasattr(para_cos2, 'unit'):
        if para_cos2.unit != u.dimensionless_unscaled:
            import warnings
            warnings.warn(
                f'para_cos2 has unit {para_cos2.unit}, expected dimensionless counts [#]. '
                f'Check inputs to split_signal_Ads_Phases.',
                UserWarning, stacklevel=2
            )
        para_cos2 = para_cos2.value
        para_sin2 = para_sin2.value

    mod2_fit_counts = np.sqrt(para_cos2**2 + para_sin2**2)
    Phase2_fit_rad  = np.arctan2(para_sin2, para_cos2) % (2 * np.pi)

    return mod2_fit_counts, Phase2_fit_rad, np.array([para_cos2, para_sin2])

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SHOT NOISELESS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def decompose_noiseless(Ad1, Phase1_true_rad, N1_in_bin,
                        Ad2, Phase2_true_rad, N2_in_bin,
                        Nbkg_in_bin,
                        period_days, T_days, dt_days,
                        fluctuation_type='integral'):
    """
    Ideal case: no noise. Build superposition, extract phasor, decompose.
    Use this to verify the calculation is self-consistent before adding noise.

    Returns
    -------
    dict with:
      total_ts, known_ts, unknown_ts, bkg_ts   : time series
      time_bins, time_bin_centers               : time arrays [days]
      para_cos, para_sin, Phase_total_rad       : LS phasor of total
      mod2_fit_counts, Phase2_fit_rad           : recovered unknown
      mod2_true_counts, Phase2_true_rad         : truth
      delta_mod2_counts, delta_Phase_deg        : residuals
      para2_cos_true, para2_sin_true, para2_cos_fit, para2_sin_fit : Cartesian phasors
      delta_para_cos, delta_para_sin            : Cartesian residuals
    """
    time_bins, time_bin_centers = make_time_bins(T_days, dt_days)

    total_ts, known_ts, unknown_ts, bkg_ts = make_superposed_timeseries(
        Ad1, Phase1_true_rad, N1_in_bin,
        Ad2, Phase2_true_rad, N2_in_bin,
        Nbkg_in_bin,
        period_days, time_bins, time_bin_centers, dt_days,
        fluctuation_type=fluctuation_type,
    )

    para_cos, para_sin, Phase_total_fit_rad = get_LS_phasor(
        period_days, time_bin_centers, total_ts
    )
    para_cos1_eff, para_sin1_eff, _ = get_LS_phasor(period_days, time_bin_centers, known_ts)
    mod1_eff   = np.sqrt(para_cos1_eff**2 + para_sin1_eff**2)
    Phase1_eff = np.arctan2(para_sin1_eff, para_cos1_eff)

    mod1_true_counts = Ad1 * N1_in_bin
    mod2_true_counts = Ad2 * N2_in_bin

    mod2_fit_counts, Phase2_fit_rad, para2_cos_sin = split_signal_Ads_Phases(
        mod1_eff, Phase1_eff,
        para_cos, para_sin,
        dt_days, period_days,
    )

    para2_cos_fit = para2_cos_sin[0]
    para2_sin_fit = para2_cos_sin[1]
    
    
    # Cartesian truth
    para2_cos_true = mod2_true_counts * np.cos(Phase2_true_rad)
    para2_sin_true = mod2_true_counts * np.sin(Phase2_true_rad)

    return dict(
        time_bins=time_bins, time_bin_centers=time_bin_centers,
        period_days=period_days,
        total_ts=total_ts, known_ts=known_ts,
        unknown_ts=unknown_ts, bkg_ts=bkg_ts,
        para_cos=para_cos, para_sin=para_sin,
        Phase_total_fit_rad=Phase_total_fit_rad,
        mod2_true_counts=mod2_true_counts,   Phase2_true_rad=Phase2_true_rad,
        mod2_fit_counts=mod2_fit_counts,     Phase2_fit_rad=Phase2_fit_rad,
        delta_mod2_counts=mod2_fit_counts - mod2_true_counts,
        delta_Phase_deg=(np.rad2deg(Phase2_fit_rad) - np.rad2deg(Phase2_true_rad)) % 360,
        para2_cos_true=para2_cos_true,   para2_sin_true=para2_sin_true,
        para2_cos_fit=para2_cos_fit,     para2_sin_fit=para2_sin_fit,
        delta_para_cos=para2_cos_fit - para2_cos_true,
        delta_para_sin=para2_sin_fit - para2_sin_true,
    )



# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_noiseless_check(noiseless_result, Ad1, Phase1_true_rad, Ad2, Phase2_true_rad,
                         N1_in_bin, N2_in_bin, save_path=None):
    """
    Single-panel time series plot for the noiseless consistency check.
    Exactly matches the reconstruct notebook (cell 9).
    """
    r = noiseless_result
    t = r['time_bin_centers']

    # modulation-only signals (already zero-mean, mean-subtraction is a no-op)
    known_ts   = r['known_ts']
    unknown_ts = r['unknown_ts']
    total_ts   = r['total_ts']

    # phasor amplitudes in counts
    mod1_true_counts = N1_in_bin * Ad1
    mod2_true_counts = N2_in_bin * Ad2

    # effective amplitude: law of cosines
    mod_total_true_counts = np.sqrt(
        mod1_true_counts**2 + mod2_true_counts**2 +
        2 * mod1_true_counts * mod2_true_counts * np.cos(Phase2_true_rad - Phase1_true_rad)
    )

    # true total phase from phasor addition
    y = mod1_true_counts * np.sin(Phase1_true_rad) + mod2_true_counts * np.sin(Phase2_true_rad)
    x = mod1_true_counts * np.cos(Phase1_true_rad) + mod2_true_counts * np.cos(Phase2_true_rad)
    Phase_total_true_rad = np.arctan2(y, x)

    omega = 2 * np.pi / r['period_days']

    fig, ax = plt.subplots(figsize=(45, 10))

    ax.axhline( mod_total_true_counts, color='grey', lw=1)
    ax.axhline(-mod_total_true_counts, color='grey', lw=1)

    ax.plot(t, known_ts   - known_ts.mean(),   marker='o', label='input KNOWN signal')
    ax.plot(t, unknown_ts - unknown_ts.mean(), marker='o', label='inputUNKNOWN signal')
    ax.plot(t, total_ts   - total_ts.mean(),
            marker='o', color='black', label='input KNOWN + UNKNOWN signals')
    ax.plot(t,
            mod_total_true_counts * np.array([np.cos(ang) for ang in omega * t - Phase_total_true_rad]),
            color='red', ls='--', lw=3, label='input KNOWN + UNKNOWN signals')
    
    mod2_fit_counts = r['mod2_fit_counts']
    Phase2_fit   = r['Phase2_fit_rad']
    ax.plot(t,
            mod2_fit_counts * np.array([np.cos(ang) for ang in omega * t - Phase2_fit]),
            color='green', ls='--', lw=3, label='fitted UNKNOWN')
 

    ax.set_xlabel('day', fontsize=25)
    ax.set_ylabel('counts in bin [#]', fontsize=25)
    ax.tick_params(labelsize=25)
    ax.legend(fontsize=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved {save_path}')
    return fig

