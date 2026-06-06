"""
toy_runner.py
=============
Noise layer — adds Poisson noise to the noiseless superposition
and runs N random trials to build the ensemble for hypothesis testing.

Also provides the noiseless single-shot check (noise_type='none')
which verifies the calculation is self-consistent before any noise is added.

This file is the only place where randomness lives.
decomp_calc.py never touches a random number.

Noise options
-------------
'poisson'  : np.random.poisson  (default, physical)
'gaussian' : np.random.normal with sigma = sqrt(counts)  (large-N approximation)
'none'     : no noise — single shot, checks input = reconstructed

Usage
-----
from toy_runner import run_toys, run_noiseless_check
"""

import numpy as np
import decomp_calc as calc


# ─────────────────────────────────────────────────────────────────────────────
# NOISE
# ─────────────────────────────────────────────────────────────────────────────

def add_noise(input_counts_ts, noise_type='poisson', rng=None):
    """
    Add noise to a noiseless counts array.

    Parameters
    ----------
    input_counts_ts : ndarray  noiseless expected number of events in each bin [#], unitless
    noise_type      : 'poisson' | 'gaussian' | 'none'
    rng             : numpy Generator (for reproducibility)

    Returns
    -------
    noisy_counts_ts : ndarray  noisy counts time series [#]
    """
    if rng is None:
        rng = np.random.default_rng()

    if noise_type == 'none':
        return input_counts_ts.copy()
    elif noise_type == 'poisson':
        #return rng.poisson(input_counts_ts).astype(float)
        return rng.poisson(np.maximum(input_counts_ts, 0)).astype(float)
    elif noise_type == 'gaussian':
        sigma = np.sqrt(np.abs(input_counts_ts))
        return rng.normal(input_counts_ts, sigma)
    else:
        raise ValueError(f"noise_type must be 'poisson', 'gaussian', or 'none'. Got: {noise_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# NOISELESS SINGLE-SHOT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def run_noiseless_check(Ad1, Phase1_true_rad, N1_in_bin,
                        Ad2, Phase2_true_rad, N2_in_bin,
                        Nbkg_in_bin,
                        period_days, T_days, dt_days,
                        fluctuation_type='integral',
                        print_check=True):
    """
    Ideal case: no noise, single shot.
    Verifies that the reconstructed UNKNOWN matches the input exactly.
    Run this before any toy study to confirm the calculation is self-consistent.

    Parameters  (all plain floats, N values are number of events in one bin [#], unitless)
    ----------
    Ad1, Phase1_true_rad, N1_in_bin  : known signal
    Ad2, Phase2_true_rad, N2_in_bin  : unknown signal  (truth)
    Nbkg_in_bin                      : number of background events in one bin [#], unitless
    period_days                      : float
    T_days                           : float  total run time [days]
    dt_days                          : float  bin width [days]

    Returns
    -------
    result dict from decomp_calc.decompose_noiseless
    """
    result = calc.decompose_noiseless(
        Ad1, Phase1_true_rad, N1_in_bin,
        Ad2, Phase2_true_rad, N2_in_bin,
        Nbkg_in_bin,
        period_days, T_days, dt_days,
        fluctuation_type=fluctuation_type,
    )

    if print_check:
        mod2_true_counts = Ad2 * N2_in_bin
        # not from LS fit — computed from truth for error calculation
        para2_cos_true = mod2_true_counts * np.cos(Phase2_true_rad)
        para2_sin_true = mod2_true_counts * np.sin(Phase2_true_rad)

        print()
        print('  ── Noiseless consistency check ──')
        print(f' input KNOWN   : Ad={Ad1:.6f}  φ={np.rad2deg(Phase1_true_rad):.4f}°  N_in_bin={N1_in_bin:.3f}')
        print(f' input UNKNOWN : Ad={Ad2:.6f}  φ={np.rad2deg(Phase2_true_rad):.4f}°  N_in_bin={N2_in_bin:.3f}')
        print(f'  Truth   : mod2_true [#] = {mod2_true_counts:.6f}  φ={np.rad2deg(Phase2_true_rad):.4f}°')
        print(f'  Fit     : mod2_fit  [#] = {result["mod2_fit_counts"]:.6f}  φ={np.rad2deg(result["Phase2_fit_rad"]):.4f}°')
        #print(f'  |mod2_fit[#] - mod2_true[#]| / mod2_true[#] * 100% = {(abs(result["delta_mod2_counts"]) / mod2_true_counts * 100):.4f} %')
        #print(f'  |φ2_fit - φ2_truth| / φ2_truth * 100% = {(abs(result["delta_Phase_deg"]) / np.rad2deg(Phase2_true_rad) * 100):.4f} %')
        total_error = (result["delta_para_cos"]**2 + result["delta_para_sin"]**2)**0.5
        print(f'  total phasor error = (delta_para_cos[#]**2 + delta_para_sin[#]**2)**0.5 = {total_error:.2e} [#]')
        
        #claude said this thing is 1/SNR, noiseless case it <~6% depends on dt, which should be 0.
        #but with noise SNR is <~3, so 1/SNR is >> 6%, so its not a problem.
        print(f'  total phasor error / mod2_true[#] * 100% = {total_error / mod2_true_counts * 100:.4f} %')
        #
        print()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TOY RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_toys(Ad1, Phase1_true_rad, N1_in_bin,
             Ad2, Phase2_true_rad, N2_in_bin,
             Nbkg_in_bin,
             period_days, T_days, dt_days,
             n_toys=2000,
             noise_type='poisson',
             seed=42,
             fluctuation_type='integral'):
    """
    Run n_toys noisy experiments and collect decomposition results.

    Purpose: test how well signal 2 (UNKNOWN) can be extracted from the
    total signal, given that signal 1 (KNOWN) is subtracted analytically.
    Each toy adds Poisson noise to the noiseless superposition, runs the
    LS phasor extraction, and subtracts the known signal 1 phasor to
    recover signal 2.

    Parameters  (all plain floats, N values are number of events in one bin [#], unitless)
    ----------
    Ad1, Phase1_true_rad, N1_in_bin  : known signal
    Ad2, Phase2_true_rad, N2_in_bin  : unknown signal  (truth)
    Nbkg_in_bin                : number of background events in one bin [#], unitless
    period_days                  : float
    T_days                       : float  total run time [days]
    dt_days                      : float  bin width [days]
    n_toys                       : int    number of random trials
    noise_type                   : 'poisson' | 'gaussian' | 'none'
                                   'none' runs one trial with no noise
                                   (equivalent to noiseless check in array form)
    seed                         : int    random seed

    Returns
    -------
    dict with arrays of length n_toys (or 1 if noise_type='none'):
      mod2_fit_counts, Phase2_fit_rad : recovered signal 2
      para2_cos_fit, para2_sin_fit    : recovered Cartesian [#]
      delta_para_cos, delta_para_sin  : residuals fit - truth [#]
    and scalar truth values:
      mod2_true_counts, Phase2_true_rad, para2_cos_true, para2_sin_true
    """
    rng = np.random.default_rng(seed)

    time_bins, time_bin_centers = calc.make_time_bins(T_days, dt_days)

    # noiseless superposition built once
    total_noiseless_ts, _, _, _ = calc.make_superposed_timeseries(
        Ad1, Phase1_true_rad, N1_in_bin,
        Ad2, Phase2_true_rad, N2_in_bin,
        Nbkg_in_bin,
        period_days, time_bins, time_bin_centers, dt_days,
        fluctuation_type=fluctuation_type,
    )

    mod1_true_counts = Ad1 * N1_in_bin
    mod2_true_counts = Ad2 * N2_in_bin #for wimp is 1 * mod2_true_counts 
    # not from LS fit — computed from truth for error calculation
    para2_cos_true   = mod2_true_counts * np.cos(Phase2_true_rad)
    para2_sin_true   = mod2_true_counts * np.sin(Phase2_true_rad)

    n_actual = 1 if noise_type == 'none' else n_toys

    out = dict(mod2_fit_counts=[], Phase2_fit_rad=[],
               para2_cos=[], para2_sin=[],
               delta_para2_cos=[], delta_para2_sin=[])

    for _ in range(n_actual):
        noisy_ts = add_noise(total_noiseless_ts, noise_type=noise_type, rng=rng)

        para_cos, para_sin, _ = calc.get_LS_phasor(  # _ = Phase_total_fit_rad, not needed here
            period_days, time_bin_centers, noisy_ts
        )

        mod2_fit_counts, Phase2_fit_rad, para2_cos_sin = calc.split_signal_Ads_Phases(
            mod1_true_counts, Phase1_true_rad,
            para_cos, para_sin,
            dt_days, period_days,
        )

        out['mod2_fit_counts'].append(mod2_fit_counts)
        out['Phase2_fit_rad'].append(Phase2_fit_rad)
        out['para2_cos'].append(para2_cos_sin[0])
        out['para2_sin'].append(para2_cos_sin[1])
        out['delta_para2_cos'].append(para2_cos_sin[0] - para2_cos_true)
        out['delta_para2_sin'].append(para2_cos_sin[1] - para2_sin_true)

    for k in out:
        out[k] = np.array(out[k])

    out.update(
        mod2_true_counts=mod2_true_counts,
        Phase2_true_rad=Phase2_true_rad,
        para2_cos_true=para2_cos_true,
        para2_sin_true=para2_sin_true,
        noise_type=noise_type,
        n_toys=n_actual,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BIAS RUNNER  —  many toys, single-signal fit only
# ─────────────────────────────────────────────────────────────────────────────

def run_bias_toys(Ad1, Phase1_true_rad, N1_in_bin,
                  Ad2, Phase2_true_rad, N2_in_bin,
                  Nbkg_in_bin,
                  period_days, T_days, dt_days,
                  n_toys=1000,
                  noise_type='poisson',
                  seed=42,
                  fluctuation_type='integral'):
    """
    Run n_toys experiments.  Each toy:
      1. Generate noisy counts from signal1 + signal2 + background.
      2. Fit assuming ONLY signal 1 exists (no decomposition).
      3. Record recovered signal-1 parameters and residuals vs truth.

    Purpose: test how much the presence of signal 2 biases the recovery
    of signal 1 when signal 2 is ignored.
 
    Returns
    -------
    dict with arrays of length n_toys:
      mod1_fit_counts        : recovered modulation amplitude of signal 1 [#]
      Phase1_fit_rad         : recovered phase of signal 1 [rad, dimensionless]
      para1_cos, para1_sin   : recovered Cartesian components [#]
      delta_para1_cos, delta_para1_sin : bias = fit - truth  (should be 0 if signal 2 is absent)
    and scalar truth values.
    """
    rng = np.random.default_rng(seed)
 
    time_bins, time_bin_centers = calc.make_time_bins(T_days, dt_days)
 
    # Noiseless superposition (signal 1 + signal 2 + background)
    total_noiseless_ts, _, _, _ = calc.make_superposed_timeseries(
        Ad1, Phase1_true_rad, N1_in_bin,
        Ad2, Phase2_true_rad, N2_in_bin,
        Nbkg_in_bin,
        period_days, time_bins, time_bin_centers, dt_days,
        fluctuation_type=fluctuation_type,
    )
 
    # Signal 1 truth in Cartesian phasor space
    mod1_true_counts = Ad1 * N1_in_bin
    # not from LS fit — computed from truth for error calculation
    para1_cos_true   = mod1_true_counts * np.cos(Phase1_true_rad)#unitless
    para1_sin_true   = mod1_true_counts * np.sin(Phase1_true_rad)#unitless
    
    # Signal 2 truth (for reference / labelling only)
    mod2_true_counts = Ad2 * N2_in_bin
 
    out = dict(mod1_fit_counts=[], Phase1_fit_rad=[],
               para1_cos=[], para1_sin=[],
               delta_para1_cos=[], delta_para1_sin=[])
 
    n_actual = 1 if noise_type == 'none' else n_toys
 
    for _ in range(n_actual):
        noisy_ts = add_noise(total_noiseless_ts, noise_type=noise_type, rng=rng)
 
        # fit total signal — treating everything as one signal (signal 2 ignored)
        mod_total_fit_counts, Phase_total_fit_rad, para_cos, para_sin = calc.fit_total_signal(
            noisy_ts, period_days, time_bin_centers
        )
        
        out['mod1_fit_counts'].append(mod_total_fit_counts)   # treated as signal 1
        out['Phase1_fit_rad'].append(Phase_total_fit_rad.value) #Phase_total_fit_rad [rad, force to value]
        out['para1_cos'].append(para_cos)
        out['para1_sin'].append(para_sin)
        out['delta_para1_cos'].append(para_cos - para1_cos_true)
        out['delta_para1_sin'].append(para_sin - para1_sin_true)
        #print(mod_total_fit_counts, np.sqrt(para_cos**2 + para_sin**2))
        #print(para_cos,para1_cos_true , para_sin, para1_sin_true)
    for k in out:
        out[k] = np.array(out[k])
    
    # sanity check: if Ad2=0, signal 2 is absent and bias must be 0
    # if this warns, there is a bug — the bias test results are meaningless
    #bias for signal 1 is not zero even under noiseless case when dt is larger. 
    #still an issue related to large dt bins, this warning pop out when dt= 30 days, under noiseless case.
    if Ad2 == 0:
        mean_bias_cos = np.mean(out['delta_para1_cos'])
        mean_bias_sin = np.mean(out['delta_para1_sin'])
        threshold = 1e-2 * max(abs(para1_cos_true), abs(para1_sin_true), 1e-10)
        if abs(mean_bias_cos) > threshold or abs(mean_bias_sin) > threshold:
            import warnings
            warnings.warn(
                f'Ad2=0 but bias is non-zero: '
                f'mean(delta_para1_cos)={mean_bias_cos:.2e}, '
                f'mean(delta_para1_sin)={mean_bias_sin:.2e}. '
                f'Bug in run_bias_toys — bias results are meaningless.',
                UserWarning, stacklevel=2
            )
    out.update(
        mod1_true_counts=mod1_true_counts,
        Phase1_true_rad=Phase1_true_rad,
        para1_cos_true=para1_cos_true,
        para1_sin_true=para1_sin_true,
        mod2_true_counts=mod2_true_counts,
        Phase2_true_rad=Phase2_true_rad,
        dphi_deg=np.rad2deg((Phase2_true_rad - Phase1_true_rad) % (2 * np.pi)),
        n_toys=n_actual,
        noise_type=noise_type,
    )
    return out



# ─────────────────────────────────────────────────────────────────────────────
#signal 1 will pump signal 2 up
# ─────────────────────────────────────────────────────────────────────────────
def run_noise_floor(Ad1, Phase1_true_rad, N1_in_bin,
               Ad2, Phase2_true_rad, N2_in_bin,
               Nbkg_in_bin,
               period_days, T_days, dt_days,
               n_toys=1000, noise_type='poisson', seed=42,
               fluctuation_type='integral'):
    """
    Run toys and inject mod1_true_counts, mod2_true_counts so inflation stats work.
    Measures how much signal 1 pumps up the recovered signal 2 amplitude.
    """
    res = run_toys(
        Ad1, Phase1_true_rad, N1_in_bin,
        Ad2, Phase2_true_rad, N2_in_bin,
        Nbkg_in_bin,
        period_days, T_days, dt_days,
        n_toys=n_toys, noise_type=noise_type, seed=seed,
        fluctuation_type=fluctuation_type,
    )
    res['mod1_true_counts'] = Ad1 * N1_in_bin   # modulation amplitude of signal 1 [#]
    res['mod2_true_counts'] = Ad2 * N2_in_bin   # modulation amplitude of signal 2 [#]
    return res
 
    
# ─────────────────────────────────────────────────────────────────────────────
#withouht signal1, noise will pump signal 2 up
# ─────────────────────────────────────────────────────────────────────────────
def run_single_signal_toys(Ad2, Phase2_true_rad, N2_in_bin,
                            Nbkg_in_bin,
                            period_days, T_days, dt_days,
                            n_toys=1000, noise_type='poisson', seed=42,
                            fluctuation_type='integral'):
    """
    Single-signal noise floor: Ad1=0, signal 1 is absent.
    Measures how much pure Poisson noise pumps up the recovered signal 2
    amplitude, with no signal 1 contribution at all.
    Wrapper around run_noise_floor with Ad1=0.
 
    The meaningful amplitude is mod2_true_counts = Ad2 * N2_in_bin [#].
    N2_in_bin already encodes R * mass * dt, so Rs is not needed here.
    """
    res = run_noise_floor(
        0.0, 0.0, 0.0,        # Ad1=0 — signal 1 absent
        Ad2, Phase2_true_rad, N2_in_bin,
        Nbkg_in_bin,
        period_days, T_days, dt_days,
        n_toys=n_toys, noise_type=noise_type, seed=seed,
        fluctuation_type=fluctuation_type,
    )
    return res
    
