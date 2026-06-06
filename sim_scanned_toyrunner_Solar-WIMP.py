"""
scan_wimp_erwindow.py
=====================
For a given Er window and signal1 sources, loop over all simulated WIMP
masses, run toys, and store results.

Edit the parameters section at the top, then run:
    python3 scan_scanned_toyrunner_wimp_erwindow.py
"""

import sys
import os
import numpy as np
import astropy.units as u
import astropy.constants as const

# ---------------------------------------------------------------------------
# Repo paths — edit once
# ---------------------------------------------------------------------------

sys.path.insert(0, '/home/echo/phasor_decomp')
sys.path.insert(0, '/home/echo/neutrino_spectrum')
sys.path.insert(0, '/home/echo/wimp_spectrum')

import convert_params as conv
import toy_runner as toys
import compute_toy_runner_stat as calctoys
import get_realistic_spectrum as spec
import get_wimp_Erwindow as wimpspec
from datetime import datetime 
# ---------------------------------------------------------------------------
# Parameters — edit here
# ---------------------------------------------------------------------------

TARGET      = 'Xe'
DETECTOR    = 'Xe1t'
CHANNEL     = 'NR'

Er_window   = np.linspace(4,6,101) * u.keV

SIGNAL1_SOURCES = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17', '8B', 'hep']   #['8B']# neutrino sources for signal1

D           = 100.0  * u.tonne
dt          = 15.0   * u.day
T           = 10.0   * u.yr
P           = 365.25 * u.day
t0          = 3.0    * u.day
t0_SHM      = 152.5  * u.day
Ad1         = 0.03342

N_TOYS           = 1000
NOISE_TYPE       = 'poisson'
SEED             = 42
SIGNIFICANCE     = 0.1
SNR_THRESHOLD    = 2.0
TOLERANCE_NSIGMA = 2.0
BIAS_TOLERANCE   = 0.1

output_folder = 'sim_data'
os.makedirs(output_folder, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
FILE_name = f'Solar-WIMP_scan_{timestamp}'+'_results.json'
SAVE_FILE = os.path.join(output_folder, FILE_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # --- get WIMP mapping ---------------------------------------------------
    m_chis, R_wimps, Ad_R_itgl_wimps = wimpspec.get_wimp_erwindow_mapping(
        target    = TARGET,
        Er_window = Er_window,
        detector  = DETECTOR)

    # --- get signal1 rate ---------------------------------------------------
    Er, rate_s1, _, _ = spec.get_realistic_spectrum(
        target   = TARGET,
        channel  = CHANNEL,
        signal1  = SIGNAL1_SOURCES,
        signal2  = [],
        bkgd     = [],
        mode     = 'realistic',
        detector = DETECTOR)

    R1_nu = wimpspec.integrate_in_window(Er, rate_s1, Er_window)
    print(f'R1_nu = {R1_nu:.4e}')

    # --- phases -------------------------------------------------------------
    phi1 = conv.peak_day_to_phase(t0,     P)
    phi2 = conv.peak_day_to_phase(t0_SHM, P)

    T_days  = conv.T_to_days(T)
    dt_days = conv.dt_to_days(dt)
    P_days  = P.to(u.day).value
    
    # --- signal 1 and background is fixed----
    Rs1 = R1_nu
    Rb  = 0.
    
    N1 = conv.rate_to_counts_per_bin(Rs1, D, dt)
    Nb = conv.rate_to_counts_per_bin(Rb,  D, dt)
    
    # --- run_bias_toys: sig1 only, without sig2, for comparison ----
    res_sig1only = toys.run_bias_toys(
        Ad1, phi1.value, N1,
        0., 0., 0.,
        Nb, P_days, T_days, dt_days,
        n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED)

    stat_sig1only = calctoys.compute_bias_stats(res_sig1only, tolerance=BIAS_TOLERANCE)
    
    # --- loop over WIMP masses ----------------------------------------------
    results = []

    for m_chi, R_wimp, Ad_R_wimp in zip(m_chis, R_wimps, Ad_R_itgl_wimps):
        Rs2 = R_wimp
        mod2_true_counts = conv.rate_to_counts_per_bin(Ad_R_wimp, D, dt)
        

        print(f'\nm_chi={m_chi:.1f}  N1={N1:.4f} integrated Ad_R_wimp={Ad_R_wimp:.4f}  mod2_true_counts={mod2_true_counts:.4f}')

        # run_toys: sig2 extraction
        #run_toys use mod2_true_counts = Ad2 * N2
        #for WIMP Ad_R_wimp is integral Ad(Er) * R(Er), 
        res_split2 = toys.run_toys(
            Ad1, phi1.value, N1,
            1., phi2.value, mod2_true_counts,
            Nb, P_days, T_days, dt_days,
            n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED)

        stat_split2 = calctoys.compute_sig2_recovery(
            res_split2,
            significance_level=SIGNIFICANCE,
            snr_threshold=SNR_THRESHOLD,
            tolerance_nsigma=TOLERANCE_NSIGMA)

        # run_bias_toys: sig1 contamination
        res_ignore2 = toys.run_bias_toys(
            Ad1, phi1.value, N1,
            1., phi2.value, mod2_true_counts,
            Nb, P_days, T_days, dt_days,
            n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED)

        stat_ignore2 = calctoys.compute_bias_stats(
            res_ignore2, tolerance=BIAS_TOLERANCE)
        
        

        results.append({
            'm_chi_GeV'                      : m_chi.to(u.GeV).value,
            'mod2_true_rate'                 :Ad_R_wimp.value,
            'R2_wimp'                        :Rs2.value,
            'mod2_true_counts'               : mod2_true_counts,
            'median_mod2_fit_counts'         : stat_split2['median_mod2_fit_counts'],
            'mean_Phase2_fit_rad'            : stat_split2['mean_Phase2_fit_rad'],
            'std_Phase2_fit_rad'             : stat_split2['std_Phase2_fit_rad'],
            'bias_mod1_counts'               : stat_ignore2['bias_mod1_counts'],
            'bias_phase1_deg'                : stat_ignore2['bias_phase1_deg'],
            
        })

        print(f'  mod2_fit={stat_split2["median_mod2_fit_counts"]:.4f}  '
              f'phase2={stat_split2["mean_Phase2_fit_rad"]:.4f} rad  '
              f'bias_mod1_counts={stat_ignore2["bias_mod1_counts"]:.4f}  '
              f'bias_phase1={stat_ignore2["bias_phase1_deg"]:.4f} deg    ')

    # --- save ---------------------------------------------------------------
    import pandas as pd
    import json

    df = pd.DataFrame(results)
    df['Ad1_nu']       = Ad1
    df['R1_nu']        = Rs1.value
    df['N1_nu_in_bin'] = N1

    meta = {
        'signal1_sources'                : SIGNAL1_SOURCES,
        'target'                         : TARGET,
        'detector'                       : DETECTOR,
        'channel'                        : CHANNEL,
        'Er_window_i'                    : float(Er_window.to(u.keV).value[0]),
        'Er_window_f'                    : float(Er_window.to(u.keV).value[-1]),
        'D_tonne'                        : float(D.to(u.tonne).value),
        'dt_day'                         : float(dt.to(u.day).value),
        'T_yr'                           : float(T.to(u.yr).value),
        'P_day'                          : float(P.to(u.day).value),
        't01_day'                        : float(t0.to(u.day).value),
        't02_day'                        : float(t0_SHM.to(u.day).value),
        'bias_mod1_counts_sig1only'      : stat_sig1only['bias_mod1_counts'],
        'bias_phase1_deg_sig1only'       : stat_sig1only['bias_phase1_deg'],
        
        'N_toys'                         : N_TOYS,
        'noise_type'                     : NOISE_TYPE,
        'seed'                           : SEED,
    }

    output = {
        'meta': meta,
        'data': df.to_dict(orient='records')
    }

    SAVE_FILE = SAVE_FILE.replace('.csv', '.json')
    with open(SAVE_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved -> {SAVE_FILE}')