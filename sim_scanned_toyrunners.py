"""
run_scan_toyrunners.py
======================
Scan sig2 reconstruction quality over a 2D grid of (dphi, Ad2/Ad1 ratio).
Saves results to a timestamped folder with metadata.csv and one CSV per stat.

Usage
-----
python run_scan_toyrunners.py
"""

import sys
sys.path.insert(0, '/WIMP_decomp')

import os
import numpy as np
import astropy.units as u
import toy_runner   as toys
import convert_params as conv
import compute_toy_runner_stat as calctoys
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

Rs1  = 20.0  / (u.tonne * u.yr)
Rs2  = Rs1
Rb   = 0.0   / (u.tonne * u.yr)
D    = 10.0  * u.tonne
dt   = 30.0  * u.day
T    = 10.0  * u.yr
P    = 365.25 * u.day

t0   = 3.0 * u.day
phi1 = conv.peak_day_to_phase(t0, P)

N1      = conv.rate_to_counts_per_bin(Rs1, D, dt)
N2      = N1
Nb      = conv.rate_to_counts_per_bin(Rb,  D, dt)
T_days  = conv.T_to_days(T)
dt_days = conv.dt_to_days(dt)
P_days  = P.to(u.day).value

Ad1 = 0.03342

# ── scan grids ───────────────────────────────────────────────────────────────
Ad2_ratio_grid = np.sort(np.unique(np.concatenate([np.logspace(-3, 2, 6), 
                                 np.logspace(-3, 1, 5) * 5, 
                                 np.logspace(-3, 1, 5) * 2,
                                 np.logspace(-3, 1, 5) * 3,
                                np.logspace(-3, 1, 5) * 6,
                                  np.logspace(-3, 1, 5) * 8 ])))
dphi_grid_deg  = np.arange(0, 360,10)

# ── toy runner settings ──────────────────────────────────────────────────────
N_TOYS           = 1000
NOISE_TYPE       = 'poisson'
SEED             = 42
SIGNIFICANCE     = 0.1
SNR_THRESHOLD    = 2.0
TOLERANCE_NSIGMA = 2.0
BIAS_TOLERANCE   = 0.1

print(f'phi1={phi1:.4f} rad')
print(f'N1={N1:.4f}  Nb={Nb:.4f}')
print(f'Ad2_ratio_grid: {Ad2_ratio_grid}')
print(f'dphi_grid_deg:  {dphi_grid_deg}')


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

col_labels = [f'Ad2/Ad1={r:.4g}' for r in Ad2_ratio_grid]
row_labels = [f'dphi={d:.0f}deg'  for d in dphi_grid_deg]

df_SNR                   = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_power_testA           = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_power_testB           = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_median_mod2_fit       = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_mean_Phase2_fit       = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_std_Phase2_fit        = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_bias_mod1_counts      = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_bias_phase1_deg       = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_median_mod2_fit_floor = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_mean_Phase2_fit_floor = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
df_std_Phase2_fit_floor  = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────────────────────────────────────

for Ad2_ratio in Ad2_ratio_grid:
    Ad2 = Ad2_ratio * Ad1
    col = f'Ad2/Ad1={Ad2_ratio:.4g}'

    for dphi_deg in dphi_grid_deg:
        Phase2_true_rad = (phi1.value + np.deg2rad(dphi_deg)) % (2 * np.pi)
        row = f'dphi={dphi_deg:.0f}deg'

        # run_toys: sig2 extraction
        res_split2 = toys.run_toys(
            Ad1, phi1.value, N1,
            Ad2, Phase2_true_rad, N2,
            Nb, P_days, T_days, dt_days,
            n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
        )
        stat_split2 = calctoys.compute_sig2_recovery(
            res_split2,
            significance_level=SIGNIFICANCE,
            snr_threshold=SNR_THRESHOLD,
            tolerance_nsigma=TOLERANCE_NSIGMA,
        )
        df_SNR.loc[row, col]             = stat_split2['SNR']
        df_power_testA.loc[row, col]     = stat_split2['power_testA']
        df_power_testB.loc[row, col]     = stat_split2['power_testB']
        df_median_mod2_fit.loc[row, col] = stat_split2['median_mod2_fit_counts']
        df_mean_Phase2_fit.loc[row, col] = stat_split2['mean_Phase2_fit_rad']
        df_std_Phase2_fit.loc[row, col]  = stat_split2['std_Phase2_fit_rad']

        # run_single_signal_toys: sig2 only, no sig1 (noise floor)
        res_sig2_only = toys.run_single_signal_toys(
            Ad2, Phase2_true_rad, N2,
            Nb, P_days, T_days, dt_days,
            n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
        )
        stats_sig2_only = calctoys.compute_floor_stats(res_sig2_only)
        df_median_mod2_fit_floor.loc[row, col] = stats_sig2_only['median_mod2_fit_counts']
        df_mean_Phase2_fit_floor.loc[row, col] = stats_sig2_only['mean_Phase2_fit_rad']
        df_std_Phase2_fit_floor.loc[row, col]  = stats_sig2_only['std_Phase2_fit_rad']

        # run_bias_toys: sig1 contamination when sig2 is ignored
        res_ignore2 = toys.run_bias_toys(
            Ad1, phi1.value, N1,
            Ad2, Phase2_true_rad, N2,
            Nb, P_days, T_days, dt_days,
            n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
        )
        stat_ignore2 = calctoys.compute_bias_stats(res_ignore2, tolerance=BIAS_TOLERANCE)
        df_bias_mod1_counts.loc[row, col]       = stat_ignore2['bias_mod1_counts']
        df_bias_phase1_deg.loc[row, col] = stat_ignore2['bias_phase1_deg']

        print(f'  Ad2/Ad1={Ad2_ratio:.4g}  dphi={dphi_deg:.0f}deg  '
              f'SNR={stat_split2["SNR"]:.2f}  '
              f'powerA={stat_split2["power_testA"]:.2f}  '
              f'bias_mod1_counts={stat_ignore2["bias_mod1_counts"]:+.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

timestamp  = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'scan_{timestamp}'
output_folder = 'sim_data'
os.makedirs(os.path.join(output_folder, output_dir), exist_ok=True)

metadata = pd.DataFrame([
    ('Rs1',           Rs1.value,        str(Rs1.unit)),
    ('Ad1',           Ad1,              'dimensionless'),
    ('Phase1_rad',    float(phi1.value),'rad'),
    ('Rb',            Rb.value,         str(Rb.unit)),
    ('D',             D.value,          str(D.unit)),
    ('T',             T.value,          str(T.unit)),
    ('dt',            dt.value,         str(dt.unit)),
    ('P',             P.value,          str(P.unit)),
    ('N1_in_bin',     float(N1),        'counts in bin [#]'),
    ('N2_in_bin',     float(N2),        'counts in bin [#]'),
    ('n_toys',        N_TOYS,           ''),
    ('noise_type',    NOISE_TYPE,       ''),
    ('seed',          SEED,             ''),
    ('significance',  SIGNIFICANCE,     ''),
    ('snr_threshold', SNR_THRESHOLD,    ''),
    ('bias_tolerance',BIAS_TOLERANCE,   ''),
], columns=['param', 'value', 'unit'])
metadata.to_csv(os.path.join(output_folder, output_dir, 'metadata.csv'), index=False)

df_SNR.to_csv(                   os.path.join(output_folder, output_dir, 'SNR.csv'))
df_power_testA.to_csv(           os.path.join(output_folder, output_dir, 'power_testA.csv'))
df_power_testB.to_csv(           os.path.join(output_folder, output_dir, 'power_testB.csv'))
df_median_mod2_fit.to_csv(       os.path.join(output_folder, output_dir, 'median_mod2_fit_counts.csv'))
df_mean_Phase2_fit.to_csv(       os.path.join(output_folder, output_dir, 'mean_Phase2_fit_rad.csv'))
df_std_Phase2_fit.to_csv(        os.path.join(output_folder, output_dir, 'std_Phase2_fit_rad.csv'))
df_bias_mod1_counts.to_csv(      os.path.join(output_folder, output_dir, 'bias_mod1_counts.csv'))
df_bias_phase1_deg.to_csv(       os.path.join(output_folder, output_dir, 'bias_phase1_deg.csv'))
df_median_mod2_fit_floor.to_csv( os.path.join(output_folder, output_dir, 'median_mod2_fit_counts_floor.csv'))
df_mean_Phase2_fit_floor.to_csv( os.path.join(output_folder, output_dir, 'mean_Phase2_fit_rad_floor.csv'))
df_std_Phase2_fit_floor.to_csv(  os.path.join(output_folder, output_dir, 'std_Phase2_fit_rad_floor.csv'))

print(f'Saved to {os.path.join(output_folder, output_dir)}/')