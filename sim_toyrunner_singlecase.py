"""
sim_toyrunner_singlecase.py
===========================
Single-case toy runner study.

"""

import sys
sys.path.insert(0, '/WIMP_decomp')

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os

import decomp_calc as calc
import toy_runner as toys
import convert_params as conv
import compute_toy_runner_stat as calctoys


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

Rs1 = 30.0 / (u.tonne * u.yr)   # signal 1 rate
Rs2 = 30.0 / (u.tonne * u.yr)   # signal 2 rate
Rb  = 0.0  / (u.tonne * u.yr)   # background rate
D   = 10.0 * u.tonne
dt  = 30.0 * u.day
T   = 10.0 * u.yr
P   = 365.25 * u.day

phi1 = conv.peak_day_to_phase(3.0   * u.day, P)
phi2 = conv.peak_day_to_phase(150.0 * u.day, P)

N1 = conv.rate_to_counts_per_bin(Rs1, D, dt)
N2 = conv.rate_to_counts_per_bin(Rs2, D, dt)
Nb = conv.rate_to_counts_per_bin(Rb,  D, dt)

T_days  = conv.T_to_days(T)
dt_days = conv.dt_to_days(dt)
P_days  = P.to(u.day).value

Ad1 = 0.03342
Ad2 = 0.01

N_TOYS           = 1000
NOISE_TYPE       = 'poisson'
SEED             = 42
SIGNIFICANCE     = 0.1
SNR_THRESHOLD    = 2.0
TOLERANCE_NSIGMA = 2.0
BIAS_TOLERANCE   = 0.1

print(f'phi1={phi1:.4f}  phi2={phi2:.4f}')
print(f'N1={N1:.4f}  N2={N2:.4f}  Nb={Nb:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# NOISELESS CHECK
# ─────────────────────────────────────────────────────────────────────────────

nl  = toys.run_noiseless_check(
    Ad1, phi1.value, N1,
    Ad2, phi2.value, N2,
    Nb,
    P_days, T_days, dt_days,
    print_check=True,
)
fig_nl = calc.plot_noiseless_check(nl, Ad1, phi1.value, Ad2, phi2.value, N1, N2)


# ─────────────────────────────────────────────────────────────────────────────
# RUN TOYS
# ─────────────────────────────────────────────────────────────────────────────

res_split2 = toys.run_toys(
    Ad1, phi1.value, N1,
    Ad2, phi2.value, N2,
    Nb, P_days, T_days, dt_days,
    n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
)

res_ignore2 = toys.run_bias_toys(
    Ad1, phi1.value, N1,
    Ad2, phi2.value, N2,
    Nb, P_days, T_days, dt_days,
    n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
)

res_1pumped2 = toys.run_noise_floor(
    Ad1, phi1.value, N1,
    Ad2, phi2.value, N2,
    Nb, P_days, T_days, dt_days,
    n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
)

res_pumped2 = toys.run_single_signal_toys(
    Ad2, phi2.value, N2,
    Nb, P_days, T_days, dt_days,
    n_toys=N_TOYS, noise_type=NOISE_TYPE, seed=SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE STATS
# ─────────────────────────────────────────────────────────────────────────────

split = calctoys.compute_sig2_recovery(
    res_split2,
    significance_level=SIGNIFICANCE,
    snr_threshold=SNR_THRESHOLD,
    tolerance_nsigma=TOLERANCE_NSIGMA,
)

bst = calctoys.compute_bias_stats(res_ignore2, tolerance=BIAS_TOLERANCE)
ist = calctoys.compute_inflation_stats(res_1pumped2)
st  = calctoys.compute_floor_stats(res_pumped2)

