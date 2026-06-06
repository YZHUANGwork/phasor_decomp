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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2 as chi2_dist, norm as norm_dist, kstest, circmean, circstd
import decomp_calc as calc
import toy_runner as toys

import argparse


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTABILITY  —  signal2 vs signal1: is the recovered signal2 real or just noise?
# ─────────────────────────────────────────────────────────────────────────────

def compute_sig2_recovery(toy_results, significance_level=0.1,
                          snr_threshold=2.0,
                          tolerance_nsigma=2.0):
    """
    Test whether sig2 is correctly extracted from the mixed signal across toys.
    Input: dict from toy_runner.run_toys()
 
    Test A: power_testA = fraction of toys where recovered phasor != 0  (want large)
    Test B: power_testB = fraction of toys where recovered phasor != truth  (want small)
    SNR = mod2_true_counts / sigma_phasor2  (>> 1: reliable, ~ 1: marginal, < 1: buried)
    """
    cov_noise_inv = np.linalg.inv(np.cov(toy_results['delta_para2_cos'], toy_results['delta_para2_sin']))
 
    sigma_para2_cos = np.std(toy_results['delta_para2_cos'], ddof=1)
    sigma_para2_sin = np.std(toy_results['delta_para2_sin'], ddof=1)
    sigma_phasor2   = np.sqrt(sigma_para2_cos**2 + sigma_para2_sin**2)
    
    
    #whether sig2 is large enough to stand above the noise before you even run the hypothesis tests. If SNR < 1, sig2 is buried and you already know power_testA will be low. If SNR >> 1, sig2 is detectable
    #SNR is really just a predictor of power_testA
    SNR             = toy_results['mod2_true_counts'] / sigma_phasor2
 
    # running hypothesis tests on Z2 squared Mahalanobis distances
    H0_rejection_threshold = chi2_dist.ppf(1 - significance_level, df=2)  # H0 is chi2 with dof=2
    para2_cos_sin_fit   = np.stack([toy_results['para2_cos'], toy_results['para2_sin']], axis=1)
    para2_cos_sin_delta = np.stack([toy_results['delta_para2_cos'], toy_results['delta_para2_sin']], axis=1)
 
    # Test A: is recovered phasor != 0?
    # H0: sig2 = 0, Z2_H0_testA ~ chi2(dof=2)
    # H1: sig2 != 0, Z2_H1_testA computed from actual recovered phasors
    # power_testA = fraction of toys where H0 is rejected (want large)
    Z2_H1_testA = np.einsum('ni,ij,nj->n', para2_cos_sin_fit, cov_noise_inv, para2_cos_sin_fit)
    power_testA = np.sum(Z2_H1_testA > H0_rejection_threshold) / len(Z2_H1_testA)
 
    # Test B: is recovered phasor consistent with sig2 truth?
    # H0: recovered phasor = sig2 truth, Z2_H0_testB ~ chi2(dof=2)
    # H1: recovered phasor != sig2 truth, Z2_H1_testB computed from residuals around truth
    # power_testB = fraction of toys where H0 is rejected (want small)
    Z2_H1_testB = np.einsum('ni,ij,nj->n', para2_cos_sin_delta, cov_noise_inv, para2_cos_sin_delta)
    power_testB = np.sum(Z2_H1_testB > H0_rejection_threshold) / len(Z2_H1_testB)
 
    # quality of fitted modulation counts
    median_mod2_fit_counts = np.median(toy_results['mod2_fit_counts'])
    fractional_recovery  = median_mod2_fit_counts / toy_results['mod2_true_counts']
    
    mean_Phase2_fit_rad = circmean(toy_results['Phase2_fit_rad'])
    std_Phase2_fit_rad = circstd(toy_results['Phase2_fit_rad'])
    
    
    # ── verdict ──────────────────────────────────────────────────────
    # allow tolerance_nsigma * sqrt(p*(1-p)/n) slack for finite-toy fluctuation
    p_target  = 1 - significance_level
    tol       = tolerance_nsigma * np.sqrt(p_target * (1 - p_target) / toy_results['n_toys'])
 
    snr_ok          = SNR              >= snr_threshold
    detection_ok    = power_testA >= p_target - tol
    consistency_ok  = (1 - power_testB) >= p_target - tol
 
    recovery_successful = snr_ok and detection_ok and consistency_ok
 
    verdict_detail = dict(
        snr_ok=snr_ok,
        detection_ok=detection_ok,
        consistency_ok=consistency_ok,
        snr_threshold=snr_threshold,
        p_target=p_target,
        tol=tol,
        tolerance_nsigma=tolerance_nsigma,
    )
 
    return dict(
        sigma_para2_cos=sigma_para2_cos, sigma_para2_sin=sigma_para2_sin,
        sigma_phasor2=sigma_phasor2, SNR=SNR,
        Z2_H1_testA=Z2_H1_testA, power_testA=power_testA,
        Z2_H1_testB=Z2_H1_testB, power_testB=power_testB,
        median_mod2_fit_counts=median_mod2_fit_counts, fractional_recovery=fractional_recovery,
        mean_Phase2_fit_rad = mean_Phase2_fit_rad, std_Phase2_fit_rad = std_Phase2_fit_rad, 
        recovery_successful=recovery_successful,
        verdict_detail=verdict_detail,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BIAS STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_bias_stats(toy_results, tolerance=0.10):
    """
    How much does ignoring sig2 bias the recovery of sig1 amplitude and phase.
    tolerance : fraction of mod1_true_counts used as bias budget (default 0.10 = 10%)
    """
    bias_mod1_counts   = np.mean(toy_results['mod1_fit_counts']) - toy_results['mod1_true_counts']
    bias_phase1_deg = np.rad2deg(np.mean(
        (toy_results['Phase1_fit_rad'] - toy_results['Phase1_true_rad'] + np.pi) % (2 * np.pi) - np.pi
    ))
    budget_counts    = tolerance * toy_results['mod1_true_counts']
    is_biased = abs(bias_mod1_counts) > budget_counts
 
    return dict(
        bias_mod1_counts=bias_mod1_counts,
        bias_phase1_deg=bias_phase1_deg,
        budget_counts=budget_counts,
        is_biased=is_biased,
        tolerance=tolerance,
    )
 

def compute_floor_stats(toy_results):
    """
    Noise-floor metrics for single-signal toys (no sig1).
    Input: dict from toy_runner.run_single_signal_toys()
    """
    inflation = np.median(toy_results['mod2_fit_counts']) / toy_results['mod2_true_counts']
    mean_Phase2_fit_rad = circmean(toy_results['Phase2_fit_rad'])
    std_Phase2_fit_rad = circstd(toy_results['Phase2_fit_rad'])
    
    return dict(
        mean_Phase2_fit_rad = mean_Phase2_fit_rad, 
        std_Phase2_fit_rad = std_Phase2_fit_rad, 
        median_mod2_fit_counts = np.median(toy_results['mod2_fit_counts']),
        inflation              = inflation,
        correction_factor      = 1.0 / inflation,
    )
 


def compute_inflation_stats(toy_results):
    """
    Noise-floor inflation of sig2 due to sig1.
    Input: dict from toy_runner.run_noise_floor()
    """
    inflation  = toy_results['mod2_fit_counts'] / toy_results['mod2_true_counts']
    rel_sig1   = toy_results['mod2_fit_counts'] / toy_results['mod1_true_counts']
    phys_ratio = toy_results['mod2_true_counts'] / toy_results['mod1_true_counts']
 
    median_inflation = np.median(inflation)
    mean_Phase2_fit_rad = circmean(toy_results['Phase2_fit_rad'])
    std_Phase2_fit_rad = circstd(toy_results['Phase2_fit_rad'])
    
    return dict(
        inflation          = inflation,
        rel_sig1           = rel_sig1,
        phys_ratio         = phys_ratio,
        median_inflation   = median_inflation,
        std_inflation      = np.std(inflation, ddof=1),
        median_rel_sig1    = np.median(rel_sig1),
        median_mod2_fit_counts = np.median(toy_results['mod2_fit_counts']),
        mean_Phase2_fit_rad = mean_Phase2_fit_rad, std_Phase2_fit_rad = std_Phase2_fit_rad, 
        correction_factor  = 1.0 / median_inflation,
    )
 