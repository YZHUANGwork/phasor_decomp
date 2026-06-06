"""
run_decomp.py
=============
Execution entry point.
Takes plain count-level numbers (output of convert_params.py setup)
and runs the full decomposition study.

Step 0 — noiseless check  (always runs first)
  Verifies input = reconstructed under ideal conditions.
  If this fails, there is a bug in the calculation, not the noise.

Step 1 — noisy toy ensemble  (if --n_toys > 0)
  Adds Poisson (or Gaussian) noise, runs N trials,
  computes pull distributions and sensitivity.

Step 2 — Δφ scan  (if --scan_dphi)
  Repeats Step 1 across a grid of phase separations.

Use convert_params.py setup to generate the correct call from physical units.

Example
-------
python run_decomp.py \\
    --Ad1 0.033  --phi1_rad 0.051607 \\
    --Ad2 0.0015 --phi2_rad 1.548218 \\
    --N1 12.9227 --N2 0.8077 --Nb 80.7666 \\
    --period_days 365.25 --T_days 3652.5 --dt_days 10 \\
    --n_toys 2000 --noise_type poisson \\
    --outdir ./results
"""

import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decomp_calc as calc
import toy_runner  as toys


def print_splittability(split):
    verdict = 'SPLIT SUCCESSFUL' if split['split_successful'] else 'BURIED IN NOISE'
    vd      = split['verdict_detail']
    tol     = vd['tol']
    p_tgt   = vd['p_target']
    n       = split['n_good']

    def check_str(ok):
        return 'pass' if ok else 'FAIL'

    print()
    print(f'  Splittability  ->  {verdict}')
    print(f'  (thresholds: SNR>={vd["snr_threshold"]}  ')
    print(f'   rate target={p_tgt:.3f}  tolerance={vd["tolerance_nsigma"]:.1f}*sqrt(p*(1-p)/n)={tol:.4f}  n_good={n})')
    print()
    print(f'  SNR            = {split["SNR"]:.3f}  ' +
          f'(>= {vd["snr_threshold"]})  [{check_str(vd["snr_ok"])}]')
    print(f'  Detection rate = {split["detection_rate"]:.3f}  ' +
          f'(>= {p_tgt:.3f} - {tol:.4f} = {p_tgt-tol:.4f})  ' +
          f'[{check_str(vd["detection_ok"])}]')
    print(f'  Consistency    = {split["consistency_rate"]:.3f}  ' +
          f'(>= {p_tgt:.3f} - {tol:.4f} = {p_tgt-tol:.4f})  ' +
          f'[{check_str(vd["consistency_ok"])}]')
    print(f'  Recovery       = {split["fractional_recovery"]:.3f}  (1.0 = perfect)')
    print(f'  sigma_phasor   = {split["sigma_phasor"]:.5f}  counts')
    print(f'  Ad_true        = {split["Ad_true"]:.5f}  counts')


def save_splittability(split, args, outdir):
    f_path = os.path.join(outdir, 'splittability.txt')
    with open(f_path, 'w') as f:
        f.write('# Splittability results\n')
        f.write(f'Ad1={args.Ad1}  phi1_rad={args.phi1_rad}\n')
        f.write(f'Ad2={args.Ad2}  phi2_rad={args.phi2_rad}\n')
        f.write(f'N1={args.N1}  N2={args.N2}  Nb={args.Nb}\n')
        f.write(f'period_days={args.period_days}  T_days={args.T_days}  '
                f'dt_days={args.dt_days}\n')
        f.write(f'n_toys={args.n_toys}  noise_type={args.noise_type}\n')
        f.write(f'Ad_true={split["Ad_true"]:.8f}\n')
        f.write(f'sigma_phasor={split["sigma_phasor"]:.8f}\n')
        f.write(f'SNR={split["SNR"]:.6f}\n')
        f.write(f'detection_rate={split["detection_rate"]:.6f}\n')
        f.write(f'fractional_recovery={split["fractional_recovery"]:.6f}\n')
        f.write(f'reasonable_frac={split["reasonable_frac"]:.6f}\n')
        f.write(f'split_successful={split["split_successful"]}\n')
    print(f'  Splittability written to {f_path}')


def parse_args():
    p = argparse.ArgumentParser(
        description='Signal decomposition hypothesis test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── signal parameters (plain counts, output of convert_params) ──
    p.add_argument('--Ad1',      type=float, required=True,
                   help='Fractional modulation amplitude of KNOWN signal')
    p.add_argument('--phi1_rad', type=float, required=True,
                   help='Phase of KNOWN signal [rad, dimensionless]')
    p.add_argument('--Ad2',      type=float, required=True,
                   help='Fractional modulation amplitude of UNKNOWN signal (truth)')
    p.add_argument('--phi2_rad', type=float, required=True,
                   help='Phase of UNKNOWN signal [rad, dimensionless]')

    p.add_argument('--N1', type=float, required=True,
                   help='number of events in one bin [#] from known signal, unitless
    p.add_argument('--N2', type=float, required=True,
                   help='number of events in one bin [#] from unknown signal, unitless
    p.add_argument('--Nb', type=float, required=True,
                   help='number of background events in one bin [#], unitless

    # ── time parameters ──
    p.add_argument('--period_days', type=float, default=365.25,
                   help='Signal period [days]  (default 365.25)')
    p.add_argument('--T_days',      type=float, required=True,
                   help='Total run time [days]')
    p.add_argument('--dt_days',     type=float, required=True,
                   help='Bin width [days]')

    # ── toy options ──
    p.add_argument('--n_toys',      type=int,   default=2000,
                   help='Number of random trials  (default 2000, 0 = noiseless only)')
    p.add_argument('--noise_type',  type=str,   default='poisson',
                   choices=['poisson', 'gaussian', 'none'],
                   help='Noise model  (default poisson)')
    p.add_argument('--seed',        type=int,   default=42)
                   help='Initial guess scan step [deg]  (default 5)')
    p.add_argument('--significance',  type=float, default=0.1,
                   help='Significance level alpha for detection test  (default 0.1)')
    p.add_argument('--snr_threshold',    type=float, default=2.0,
                   help='SNR threshold for successful split verdict  (default 2.0)')
    p.add_argument('--tolerance_nsigma', type=float, default=2.0,
                   help='Sigma tolerance on rate checks to absorb finite-toy fluctuation  (default 2.0)')
    p.add_argument('--diagnostics', action='store_true',
                   help='Also run pull/chi2/coverage error model diagnostics')
    p.add_argument('--scan_ratio', action='store_true',
                   help='Scan Ad2/Ad1 ratio to find minimum detectable amplitude')
    p.add_argument('--ratio_toys', type=int, default=500,
                   help='Toys per point in ratio scan  (default 500)')

    # ── scan ──
    p.add_argument('--scan_dphi',  action='store_true',
                   help='Run sensitivity scan over Δφ = 0–180°')
    p.add_argument('--scan_step',  type=float, default=15.0,
                   help='Δφ scan step [deg]  (default 15)')
    p.add_argument('--scan_toys',  type=int,   default=500,
                   help='Toys per Δφ point  (default 500)')

    p.add_argument('--outdir', type=str, default='.',
                   help='Output directory  (default current dir)')
    return p.parse_args()


def print_summary_header(args):
    dphi = np.rad2deg((args.phi2_rad - args.phi1_rad) % (2 * np.pi))
    print()
    print('=' * 66)
    print('  Signal Decomposition Study')
    print('=' * 66)
    print(f'  Known   signal : Ad={args.Ad1}  φ={args.phi1_rad:.5f} rad  '
          f'({np.rad2deg(args.phi1_rad):.2f}°)  N_in_bin={args.N1:.4f}')
    print(f'  Unknown signal : Ad={args.Ad2}  φ={args.phi2_rad:.5f} rad  '
          f'({np.rad2deg(args.phi2_rad):.2f}°)  N_in_bin={args.N2:.4f}')
    print(f'  Background     : Nb/bin={args.Nb:.4f}')
    print(f'  Period / dt    : {args.period_days} days / {args.dt_days} days')
    print(f'  Run time       : {args.T_days} days  '
          f'({args.T_days/365.25:.2f} yr)')
    print(f'  Δφ             : {dphi:.2f}°')
    print('=' * 66)


def print_toy_summary(stats, n_toys):
    print()
    print('  Toy ensemble results')
    print(f'  n_good         = {stats["n_good"]} / {n_toys}')
    print()
    print('  Test 1 — Pull distributions  (should be N(0,1))')
    print(f'  bias  (a, b)   = ({stats["bias_para_cos"]:.5f},  {stats["bias_para_sin"]:.5f})  counts')
    print(f'  sigma (a, b)   = ({stats["sigma_para_cos"]:.5f},  {stats["sigma_para_sin"]:.5f})  counts')
    print(f'  Pull a         : mean={stats["pull_para_cos"].mean():.4f}  std={stats["pull_para_cos"].std():.4f}'
          f'  KS p={stats["ks_pull_para_cos_pval"]:.4f}')
    print(f'  Pull b         : mean={stats["pull_para_sin"].mean():.4f}  std={stats["pull_para_sin"].std():.4f}'
          f'  KS p={stats["ks_pull_para_sin_pval"]:.4f}')
    print()
    print('  Test 2 — χ²₂ phasor residual  (p-values should be uniform)')
    print(f'  KS stat        = {stats["ks_stat"]:.4f}  KS p={stats["ks_pval"]:.4f}')
    print()
    print('  Test 3 — Coverage  (observed fraction should equal stated CL)')
    for cl, cov in sorted(stats['coverage'].items()):
        status = 'OK' if abs(cov['n_sigma']) < 2.0 else 'WARNING'
        print(f'  {int(cl*100)}% CL : observed={cov["fraction"]:.4f}  '
              f'expected={cov["expected_CL"]:.4f}  '
              f'deviation={cov["deviation"]:+.4f}  '
              f'({cov["n_sigma"]:+.2f}σ)  [{status}]')
    print()
    print('  Sensitivity')
    print(f'  σ_phasor       = {stats["sigma_phasor"]:.5f}  counts')
    print(f'  A_min  90% CL  = {stats["A_min_90CL"]:.5f}  counts')


def save_results(stats, args, outdir):
    result_file = os.path.join(outdir, 'decomp_results.txt')
    with open(result_file, 'w') as f:
        f.write('# Signal Decomposition Results\n')
        f.write(f'Ad1={args.Ad1}  phi1_rad={args.phi1_rad}\n')
        f.write(f'Ad2={args.Ad2}  phi2_rad={args.phi2_rad}\n')
        f.write(f'N1={args.N1}  N2={args.N2}  Nb={args.Nb}\n')
        f.write(f'period_days={args.period_days}  T_days={args.T_days}  dt_days={args.dt_days}\n')
        f.write(f'n_toys={args.n_toys}  noise_type={args.noise_type}  n_good={stats["n_good"]}\n')
        f.write(f'bias_para_cos={stats["bias_para_cos"]:.8f}  bias_para_sin={stats["bias_para_sin"]:.8f}\n')
        f.write(f'sigma_para_cos={stats["sigma_para_cos"]:.8f}  sigma_para_sin={stats["sigma_para_sin"]:.8f}\n')
        f.write(f'sigma_phasor={stats["sigma_phasor"]:.8f}\n')
        f.write(f'A_min_90CL={stats["A_min_90CL"]:.8f}\n')
        f.write(f'pull_para_cos_mean={stats["pull_para_cos"].mean():.6f}  pull_para_cos_std={stats["pull_para_cos"].std():.6f}\n')
        f.write(f'pull_para_sin_mean={stats["pull_para_sin"].mean():.6f}  pull_para_sin_std={stats["pull_para_sin"].std():.6f}\n')
        f.write(f'chi2_KS={stats["ks_stat"]:.6f}  chi2_KS_p={stats["ks_pval"]:.6f}\n')
        f.write(f'pull_para_cos_ks_p={stats["ks_pull_para_cos_pval"]:.6f}  pull_para_sin_ks_p={stats["ks_pull_para_sin_pval"]:.6f}\n')
        for cl, cov in sorted(stats['coverage'].items()):
            f.write(f'coverage_{{}}'.format(int(cl*100)) + 'pct=' + 
                    f'{cov["fraction"]:.6f}  expected={cov["expected_CL"]:.6f}  '
                    f'deviation={cov["deviation"]:+.6f}  n_sigma={cov["n_sigma"]:+.4f}\n')
    print(f'  Results written to {result_file}')


def save_scan(curve, outdir):
    scan_file = os.path.join(outdir, 'scan_results.txt')
    with open(scan_file, 'w') as f:
        f.write('# dphi_deg  phi2_rad  sigma_phasor  A_min_90CL  '
                'sigma_para_cos  sigma_para_sin  bias_para_cos  bias_para_sin  ks_pval\n')
        for r in curve:
            f.write(f'{r["dphi_deg"]:.2f}  {r["phi2_rad"]:.6f}  '
                    f'{r["sigma_phasor"]:.8f}  {r["A_min_90CL"]:.8f}  '
                    f'{r["sigma_para_cos"]:.8f}  {r["sigma_para_sin"]:.8f}  '
                    f'{r["bias_para_cos"]:.8f}  {r["bias_para_sin"]:.8f}  '
                    f'{r["ks_pval"]:.6f}\n')
    print(f'  Scan table written to {scan_file}')


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print_summary_header(args)

    # ── Step 0: noiseless check — always runs ──
    print()
    print('  Step 0 — Noiseless consistency check')
    nl = toys.run_noiseless_check(
        args.Ad1, args.phi1_rad, args.N1,
        args.Ad2, args.phi2_rad, args.N2,
        args.Nb,
        args.period_days, args.T_days, args.dt_days,
        
        print_check=True,
    )
    calc.plot_noiseless_check(
        nl,
        args.Ad1, args.phi1_rad,
        args.Ad2, args.phi2_rad,
        args.N1,  args.N2,
        save_path=os.path.join(args.outdir, 'noiseless_check.png'),
    )

    # ── Step 1: noisy toy ensemble ──
    if args.n_toys > 0:
        print()
        print(f'  Step 1 — Toy ensemble  '
              f'({args.n_toys} toys, noise={args.noise_type})')
        toy_res = toys.run_toys(
            args.Ad1, args.phi1_rad, args.N1,
            args.Ad2, args.phi2_rad, args.N2,
            args.Nb,
            args.period_days, args.T_days, args.dt_days,
            n_toys=args.n_toys,
            noise_type=args.noise_type,
            seed=args.seed,
            
        )

        # ── primary output: can we actually split the signal? ──
        split = calc.compute_splittability(toy_res,
                                           significance_level=args.significance,
                                           snr_threshold=args.snr_threshold,
                                           tolerance_nsigma=args.tolerance_nsigma)
        print_splittability(split)
        save_splittability(split, args, args.outdir)

        calc.plot_splittability(
            toy_res, split,
            args.Ad1, args.phi1_rad,
            args.Ad2, args.phi2_rad,
            save_path=os.path.join(args.outdir, 'splittability.png'),
        )

        # ── optional: error model diagnostics ──
        if args.diagnostics:
            stats = calc.compute_stats(toy_res)
            print_toy_summary(stats, args.n_toys)
            save_results(stats, args, args.outdir)
            calc.plot_diagnostics(
                toy_res, stats,
                args.Ad1, args.phi1_rad,
                args.Ad2, args.phi2_rad,
                save_path=os.path.join(args.outdir, 'decomp_diagnostics.png'),
            )

    # ── Step 1b: amplitude ratio scan ──
    if args.scan_ratio:
        print()
        print('  Step 1b — Amplitude ratio scan  (Ad2/Ad1)')
        ratio_results, Ad2_grid, ratio_grid = toys.scan_amplitude_ratio(
            args.Ad1, args.phi1_rad, args.N1,
            args.phi2_rad,           args.N2,
            args.Nb,
            args.period_days, args.T_days, args.dt_days,
            n_toys=args.ratio_toys,
            noise_type=args.noise_type,
            seed=args.seed,
            
        )
        summary, fig_ratio = calc.scan_splittability(
            ratio_results,
            scan_param_values=ratio_grid,
            scan_param_name='Ad2 / Ad1  (Kr / Solar)',
            significance_level=args.significance,
            save_path=os.path.join(args.outdir, 'scan_ratio.png'),
        )

    # ── Step 2: Δφ scan ──
    if args.scan_dphi:
        print()
        print('  Step 2 — Δφ sensitivity scan')
        curve = toys.scan_phase_separation(
            args.Ad1, args.phi1_rad, args.N1,
            args.Ad2,               args.N2,
            args.Nb,
            args.period_days, args.T_days, args.dt_days,
            dphi_grid_deg=np.arange(0, 181, args.scan_step),
            n_toys=args.scan_toys,
            noise_type=args.noise_type,
            seed=args.seed,
            
        )
        calc.plot_sensitivity_vs_dphi(
            curve,
            save_path=os.path.join(args.outdir, 'sensitivity_vs_dphi.png'),
        )
        save_scan(curve, args.outdir)

    print()
    print(f'  All outputs in {args.outdir}/')
    print('Done.')


if __name__ == '__main__':
    main()
