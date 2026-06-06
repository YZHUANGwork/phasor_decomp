"""
convert_params.py
=================
Unit conversion between physical quantities (astropy units)
and the plain count-level numbers expected by decomp_calc.py / toy_runner.py.

All inputs/outputs here carry astropy units.
The outputs of peak_day_to_phase and rate_to_counts_per_bin
are plain floats ready to pass directly into run_toys() or decompose_noiseless().

Subcommands
-----------
phase        peak day [days] → phase [rad]
rate2counts  rate [#/(tonne·yr)] + mass + dt → counts/bin
counts2rate  counts/bin + mass + dt → rate [#/(tonne·yr)]
setup        full conversion → prints ready-to-run run_decomp.py call

Examples
--------
python convert_params.py phase --t_peak 3 --period 365.25

python convert_params.py rate2counts --rate 80 --mass 5.9 --dt 10

python convert_params.py counts2rate --counts_per_bin 13.0 --mass 5.9 --dt 10

python convert_params.py setup \\
    --rate1 80  --Ad1 0.033  --t_peak1 3   \\
    --rate2 5   --Ad2 0.0015 --t_peak2 90  \\
    --rate_bkg 500 --mass 5.9 --dt 10 --period 365.25
"""

import argparse
import numpy as np
import astropy.units as u


# ─────────────────────────────────────────────────────────────────────────────
# conversion functions  (pure, no side effects)
# ─────────────────────────────────────────────────────────────────────────────

def peak_day_to_phase(t_peak, period):
    """
    t_peak : Quantity [time]   time of signal peak from start of year/run
    period : Quantity [time]
    → phase_rad : Quantity [rad]

    Convention: signal = A * cos(2π/P * t - φ)
                peak at t = t_peak  →  φ = 2π * t_peak / P
    """
    return ((t_peak / period).decompose() * 2 * np.pi * u.rad).to(u.rad)


def phase_to_peak_day(phase_rad, period):
    """Inverse: phase [rad] → peak time [days]."""
    return (phase_rad.to(u.rad).value / (2 * np.pi) * period).to(u.day)


def rate_to_counts_per_bin(Rs, D, dt):
    """
    Rs : Quantity [1/(tonne·yr)]   event rate
    D  : Quantity [tonne]          detector mass
    dt : Quantity [time]           bin width
    → dimensionless float  (mean counts per bin from this component)
    """
    return float((Rs * D * dt).decompose().value)


def counts_per_bin_to_rate(N_per_bin, D, dt):
    """
    N_per_bin : float   mean counts per bin
    D  : Quantity [tonne]
    dt : Quantity [time]
    → rate : Quantity [1/(tonne·yr)]
    """
    return (N_per_bin / (D * dt)).to(1 / (u.tonne * u.yr))


def T_to_days(T):
    """Any time Quantity → float in days."""
    return T.to(u.day).value


def dt_to_days(dt):
    """Any time Quantity → float in days."""
    return dt.to(u.day).value


# ─────────────────────────────────────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_phase(args):
    t_peak = args.t_peak * u.day
    period = args.period * u.day
    phi    = peak_day_to_phase(t_peak, period)
    t_back = phase_to_peak_day(phi, period)

    print()
    print('  Phase conversion')
    print(f'  t_peak  = {t_peak}  (period = {period})')
    print(f'  phi     = {phi:.6f}')
    print(f'  phi     = {phi.to(u.deg):.4f}')
    print(f'  check   : phi → t_peak = {t_back:.4f}')
    print()


def cmd_rate2counts(args):
    Rs = args.rate / (u.tonne * u.yr)
    D  = args.mass * u.tonne
    dt = args.dt   * u.day
    N  = rate_to_counts_per_bin(Rs, D, dt)

    print()
    print('  Rate → counts/bin')
    print(f'  rate    = {Rs}')
    print(f'  mass    = {D}')
    print(f'  dt      = {dt}  =  {dt.to(u.yr):.6f}')
    print(f'  N/bin   = {N:.6f}')
    print()


def cmd_counts2rate(args):
    D  = args.mass * u.tonne
    dt = args.dt   * u.day
    Rs = counts_per_bin_to_rate(args.counts_per_bin, D, dt)

    print()
    print('  Counts/bin → rate')
    print(f'  N/bin   = {args.counts_per_bin}')
    print(f'  mass    = {D}')
    print(f'  dt      = {dt}')
    print(f'  rate    = {Rs:.6f}')
    print()


def cmd_setup(args):
    D      = args.mass    * u.tonne
    dt     = args.dt      * u.day
    period = args.period  * u.day
    T      = args.T       * u.yr

    Rs1  = args.rate1    / (u.tonne * u.yr)
    Rs2  = args.rate2    / (u.tonne * u.yr)
    Rb   = args.rate_bkg / (u.tonne * u.yr)

    phi1 = peak_day_to_phase(args.t_peak1 * u.day, period)
    phi2 = peak_day_to_phase(args.t_peak2 * u.day, period)

    N1   = rate_to_counts_per_bin(Rs1, D, dt)
    N2   = rate_to_counts_per_bin(Rs2, D, dt)
    Nb   = rate_to_counts_per_bin(Rb,  D, dt)

    T_days  = T_to_days(T)
    dt_days = dt_to_days(dt)
    P_days  = dt_to_days(period)
    dphi    = (phi2.to(u.deg) - phi1.to(u.deg)).value % 360

    print()
    print('=' * 64)
    print('  Experiment Setup Conversion')
    print('=' * 64)
    print(f'  detector mass  = {D}')
    print(f'  bin width dt   = {dt}  =  {dt.to(u.yr):.6f}')
    print(f'  period         = {period}')
    print(f'  run time T     = {T}  =  {T_days:.2f} days')
    print()
    print('  Known signal (signal 1)')
    print(f'    rate         = {Rs1}')
    print(f'    Ad1          = {args.Ad1}')
    print(f'    t_peak1      = {args.t_peak1} day')
    print(f'    phi1         = {phi1:.6f}  ({phi1.to(u.deg):.3f})')
    print(f'    N1/bin       = {N1:.6f}  counts/bin')
    print(f'    Ad1*N1       = {args.Ad1 * N1:.6f}  counts  (phasor amplitude)')
    print()
    print('  Unknown signal (signal 2)')
    print(f'    rate         = {Rs2}')
    print(f'    Ad2          = {args.Ad2}')
    print(f'    t_peak2      = {args.t_peak2} day')
    print(f'    phi2         = {phi2:.6f}  ({phi2.to(u.deg):.3f})')
    print(f'    N2/bin       = {N2:.6f}  counts/bin')
    print(f'    Ad2*N2       = {args.Ad2 * N2:.6f}  counts  (phasor amplitude)')
    print()
    print('  Background')
    print(f'    rate         = {Rb}')
    print(f'    Nb/bin       = {Nb:.6f}  counts/bin')
    print()
    print(f'  Δφ (φ2 − φ1)  = {dphi:.3f}°')
    print()
    print('  ── run_decomp.py call ──')
    print( '  python run_decomp.py \\')
    print(f'    --Ad1 {args.Ad1}  --phi1_rad {phi1.value:.6f} \\')
    print(f'    --Ad2 {args.Ad2}  --phi2_rad {phi2.value:.6f} \\')
    print(f'    --N1 {N1:.6f}  --N2 {N2:.6f}  --Nb {Nb:.6f} \\')
    print(f'    --period_days {P_days}  --T_days {T_days:.2f}  --dt_days {dt_days} \\')
    print( '    --n_toys 2000  --noise_type poisson')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Unit conversion helper for signal decomposition study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('phase', help='Peak day [days] → phase [rad]')
    sp.add_argument('--t_peak',  type=float, required=True, help='Peak day [days]')
    sp.add_argument('--period',  type=float, default=365.25, help='Period [days]')

    sp = sub.add_parser('rate2counts', help='Rate + mass + dt → counts/bin')
    sp.add_argument('--rate', type=float, required=True, help='Rate [#/(tonne·yr)]')
    sp.add_argument('--mass', type=float, required=True, help='Mass [tonne]')
    sp.add_argument('--dt',   type=float, required=True, help='Bin width [days]')

    sp = sub.add_parser('counts2rate', help='Counts/bin + mass + dt → rate')
    sp.add_argument('--counts_per_bin', type=float, required=True)
    sp.add_argument('--mass', type=float, required=True, help='Mass [tonne]')
    sp.add_argument('--dt',   type=float, required=True, help='Bin width [days]')

    sp = sub.add_parser('setup', help='Full setup → run_decomp.py call')
    sp.add_argument('--rate1',    type=float, required=True, help='Known rate [#/(tonne·yr)]')
    sp.add_argument('--Ad1',      type=float, required=True, help='Known fractional amplitude')
    sp.add_argument('--t_peak1',  type=float, required=True, help='Known peak day [days]')
    sp.add_argument('--rate2',    type=float, required=True, help='Unknown rate [#/(tonne·yr)]')
    sp.add_argument('--Ad2',      type=float, required=True, help='Unknown fractional amplitude')
    sp.add_argument('--t_peak2',  type=float, required=True, help='Unknown peak day [days]')
    sp.add_argument('--rate_bkg', type=float, required=True, help='Background rate [#/(tonne·yr)]')
    sp.add_argument('--mass',     type=float, required=True, help='Detector mass [tonne]')
    sp.add_argument('--dt',       type=float, required=True, help='Bin width [days]')
    sp.add_argument('--period',   type=float, default=365.25, help='Period [days]')
    sp.add_argument('--T',        type=float, default=10.0, help='Run time [yr]')

    args = p.parse_args()
    {'phase': cmd_phase,
     'rate2counts': cmd_rate2counts,
     'counts2rate': cmd_counts2rate,
     'setup': cmd_setup}[args.cmd](args)


if __name__ == '__main__':
    main()
