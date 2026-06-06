"""
get_realistic_spectrum.py
=========================
Applies smearing and detector efficiency to raw summed spectra from
get_sumraw_spectrum.py, and returns Er, rate_signal1, rate_signal2, rate_bkgd
on a common grid.

Each of signal1, signal2, bkgd is a list of sources — any mix of neutrino
source names (str), WIMP mass (Quantity), or background names (str).
The function sums each bucket separately.

signal1, signal2, bkgd can all be None or []. The returned Er grid spans
the union of all non-None bucket grids.

Usage
-----
    from get_realistic_spectrum import get_realistic_spectrum
    import astropy.units as u

    Er, rate_s1, rate_s2, rate_bkgd = get_realistic_spectrum(
        target   = 'Xe',
        channel  = 'NR',
        signal1  = ['pp', '8B', 'hep'],
        signal2  = [6 * u.GeV],
        bkgd     = ['Kr85'],
        mode     = 'realistic',
        detector = 'Xe1t')
"""

import os
import sys
import glob
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Repo paths — edit once
# ---------------------------------------------------------------------------

_NU_REPO      = '/home/echo/neutrino_spectrum'
_WIMP_REPO    = '/home/echo/wimp_spectrum'
_DET_EFF_REPO = '/home/echo/detector_efficiency'

sys.path.insert(0, _NU_REPO)
sys.path.insert(0, _WIMP_REPO)

from get_sumraw_spectrum import get_sumraw_spectrum

_RATE_UNIT = 1 / u.tonne / u.yr / u.keV

# ---------------------------------------------------------------------------
# Smearing and efficiency
# ---------------------------------------------------------------------------

def _smear(Er, rate, target, sigma_percentage=0.07):
    """
    Xe: σ = 0.31√Er + 0.0035·Er  [keV]  (arXiv:1807.07169)
    Ar: σ = sigma_percentage · Er
    """
    Er_keV = Er.to(u.keV)
    if target == 'Xe':
        sigmas = (0.31 * np.sqrt(Er_keV / u.keV) + 0.0035 * Er_keV / u.keV) * u.keV
    elif target == 'Ar':
        sigmas = sigma_percentage * Er_keV
    else:
        raise ValueError(f"Smearing not defined for target '{target}'")

    dEr     = np.diff(Er_keV)
    smeared = np.zeros(len(dEr)) * rate.unit
    for i, (Er_i, dEr_i, sig_i) in enumerate(zip(Er_keV, dEr, sigmas)):
        kernel     = np.exp(-(Er_keV - Er_i)**2 / (2 * sig_i**2)) / (np.sqrt(2 * np.pi) * sig_i)
        smeared[i] = np.sum(rate * kernel * dEr_i)
    return Er_keV[:-1], smeared


def _efficiency(Er_keV, detector, interaction, signal_type='S1S2'):
    """Return efficiency array (dimensionless) evaluated at Er_keV."""
    if detector == 'LUX03':
        A, B, C, D, E, F = 17.106, 1.8223, 0.65911, 18.292, 20869, -2.35
        x = Er_keV.value
        return 10 ** (2 - A * np.exp(-B * x**C)
                        - D * np.exp(-E * x**F)) / 100

    if 'ideal' in detector and 'Ethrd' in detector:
        Ethrd    = float(detector.split('Ethrd')[-1].replace('keV', '')) * u.keV
        eff_grid = np.logspace(np.log10(Er_keV.value.min()),
                               np.log10(Er_keV.value.max()), 1001) * u.keV
        eff      = np.heaviside(eff_grid - Ethrd, 1.)
        return interp1d(eff_grid, eff, fill_value=(0, 1),
                        bounds_error=False)(Er_keV)

    pattern = os.path.join(os.path.join(_DET_EFF_REPO, 'detector_efficiency'),
                           f'det_eff_{detector}_{interaction}.*')
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No efficiency file matching: {pattern}')
    fpath = sorted(files)[0]
    if fpath.endswith('.csv'):
        import pandas as pd
        df   = pd.read_csv(fpath, header=0, index_col=0)
        E_   = list(df['E_center [keV]'])
        eff_ = list(df[f'eff {signal_type}'])
    else:
        data     = np.loadtxt(fpath, skiprows=1)
        E_, eff_ = data[:, 0], data[:, 1]
    return interp1d(E_, eff_, fill_value=(0, eff_[-1]),
                    bounds_error=False)(Er_keV)


# ---------------------------------------------------------------------------
# Source dispatcher
# ---------------------------------------------------------------------------

def _sum_bucket(sources, target, channel, metallicity, Er_range_keV, Er_bins):
    """
    Sum a bucket of sources onto a common Er grid.
    Returns (Er, rate) or None if sources is empty.
    """
    if not sources:
        return None

    nu_list   = [s for s in sources if isinstance(s, str) and not _is_bkgd(s, target)]
    bkgd_list = [s for s in sources if isinstance(s, str) and     _is_bkgd(s, target)]
    wimp_list = [s for s in sources if isinstance(s, u.Quantity)]
    wimp_mass = wimp_list[0] if wimp_list else None

    Er, rate_nu, rate_wimp, rate_bkgd = get_sumraw_spectrum(
        target       = target,
        channel      = channel,
        nu_sources   = nu_list,
        wimp_mass    = wimp_mass,
        bkgd_sources = bkgd_list,
        metallicity  = metallicity,
        Er_range_keV = Er_range_keV,
        Er_bins      = Er_bins)

    total = None
    for r in [rate_nu, rate_wimp, rate_bkgd]:
        if r is not None:
            total = r if total is None else total + r

    return Er, total


def _is_bkgd(source, target):
    from get_sumraw_spectrum import _BKGD_FOLDER
    fpath = os.path.join(_BKGD_FOLDER, f'{source}-{target}_pdf.txt')
    return os.path.exists(fpath)


def _rebin_onto(Er_src, rate_src, Er_target):
    """Interpolate rate_src onto Er_target, zero-padding outside range."""
    if rate_src is None:
        return None
    f = interp1d(Er_src.to(u.keV).value, rate_src.value,
                 fill_value=(0, 0), bounds_error=False)
    return f(Er_target.to(u.keV).value) * _RATE_UNIT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_realistic_spectrum(
        target,
        channel,
        signal1          = None,
        signal2          = None,
        bkgd             = None,
        mode             = 'realistic',
        detector         = None,
        metallicity      = 'high',
        Er_range_keV     = None,
        Er_bins          = 501,
        sigma_percentage = 0.07,
        interaction      = None,
        signal_type      = 'S1S2'):
    """
    Parameters
    ----------
    target           : str        'Xe' or 'Ar'
    channel          : str        'NR' or 'ER'
    signal1          : list|None  sources for signal1 bucket
    signal2          : list|None  sources for signal2 bucket
    bkgd             : list|None  sources for background bucket
    mode             : str        'ideal' or 'realistic'
    detector         : str        required for mode='realistic'
    metallicity      : str        'high' or 'low'
    sigma_percentage : float      Ar smearing fraction
    interaction      : str        override interaction label for eff. file
    signal_type      : str        column label in CSV efficiency files

    Returns
    -------
    Er           : astropy Quantity [keV]  — universal grid, or None if all empty
    rate_signal1 : astropy Quantity or None
    rate_signal2 : astropy Quantity or None
    rate_bkgd    : astropy Quantity or None
    """
    if mode == 'realistic' and detector is None:
        raise ValueError("detector is required for mode='realistic'")
    if interaction is None:
        interaction = 'NR' if channel == 'NR' else 'beta'

    kwargs = dict(target=target, channel=channel, metallicity=metallicity,
                  Er_range_keV=Er_range_keV, Er_bins=Er_bins)

    res1 = _sum_bucket(signal1 or [], **kwargs)
    res2 = _sum_bucket(signal2 or [], **kwargs)
    res3 = _sum_bucket(bkgd    or [], **kwargs)

    if res1 is None and res2 is None and res3 is None:
        return None, None, None, None

    # universal Er grid: union of all bucket ranges
    all_Er = [r[0] for r in [res1, res2, res3] if r is not None]
    Er_min = min(e.to(u.keV).value.min() for e in all_Er)
    Er_max = max(e.to(u.keV).value.max() for e in all_Er)
    Er_raw = np.logspace(np.log10(Er_min), np.log10(Er_max), Er_bins) * u.keV

    # rebin all buckets onto the universal grid
    rate_s1_raw   = _rebin_onto(res1[0], res1[1], Er_raw) if res1 is not None else None
    rate_s2_raw   = _rebin_onto(res2[0], res2[1], Er_raw) if res2 is not None else None
    rate_bkgd_raw = _rebin_onto(res3[0], res3[1], Er_raw) if res3 is not None else None

    if mode == 'realistic':
        # smear a reference bucket to get the post-smear Er grid and eff
        ref_rate = next(r for r in [rate_s1_raw, rate_s2_raw, rate_bkgd_raw] if r is not None)
        Er, _    = _smear(Er_raw, ref_rate, target, sigma_percentage)
        eff      = _efficiency(Er, detector, interaction, signal_type)

        if rate_s1_raw is not None:
            _, rate_signal1 = _smear(Er_raw, rate_s1_raw, target, sigma_percentage)
            rate_signal1    = rate_signal1 * eff
        else:
            rate_signal1 = None

        if rate_s2_raw is not None:
            _, rate_signal2 = _smear(Er_raw, rate_s2_raw, target, sigma_percentage)
            rate_signal2    = rate_signal2 * eff
        else:
            rate_signal2 = None

        if rate_bkgd_raw is not None:
            _, rate_bkgd = _smear(Er_raw, rate_bkgd_raw, target, sigma_percentage)
            rate_bkgd    = rate_bkgd * eff
        else:
            rate_bkgd = None

        return Er, rate_signal1, rate_signal2, rate_bkgd

    else:
        return Er_raw, rate_s1_raw, rate_s2_raw, rate_bkgd_raw


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    Er, rate_s1, rate_s2, rate_bkgd = get_realistic_spectrum(
        target   = 'Xe',
        channel  = 'NR',
        signal1  = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17', '8B', 'hep'],
        signal2  = [6 * u.GeV],
        bkgd     = [],
        mode     = 'realistic',
        detector = 'Xe1t')

    print('Done.')
