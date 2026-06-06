"""
get_realistic_mapping_wimp_Erwindow.py
=======================================
Reads all simulated WIMP masses from git2026_wimp_spectrum/WIMP_N_el_spectra/,
applies smearing and detector efficiency, integrates over a user-defined Er
window, and returns the WIMP masses, integrated rates, and average Ad factor.

Usage
-----
    from get_realistic_mapping_wimp_Erwindow import get_wimp_erwindow_mapping
    import numpy as np
    import astropy.units as u

    m_chis, R_wimps, Ad_avgs = get_wimp_erwindow_mapping(
        target     = 'Xe',
        Er_window  = np.linspace(1, 10, 102) * u.keV,
        detector   = 'Xe1t')
"""

import os
import sys
import glob
import re
import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Repo paths — edit once
# ---------------------------------------------------------------------------

_NU_REPO      = '/home/echo/neutrino_spectrum'
_WIMP_REPO    = '/home/echo/wimp_spectrum'
_DET_EFF_REPO = '/home/echo/detector_efficiency'
_PHASOR_REPO  = '/home/echo/phasor_decomp'

sys.path.insert(0, _NU_REPO)
sys.path.insert(0, _WIMP_REPO)
sys.path.insert(0, _PHASOR_REPO)

from get_sumraw_spectrum import _read_wimp
from get_realistic_spectrum import _smear, _efficiency
from DM_modulation_model import get_SHM_Ads
from scatter_target import get_target_info

_WIMP_FOLDER = os.path.join(_WIMP_REPO, 'WIMP_N_el_spectra')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_simulated_masses(target):
    """Read all simulated WIMP masses for a given target from the folder."""
    pattern = os.path.join(_WIMP_FOLDER, f'WIMP-Nel_*-{target}_*_pdf.txt')
    files   = sorted(glob.glob(pattern))
    masses  = []
    for f in files:
        match = re.search(r'WIMP-Nel_(\d+)(GeV|TeV|MeV)-' + target, os.path.basename(f))
        if match:
            val  = float(match.group(1))
            unit = match.group(2)
            if unit == 'GeV':
                masses.append(val * u.GeV)
            elif unit == 'TeV':
                masses.append(val * u.TeV)
            elif unit == 'MeV':
                masses.append(val * u.MeV)
    # deduplicate
    seen = set()
    unique = []
    for m in masses:
        key = m.to(u.GeV).value
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return [m.to(u.GeV).value for m in unique] * u.GeV


def integrate_in_window(Er, rate, Er_window):
    """Interpolate rate of rate modulation onto Er_window and integrate with trapezoid rule."""
    f = interp1d(Er.to(u.keV).value, rate.value,
                 fill_value=0, bounds_error=False)
    return np.trapz(f(Er_window.to(u.keV).value),
                        Er_window.to(u.keV).value) * rate.unit * u.keV


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_wimp_erwindow_mapping(
        target,
        Er_window,
        detector,
        mode             = 'realistic',
        sigma_percentage = 0.07,
        interaction      = 'NR',
        signal_type      = 'S1S2'):
    """
    Parameters
    ----------
    target           : str               'Xe' or 'Ar'
    Er_window        : Quantity array    energy window e.g. np.linspace(1,10,102)*u.keV
    detector         : str               detector name
    mode             : str               'ideal' or 'realistic'
    metallicity      : str               'high' or 'low'
    Er_range_keV     : tuple             (E_min, E_max) in keV
    Er_bins          : int               number of energy bins
    sigma_percentage : float             Ar smearing fraction
    interaction      : str               interaction label for eff. file
    signal_type      : str               column label in CSV efficiency files

    Returns
    -------
    m_chis  : list of Quantity   WIMP masses [GeV]
    R_wimps : list of Quantity   integrated rate in Er_window [ton⁻¹ yr⁻¹]
    Ad_avgs : list of float      average Ad factor in Er_window
    """
    A, Z  = get_target_info(target)
    m_T   = A * (const.m_n * const.c**2).to(u.GeV)

    m_chis  = _get_simulated_masses(target)
    R_wimps_window = []
    Ad_R_itgl_wimps = []
    m_chis_valid = []
    for m_chi in m_chis:
        # --- get raw WIMP spectrum ------------------------------------------
        Er_raw, rate_wimp_raw = _read_wimp(m_chi, target)

        if rate_wimp_raw is None:
            R_wimps.append(None)
            Ad_avgs.append(None)
            continue

        
        # --- smear + efficiency ---------------------------------------------
        if mode == 'realistic':
            Er_sm, rate_wimp = _smear(Er_raw, rate_wimp_raw, target, sigma_percentage)
            eff              = _efficiency(Er_sm, detector, interaction, signal_type)
            rate_wimp        = rate_wimp * eff
        else:
            Er_sm    = Er_raw
            rate_wimp = rate_wimp_raw

        # --- integrate in window --------------------------------------------
        R_wimp_window = integrate_in_window(Er_sm, rate_wimp, Er_window)
        R_wimps_window.append(R_wimp_window)
        f_rate  = interp1d(Er_sm.to(u.keV).value, rate_wimp.value,
                   fill_value=0, bounds_error=False)
        Rs_window = f_rate(Er_window.to(u.keV).value) * rate_wimp.unit

        # --- Ad factor ------------------------------------------------------
        Ads_window = get_SHM_Ads(Er_window, m_chi, m_T)
        
        if any(np.isnan(Ads_window)):
            print(f'for {m_chi} the selected Er window includes Ad outside SHM model, dropped')
            continue   

        Ad_R_itgl_window    = integrate_in_window(Er_window, Ads_window * Rs_window, Er_window)
        if any(np.sign(Ads_window) < 0) and any(np.sign(Ads_window) > 0):
            print(f'Warning: Ad changes sign for m_chi={m_chi} in this Er window — reaching SHM phase reversal region. Ad_avg may not be meaningful.')

        #Ad_avgs.append(Ad_avg)
        Ad_R_itgl_wimps.append(Ad_R_itgl_window)
        m_chis_valid.append(m_chi)
        
    R_unit  = R_wimps_window[0].unit if R_wimps_window else u.Unit('1/tonne/yr')
    Ad_R_itgl_wimps = np.array([r.value for r in Ad_R_itgl_wimps]) * R_unit
    #Ad_avgs = np.array(Ad_avgs)

    return m_chis_valid, R_wimps_window, Ad_R_itgl_wimps


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    Er_window = np.linspace(1, 10, 102) * u.keV

    m_chis, R_wimps, Ad_avgs = get_wimp_erwindow_mapping(
        target    = 'Xe',
        Er_window = Er_window,
        detector  = 'Xe1t')

    for m, R, Ad in zip(m_chis, R_wimps, Ad_avgs):
        print(f'm={m:.1f}  R={R:.3e}  Ad_avg={Ad:.4f}')