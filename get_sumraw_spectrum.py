"""
get_sumraw_spectrum.py
======================
Reads neutrino and WIMP recoil spectra from their respective repositories,
sums them onto a common Er grid, and returns them separately.
No smearing or detector efficiency is applied here.

Repo paths
----------
Edit the three path variables below once to match your machine.

Returns
-------
    Er         : astropy Quantity [keV]         — common energy grid
    rate_nu    : astropy Quantity [ton⁻¹ yr⁻¹ keV⁻¹]  — neutrino sum (or None)
    rate_wimp  : astropy Quantity [ton⁻¹ yr⁻¹ keV⁻¹]  — WIMP spectrum (or None)
    rate_bkgd  : astropy Quantity [ton⁻¹ yr⁻¹ keV⁻¹]  — background sum (or None)

Example
-------
    from get_sumraw_spectrum import get_sumraw_spectrum
    import astropy.units as u

    Er, rate_nu, rate_wimp, rate_bkgd = get_sumraw_spectrum(
        target       = 'Xe',
        channel      = 'NR',
        nu_sources   = ['pp', 'Be7_384', 'Be7_861', 'pep',
                        'N13', 'O15', 'F17', '8B', 'hep',
                        'dsnb', 'atmNu_SURF_avg'],
        wimp_mass    = 6 * u.GeV,
        bkgd_sources = None)
"""

import os
import sys
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Repo paths — edit once
# ---------------------------------------------------------------------------

_NU_REPO      = '/home/echo/neutrino_spectrum'
_WIMP_REPO    = '/home/echo/wimp_spectrum'
_DET_EFF_REPO = '/home/echo/detector_efficiency'

if _NU_REPO not in sys.path:
    sys.path.insert(0, _NU_REPO)

from cross_section_to_rate import nr_rate, er_rate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RATE_UNIT   = 1 / u.tonne / u.yr / u.keV
_BKGD_FOLDER = os.path.join(_NU_REPO, 'measured_spectrum')
_WIMP_FOLDER = os.path.join(_WIMP_REPO, 'WIMP_N_el_spectra')

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read(filepath):
    """Read a two-column spectrum file, skipping all non-numeric header lines."""
    with open(filepath) as f:
        lines = f.readlines()
    skiprows = 0
    for line in lines:
        try:
            float(line.split()[0])
            break
        except (ValueError, IndexError):
            skiprows += 1
    data = np.loadtxt(filepath, skiprows=skiprows)
    return data[:, 0], data[:, 1]


def _read_background(source, target):
    """Read a pre-normalised radioactive background spectrum."""
    fpath = os.path.join(_BKGD_FOLDER, f'{source}-{target}_pdf.txt')
    if not os.path.exists(fpath):
        print(f'background {source} not available for {target}, skipping.')
        return None, None
    Er_arr, rate_arr = _read(fpath)
    Er   = Er_arr   * u.keV
    rate = rate_arr * _RATE_UNIT
    if target == 'Ar' and source == 'Rn222':
        rate = rate / 4000
        Er   = Er   * 1000
    return Er, rate


def _read_wimp(m_wimp, target,
               sigma_i      = 1e-45 * u.cm**2,
               xsec_file    = None,
               xsec_default = 4.4e-45 * u.cm**2):
    """
    Load a pre-computed WIMP-nucleus spectrum, look up the exclusion
    cross-section for m_wimp, and return the rescaled rate.
    """
    if xsec_file is None:
        xsec_file = os.path.join(_WIMP_REPO, 'WIMP_xsec_LZ_excludedtupperlimit.txt')
    masses_val, xsecs_val = _read(xsec_file)
    interp_xsec = interp1d(masses_val, xsecs_val)
    try:
        sigma_f = float(interp_xsec(m_wimp.to(u.GeV).value)) * u.cm**2
    except ValueError:
        sigma_f = xsec_default

    mass_str  = f"{int(m_wimp.to(u.GeV).value)}GeV"
    xsec_str  = f"{sigma_i.value}{sigma_i.unit}".replace(" ", "")
    file_name = f"WIMP-Nel_{mass_str}-{target}_{xsec_str}_pdf.txt"
    fpath     = os.path.join(_WIMP_FOLDER, file_name)

    if not os.path.exists(fpath):
        print(f'WIMP spectrum file not found: {fpath}')
        return None, None

    Er_val, dN_dEr_val = _read(fpath)
    scale = (sigma_f / sigma_i).decompose().value
    return Er_val * u.keV, scale * dN_dEr_val * _RATE_UNIT


def _sum(spectra, Er_common):
    """Interpolate all (Er, rate) spectra onto Er_common and sum."""
    total = np.zeros(len(Er_common))
    for Er, rate in spectra:
        f      = interp1d(Er.to(Er_common.unit).value, rate.value,
                          fill_value=(0, 0), bounds_error=False)
        total += f(Er_common.value)
    return total * _RATE_UNIT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sumraw_spectrum(
        target,
        channel,
        nu_sources   = None,
        wimp_mass    = None,
        bkgd_sources = None,
        metallicity  = 'high',
        Er_range_keV = None,
        Er_bins      = 501,
        NR_folder    = None,
        ER_folder    = None):
    """
    Parameters
    ----------
    target       : str        'Xe' or 'Ar'
    channel      : str        'NR' or 'ER'
    nu_sources   : list[str]  neutrino sources (None / [] to skip)
    wimp_mass    : Quantity   single WIMP mass e.g. 6 * u.GeV (NR only)
    bkgd_sources : list[str]  radioactive backgrounds (ER only)
    metallicity  : str        'high' (GS98) or 'low' (AGSS09)
    Er_range_keV : tuple      (E_min, E_max) in keV; auto-set if None
    Er_bins      : int        number of log-spaced energy bins
    NR_folder    : str        override path to neutrino-Nucleus_el/
    ER_folder    : str        override path to neutrino-electron_el/

    Returns
    -------
    Er         : astropy Quantity [keV]
    rate_nu    : astropy Quantity [ton⁻¹ yr⁻¹ keV⁻¹]  or None
    rate_wimp  : astropy Quantity [ton⁻¹ yr⁻¹ keV⁻¹]  or None
    rate_bkgd  : astropy Quantity [ton⁻¹ yr⁻¹ keV⁻¹]  or None
    """
    nu_sources   = nu_sources   or []
    bkgd_sources = bkgd_sources or []

    if Er_range_keV is None:
        Er_range_keV = (0.001, 1000) if channel == 'NR' else (0.001, 5000)
    if NR_folder is None:
        NR_folder = os.path.join(_NU_REPO, 'neutrino-Nucleus_el')
    if ER_folder is None:
        ER_folder = os.path.join(_NU_REPO, 'neutrino-electron_el')

    Er_common = np.logspace(np.log10(Er_range_keV[0]),
                            np.log10(Er_range_keV[1]), Er_bins) * u.keV

    # --- neutrino sum -------------------------------------------------------
    nu_spectra = []
    if channel == 'NR':
        for source in nu_sources:
            Er, rate, _ = nr_rate(source, target,
                                  folder=NR_folder, metallicity=metallicity)
            nu_spectra.append((Er, rate))
    elif channel == 'ER':
        for source in nu_sources:
            Er, rate, _ = er_rate(source, target,
                                  folder=ER_folder, metallicity=metallicity)
            nu_spectra.append((Er, rate))
    else:
        raise ValueError(f"channel must be 'NR' or 'ER', got '{channel}'")

    rate_nu = _sum(nu_spectra, Er_common) if nu_spectra else None

    # --- WIMP ---------------------------------------------------------------
    if wimp_mass is not None:
        if channel == 'ER':
            raise ValueError("WIMP spectra are NR only.")
        Er_w, rate_w = _read_wimp(wimp_mass, target)
        if Er_w is not None:
            f         = interp1d(Er_w.to(Er_common.unit).value, rate_w.value,
                                 fill_value=(0, 0), bounds_error=False)
            rate_wimp = f(Er_common.value) * _RATE_UNIT
        else:
            rate_wimp = None
    else:
        rate_wimp = None

    # --- backgrounds --------------------------------------------------------
    bkgd_spectra = []
    for source in bkgd_sources:
        Er_b, rate_b = _read_background(source, target)
        if Er_b is not None:
            bkgd_spectra.append((Er_b, rate_b))

    rate_bkgd = _sum(bkgd_spectra, Er_common) if bkgd_spectra else None

    return Er_common, rate_nu, rate_wimp, rate_bkgd
