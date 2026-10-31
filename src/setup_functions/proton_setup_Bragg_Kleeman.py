# July 2026
# Supporting functions for proton transport (initialisation, scheme execution)

import numpy as np
from math import cos, sin
from src.parameterisation._fixed_params import *

def choose_Omega_0():
    if SPATIAL_DIM == 3:
        Omega0 = np.array([1.0, 0.0, 0.0]) #Must be on the unit sphere
    elif SPATIAL_DIM == 2:
        Omega0 = np.array([1.0, 0.0]) #Beam axis is the x axis, must be on the unit sphere
    return Omega0

def choose_kappa():
    """
    None, Light, Moderate, Strong
    """
    if straggling_severity == "None":
        return 0
    elif straggling_severity == "Light":
        return 1e-6
    elif straggling_severity == "Moderate":
        return 4e-5
    elif straggling_severity == "Strong":
        return 1e-3

def exponential_map_sphere(Omega_n, strat_proj_brownian):
    """Update a direction vector along a geodesic on the unit sphere.

    Parameters
    ----------
    Omega_n : numpy.ndarray, shape (d,)
        Current unit direction vector.
    strat_proj_brownian : numpy.ndarray, shape (d,)
        Velocity travelled along geodesic.

    Returns
    -------
    numpy.ndarray, shape (d,)
        Updated direction vector on the sphere.
        If the increment is zero, this is equal to ``Omega_n``.
    """
    norm = np.linalg.norm(strat_proj_brownian)

    if norm == 0:
        return np.asarray(Omega_n).copy()  # copy is used so Omega_n is not modified

    return cos(norm) * Omega_n + sin(norm) * (strat_proj_brownian / norm)

def stopping_power(E):
    """Calculate the Bragg Kleeman stopping power.
    Given by ``1/alpha*p * E^(1-p)``, where alpha, p are medium dependent parameters.

    Parameters
    ----------
    E : float
        Energy, must be nonegative.

    Returns
    -------
    float
        The value of the Bragg Kleeman stopping power.
    """
    return 1/(ALPHA*p) * E**(1-p)

def reciprocal_stopping_power(E):
    """Calculate the reciprocal of the Bragg Kleeman stopping power.
    Given by ``alpha*p*E^(p-1)``, where alpha, p are medium dependent parameters.

    Parameters
    ----------
    E : float
        Energy, must be nonegative.

    Returns
    -------
    float
        The reciprocal of the value of the Bragg Kleeman stopping power.
    """
    return ALPHA * p * E ** (p - 1)

def calculate_R0():
    """
    Calculate the proton range using the Bragg Kleeman model.
    Parameters are imported and set to defaults in src/parameterisation.

    Returns
    -------
    float
        Value of the Bragg Kleeman proton range.
    """
    return ALPHA*E0**p

def calculate_coeff(E_n):
    if sde_model == 'track_length':
        coeff = np.sqrt(2*EPS_0)
    elif sde_model == 'energy':
        coeff = np.sqrt(2*EPS_0 * ALPHA * p) * E_n**(p/2 - 0.5)
    else:
        raise ValueError(f"sde_model {sde_model} does not exist - try energy or track_length.")
    return coeff

def choose_geometric_dE_mu_ds(level: int, mu=1):
    """
    This is chosen such that dE aligns with a mlmc stepsize, purely for convenience.
    And dE and ds align wrt the stopping power.
    And so that the values align relatively closely to those used in Chronholm+Pryer 2026

    mu is in [0,1], chosen freely.
    level is the geometric stepsize level, 9 for MC standard (chosen in main text)
    """
    dE = -T * base ** (-level)
    ds = -dE / (mu * stopping_power(E0))
    return dE, ds
