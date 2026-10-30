# July 2026
# Supporting functions for proton transport (initialisation, scheme execution)

import numpy as np
from math import cos, sin
from pbt_mlmc.simulation_config import (
    SPATIAL_DIM, straggling_severity, ALPHA, p, EPS_0, sde_model,
)

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

def stopping_power(E, a=ALPHA, p=p):
    """Calculate the Bragg Kleeman stopping power.
    Given by ``1/ap * E^(1-p)``.

    Parameters
    ----------
    E : float
        Energy, must be nonegative.
    a : float, optional
        Medium dependent coefficient.
        Default medium is water, default value is ``ALPHA`` from ``pbt_mlmc.simulation_config``.
    p : float, optional
        Medium dependent coefficient.
        Default medium is water, default value is ``p`` from ``pbt_mlmc.simulation_config``.

    Returns
    -------
    float
        The value of the Bragg Kleeman stopping power.
    """
    return 1/(a*p) * E**(1-p)

def reciprocal_stopping_power(E, a=ALPHA, p=p):
    """Calculate the reciprocal of the Bragg Kleeman stopping power.
    Given by ``apE^(p-1)``.

    Parameters
    ----------
    E : float
        Energy, must be nonegative.
    a : float, optional
        Medium dependent coefficient.
        Default medium is water, default value is ``ALPHA`` from ``pbt_mlmc.simulation_config``.
    p : float, optional
        Medium dependent coefficient.
        Default medium is water, default value is ``p`` from ``pbt_mlmc.simulation_config``.

    Returns
    -------
    float
        The reciprocal of the value of the Bragg Kleeman stopping power.
    """
    return a * p * E ** (p - 1)

def calculate_coeff(E_n, EPS_0 = EPS_0, sde_model = sde_model, a=ALPHA, p=p):
    if sde_model == 'track_length':
        coeff = np.sqrt(2*EPS_0)
    elif sde_model == 'energy':
        coeff = np.sqrt(2*EPS_0 * a * p) * E_n**(p/2 - 0.5)
    else:
        raise ValueError(f"sde_model {sde_model} does not exist - try energy or track_length.")
    return coeff
