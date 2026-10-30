# July 2026
# Initialisation and parameterisation of Gaussian beam

import numpy as np
from pbt_mlmc.parameterisation import *

def initial_energy_spread():
    """Draw a random initial proton energy from a Gaussian (normal) distribution.

    The distribution is centered on ``E0`` and uses ``energy_sdev`` from
    ``doseparams`` to set its spread.

    Returns
    -------
    float
        A sampled initial proton energy.
    """
    return np.random.normal(E0, energy_sdev**2)

def sample_initial_position(meshgrid, rng, side_len):
    """
    Gaussian sample the initial beam position given the X0 mean.
    The spread (i.e. the standard dev) should be order side_len
    """
    X0_mean, _, _ = choose_X0(meshgrid)

    if SPATIAL_DIM == 2:
        # we only need to sample the back "wall" where the beam originates
        x0, y0 = X0_mean
        width_sdev = width_sdev_factor * side_len
        y = rng.normal(loc=y0, scale=width_sdev)
        return np.array([x0, y], dtype=float)

    elif SPATIAL_DIM == 3:
        x0, y0, z0 = X0_mean
        width_sdev = width_sdev_factor * side_len
        y = rng.normal(loc=y0, scale=width_sdev)
        z = rng.normal(loc=z0, scale=width_sdev)
        return np.array([x0, y, z], dtype=float)
