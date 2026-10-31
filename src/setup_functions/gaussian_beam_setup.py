# July 2026
# Initialisation and parameterisation of Gaussian beam

import numpy as np
from src.parameterisation import *
from math import log

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
    Sample the initial beam position from a transverse Gaussian distribution.

    The longitudinal coordinate is fixed at the position returned by
    ``choose_X0``. Each transverse coordinate is sampled independently with
    standard deviation ``width_sdev_factor * side_len``, set inside src/parameterisation.

    Parameters
    ----------
    meshgrid : array_like
        Must be in the form of the output of np.meshgrid, as determined by
        nodal_mesh_setup/nodes()
    rng : numpy.random.Generator
        Random number generator used to sample the transverse coordinates.
    side_len : float
        The distance between two adjacent nodes along the same fundamental direction.

    Returns
    -------
    X0 : numpy.ndarray of shape (d,)
        Sampled initial position for one proton, containing coordinates ``[x0, y]``
        in 2D or ``[x0, y, z]`` in 3D.
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

def sample_gauss_beam(rng, s0=0):
    s = float(s0)
    X = sample_initial_position(meshgrid, rng, side_len)
    Omega = Omega0
    E = initial_energy_spread()
    Y = log(E)
    return s, X, Omega, E, Y
