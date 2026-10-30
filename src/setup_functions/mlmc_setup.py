# July 2026
# Supporting functions for MLMC execution

import numpy as np
from math import log
from src.parameterisation import *

def coarsen_step(brownian_incs):
    """
    For geometric MLMC where step sizes are of the form h = T * base ** (-L), for integer L.
    Geometric MLMC is the default type used in src/monte_carlo_methods/mlmc_driver.
    Coarsen Brownian path by summing groups of ``base`` steps to construct coarse path from fine path.

    Parameters
    ----------
    brownian_incs : array_like

    Returns
    -------
    numpy.ndarray
        Coarse Brownian path, coupled to fine path for MLMC variance reduction.
    """
    brownian_incs = np.asarray(brownian_incs)
    b_shape = brownian_incs.shape
    level_shape = log(b_shape[0]) / log(base)
    if not (abs(level_shape - round(level_shape)) < 1e-6):
        raise ValueError(
            f"input (fine) brownian increments do not have geometric step size: "
            f"num_steps: {b_shape[0]}, level reading as {level_shape}, base: {base}."
        )
    if len(b_shape) == 1:  # 1D array
        return brownian_incs.reshape(-1, base).sum(axis=1)
    elif len(b_shape) == 2:
        dim = b_shape[1]
        incs_list = []
        for idx in range(dim):  # 2D, 3D
            coarsened_inc = brownian_incs[:, idx].reshape(-1, base).sum(axis=1)
            incs_list.append(coarsened_inc)
        incs_tuple = tuple(incs_list)
        coarsed_brownian = np.column_stack(incs_tuple)
        level_shape = log(coarsed_brownian.shape[0]) / log(base)
        if not (abs(level_shape - round(level_shape)) < 1e-6):
            raise ValueError(
                f"coarsened brownian increments do not have geometric step size, but fine brownian incremements did. "
                f"num_steps: {coarsed_brownian.shape[0]}, level reading as {level_shape}, base: {base}."
            )
        return np.column_stack(incs_tuple)
    else:
        raise ValueError(
            f"brownian_incs shape is incompatible, should be (num_steps,) or (num_steps, dimension), is: {b_shape}"
        )
