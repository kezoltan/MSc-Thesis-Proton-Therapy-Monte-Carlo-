# July 2026
# Supporting functions for MLMC execution

import numpy as np
from math import log
from pbt_mlmc.parameterisation import *

def coarsen_step(brownian_incs):
    """
    Take an input list/array of Brownian increments dB over some energy step, say h, can be any dimension of Brownians
    Generate path one level more coarse than the input (each set of base fine steps -> sum -> one coarse step)

    Must be sqrt(h)*dZ = dB! Brownian increments
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
