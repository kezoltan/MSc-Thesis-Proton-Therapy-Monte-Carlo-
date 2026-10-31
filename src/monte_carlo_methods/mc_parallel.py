# July 2026
# Monte Carlo Dose Execution Files (parallelised)

from matplotlib.pylab import ArrayLike
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numpy.random import default_rng
from src.parameterisation import *
from src.dose_computation.dosemap1_shape_function_geoEM import (
    track_len_one_path_load_contribution,
    energy_one_path_load_contribution,
)
from src.dose_computation.dosemap2_spatial_kernel_geoEM import (
    track_len_one_path_dose_contribution,
    energy_one_path_dose_contribution,
)
from src.sde_model_simulation.energy_geometric_EM import full_energy_model_setup

def mc_worker(path_seed_chunk: list, level: int):
    """
    Worker function that runs independent Monte Carlo simulations using each seed in
    `path_seed_chunk` to simulate one proton. Then sums their outputs vectors.

    Parameters
    ----------
    path_seed_chunk : list
    level : int
        Positive integer MLMC level. 

    Returns
    -------
    np.ndarray, shape (dose_shape,)
        Summed total of all worker contributions across all nodes.
        Depending on the dose computation method used, the contribution may be the total
        dose added to the nodes from those simulations, or the total load (prior to mass 
        matrix rescaling).
    """
    wc = np.zeros(dose_shape)  # worker contribution
    if dose_method == "SF":
        for path_seed in path_seed_chunk:
            rng=default_rng(path_seed)
            if sde_model=='energy': #Generate full brownian paths here
                setup_tuple, (dZ1, dZ2_spatial_dim) = full_energy_model_setup(level, rng)
                _,_,_,_,_, h,_ = setup_tuple
                sqrt_h = np.sqrt(abs(h))
                contribution = energy_one_path_load_contribution(
                  sde_model, (sqrt_h * dZ1,sqrt_h * dZ2_spatial_dim), setup_tuple
                )
                wc += contribution

            elif sde_model == "track_length":
                contribution = track_len_one_path_load_contribution(sde_model, rng, level)
                wc += contribution
    if dose_method == "SK":
        for path_seed in path_seed_chunk:
            rng=default_rng(path_seed)
            if sde_model == 'energy':
                setup_tuple, (dZ1, dZ2_spatial_dim) = full_energy_model_setup(level, rng)
                _,_,_,_,_, h,_ = setup_tuple
                sqrt_h = np.sqrt(abs(h))
                contribution = energy_one_path_dose_contribution(
                  sde_model, (sqrt_h * dZ1, sqrt_h * dZ2_spatial_dim), setup_tuple
                )
                wc += contribution

            if sde_model == "track_length":
                contribution = track_len_one_path_dose_contribution(sde_model, rng, level)
                wc += contribution
    return wc


def mc_parallel(level: int):
    """
    Compute the sample mean of all nodal outputs from all CPUs. Nodal outputs are computed
    by the mc_worker function.

    Parameters
    ----------
    level : int
        Positive integer MLMC level.

    Returns
    -------
    np.ndarray, shape (dose_shape,)
        Sample mean of all mc_worker outputs executed in parallel. 
    """
    total_sims = SIMS_PER_CPU * NUM_CPUS
    total_out = None

    # Setting seeds per worker
    path_seeds = master_seed_seq.spawn(total_sims)  # Make enough for all paths
    path_seed_chunks = [path_seeds[i * SIMS_PER_CPU : (i + 1) * SIMS_PER_CPU] for i in range(NUM_CPUS)]

    with ProcessPoolExecutor(max_workers=NUM_CPUS) as ex:
        for wc in ex.map(
            mc_worker, path_seed_chunks, [level] * NUM_CPUS
        ):
            if total_out is None:
                total_out = np.array(wc, copy=True)
            else:
                total_out += wc
    out = total_out / total_sims  # type: ignore
    return out
