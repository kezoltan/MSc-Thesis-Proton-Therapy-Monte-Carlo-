# July 2026
# Monte Carlo Dose Execution Files (parallelised)

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numpy.random import default_rng
from doseparams import *
from dose_computation.dosemap1_shape_function_geoEM import (
    track_len_one_path_load_contribution,
    energy_one_path_load_contribution,
)
from dose_computation.dosemap2_spatial_kernel_geoEM import (
    track_len_one_path_dose_contribution,
    energy_one_path_dose_contribution,
)
from src.sde_model_simulation.energy_geometric_EM import full_energy_model_setup


def mc_worker(sde_model, dose_method, path_seed_chunk, level):
    """
    This is the worker function telling each CPU what to do
    Each CPU will run SIMS_PER_CPU independent simulations, and add up the load vectors.
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


def mc_parallel(sde_model, level, SIMS_PER_CPU=SIMS_PER_CPU, NUM_CPUS=NUM_CPUS):
    """
    Retrieves the load vector sums from all CPUs, adds them up, takes expectation.
    Returns the coefficients c_i for the function approximation.
    """
    total_sims = SIMS_PER_CPU * NUM_CPUS
    total_out = None

    # Setting seeds per worker
    path_seeds = master_seed_seq.spawn(total_sims)  # Make enough for all paths
    path_seed_chunks = [path_seeds[i * SIMS_PER_CPU : (i + 1) * SIMS_PER_CPU] for i in range(NUM_CPUS)]

    with ProcessPoolExecutor(max_workers=NUM_CPUS) as ex:
        for wc in ex.map(
            mc_worker, [sde_model] * NUM_CPUS, [dose_method] * NUM_CPUS, path_seed_chunks, [level] * NUM_CPUS
        ):
            if total_out is None:
                total_out = np.array(wc, copy=True)
            else:
                total_out += wc
    out = total_out / total_sims  # type: ignore
    return out
