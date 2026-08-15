
#August 2026 
#Multilevel Dose Contributionn User Input (at one level, for driver code)

from dosemap1_shape_function_geoEM import KZ_one_path_load_contribution
from dosemap2_spatial_kernel_geoEM import KZ_one_path_dose_contribution
from dosesetup import *
from numpy.random import default_rng
from concurrent.futures import ProcessPoolExecutor
import numpy as np 
import os
#from dosesetup 

#Don't import dE and ds
from doseparams import method, dose_method, dose_shape, energy_sdev

#---------------------------------MLMC LEVEL WORKER FUNCTIONS-----------------------------------------

def mlmc_coarse_fine_worker(level:int, path_seed_chunk_l, M_diag):
    """
    Runs the dose method on level l of MLMC + calculates coarse/fine diff, + other statistics for testing
    Pl is fine, P_{l-1} is coarse, case of l=0 is distinct: only P0
    Parallelises, given sim num Ml at level l 
    
    Inputs:
        l: level number >= 0 of the finer path 
        path_seed_chunk_l: path seeds for each path run at level l 

    Returns:
        statistics PER OUTPUT (i.e. per dose node), worker contribution, which are (in order):
            raw diff, diff**2, diff**3, diff**4, fine payoff, fine payoff**2
    """

    if energy_spread:
        raise ValueError(f"Gaussian beam energy sampling cannot be active for MLMC. Energy deviation is {energy_sdev}, E0 is {E0}.")
    if method!="KZ":
        raise ValueError("mlmc can only be run with the energy model (KZ).")
    sim_counter = 0

    wc_stats = np.zeros((dose_shape, 6)) #6 statistics, hardcode

    #mlmc level 0 will infact be stepsize level mlmc_offset
    #i.e. we will begin the mlmc from a finer level than actual 0
    stepsize_level = level + MLMC_LEVEL_OFFSET

    for path_seed in path_seed_chunk_l: #number of monte carlo sims 
        rng=default_rng(path_seed)

        #initialise the path at level l, the fine(r) path, this gets fed straight in
        setup_tuple, (dZ1_fine, dZ2_3D_fine) = full_energy_model_setup(stepsize_level, rng)

        if sim_counter == 0:
            s, X, Omega, E, Y, hf, coeff_prefactor, nf = setup_tuple
            hc = hf * base if level > 0 else 0.0
            nc = int(T // (-hc)) if level > 0 else 0
            if not(abs(stepsize_level - round(-log(-hf/T)/log(base))) < 1e-6):
                raise ValueError(f'hf is not geometric, hf: {hf}, stepsize level: {stepsize_level} reading as {-log(-hf/T)/log(base)}, base: {base}, T: {T}, mlmc level {level}')
        else:
            s, X, Omega, E, Y, h, coeff_prefactor, num_steps = setup_tuple
            if h!=hf:
                raise ValueError(f"hf should not change during one mlmc level. initially sampled as {hf} at mlmc level {level}, now reads as {h}.")
            if num_steps!=nf:
                raise ValueError(f"nf should not change during one mlmc level. initially sampled as {nf} at mlmc level {level}, now reads as {num_steps}.")

        #convert to brownian from standard normal paths
        sqrt_hf = np.sqrt(abs(hf))
        dB1_fine = sqrt_hf * dZ1_fine
        dB2_3D_fine = sqrt_hf * dZ2_3D_fine

        #Generate final dose payoff, fine path
        if dose_method == "SF":
            load_f = KZ_one_path_load_contribution(method, (dB1_fine, dB2_3D_fine), setup_tuple)
            dose_f = load_f / M_diag
        elif dose_method == "SK":
            dose_f = KZ_one_path_dose_contribution(method, (dB1_fine, dB2_3D_fine), setup_tuple)
        else:
            raise ValueError(f"invalid dose method: {dose_method}")
        if level == 0: #mlmc level!
            payoff_diff = dose_f 
        elif level > 0:
            dB1_coarse = coarsen_step(dB1_fine)
            dB2_3D_coarse = coarsen_step(dB2_3D_fine)

            #change the input stepsize -- inefficient, update this
            #keep all the same other ics
            setup_tuple = s, X, Omega, E, Y, hc, coeff_prefactor, nc

            if nc != len(dB1_coarse):
                raise ValueError(f"nc should not change during one mlmc level. initially sampled as {nc} at level {level}, now reads as {len(dB1_coarse)}")
            
            if dose_method == "SF":
                load_c = KZ_one_path_load_contribution(method, (dB1_coarse, dB2_3D_coarse), setup_tuple)
                dose_c = load_c / M_diag
                payoff_diff = dose_f - dose_c
            elif dose_method == "SK":
                dose_c = KZ_one_path_dose_contribution(method, (dB1_coarse, dB2_3D_coarse), setup_tuple)
                payoff_diff = dose_f - dose_c
        else:
            raise ValueError(f"mlmc level should not be negative: {level}.")

        if payoff_diff.shape[0] != dose_shape or len(payoff_diff.shape) != 1:
            raise ValueError(f"payoff shape is {payoff_diff.shape}, should be 1D array of shape: {(dose_shape,)}.")

        sim_counter+=1

        #for each sim, sum up (for mc estimator)
        for k in range(6):
            if k <= 3:
                wc_stats[:, k] += payoff_diff**(k+1) 
            else:
                wc_stats[:, k] += dose_f**(k-3)
                           
    return wc_stats, nf

def mlmc_parallel(level, Ml:int, M_diag):
    """
    Retrieves the load vector sums from all CPUs, adds them up, takes expectation.
    Returns the coefficients c_i for the function approximation. 

    Ml, int: number of simulations performed at level l

    Returns sum of full stats from all workers
    """
    total_sims = Ml
    if NUM_CPUS==0:
        raise ValueError("NUM_CPUS cannot be zero, must be a positive int.")
    if Ml == 0:
        raise ValueError(f"number of sims at mlmc level {level} is zero.")
    use_cpus = min(NUM_CPUS, Ml) #if not enough sims, reduce num cpus
    total_out = None 

    #Setting seeds per worker
    path_seeds = master_seed_seq.spawn(total_sims) #Make enough for all paths 
    path_seed_chunks = np.array_split(path_seeds, use_cpus)

    with ProcessPoolExecutor(max_workers=use_cpus) as ex: 
        for wc in ex.map(mlmc_coarse_fine_worker, [level]*use_cpus, path_seed_chunks, [M_diag]*use_cpus):
            if total_out is None: 
                total_out, nf = np.array(wc[0], copy=True), wc[1]
            else:
                total_out += wc[0]
                if wc[1] != nf:
                    raise ValueError(f"workers must use the sample number of fine steps per sim: {nf}, but another worker is reading as {wc[1]}.")
    out_l = total_out
    cost_l = Ml * nf
    return out_l, cost_l
