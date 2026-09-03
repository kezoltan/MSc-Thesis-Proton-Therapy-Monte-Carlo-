
#June 2026 -- Updated August 2026
#Strong Convergence for EM Scheme Energy Model 

import numpy as np 
from concurrent.futures import ProcessPoolExecutor
import os
from doseparams import Omega0, E0, p, base, master_seed_seq, file_path, SPATIAL_DIM, T, MLMC_LEVEL_OFFSET
from doseparams import NUM_CPUS as num_CPUs
from doseparams import X0_mean as X0
from doseparams import ALPHA as alpha
from doseparams import EPS_0 as eps_0
from dosesetup import coarsen_step, one_step_KZ_g_euler
import matplotlib.pyplot as plt

largest_exponent = 22 #if largest_exp = m then the reference/"exact" timestep size is base**(-m)
sims_per_CPU = 10  
print(f"Base is {base}")

#Generate the correct number of energy steps for the base/exponent chosen 

hs = [-T * base**(-i) for i in range(largest_exponent + 1)]
fine_num_E_steps = base**largest_exponent

#Reverse hs to match what we are plotting i.e. plot this on your x axis not normal hs 

hs_rev = np.asarray(hs[::-1]) #flip order so it corresponds to the above order
hs_plot = np.abs(hs_rev[1::]) #Cut off the finest step since the error (which is 0) was removed above 

def one_sim_strong_conv_EM_KZ(fine_num_E_steps, hs, rng, E0=E0, Omega0=Omega0, X0=X0, largest_exponent=largest_exponent, s0 = 0, a=alpha, p=p):
    """
    """
    fine_dE = hs[-1] #smallest value
    final_lengths = []
    final_omegas = []
    final_pos = []
    
    dZ1 = rng.standard_normal(fine_num_E_steps) 
    dZ2_3D = rng.standard_normal((fine_num_E_steps, SPATIAL_DIM))

    #Generate new Gaussians ahead of time so they are the same for every h:
    dB1 = np.sqrt(abs(fine_dE)) * dZ1 
    dB2_3D = np.sqrt(abs(fine_dE)) * dZ2_3D
    coeff_prefactor = np.sqrt(2*eps_0 * a*p)
        
    for i in range(largest_exponent + 1): #We will reduce the timestep repeatedly 
        if i != 0:
            dB1 = coarsen_step(dB1) #now increase the energy step size
            dB2_3D = coarsen_step(dB2_3D)

        s = float(s0)
        E = float(E0)
        Omega = np.array(Omega0)
        X = np.array(X0)
        h = hs[-1-i] #set the energy step size, starting with the finest
        for k in range(len(dB1)): #Simulate lengths but only store the end point
            dB1_onestep = dB1[k] 
            dB2_3D_onestep = dB2_3D[k]
            E, Omega, s, X = one_step_KZ_g_euler(h, dB1_onestep, dB2_3D_onestep, coeff_prefactor, E, Omega, s, X)
        final_lengths.append(s) #store the final path length for that step size and that sim 
        final_omegas.append(Omega)
        final_pos.append(X)

    final_lengths = np.asarray(final_lengths)
    final_omegas = np.asarray(final_omegas)
    final_pos = np.asarray(final_pos)
    one_sim_abs_path_errors = (np.abs(final_lengths - final_lengths[0])[1::])**2 #Cut off the first error which is trivially 0
    one_sim_abs_angle_errors = (np.linalg.norm(final_omegas - final_omegas[0], axis = 1)[1::])**2
    one_sim_abs_pos_errors = (np.linalg.norm(final_pos - final_pos[0], axis = 1)[1::])**2
    return one_sim_abs_path_errors, one_sim_abs_angle_errors, one_sim_abs_pos_errors #These are arrays of length largest_exponent

def worker(path_seed_chunk, fine_num_E_steps, hs):
    """
    This is the worker function telling each CPU what to do 
    Each CPU will run sim_num simulations, and add up the total abs errors

    Note: Chat is saying we need to set a seed per CPU ? Maybe add in later. 
    """
    abs_path_error_sum = np.zeros(largest_exponent)
    abs_angle_error_sum = np.zeros(largest_exponent)
    abs_pos_error_sum = np.zeros(largest_exponent)

    for path_seed in path_seed_chunk:
        rng=np.random.default_rng(path_seed)
        path_update, angle_update, pos_update = one_sim_strong_conv_EM_KZ(fine_num_E_steps, hs, rng)
        abs_path_error_sum += path_update  
        abs_angle_error_sum += angle_update
        abs_pos_error_sum += pos_update
    return np.array([abs_path_error_sum, abs_angle_error_sum, abs_pos_error_sum])

#Chat is helping with the next part, which is the actual parallelisation:

def parallel_KZ_strong_conv(fine_num_E_steps, hs):
    """
    """
    total_sims = sims_per_CPU * num_CPUs

    #Setting seeds per worker
    path_seeds = master_seed_seq.spawn(total_sims) #Make enough for all paths 
    path_seed_chunks = np.array_split(path_seeds, num_CPUs)

    with ProcessPoolExecutor(max_workers = num_CPUs) as ex:
        worker_outputs = list(ex.map(worker, path_seed_chunks, [fine_num_E_steps]*num_CPUs, [hs]*num_CPUs))
        worker_outputs = np.stack(worker_outputs)
        print(worker_outputs.shape) #Expect shape (num_CPUs, 2 (num equations i think), largest_exponent) - ChatGPT

        totals = np.sum(worker_outputs, axis=0) #should sum down the columns
        abs_path_errors_total, abs_angle_errors_total, abs_pos_errors_total = totals[0] / total_sims, totals[1] / total_sims, totals[2] / total_sims 

    return abs_path_errors_total, abs_angle_errors_total, abs_pos_errors_total 

plot_only=True #only set to true to plot smthg you already have the parameters for

if __name__ == "__main__":

    def plot_strong_conv(hs, results_list, results_labels, title_indicator, total_sims, largest_exponent):
        """
        Creates log-log convergence plot and pairwise slopes of X.
        """
        transient_start, transient_end = 4, -4
        print(f"{hs} became {hs[transient_start:transient_end]} for plot.")
        fit_hs = hs[transient_start:transient_end]
        log_hs = np.log(fit_hs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Strong Convergence Estimates (n={total_sims}, L={largest_exponent})", fontsize=16)

        for i in range(len(results_list)):
            results = results_list[i]**(1/2)
            log_results = np.log(results[transient_start:transient_end])
            slope, intercept = np.polyfit(log_hs, log_results, 1)
            fitted_results = np.exp(intercept + slope*log_hs)
            ax1.scatter(fit_hs, results[transient_start:transient_end], s=20)

            if results_labels[i] == r"$\boldsymbol{X}$":
                ax1.loglog(hs, results, linestyle='--', label=results_labels[i])
                pairwise_slopes = np.diff(log_results)/np.diff(log_hs)
                pairwise_hs = np.sqrt(fit_hs[:-1]*fit_hs[1:])
                ax2.plot(pairwise_hs, pairwise_slopes, marker='o', color='C2')
            else:
                ax1.loglog(fit_hs, fitted_results, label=f'{results_labels[i]}, Fit: {slope:.3f}')

        ax1.set_xlim(hs[transient_start], hs[transient_end-1])
        ax1.set_title(f"{title_indicator}", fontsize=14)
        ax1.set_xlabel("Energy Loss Step Size", fontsize=12)
        ax1.set_ylabel("RMSE", fontsize=14)
        ax1.legend(fontsize=12, loc="upper left")
        ax1.tick_params(axis='both', labelsize=12)
        ax1.grid(True)

        ax2.set_xscale('log')
        ax2.set_xlabel("Energy Loss Step Size", fontsize=12)
        #ax2.set_ylabel(r"Pairwise slope of $\boldsymbol{X}$", fontsize=14)
        ax2.set_title(r"Pairwise Convergence Rate of $\boldsymbol{X}$", fontsize=14)
        ax2.grid(True)
        ax2.tick_params(axis='both', labelsize=12)

        fig.tight_layout()
        save_path = os.path.join(file_path, f"plot_full_KZ_EM_strong_conv_X_pairwise_{num_CPUs * sims_per_CPU}_eps0_{eps_0}_L_{largest_exponent}.png")
        fig.savefig(save_path, dpi=300)

    save_loc_name = os.path.join(file_path, f"full_KZ_EM_strong_conv_{num_CPUs * sims_per_CPU}_eps0_{eps_0}_L_{largest_exponent}.npz")

    if plot_only:
        print("Plotting only...")
        data = np.load(save_loc_name)
        hs_plot = data["hs"]
        path_errors=data["path_errors"]
        exp_angle_errors=data["exp_angle_errors"]
        exp_position_errors=data["exp_pos_errors"]
        largest_exponent=data['largest_exponent']

        results_list = [path_errors, exp_angle_errors, exp_position_errors]
        results_labels = [r"$s$", r"$\boldsymbol{\Omega}$", r"$\boldsymbol{X}$"]
        title = "Geometric EM Scheme"
        total_sims = sims_per_CPU * num_CPUs
        plot_strong_conv(hs_plot, results_list, results_labels, title, total_sims, largest_exponent)

    else:
        print("Running full sims with geometric euler: path angle and position.") #
        path_errors, exp_angle_errors, exp_position_errors = parallel_KZ_strong_conv(fine_num_E_steps, hs)
        np.savez(save_loc_name, total_simulations=np.array([sims_per_CPU * num_CPUs]),
                hs = hs_plot, path_errors=path_errors, exp_angle_errors=exp_angle_errors, exp_pos_errors=exp_position_errors, largest_exponent=largest_exponent)  

        results_list = [path_errors, exp_angle_errors, exp_position_errors]
        results_labels = [r"$s$", r"$\boldsymbol{\Omega}$", r"$\boldsymbol{X}$"]
        title = "Geometric EM Scheme"
        total_sims = sims_per_CPU * num_CPUs

        plot_strong_conv(hs_plot, results_list, results_labels, title, total_sims, largest_exponent)