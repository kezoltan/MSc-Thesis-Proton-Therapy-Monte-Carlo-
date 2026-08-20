
#August 2026

#Run this file for full simulations
#Parameters + Methods are assigned in doseparams

import numpy as np
from doseparams import *
from doseplot import dose_plot_2D, dose_plot_3D
from dosesetup import choose_dE_ds, load_mass_matrix
from dose_mc import mc_parallel
from dose_mlmc import mlmc_parallel
from _driver_mlmc_testv import mlmc_testv
from _driver_mlmc_plot import mlmc_plot
from functools import partial

if __name__ == "__main__":
    print(f"Gaussian energy spread is {'active' if energy_spread else 'inactive'}: energy deviation is {energy_sdev}, E0 is {E0}.")
    print(f"Gaussian initial position sampling is {'active' if width_spread else 'inactive'}.")

    #For Monte Carlo, we select (it is simply convenient to ensure the stepsize is compatible with mlmc)
    mc_level= 9
    if sampling_type=="mc":
        dE, ds = choose_dE_ds(mc_level)
        if method=='V':
            h = round(abs(ds), 3)
        elif method=='KZ':
            h = round(abs(dE), 3)
        print(f"Simulation details: {sampling_type} dose method {dose_method}, model {method}, {h} h, EO {E0}, l {l}, N {SIMS_PER_CPU*NUM_CPUS}, KAPPA {KAPPA}")

        if dose_method=="SF":
            M_diag = load_mass_matrix()

            mean_load_est = mc_parallel(method, mc_level)
            mean_coeffs_est = mean_load_est / M_diag
            l=round(l,3)
            path_3D = os.path.join(file_path, f"{sampling_type}_{dose_method}_{method}_{h}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{l}_N_{SIMS_PER_CPU*NUM_CPUS}.npz") 
            np.savez(path_3D, coeffs_expected=mean_coeffs_est, sim_num=sim_num, method=method, absolute_h=h, SPATIAL_DIM=SPATIAL_DIM,  l = l, X = X_meshgrid)

        if dose_method=="SK":
            dose_expected = mc_parallel(method, mc_level)
            l=round(l,3)
            path_3D = os.path.join(file_path, f"{sampling_type}_{dose_method}_{method}_{h}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{l}_N_{SIMS_PER_CPU*NUM_CPUS}.npz") 
            np.savez(path_3D, dose_expected=dose_expected, sim_num=sim_num, method=method, absolute_h=h, SPATIAL_DIM=SPATIAL_DIM, l=l, X_meshgrid = X_meshgrid, sigma=SIGMA)

    if sampling_type=='mlmc':
        if mc_level < MLMC_LEVEL_OFFSET:
            print(f"CAUTION: mc_level {mc_level} strictly less than mlmc offset {MLMC_LEVEL_OFFSET}.")

        print(f"Simulation details: {sampling_type} dose method {dose_method}, model {method}, EO {E0}, l {l}, KAPPA {KAPPA} \n" \
              f"mlmc_test details: mlmc L_conv {L_conv_test} (finest step level {L_conv_test + MLMC_LEVEL_OFFSET}), mc sims per level {N_conv_test} \n mlmc details (main): mlmc Lmin {Lmin}, mlmc Lmax {Lmax}, N0 {N0}, base {base}")

        if SPATIAL_DIM not in [2, 3]:
            raise ValueError("spatial dimension must be either 2 or 3.")
        if dose_method not in ['SF', 'SK']:
            raise ValueError("dose method must be either SF or SK.")
        
        #This is hardcoded from the dose_mlmc file
        print("The following stats will be calculated by MLMC test:"
            "\n  E[P_l - P_{l-1}]"
            "\n  E[P_l]"
            "\n  Var[P_l - P_{l-1}]"
            "\n  Var[P_l]"
            "\n  Kurtosis: E[(Y_l - E[Y_l])⁴] / Var(Y_l)²,  Y_l = P_l - P_{l-1}"
            "\n  Consistency: E[P_l - P_{l-1}] + E[P_{l-1}] ≈ E[P_l]")

        txt_save_path = os.path.join(file_path, f"mlmc_{dose_method}_base_{base}_N0_{N0}_Lmin_{Lmin}_Lmax_{Lmax}_offset_{MLMC_LEVEL_OFFSET}_EO_{E0}_l_{l}_KAPPA_{KAPPA}.txt")

        #we just load this every time -- please update
        M_diag = load_mass_matrix()
        partial_mlmc_parallel = partial(mlmc_parallel, M_diag=M_diag)

        with open(txt_save_path, 'w') as fp:
            all_dose_estimates, Eps, Nl, Cl, _ = mlmc_testv(partial_mlmc_parallel, N_conv_test, L_conv_test, N0, Eps, Lmin, Lmax, fp)
        
        plot_folder = f"mlmc_{dose_method}_Nfull_{np.sum(Nl)}_lvls_{MLMC_LEVEL_OFFSET}_{len(Nl)-1+MLMC_LEVEL_OFFSET}_l_{side_len}_eps_{Eps[0]}_{Eps[-1]}"
        folder_path = os.path.join(file_path, plot_folder)
        os.makedirs(folder_path, exist_ok=True)
        sim_num = np.sum(Nl)
        print(f"Total sims done by mlmc across all levels (exlcuding mlmc test): {sim_num}.")

        if dose_method=='SF':
            l=round(l,3)
            path_3D = os.path.join(folder_path, f"{sampling_type}_{dose_method}_{method}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{l}_lvls_{Lmin + MLMC_LEVEL_OFFSET}_{len(Nl) - 1 + MLMC_LEVEL_OFFSET}.npz") 
            np.savez(path_3D, coeffs_expected=all_dose_estimates[-1], accuracy=Eps[-1], all_dose_estimates=all_dose_estimates, all_accuracies=Eps, min_step_lvl = MLMC_LEVEL_OFFSET + Lmin, mlmc_offset = MLMC_LEVEL_OFFSET, final_samples_per_lvl = Nl, final_costs_per_lvl = Cl, sim_num=sim_num, dose_method=dose_method, SPATIAL_DIM=SPATIAL_DIM, l=l, X_meshgrid = X_meshgrid)
        if dose_method=='SK':
            l=round(l,3)
            path_3D = os.path.join(folder_path, f"{sampling_type}_{dose_method}_{method}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{l}_lvls_{Lmin + MLMC_LEVEL_OFFSET}_{len(Nl) - 1 + MLMC_LEVEL_OFFSET}_sigma_{SIGMA:.3f}.npz") 
            np.savez(path_3D, dose_expected=all_dose_estimates[-1], accuracy=Eps[-1], all_dose_estimates=all_dose_estimates, all_accuracies=Eps, min_step_lvl = MLMC_LEVEL_OFFSET + Lmin, mlmc_offset = MLMC_LEVEL_OFFSET, final_samples_per_lvl = Nl, final_costs_per_lvl = Cl, sim_num=sim_num, dose_method=dose_method, SPATIAL_DIM=SPATIAL_DIM, l=l, X_meshgrid = X_meshgrid, sigma=SIGMA)

        #Don't call this!
            #mlmc_plot(txt_save_path, nvert=3, error_bars=True)

    print(f"Raw output array saved at {path_3D}.")
    print("Now plotting...")
    if SPATIAL_DIM==2:
        plot = dose_plot_2D(method, dose_method, path_3D)
    if SPATIAL_DIM==3:
        plot = dose_plot_3D(method, dose_method, path_3D)
