
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
from functools import partial
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    
    plot_only=False

    if plot_only:

        print("Caution: plot will only compute correctly if all parameter values are unchanged in doseparams.py")
        path_3D = input("Please supply path to plot data: ")
        method = input("Please supply the matching scheme method (KZ/V) ")
        dose_method = input("And the matching dose method (SF/SK): ")

        if dose_method not in path_3D or method not in path_3D:
            raise ValueError("Path details do not match method and/or dose_method. Please check and try again.")
        if SPATIAL_DIM==2:
            plot = dose_plot_2D([path_3D])
        if SPATIAL_DIM==3:
            plot = dose_plot_3D([path_3D])

        #eps 0.6 with SF method mlmc:
            #path_3D=/home/zoltan/Documents/mlmc_SF_KZ_eps_0.6_1.2_offset_5_l_0.05_2D_shape_13090_E0_62_sigma_0.04_kappa_4e-05_eps0_0.005_theta_0.25_width_sdev_3.00_EMIN_0.05/mlmc_SF_eps_0.6_minstep_5_maxstep_15_Nfull_2522128.npz

    else:
        print(f"Gaussian energy spread is {'active' if energy_spread else 'inactive'}: energy deviation is {energy_sdev}, E0 is {E0}.")
        print(f"Gaussian initial position sampling is {'active' if width_spread else 'inactive'}.")

        #For Monte Carlo, we select (it is simply convenient to ensure the stepsize is compatible with mlmc)
        mc_level= 11#9
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
                dose_expected = mean_load_est / M_diag

            if dose_method=="SK":
                dose_expected = mc_parallel(method, mc_level)

            sim_num=SIMS_PER_CPU*NUM_CPUS
            save_folder = f"{sampling_type}_{dose_method}_{method}_h_{h}_l_{l}_N_{sim_num}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_sigma_{SIGMA:.2f}_kappa_{KAPPA}_eps0_{EPS_0}_width_sdev_{width_sdev_factor:.2f}_EMIN_{EMIN}_energy_sdev_{energy_sdev}"
            folder_path = os.path.join(file_path, save_folder)
            os.makedirs(folder_path, exist_ok=True)

            path_3D = os.path.join(folder_path, f"{sampling_type}_{dose_method}_{method}.npz")  
            np.savez(path_3D, dose_expected=dose_expected, sim_num=sim_num, method=method, dose_method=dose_method, absolute_h=h, SPATIAL_DIM=SPATIAL_DIM, l=l, X_meshgrid = X_meshgrid, Y_meshgrid = Y_meshgrid, sigma=SIGMA, folder_path=folder_path, dose_shape=dose_shape, KAPPA=KAPPA, EPS_0=EPS_0, sampling_type=sampling_type, EMIN=EMIN, E0=E0)

            print("Now plotting MC data...")
            if SPATIAL_DIM==2:
                plot = dose_plot_2D([path_3D])
            if SPATIAL_DIM==3:
                plot = dose_plot_3D([path_3D])

        if sampling_type=='mlmc':
            if mc_level < MLMC_LEVEL_OFFSET:
                print(f"CAUTION: mc_level {mc_level} strictly less than mlmc offset {MLMC_LEVEL_OFFSET}.")

            print(f"Simulation details: {sampling_type} dose method {dose_method}, model {method}, EO {E0}, l {l}, KAPPA {KAPPA}, eps_0 {EPS_0}\n" \
                f"mlmc_test details: mlmc L_conv {L_conv_test} (finest step level {L_conv_test + MLMC_LEVEL_OFFSET}), mc sims per level {N_conv_test} \n mlmc details (main): mlmc Lmin {Lmin}, mlmc Lmax {Lmax}, N0 {N0}, base {base}")

            if SPATIAL_DIM not in [2, 3]:
                raise ValueError("spatial dimension must be either 2 or 3.")
            if dose_method not in ['SF', 'SK']:
                raise ValueError("dose method must be either SF or SK.")
            
            #This is hardcoded from the dose_mlmc file
            #print("The following stats will be calculated by MLMC test:"
            #    "\n  E[P_l - P_{l-1}]"
            #    "\n  E[P_l]"
            #    "\n  Var[P_l - P_{l-1}]"
            #    "\n  Var[P_l]"
            #    "\n  Kurtosis: E[(Y_l - E[Y_l])⁴] / Var(Y_l)²,  Y_l = P_l - P_{l-1}"
            #    "\n  Consistency: E[P_l - P_{l-1}] + E[P_{l-1}] ≈ E[P_l]")

            #we just load this every time -- please update
            M_diag = load_mass_matrix()
            partial_mlmc_parallel = partial(mlmc_parallel, M_diag=M_diag)

            print(f"Running full MLMC and MLMC test with Eps = {Eps}")
            folder_path, title_seg, paths_file = mlmc_testv(partial_mlmc_parallel, N_conv_test, L_conv_test, N0, Eps, Lmin, Lmax)

            print(f"Accessing mlmc data from within folder: {folder_path}")

            #Retrieve the data
            all_Nls=[]
            all_Cls=[]
            std_MC_costs=[]
            std_mlmc_costs=[]

            f = open(paths_file, "r")
            results_paths = [line.strip() for line in f if line.strip()]
            f.close()

            for path_3D in results_paths:
                mlmc_data=np.load(path_3D, allow_pickle=True)
                all_Nls.append(mlmc_data['Nl'])
                all_Cls.append(mlmc_data['Cl'])
                std_MC_costs.append(mlmc_data['std_cost'])
                std_mlmc_costs.append(mlmc_data['std_mlmc_cost'])
            std_MC_costs=np.asarray(std_MC_costs)
            std_mlmc_costs=np.asarray(std_mlmc_costs)
            Eps=np.asarray(Eps) 

            #To save as arrays, need to standardise the length:
            max_num_lvls = max(len(Nl) for Nl in all_Nls) #since Nl includes lvl 0
            step_levels = np.arange(MLMC_LEVEL_OFFSET, max_num_lvls + MLMC_LEVEL_OFFSET)
            for i, Nl in enumerate(all_Nls):
                Nl_extend = np.zeros(max_num_lvls)
                Nl_extend[0:len(Nl)] = Nl #and any outlying elts are 0
                all_Nls[i] = Nl_extend #update saved data
            for i, Cl in enumerate(all_Cls):
                Cl_extend = np.zeros(max_num_lvls)
                Cl_extend[0:len(Cl)] = Cl #and any outlying elts are 0
                all_Cls[i] = Cl_extend

            def plot_mlmc():
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"{title_seg} MLMC Results")

                max_num_lvls = max(len(Nl) for Nl in all_Nls) #since Nl includes lvl 0
                step_levels = np.arange(MLMC_LEVEL_OFFSET, max_num_lvls + MLMC_LEVEL_OFFSET)

                #Log scale on y axis 
                for i, Nl in enumerate(all_Nls):
                    Nl_extend = np.zeros(max_num_lvls)
                    Nl_extend[0:len(Nl)] = Nl #and any outlying elts are 0

                    #Avoid zeros + replace with crosses
                    pos_mask = Nl_extend >= 1
                    Nl_line_plot=np.where(pos_mask, Nl, np.nan) #so it doesnt plot anything if it was 0
                    axes[0].semilogy(step_levels, Nl_line_plot, label=f"{Eps[i]:.2f}")
                    #if np.any(~pos_mask):
                    #    y_axis = np.min(Nl[pos_mask]) if np.any(pos_mask) else 1.0
                    #    axes[0].scatter(step_levels[~pos_mask], np.full(np.sum(~pos_mask), y_axis), marker='x')
                    axes[0].set_xlabel(r"Step Level $\ell$")
                    axes[0].set_xticks(step_levels)
                    axes[0].set_ylabel(r"Number of Proton Paths $M_\ell$")
                #axes[0].scatter([], [], marker="x", color="black",label=r"($N_\ell=0$)")
                axes[0].legend(title=r"$\epsilon$")


                axes[1].loglog(Eps, Eps**2 * std_MC_costs, "-*", label="Standard MC")
                axes[1].loglog(Eps, Eps**2 * std_mlmc_costs, ":*", label="Standard MLMC")
                axes[1].set_xlabel(r"Accuracy $\epsilon$")
                axes[1].set_xticks(Eps)
                axes[1].set_xticklabels([f"{eps:g}" for eps in Eps])
                axes[1].set_ylabel(r"$\epsilon^2$ Cost")
                axes[1].legend()

                out_mlmc_plot_path=os.path.join(folder_path, f"mlmc_plot_{dose_method}_{method}_eps_{Eps[0]}_{Eps[-1]}_minstep_{step_levels[0]}_maxstep_{step_levels[-1]}.png") 
                plt.tight_layout()
                plt.savefig(out_mlmc_plot_path, dpi=300)
                plt.show()    
            plot_mlmc()

            #Find the highest accuracy result to plot
            Eps=np.asarray(Eps)
            eps_idx=np.argmin(Eps)
            path_3D=results_paths[eps_idx]

            print(f"Now plotting MLMC with RMSE eps={Eps[eps_idx]}...")
            if SPATIAL_DIM==2:
                plot = dose_plot_2D([path_3D])
            if SPATIAL_DIM==3:
                plot = dose_plot_3D([path_3D])
