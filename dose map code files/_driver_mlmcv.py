import numpy as np
from doseparams import MLMC_LEVEL_OFFSET, dose_shape, nodes_array, Lmin, Lmax, N0, Eps, X_meshgrid, Y_meshgrid, SIGMA, SPATIAL_DIM, dose_shape, dose_method, method, sampling_type, E0, N_conv_test, L_conv_test, file_path, theta
from doseparams import l as side_len
from dose_mlmc import mlmc_parallel as mlmc_parallel_l
from dosesetup import load_mass_matrix
from math import ceil 
from dosemap1_shape_function_geoEM import storage_position_convention
import os
from doseplot import dose_plot_2D, dose_plot_3D
from functools import partial

def mlmcv(mlmc_parallel_l, N0, eps, Lmin, Lmax, alpha0=1.0, beta0=1.0, gamma0=1.0, *args):
    """
    Multi-level Monte Carlo estimation.
    Dynamically vectorized to handle the Primal value and an arbitrary 
    number of sensitivities simultaneously.

    num_q: number of outputs, supplied by mlmc test
    mask: the node mask, also supplied by mlmc test
    """
    # Check input parameters
    if Lmin < 2:
        raise ValueError("error: needs Lmin >= 2")
    if Lmax < Lmin:
        raise ValueError("error: needs Lmax >= Lmin")
    if N0 <= 0 or eps <= 0:
        raise ValueError("error: needs N0 > 0, eps > 0")

    # Initialization -- set to 1.0 for dose EM
    alpha = 1.0 #max(0.0, alpha0) 
    beta  = 1.0 #max(0.0, beta0)
    gamma = max(0.0, gamma0)

    print(f"Alpha, beta, gamma: {alpha, beta, gamma}")
    L = Lmin

    # Arrays directly mapping to levels l = 0, 1, ..., L
    Nl = np.zeros(L + 1, dtype=int)
    costl = np.zeros(L + 1, dtype=float)
    dNl = np.full(L + 1, int(N0), dtype=int)
    suml = np.zeros((4 * dose_shape, L + 1), dtype=float)

    while np.sum(dNl) > 0:
        for l in range(L + 1):
            if dNl[l] > 0:
                #print(f"START mlmc level {l}, step level {l+MLMC_LEVEL_OFFSET}: running {dNl[l]} simulations", flush=True)
                sums, cost = mlmc_parallel_l(l, dNl[l], *args)

                #check the dose shape 
                if sums.shape != (dose_shape, 6):
                    raise ValueError(f"mlmc sums vs. dose shape mismatch: expected {(dose_shape, 6)}, got {sums.shape}.")

                Nl[l] += dNl[l]
                costl[l] += cost
                
                for k in range(dose_shape):
                    suml[4 * k, l]     += sums[k,0]      # diff accumulation
                    suml[4 * k + 1, l] += sums[k,1]      # diff**2 accumulation

                    #Edit-- for the dose application
                    #   The MLMC estimator can be negative which we do not want 
                    #   This is likely to occur at nodes with very low payoff/variance where the hit rate is ~0
                    #   In these cases we should able to replace with the MC estimator with no loss of accuracy

                    #Also store the MC data:
                    suml[4 * k + 2, l] += sums[k, -2]    # mc payoff accumulation
                    suml[4 * k + 3, l] += sums[k, -1]    # mc payoff ** 2 accumulation

        # Reshape suml to compute moments across all quantities simultaneously
        suml_reshaped = suml.reshape(dose_shape, 4, L + 1)
        ml = np.abs(suml_reshaped[:, 0, :] / Nl)
        Vl = np.maximum(0.0, suml_reshaped[:, 1, :] / Nl - ml**2)

        #Store the mc values as well:
        mc_ml = np.abs(suml_reshaped[:, 2, :] / Nl)
        mc_variances = np.maximum(0.0, suml_reshaped[:, 3, :] / Nl - mc_ml**2)
        
        #Using acta numerica we maximise to ensure criteria met for all nodes 
        for l in range(L + 1):
            k = np.argmax(Vl[:, l])
            print(f"level {l}: max variance node={nodes_array[k]}, Vl={Vl[k,l]:.6e}, mean={ml[k,l]:.6e}")

        #maximum arrays, one output per level
        ml_max = np.max(ml, axis=0)
        Vl_max = np.max(Vl, axis=0)
        Cl = costl / Nl

        # Fix to cope with possible zero values for extrapolated means/variances
        for l in range(2, L + 1):
            ml_max[l] = max(ml_max[l], 0.5 * ml_max[l-1] / (2.0**alpha))
            Vl_max[l] = max(Vl_max[l], 0.5 * Vl_max[l-1] / (2.0**beta))

        # Use linear regression on the maximum bounds to estimate parameters
        A = np.column_stack((np.arange(1, L + 1), np.ones(L)))

        if alpha0 <= 0:
            y = np.log2(ml_max[1:])
            x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            alpha = max(0.5, -x[0])

        if beta0 <= 0:
            y = np.log2(Vl_max[1:])
            x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            beta = max(0.5, -x[0])

        if gamma0 <= 0:
            y = np.log2(Cl[1:])
            x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            gamma = max(0.5, x[0])

        # Set optimal number of additional samples using worst-case variance bounds
        Ns = np.ceil(np.sqrt(Vl_max / Cl) * np.sum(np.sqrt(Vl_max * Cl)) / ((1 - theta) * eps**2))

        print("\n--- NEW SAMPLE ALLOCATION ---", flush=True)
        print("Nl     =", Nl, flush=True)
        print("ml_max =", ml_max, flush=True)
        print("Vl_max =", Vl_max, flush=True)
        print("Cl     =", Cl, flush=True)
        print("Ns     =", Ns, flush=True)

        dNl = np.maximum(0, Ns - Nl).astype(int)

        # Weak convergence verification
        if np.sum(dNl > 0.01 * Nl) == 0:
            rng = np.arange(0, min(2, L - 1) + 1) #num points to extrapolate back
            rem = np.max(ml_max[L - rng] / (2.0**(rng * alpha))) / (2.0**alpha - 1.0)

            if rem > np.sqrt(theta) * eps: #testing that weak conv allowance is met
                if L == Lmax:
                    print("*** failed to achieve weak convergence ***")
                else:
                    #adds another level if weak convergence not met
                    L += 1
                    
                    # Expand arrays dynamically for the new level

                    #Extrapolate !
                    Vl_max = np.append(Vl_max, Vl_max[-1] / (2.0**beta))
                    Cl = np.append(Cl, Cl[-1] * (2.0**gamma))
                    
                    Nl = np.append(Nl, 0)
                    suml = np.column_stack((suml, np.zeros(4 * dose_shape)))
                    costl = np.append(costl, 0.0)

                    # Recompute targets
                    Ns = np.ceil(np.sqrt(Vl_max / Cl) * np.sum(np.sqrt(Vl_max * Cl)) / ((1 - theta) * eps**2))
                    dNl = np.maximum(0, Ns - Nl).astype(int)

                    #Added a check! Because it was taking too long 
                    print("\nMLMC allocation:")
                    print("eps =", eps)
                    print("current Nl =", Nl)
                    print("target Ns =", Ns.astype(int))
                    print("additional dNl =", dNl)

    # Evaluate final multilevel estimators for all tracked outputs
    P_estimates = np.sum(suml[::4, :] / Nl, axis=1) #every 4th row

    #Finally add an extra catch for negative payoff using the mc data:

    P_estimates_hybrid = P_estimates.copy()

    P_neg_mask = P_estimates_hybrid < 0.0
    neg_idxs = np.arange(dose_shape)[P_neg_mask] #get the node index
    
    num_negative_nodes=np.count_nonzero(P_neg_mask)
    print(f"MLMC with eps={eps} finished, number of negative payoff nodes={num_negative_nodes}."
          f"\nTesting for MC estimator replacement (no new simulations) for hybrid estimator...")
    mc_replacements = 0
    for idx in neg_idxs:
        mc_variances_idx = mc_variances[idx, L] #finest level only where weak error was tested
        mc_cost_mask = mc_variances_idx / Nl[L] <= eps**2 * (1-theta) #checks if accuracy is good enough at level L where weak error is verified
        if mc_cost_mask: 
            P_estimates_hybrid[idx] = mc_ml[idx, L]
            mc_replacements+=1

    P_neg_mask = P_estimates_hybrid < 0.0   

    # Return estimates as a flattened tuple followed by Nl and Cl for clean unpacking
    return tuple(P_estimates) + tuple(P_estimates_hybrid) + (Nl, Cl, num_negative_nodes, mc_replacements) 


if __name__ == "__main__":

    #You can run only the mlmc directly here

    M_diag = load_mass_matrix()
    partial_mlmc_parallel = partial(mlmc_parallel_l, M_diag=M_diag)

    eps = 5.0
    results = mlmcv(partial_mlmc_parallel, N0, eps, Lmin, Lmax)
    dose_results=results[:-2]
    mlmc_dose=dose_results[:dose_shape]
    hybrid_dose=dose_results[dose_shape:]
    Nl = results[-2]
    Cl = results[-1]

    max_num_lvls = len(Nl) #since Nl includes lvl 0
    step_levels = np.arange(MLMC_LEVEL_OFFSET, max_num_lvls + MLMC_LEVEL_OFFSET)

    #set up title labels
    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function Dose"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel Dose"

    plot_folder = f"notest_mlmc_{dose_method}_Nfull_{np.sum(Nl)}_lvls_{MLMC_LEVEL_OFFSET}_{len(Nl)-1+MLMC_LEVEL_OFFSET}_l_{side_len}_eps_{eps}"
    folder_path = os.path.join(file_path, plot_folder)
    os.makedirs(folder_path, exist_ok=True)
    
    #We save one for the doseplot function separately
    sim_num = np.sum(Nl)
    path_3D = os.path.join(folder_path, f"notest_{sampling_type}_{dose_method}_{method}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{side_len:.3f}_eps_{eps}_minstep_{step_levels[0]}_maxstep_{step_levels[-1]}_sigma_{SIGMA:.3f}.npz") 
    np.savez(path_3D, dose_expected=mlmc_dose, hybrid_dose_expected=hybrid_dose, accuracy=eps, Nl = Nl, Cl = Cl, step_levels=step_levels, sim_num=sim_num, dose_method=dose_method, SPATIAL_DIM=SPATIAL_DIM, l=side_len, X_meshgrid = X_meshgrid, title_seg=title_seg, folder_path=folder_path, sigma=SIGMA)
    print(f"MLMC data saved at {path_3D}")
    print("Now plotting...")
    if SPATIAL_DIM==2:
        plot = dose_plot_2D([path_3D])
    if SPATIAL_DIM==3:
        plot = dose_plot_3D([path_3D])
