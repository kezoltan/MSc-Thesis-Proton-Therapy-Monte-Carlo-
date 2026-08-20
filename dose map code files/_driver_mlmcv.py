import numpy as np
from doseparams import MLMC_LEVEL_OFFSET, dose_shape, nodes_array, Lmin, Lmax, N0, Eps, X_meshgrid, Y_meshgrid, SIGMA, SPATIAL_DIM, dose_shape, dose_method, method, sampling_type, E0, N_conv_test, L_conv_test, file_path
from doseparams import l as side_len
from dose_mlmc import mlmc_parallel as mlmc_parallel_l
from dosesetup import load_mass_matrix
from math import ceil 
from dosemap1_shape_function_geoEM import storage_position_convention
import os
from doseplot import dose_plot_2D, dose_plot_3D

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

    # Initialization
    alpha = 1.0 #max(0.0, alpha0) 
    beta  = 1.0 #max(0.0, beta0)
    gamma = max(0.0, gamma0)

    print(f"Alpha, beta, gamma: {alpha, beta, gamma}")

    theta = 0.25
    L = Lmin

    # Arrays directly mapping to levels l = 0, 1, ..., L
    Nl = np.zeros(L + 1, dtype=int)
    costl = np.zeros(L + 1, dtype=float)
    dNl = np.full(L + 1, int(N0), dtype=int)
    suml = np.zeros((2 * dose_shape, L + 1), dtype=float)

    while np.sum(dNl) > 0:
        for l in range(L + 1):
            if dNl[l] > 0:
                print(f"START mlmc level {l}, step level {l+MLMC_LEVEL_OFFSET}: running {dNl[l]} simulations", flush=True)
                sums, cost = mlmc_parallel_l(l, dNl[l], *args)

                #check the dose shape 
                if sums.shape != (dose_shape, 6):
                    raise ValueError(f"mlmc sums vs. dose shape mismatch: expected {(dose_shape, 6)}, got {sums.shape}.")

                Nl[l] += dNl[l]
                costl[l] += cost
                
                for k in range(dose_shape):
                    suml[2 * k, l]     += sums[k,0]      # diff accumulation
                    suml[2 * k + 1, l] += sums[k,1]      # diff**2 accumulation

        # Reshape suml to compute moments across all quantities simultaneously
        suml_reshaped = suml.reshape(dose_shape, 2, L + 1)
        ml = np.abs(suml_reshaped[:, 0, :] / Nl)
        Vl = np.maximum(0.0, suml_reshaped[:, 1, :] / Nl - ml**2)
        
        #Using acta numerica we maximise to ensure criteria met for all nodes 
        for l in range(L + 1):
            k = np.argmax(Vl[:, l])
            print(f"level {l}: max variance node={nodes_array[k]}, Vl={Vl[k,l]:.6e}, mean={ml[k,l]:.6e}")

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
            rng = np.arange(0, min(2, L - 1) + 1)
            rem = np.max(ml_max[L - rng] / (2.0**(rng * alpha))) / (2.0**alpha - 1.0)

            if rem > np.sqrt(theta) * eps:
                if L == Lmax:
                    print("*** failed to achieve weak convergence ***")
                else:
                    L += 1
                    
                    # Expand arrays dynamically for the new level
                    Vl_max = np.append(Vl_max, Vl_max[-1] / (2.0**beta))
                    Cl = np.append(Cl, Cl[-1] * (2.0**gamma))
                    Nl = np.append(Nl, 0)
                    suml = np.column_stack((suml, np.zeros(2 * dose_shape)))
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

    #Diagnostic for the outlying nodes, same ones used in mlmc test:
    ys = np.unique(Y_meshgrid.flatten())
    outlier_y = ceil(len(ys)/5)*side_len
    nodes=[]
    names=[]
    for k in range(1,8):
        node_coord = np.array([ceil(len(np.unique(X_meshgrid.flatten()))/8)*side_len*k, outlier_y])
        node = storage_position_convention(node_coord)
        nodes.append(node)
        names.append(None)

    named_nodes = [(node_number,name) for node_number, name in zip(nodes, names)]
    #Sorting this in increasing x
    named_nodes.sort(key=lambda x: nodes_array[x[0]][0])  

    for node, name in named_nodes:
        print(f"\nNode {node}: {nodes_array[node]}")
        print("suml first moment :", suml[2 * node, :])
        print("suml second moment:", suml[2 * node + 1, :])

    # Evaluate final multilevel estimators for all tracked outputs
    P_estimates = np.sum(suml[::2, :] / Nl, axis=1)
    
    # Return estimates as a flattened tuple followed by Nl and Cl for clean unpacking
    return tuple(P_estimates) + (Nl, Cl) 


if __name__ == "__main__":
    theta = 0.25

    #==========
    all_dose_estimates = []
    M_diag=load_mass_matrix()
    part_mlmc_parallel_l = partial(mlmc_parallel_l, M_diag=M_diag)
    for eps in Eps:
        # Dynamic unpacking of arbitrary lengths via index filtering slices
        results = mlmcv(part_mlmc_parallel_l, N0, eps, Lmin, Lmax)
        P_estimates = results[:-2]
        all_dose_estimates.append(np.asarray(P_estimates))
        Nl = results[-2]
        Cl = results[-1]

        mlmc_cost = np.sum(Nl * Cl)
        
        #idx = min(len(cost) - 1, len(Nl) - 1)
        #var2_max = max([var2[k, idx] for k in range(dose_shape)])
        #std_cost = var2_max * Cl[-1] / ((1.0 - theta) * eps**2)

    sim_num = np.sum(Nl)
    print(f"Total sims done by mlmc across all levels (exlcuding mlmc test): {sim_num}.")

    plot_folder = f"mlmc_{dose_method}_Nfull_{np.sum(Nl)}_lvls_{MLMC_LEVEL_OFFSET}_{len(Nl)-1+MLMC_LEVEL_OFFSET}_l_{side_len}_eps_{Eps[0]}_{Eps[-1]}"
    folder_path = os.path.join(file_path, plot_folder)
    os.makedirs(folder_path, exist_ok=True)

    if dose_method=='SF':
        l=round(side_len,3)
        path_3D = os.path.join(folder_path, f"{sampling_type}_{dose_method}_{method}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{side_len}_lvls_{Lmin + MLMC_LEVEL_OFFSET}_{len(Nl) - 1 + MLMC_LEVEL_OFFSET}_eps_{Eps[0]}_{Eps[-1]}.npz") 
        np.savez(path_3D, coeffs_expected=all_dose_estimates[-1], accuracy=Eps[-1], all_dose_estimates=all_dose_estimates, all_accuracies=Eps, min_step_lvl = MLMC_LEVEL_OFFSET + Lmin, mlmc_offset = MLMC_LEVEL_OFFSET, final_samples_per_lvl = Nl, final_costs_per_lvl = Cl, sim_num=sim_num, dose_method=dose_method, SPATIAL_DIM=SPATIAL_DIM, l=l, X_meshgrid = X_meshgrid)
    if dose_method=='SK':
        l=round(side_len,3)
        path_3D = os.path.join(folder_path, f"{sampling_type}_{dose_method}_{method}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{side_len}_lvls_{Lmin + MLMC_LEVEL_OFFSET}_{len(Nl) - 1 + MLMC_LEVEL_OFFSET}_sigma_{SIGMA:.3f}_eps_{Eps[0]}_{Eps[-1]}.npz") 
        np.savez(path_3D, dose_expected=all_dose_estimates[-1], accuracy=Eps[-1], all_dose_estimates=all_dose_estimates, all_accuracies=Eps, min_step_lvl = MLMC_LEVEL_OFFSET + Lmin, mlmc_offset = MLMC_LEVEL_OFFSET, final_samples_per_lvl = Nl, final_costs_per_lvl = Cl, sim_num=sim_num, dose_method=dose_method, SPATIAL_DIM=SPATIAL_DIM, l=l, X_meshgrid = X_meshgrid, sigma=SIGMA)

        #Don't call this!
            #mlmc_plot(txt_save_path, nvert=3, error_bars=True)

    print(f"Raw output array saved at {path_3D}.")
    print("Now plotting...")
    if SPATIAL_DIM==2:
        plot = dose_plot_2D(method, dose_method, path_3D)
    if SPATIAL_DIM==3:
        plot = dose_plot_3D(method, dose_method, path_3D)