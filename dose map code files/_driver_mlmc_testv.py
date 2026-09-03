import sys
import time
import os
import numpy as np
from datetime import datetime
from _driver_mlmcv import mlmcv  # Vectorized controller
from doseparams import dose_shape, nodes_array, MLMC_LEVEL_OFFSET
import matplotlib.pyplot as plt
from doseparams import base, T, dose_method, L_conv_test, N_conv_test, file_path, SPATIAL_DIM, X_meshgrid, Y_meshgrid, dose_shape, nodes_array, l_reciprocal, EPS_0, KAPPA, method, E0, SIGMA, sampling_type, theta, width_sdev_factor, EMIN, title_seg #do not import l side length 
from doseparams import l as side_len
from mlmc_test_plot import *

def mlmc_testv(mlmc_parallel_l, N, L, N0, Eps, Lmin, Lmax, *args):
    """
    Multilevel Monte Carlo test routine.
    Dynamically vectorized to process and profile the Primal value along with
    an arbitrary number of sensitivities simultaneously.

    Inputs:
        mlmc_parallel_l: output of the user file (sums, cost)
        N: sample number for convergence testing (complexity theorem)
        L: max lvl in conv test
        N0: level 0 starting sample count
        Eps, array: error tolerances 
        Lmin/max: min max level numbers allowed
        fp: output file (?)
        (*args means any other outputs e.g. params from user file)
    """

    #Writes things to the text file as they print
    def printf(file_ptr, fmt, *pargs, printout=True):
        """Helper to print to both stdout and a file pointer simultaneously."""
        text = fmt % pargs if pargs else fmt
        if printout:
            sys.stdout.write(text)
            sys.stdout.flush()
        if file_ptr is not None:
            file_ptr.write(text)
            file_ptr.flush()
    start_time = time.perf_counter()

    #Create a plot folder:
    plot_folder = f"mlmc_test_{dose_method}_{method}_N_{N_conv_test}_lvls_{MLMC_LEVEL_OFFSET}_{L_conv_test+MLMC_LEVEL_OFFSET}_l_{side_len}_E0_{E0}_KAPPA_{KAPPA}_EPS0_{EPS_0}_SIGMA_{SIGMA:.3f}_shape_{dose_shape}"
    folder_path = os.path.join(file_path, plot_folder)
    os.makedirs(folder_path, exist_ok=True)

    txt_save_path = os.path.join(folder_path, "mlmc_test_data.txt")
    fp=open(txt_save_path, "w")

    if N <= 0:
        raise ValueError("N must be a positive integer.")

    printf(fp, '\n')
    printf(fp, '**********************************************************\n')
    printf(fp, '*** MLMC file version 1.0     produced by PW & KZ      ***\n')
    printf(fp, '*** Python mlmc_testf on %s            ***\n', datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
    printf(fp, '**********************************************************\n')

    #Statistics storage
    del1 = np.zeros((dose_shape, L + 1), dtype=float)
    del2 = np.zeros((dose_shape, L + 1), dtype=float)
    var1 = np.zeros((dose_shape, L + 1), dtype=float)
    var2 = np.zeros((dose_shape, L + 1), dtype=float) 
    var2_trunc = np.zeros((dose_shape, L + 1), dtype=float) #this one will get truncated earlier
    trunc_tol = 1e-15
    kur1 = np.zeros((dose_shape, L + 1), dtype=float)
    chk1 = np.zeros((dose_shape, L + 1), dtype=float)
    cost = np.zeros(L + 1, dtype=float) #store the per level cost 

    for l in range(L + 1):
        print(f'Starting parameter testing for mlmc level {l}, stepsize level {l+MLMC_LEVEL_OFFSET}')

        stats_sum_l, cst_sum_l = mlmc_parallel_l(l, N, *args)
        if stats_sum_l.shape != (dose_shape,6):
            raise ValueError(f"shape mismatch between dose_mlmc output {stats_sum_l.shape} and dose_shape stats (global) {(dose_shape, 6)}.") 

        E_stats_sums_l = np.array(stats_sum_l, dtype=float) / N
        E_cost_l = cst_sum_l / N
        cost[l] = E_cost_l

        for k in range(dose_shape):
            s0, s1, s2, s3, s4, s5 = E_stats_sums_l[k, :]
            
            del1[k, l] = s0 #expected payoff diff
            del2[k, l] = s4 #expected payoff
            var1[k, l] = s1 - s0**2 #diff variance
            var2[k, l] = s5 - s4**2
            var2_trunc[k, l] = max(s5 - s4**2, trunc_tol) #Pl variance, truncation

            if l == 0:
                kur1[k, l] = 0.0
                chk1[k, l] = 0.0
            else:
                # Protect against division-by-zero errors in case variance hits zero floor
                denom = max(s1 - s0**2, 1e-12)
                kur1[k, l] = (s3 - 4*s2*s0 + 6*s1*(s0**2) - 3*(s0**4)) / (denom**2)
                
                chk1[k, l] = (del1[k, l] + del2[k, l-1] - del2[k, l]) / \
                             (3.0 * (np.sqrt(var1[k, l]) + np.sqrt(var2_trunc[k, l-1]) + np.sqrt(var2_trunc[k, l])) / np.sqrt(N)) #div by sampling error 

    #Call all the plots you want

    plot_kurtosis_variance_heatmap(L, kur1, var1, del1, del2, folder_path, level=9-MLMC_LEVEL_OFFSET)
    plot_consistency_error_heatmap(chk1, L, folder_path)
    plot_consistency_error_heatmap(chk1, L, folder_path, log_error=True)
    plot_fine_payoff(del2, N, folder_path)

    # Dynamic system warning evaluations
    for k in range(dose_shape):
        name = f"NODE: {nodes_array[k]}"
        #for the nodes high kurtosis at the edges is inevitable - replace this with a heatmap
        if kur1[k, -1] > 100.0: 
            printf(fp, '\n WARNING: mask kurtosis on finest level for %s = %f \n', name, kur1[k, -1], printout=False)
            printf(fp, ' indicates MLMC correction dominated by a few rare paths. \n', printout=False)
        if max(chk1[k, :]) > 1.0: #i.e. the error in the tele sum >> sampling error
            printf(fp, '\n WARNING: mask maximum consistency error for %s = %f \n', name, max(chk1[k, :]))
            printf(fp, ' indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied. \n')
        if min(chk1[k, :]) < -1.0: #i.e. the error in the tele sum >> sampling error
            printf(fp, '\n WARNING: mask minimum consistency error for %s = %f \n', name, min(chk1[k, :]))
            printf(fp, ' indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied. \n')

    #Edit ------ (for piecewise Lipschitz dose computation)
    #   For the dose application we should not estimate all node alpha and beta by regression
    #   Reason is because some nodes will naturally have smaller hit rates due to their position
    #   However in these instances we also expect payoff + variance to be ~0 -- these are not the nodes we care about for cost reduction
    #   In these cases the beta/alpha conditions may already be satisfied
    #   Per node we will check for this first, then regression on the other data points
    #   Because the payoff is piecewise Lipschitz there should be sufficiently regularity in the data
    #   Note: the MLMC complexity theorem applies to all nodes independently, so constant c2 is set each time indepedent of h

    step_levels = np.arange(MLMC_LEVEL_OFFSET, L + 1 + MLMC_LEVEL_OFFSET)[1:]
    alpha = np.zeros(dose_shape)
    beta = np.zeros(dose_shape)
    pg = np.polyfit(step_levels, np.log2(np.abs(cost[1:L+1])), 1)
    gamma = pg[0]

    #Since underlying is EM + payoff is piecewise cont, we expect:
    alpha_expected = 1.0
    beta_expected = 1.0
    print(f"Step levels check: {step_levels} (not inc. steplvl={MLMC_LEVEL_OFFSET})")
    step_sizes = np.array([float(base)**(-l) for l in step_levels])

    #c2, c3 can be any finite positive constant -- edit: the only purpose of this is to deal with the outliers, however this may need changing
    #At 10.0 most internal beam values still get regressed, which is what is intended
    c2 = 10.0
    c3 = 10.0

    #For the plots this will tell us which were not regressed
    a_regression_mask = np.full(dose_shape, False)
    b_regression_mask = np.full(dose_shape, False)

    a_nan_total = 0
    b_nan_total = 0
    for k in range(dose_shape):
        if np.any(var1[k, 1:L+1] < -trunc_tol):
            print(f"Node {nodes_array[k]} has negative variance diff estimates.")
        #First shield from taking log of 0
        del_filtered = np.where(np.abs(del1[k, 1:L+1]) > trunc_tol, np.abs(del1[k, 1:L+1]), trunc_tol)
        var_filtered = np.where(np.abs(var1[k, 1:L+1]) > trunc_tol, np.abs(var1[k, 1:L+1]), trunc_tol)
        
        #at each data point test the alpha/beta condition
        a_mask = del_filtered <= c2 * step_sizes ** alpha_expected
        b_mask = var_filtered <= c3 * step_sizes ** beta_expected

        #regress on only the non-floored values
        del_mask = del_filtered > trunc_tol
        var_mask = var_filtered > trunc_tol

        if np.all(a_mask): #if all pass the test already
            alpha[k] = alpha_expected
        else: 
            #reject regression if insufficient data
            if np.count_nonzero(del_mask) < 3: 
                alpha[k] = np.nan
                a_nan_total+=1 
            else:
                pa = np.polyfit(step_levels[del_mask], np.log2(del_filtered[del_mask]), 1)
                alpha[k] = -pa[0] #keep the raw data for the heatmap + reconcile global alpha/beta later
                a_regression_mask[k] = True
        
        if np.all(b_mask):
            beta[k] = beta_expected
        else: 
            #the regression assumes sufficient sampling/regularity, place a catch here
            if np.count_nonzero(var_mask) < 3:
                beta[k] = np.nan
                b_nan_total+=1
            else:
                pb = np.polyfit(step_levels[var_mask], np.log2(var_filtered[var_mask]), 1)
                beta[k] = -pb[0]
                b_regression_mask[k] = True

    printf(fp, f"\n\nNumber of alphas estimated using regression = {np.count_nonzero(a_regression_mask)}/{dose_shape}; Number where regression data was insufficient, set to nan: {a_nan_total}.")
    printf(fp, f"\nNumber of betas estimated using regression = {np.count_nonzero(b_regression_mask)}/{dose_shape}; Number where regression data was insufficient, set to nan: {b_nan_total}.")

    #Find node with maximum V_l at each level
    max_var_nodes = np.argmax(var1[:, 1:], axis=0)
    candidate_nodes = np.unique(max_var_nodes)

    printf(fp, f"\nBeta candidate nodes: \n{nodes_array[candidate_nodes], beta[candidate_nodes]}")

    #Calculate global beta
    #This is a regression over several nodes taking the worst var

    V_l = np.max(var1[:, 1:], axis=0)
    valid = V_l > trunc_tol
    beta_fit = np.polyfit(step_levels[valid], np.log2(V_l[valid]), 1)
    beta_global = -beta_fit[0]

    #set up title labels
    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel"

    def plot_global_beta():
        l_ticks = np.arange(0, L + 1)

        V_l = np.max(var1[:, 1:], axis=0)
        valid = V_l > trunc_tol
        levels = l_ticks[1:][valid]
        log_V_l = np.log2(V_l[valid])

        beta_fit = np.polyfit(levels, log_V_l, 1)
        beta_global = -beta_fit[0]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(levels, log_V_l, color="C0", label=r"$V_\ell = \max_i V_{\ell,i}$")
        ax.plot(levels, np.polyval(beta_fit, levels), color="C0", linestyle="--", label=rf"Fit: $\beta={beta_global:.3f}$")

        ax.set_xlabel(r"MLMC Level $\ell$", fontsize=12)
        ax.set_ylabel(r"$\log_2 V_\ell$", fontsize=12)
        ax.set_title(rf"{title_seg} MLMC Test: Global $\beta$ Regression", fontsize=14)
        ax.set_xticks(l_ticks)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=12)
        save_path=os.path.join(folder_path, "global_beta_Vl_regression.png")
        plt.savefig(save_path, dpi=300)
        plt.tight_layout()
        plt.show()
    plot_global_beta()
    printf(fp, f"\n\nBeta globally regressed as {beta_global}.\n\n")

    #More plots
    plot_alpha_beta_map(alpha, beta, folder_path)
    print("\nPlotting regression data...")
    plot_beam_axis_nodes(L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path)
    plot_negative_alpha_beta_nodes(L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path)
    plot_negative_estimator_nodes(L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path)

    npz_save_path = os.path.join(folder_path,f"mlmc_test_data_{dose_method}_N_{N}_L_{L}_offset_{MLMC_LEVEL_OFFSET}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_sigma_{SIGMA:.2f}_kappa_{KAPPA}_eps0_{EPS_0}_theta_{theta}_width_sdev_{width_sdev_factor:.2f}_EMIN_{EMIN}_l_{side_len}.npz")
    np.savez(npz_save_path,nodes_array=nodes_array,mean_diff=del1,mean_fine=del2, var_diff=var1, var_fine=var2, kurtosis_diff=kur1,
            consistency_check=chk1,alpha=alpha,beta=beta,gamma=gamma,cost=cost,N=N,L=L,Lmin=Lmin,Lmax=Lmax,MLMC_LEVEL_OFFSET=MLMC_LEVEL_OFFSET,base=base, beta_global=beta_global)
    print(f"MLMC test data saved to {npz_save_path}.")

    node = np.argmax(np.abs(del2[:, -1]))
    print("\n--- HIGH-DOSE NODE CONVERGENCE ---")
    print("node:", node, nodes_array[node])
    print("E[Pf] on finest level:", del2[node, -1]) #should be in the realm of 4-20 with > about 4k sims
    print("E[Pf-Pc] by level:", del1[node, :])
    print("Var(Pf-Pc) by level:", var1[node, :]) #should be reducing with level
    print("E[Pf] by level:", del2[node, :])

    printf(fp, '\n******************************************************\n')
    printf(fp, '*** Linear regression estimates of MLMC parameters ***\n')
    printf(fp, '******************************************************\n')
    printf(fp, ' gamma = %f  (exponent for MLMC cost) \n', gamma)
    
    for k in range(dose_shape):
        name = f"NODE {k}, {nodes_array[k]}"
        printf(fp, '\n --- %s ---\n', name, printout=False)
        printf(fp, ' alpha = %f  (exponent for MLMC weak convergence)\n', alpha[k], printout=False)
        printf(fp, ' beta  = %f  (exponent for MLMC variance) \n', beta[k], printout=False)

    #Only print the best+worst n_show nodes by beta to inspect
    n_show = 5

    #Ignore nan values
    nanmask = np.flatnonzero(~np.isnan(beta))
    ranking = nanmask[np.argsort(beta[nanmask])]
    worst_nodes = ranking[:n_show]
    best_nodes = ranking[-n_show:][::-1]

    printf(fp, '\n')
    printf(fp, '******************************************************\n')
    printf(fp, '*** Best/worst MLMC nodes by beta                  ***\n')
    printf(fp, '******************************************************\n')

    printf(fp, '\nWorst nodes (smallest beta):\n')
    printf(fp, ' node       alpha        beta\n')

    for k in worst_nodes:
        printf(fp, '%5d  %-20s  %11.4e  %11.4e\n', k, nodes_array[k], alpha[k], beta[k])

    printf(fp, '\nBest nodes (largest beta):\n')
    printf(fp, ' node               alpha        beta\n')

    for k in best_nodes:
        printf(fp, '%5d  %-20s  %11.4e  %11.4e\n', k, nodes_array[k], alpha[k], beta[k])

    #Create a new folder for the mlmc data associated to this run - folder name must contain all key params 
    save_folder = f"{sampling_type}_{dose_method}_{method}_eps_{min(Eps)}_{max(Eps)}_offset_{MLMC_LEVEL_OFFSET}_l_{side_len}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_sigma_{SIGMA:.2f}_kappa_{KAPPA}_eps0_{EPS_0}_theta_{theta}_width_sdev_{width_sdev_factor:.2f}_EMIN_{EMIN}"
    folder_path = os.path.join(file_path, save_folder)
    os.makedirs(folder_path, exist_ok=True)

    #Write the path to each npz into a text file to assist with plotting in main
    paths_file = os.path.join(folder_path, "mlmc_npz_results_paths.txt")

    #First, check if any of the eps in Eps have already been completed and saved in this folder
    completed_eps = set()
    if os.path.exists(paths_file):
        f = open(paths_file, "r")
        existing_data_paths = [line.strip() for line in f if line.strip()]
        f.close()

        for path in existing_data_paths:
            if not os.path.exists(path):
                continue
            else:
                data = np.load(path, allow_pickle=True)
                completed_eps.add(float(data["accuracy"]))
                data.close()

    for eps in Eps:
        if eps in completed_eps:
            print(f"Data found for eps={eps} in folder {folder_path}, skipping.")
            continue

        # Dynamic unpacking of arbitrary lengths via index filtering slices
        results = mlmcv(mlmc_parallel_l, N0, eps, Lmin, Lmax, beta0 = beta_global, gamma0=1.0, alpha0=1.0, *args) #alpha_max, beta_max,

        #All results
        mlmc_dose=np.asarray(results[:dose_shape])
        hybrid_dose=np.asarray(results[dose_shape:2*dose_shape])
        Nl = np.asarray(results[-4])
        Cl = np.asarray(results[-3])
        num_neg_nodes=results[-2]
        mc_replacements=results[-1]

        printf(fp, f"Number of negative payoff nodes with mlmc={num_neg_nodes}.\n", printout=False)
        printf(fp, f"Hybrid: mc estimator replacements made={mc_replacements}.")

        std_mlmc_cost = np.sum(Nl * Cl)

        idx = min(L_conv_test, len(Nl) - 1)
        var2_max = max([var2[k, idx] for k in range(dose_shape)]) #finds the max variance across all outputs at this eps
        std_cost = var2_max * Cl[-1] / ((1.0 - theta) * eps**2)

        #Save the data per epsilon: 
        max_num_lvls = len(Nl) #since Nl includes lvl 0
        step_levels = np.arange(MLMC_LEVEL_OFFSET, max_num_lvls + MLMC_LEVEL_OFFSET)
        sim_num = np.sum(Nl)
        path_3D=os.path.join(folder_path, f"{sampling_type}_{dose_method}_eps_{eps}_minstep_{step_levels[0]}_maxstep_{step_levels[-1]}_Nfull_{np.sum(Nl)}.npz")
        np.savez(path_3D, dose_expected=mlmc_dose, hybrid_dose_expected=hybrid_dose, accuracy=eps, Nl = Nl, Cl = Cl, std_mlmc_cost=std_mlmc_cost, std_cost=std_cost, step_levels=step_levels, sim_num=sim_num, dose_method=dose_method, method=method, SPATIAL_DIM=SPATIAL_DIM, l=side_len, X_meshgrid=X_meshgrid, Y_meshgrid=Y_meshgrid, title_seg=title_seg, folder_path=folder_path, sigma=SIGMA, dose_shape=dose_shape, KAPPA=KAPPA, EPS_0=EPS_0, sampling_type=sampling_type, num_neg_nodes=num_neg_nodes, mc_replacements=mc_replacements)
        
        #We already checked that this is not a duplicate
        f = open(paths_file, "a")
        f.write(path_3D + "\n")
        f.close()
        
        print(f"\nMLMC data for eps={eps} saved at {path_3D}.")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    printf(fp, '\n===================================================================\n')
    printf(fp, '>>> MLMC and MLMC test successfully completed in %.2f seconds.\n', elapsed)
    printf(fp, '===================================================================\n\n')
    printf(fp, '\n')

    fp.close()
    return folder_path, title_seg, paths_file
