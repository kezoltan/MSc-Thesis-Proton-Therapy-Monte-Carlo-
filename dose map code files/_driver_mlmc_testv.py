import sys
import time
import os
import numpy as np
from datetime import datetime
from _driver_mlmcv import mlmcv  # Vectorized controller
from doseparams import dose_shape, nodes_array, MLMC_LEVEL_OFFSET
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from math import ceil, log
from doseparams import base, T, dose_method, L_conv_test, N_conv_test, file_path, SPATIAL_DIM, X_meshgrid, Y_meshgrid, dose_shape, nodes_array, l_reciprocal, EPS_0, KAPPA, method, E0, SIGMA, sampling_type, theta, width_sdev_factor, EMIN #do not import l side length 
from doseparams import l as side_len
from dosemap1_shape_function_geoEM import storage_position_convention

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

    #Moments storage (for plots)
    #cent2 = np.zeros((dose_shape, L + 1))
    #cent4 = np.zeros((dose_shape, L + 1))

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

            #cent2[k, l] = (s1 - s0**2)**2 #second non centerd moment
            #cent4[k, l] = s3 - 4*s2*s0 + 6*s1*(s0**2) - 3*(s0**4) #fourth 

            if l == 0:
                kur1[k, l] = 0.0
                chk1[k, l] = 0.0
            else:
                # Protect against division-by-zero errors in case variance hits zero floor
                denom = max(s1 - s0**2, 1e-12)
                kur1[k, l] = (s3 - 4*s2*s0 + 6*s1*(s0**2) - 3*(s0**4)) / (denom**2)
                
                chk1[k, l] = (del1[k, l] + del2[k, l-1] - del2[k, l]) / \
                             (3.0 * (np.sqrt(var1[k, l]) + np.sqrt(var2_trunc[k, l-1]) + np.sqrt(var2_trunc[k, l])) / np.sqrt(N)) #div by sampling error 

    #set up title labels
    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function Dose"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel Dose"

    def plot_kurtosis_heatmap(log_kurt=False):
        print(f"Plotting kurtosis heatmap for mlmc estimator from mlmc levels 1 to {L}...")
        kurt_all = kur1[:, 1:]
        if log_kurt:
            kurt_plot = np.log10(1 + np.maximum(kurt_all, 0)) #make the kurts all >=0
            vmin = np.min(kurt_plot) #colour bar endpoints
            vmax = np.max(kurt_plot)
        else:
            kurt_plot=kurt_all
            vmin = np.min(kurt_all) 
            vmax = np.max(kurt_all)
        #Create the grid for heatmaps 
        ncols = 3
        nrows = int(np.ceil(L / ncols)) #plot all the levels
        fig, axes = plt.subplots(nrows,ncols,figsize=(5 * ncols, 4 * nrows))

        axes = np.atleast_1d(axes).ravel()
        for idx, l in enumerate(range(1, L + 1)):
            #Create scatter over the nodes array for kurt values 
            sc = axes[idx].scatter(nodes_array[:, 0],nodes_array[:, 1],c=kurt_plot[:, l - 1],vmin=vmin,vmax=vmax)
            axes[idx].set_title(rf"$\ell = {l}: h_\ell = T \cdot {base}^{{-{MLMC_LEVEL_OFFSET + l}}}$")
            axes[idx].set_aspect("equal")
        #Remove any empty subplot slots
        for idx in range(L, len(axes)):
            fig.delaxes(axes[idx])
        #Shared colour bar
        label = "log10(1 + kurtosis)" if log_kurt else r"$Y_\ell$ Kurtosis"
        fig.colorbar(sc,ax=axes[:L].tolist(),label=label)
        fig.suptitle(f"{title_seg} MLMC Test:{'Log' if log_kurt else ''} Kurtosis Heatmap (M={N_conv_test})")
        kurt_save_path = os.path.join(folder_path, f"{'log_' if log_kurt else ''}kurtosis_map_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(kurt_save_path, dpi=300)
        plt.show()
    plot_kurtosis_heatmap()
    #plot_kurtosis_heatmap(log_kurt=True)

    def plot_consistency_error_heatmap(log_error=False):
        print(f"Plotting consistency error heatmap from MLMC levels 1 to {L}...")
        consistency_all = chk1[:, 1:]
        if log_error:
            consistency_plot = np.log10(1 + np.abs(consistency_all))
            vmin, vmax = np.min(consistency_plot), np.max(consistency_plot)
        else:
            consistency_plot = consistency_all
            vmax = np.max(np.abs(consistency_all))
            vmin = -vmax

        ncols = 3
        nrows = int(np.ceil(L / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.atleast_1d(axes).ravel()

        for idx, l in enumerate(range(1, L + 1)):
            sc = axes[idx].scatter(nodes_array[:, 0], nodes_array[:, 1], c=consistency_plot[:, l-1], vmin=vmin, vmax=vmax)
            axes[idx].set_title(rf"$\ell = {l}: h_\ell = T \cdot {base}^{{-{MLMC_LEVEL_OFFSET + l}}}$")
            axes[idx].set_aspect("equal")

        for idx in range(L, len(axes)):
            fig.delaxes(axes[idx])

        label = "log10(1 + |consistency error|)" if log_error else "Consistency error"
        fig.colorbar(sc, ax=axes[:L].tolist(), label=label)
        fig.suptitle(f"{title_seg} MLMC Test: {'Log ' if log_error else ''}Consistency Error Heatmap (M={N_conv_test})")
        save_path = os.path.join(folder_path, f"{'log_' if log_error else ''}consistency_error_map_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
    plot_consistency_error_heatmap()
    plot_consistency_error_heatmap(log_error=True)

    #2. Fine Payoff Check - excluding sampling error, should match original MLMC 

    def plot_fine_payoff():
        mc_level = 9 #for fair comparison
        print(f"Plotting fine payoff from mlmc test with {N} samples at step size level {mc_level}")
        mlmc_level = mc_level - MLMC_LEVEL_OFFSET
        fine_payoff = del2[:, mlmc_level]
        plt.figure(figsize=(8, 5))
        sc = plt.scatter(nodes_array[:, 0],nodes_array[:, 1],c=fine_payoff)
        plt.colorbar(sc, label=r"$\mathbb{E}[P_\ell]$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(rf"{title_seg} MLMC Test: Fine Payoff, $h_\ell = T \cdot {base}^{{-{mc_level}}}$ (M={N_conv_test})")
        plt.gca().set_aspect("equal")

        print("Maximum fine payoff:", np.max(fine_payoff))
        print("Node of max payoff:", nodes_array[np.argmax(fine_payoff)])
        fine_P_save_path = os.path.join(folder_path, f"fine_payoff_mlmc_test_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(fine_P_save_path, dpi=300)
        plt.show()
    plot_fine_payoff()

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

    #c2, c3 can be any finite positive constant
    #If alpha, beta show up negative try increasing this first
    c2 = 10.0
    c3 = 10.0

    #For the plots this will tell us which were not regressed
    a_regression_mask = np.full(dose_shape, False)
    b_regression_mask = np.full(dose_shape, False)

    a_nan_total = 0
    b_nan_total = 0
    #First shield from taking log of 0
    for k in range(dose_shape):
        if np.any(var1[k, 1:L+1] < -1e-15):
            print(f"Node {nodes_array[k]} has negative variance diff estimates.")
        del_filtered = np.where(np.abs(del1[k, 1:L+1]) > 1e-15, np.abs(del1[k, 1:L+1]), 1e-15)
        var_filtered = np.where(np.abs(var1[k, 1:L+1]) > 1e-15, np.abs(var1[k, 1:L+1]), 1e-15)
        
        #at each data point test the alpha/beta condition
        a_mask = del_filtered <= c2 * step_sizes ** alpha_expected
        b_mask = var_filtered <= c3 * step_sizes ** beta_expected

        if np.all(a_mask):
            alpha[k] = alpha_expected
        else: 
            #the regression assumes sufficient sampling/regularity, place a catch here
            if np.count_nonzero(~a_mask) < 3:
                alpha[k] = np.nan
                a_nan_total+=1 
            else:
                pa = np.polyfit(step_levels[~a_mask], np.log2(del_filtered[~a_mask]), 1)
                alpha[k] = -pa[0] #keep the raw data for the heatmap + reconcile global alpha/beta later
                a_regression_mask[k] = True
        if np.all(b_mask):
            beta[k] = beta_expected
        else: 
            #the regression assumes sufficient sampling/regularity, place a catch here
            if np.count_nonzero(~b_mask) < 3:
                beta[k] = np.nan
                b_nan_total+=1
            else:
                pb = np.polyfit(step_levels[~b_mask], np.log2(var_filtered[~b_mask]), 1)
                beta[k] = -pb[0]
                b_regression_mask[k] = True

    printf(fp, f"\n\nNumber of alphas estimated using regression = {np.count_nonzero(a_regression_mask)}/{dose_shape}; Number where regression data was insufficient, set to nan: {a_nan_total}.")
    printf(fp, f"\nNumber of betas estimated using regression = {np.count_nonzero(b_regression_mask)}/{dose_shape}; Number where regression data was insufficient, set to nan: {b_nan_total}.")
    
    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function Dose"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel Dose"

    def plot_alpha_beta_map():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmin = min(np.nanmin(alpha), np.nanmin(beta))
        vmax = max(np.nanmax(alpha), np.nanmax(beta))
        
        sc_alpha = axes[0].scatter(nodes_array[:, 0],nodes_array[:, 1],c=alpha, vmin=vmin, vmax=vmax)
        axes[0].set_title(r"$\alpha$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_aspect("equal")
        fig.colorbar(sc_alpha, ax=axes[0], label=r"$\alpha$")

        sc_beta = axes[1].scatter(nodes_array[:, 0],nodes_array[:, 1],c=beta, vmin=vmin, vmax=vmax)
        axes[1].set_title(r"$\beta$")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_aspect("equal")
        fig.colorbar(sc_beta, ax=axes[1], label=r"$\beta$")
        fig.suptitle(f"{title_seg} MLMC Test: Convergence Rates (M={N_conv_test})")
        plt.tight_layout()
        ab_save_path = os.path.join(folder_path,f"alpha_beta_map_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(ab_save_path, dpi=300, bbox_inches="tight")
        plt.show()

        #Beam axis plot showing how they vary 
        ys = np.unique(Y_meshgrid.flatten())
        y_center = ceil(len(ys)/2)*side_len
        center_mask = np.isclose(nodes_array[:, 1], y_center)
        x_center = nodes_array[center_mask, 0]
        order = np.argsort(x_center)

        fig_line, ax_line = plt.subplots(figsize=(8, 5))
        ax_line.plot(x_center[order], alpha[center_mask][order], label=r"$\alpha$", alpha=0.7)
        ax_line.plot(x_center[order], beta[center_mask][order], label=r"$\beta$", alpha=0.7)
        ax_line.set_xlabel("Depth x, cm")
        ax_line.set_ylabel("Convergence Rate")
        ax_line.set_title(f"{title_seg} MLMC Test: Beam Axis Convergence Estimates (M={N_conv_test}, y={y_center})")
        ax_line.legend()
        #Integer ticks + gridlines
        ax_line.xaxis.set_major_locator(MultipleLocator(1))
        ax_line.yaxis.set_minor_locator(MultipleLocator(1))
        ax_line.grid(True, which="major", alpha=0.3)
        ax_line.grid(True, which="minor", alpha=0.3)
        plt.tight_layout()

        ab_line_save_path = os.path.join(folder_path, f"alpha_beta_centerline_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(ab_line_save_path, dpi=300, bbox_inches="tight")
        plt.show()
    plot_alpha_beta_map()
    
    def plot_mlmc_test(y_coord, plot_keyword):
        """
        Produce the standard report style mlmc plots. 
        Named nodes should be in the order you want it to come out from left to right 
        """
        if SPATIAL_DIM!=2:
            raise NotImplementedError(f"alpha/beta regression plots not implemented in {SPATIAL_DIM}D.")     

        beamline_mask = nodes_array[:,1] == y_coord
        beam_nodes = nodes_array[beamline_mask] 
        beam_nodes_numbers = np.flatnonzero(beamline_mask) #gives you the node numbers on the beamline
        payoff_fine = del2[beamline_mask,-1]

        #Select nodes
        node1 = beam_nodes_numbers[np.argmax(payoff_fine)]
        node3 = beam_nodes_numbers[np.argmax(alpha[beamline_mask])]
        node4 = beam_nodes_numbers[np.argmax(beta[beamline_mask])]

        same_node=False
        if node4 == node3: #likely this could happen
            nodes=[node1, node3] #node2, 
            names=[r"Max $P_\ell$", r"Max $\alpha$, $\beta$"] #r"Min $P_\ell$",
            same_node=True
        else:
            nodes=[node1,node3,node4] #node2,
            names=[r"Max $P_\ell$", r"Max $\alpha$", r"Max $\beta$"] #r"Min $P_\ell$",
        for k in range(2,8):
            node_coord = np.array([ceil(len(np.unique(X_meshgrid.flatten()))/8)*side_len*k, y_coord])
            node = storage_position_convention(node_coord)
            nodes.append(node)
            names.append(None)

        #If there is a node with nonzero but small payoff, show it
        domain_mask = beam_nodes[:, 0] >= 0 
        domain_payoff_fine = payoff_fine[domain_mask]
        valid = domain_payoff_fine > 0.5
        if np.any(valid):
            node2_beam_idx = np.flatnonzero(valid)[np.argmin(domain_payoff_fine[valid])]
            node2 = beam_nodes_numbers[node2_beam_idx]
            nodes.append(node2)
            names.append(None)

        #If there is a node with a negative mlmc estimator, show it:
        mlmc_estimator = del2[:, 0] + np.sum(del1[:, 1:], axis=1)
        beam_mlmc_estimator = mlmc_estimator[beamline_mask]
        negative_nodes = np.flatnonzero(beam_mlmc_estimator < 0)
        if len(negative_nodes) == 0:
            printf(fp, f"\nNo nodes found along y = {y_coord} with negative estimator")
        if len(negative_nodes) > 0:
            node_idx=beam_nodes_numbers[negative_nodes[0]]
            if nodes_array[node_idx][0] >= 0: #exclude if it is behind the beam axis
                nodes.append(node_idx)
                names.append(f'Y < 0')
        if len(negative_nodes) > 1:
            node_idx=beam_nodes_numbers[negative_nodes[-1]]
            if nodes_array[node_idx][0] >= 0:
                nodes.append(node_idx)
                names.append(f'Y < 0')            

        named_nodes = [(node_number,name) for node_number, name in zip(nodes, names)]
        #Sorting this in increasing x
        named_nodes.sort(key=lambda x: nodes_array[x[0]][0])  
        l_ticks = np.arange(MLMC_LEVEL_OFFSET,L + 1 + MLMC_LEVEL_OFFSET)

        #Print the raw data into the file 

        output_txt_path = os.path.join(folder_path, f"plot_mlmc_raw_regression_data_N_{N_conv_test}.txt")
        file_ptr = open(output_txt_path, "a")

        printf(file_ptr, "\n" + "=" * 156 + "\n", printout=False)
        printf(file_ptr, "RAW MLMC DATA AT SELECTED NODES — y = %.6f\n", y_coord, printout=False)
        printf(file_ptr, "=" * 156 + "\n", printout=False)

        printf(file_ptr, "\n" + "=" * 156 + "\n", printout=False)
        printf(file_ptr, "%s\n", plot_keyword.upper() + " NODES", printout=False)
        printf(file_ptr, "=" * 156 + "\n", printout=False)

        for node, name in named_nodes:
            printf(
                file_ptr,
                "\nNode %d: %s  %s\n",
                node,
                str(nodes_array[node]),
                f"({name})" if name is not None else "",
                printout=False
            )
            printf(file_ptr, "alpha = %.12e\n", alpha[node], printout=False)
            printf(file_ptr, "beta  = %.12e\n", beta[node], printout=False)

            printf(
                file_ptr,
                "%8s %20s %20s %25s %20s %25s %20s\n",
                "level",
                "P_l - P_{l-1}",
                "P_l",
                "Var(P_l - P_{l-1})",
                "Var(P_l)",
                "Kurt(P_l - P_{l-1})",
                "Consistency",
                printout=False
            )
            printf(file_ptr, "-" * 156 + "\n", printout=False)

            for j, level in enumerate(l_ticks):
                printf(
                    file_ptr,
                    "%8d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e\n",
                    level,
                    del1[node, j],
                    del2[node, j],
                    var1[node, j],
                    var2[node, j],
                    kur1[node, j],
                    chk1[node, j],
                    printout=False
                )

        file_ptr.close()

        fig, axes = plt.subplots(4, len(nodes), figsize=(3.1 * len(nodes),12),sharex=True, constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 0.70, 0.70]})
        for i, (node, name) in enumerate(named_nodes):
            ls = range(1, L + 1)    

            axes[0, i].set_ylabel(r"$\log_2 |mean|$")
            axes[1, i].set_ylabel(r"$\log_2 variance$")
            axes[2, i].set_ylabel("Kurtosis")
            axes[3, i].set_ylabel("Consistency Check")

            #Show which node is being plotted + why
            name_label=f" ({name})" if name != None else ""
            axes[0, i].set_title(rf"Node {nodes_array[node]}{name_label}" + "\n" + rf"$\alpha={alpha[node]:.3f}$")
            axes[1, i].set_title(rf"$\beta={beta[node]:.3f}$") 

            mean_data = np.abs(del1[node, 1:])
            payoff_data = np.abs(del2[node, :])
            var_data = var1[node, 1:]
            var2_data = var2[node, :]
            mean_masks = mean_data < trunc_tol
            payoff_masks = payoff_data < trunc_tol
            var_masks = var_data < trunc_tol
            var2_masks = var2_data < trunc_tol 

            #Cut them off to avoid 0 again
            mean_data = np.maximum(mean_data, trunc_tol)
            payoff_data = np.maximum(payoff_data, trunc_tol)
            var_data = np.maximum(var_data, trunc_tol)
            var2_data = np.maximum(var2_data, trunc_tol)

            kurt_data = kur1[node, :] 
            chk_data = chk1[node, :]

            axes[0, i].scatter(l_ticks[1:][~mean_masks], np.log2(mean_data[~mean_masks]))
            axes[0, i].scatter(l_ticks[1:][mean_masks], np.log2(mean_data[mean_masks]), marker='x')
            axes[0, i].scatter(l_ticks[~payoff_masks], np.log2(payoff_data[~payoff_masks]), color='red')
            axes[0, i].scatter(l_ticks[payoff_masks], np.log2(payoff_data[payoff_masks]), marker='+', color='gray')
            axes[0, i].ticklabel_format(axis='y', style='plain', useOffset=False)

            axes[1, i].scatter(l_ticks[1:][~var_masks], np.log2(var_data[~var_masks]))
            axes[1, i].scatter(l_ticks[1:][var_masks], np.log2(var_data[var_masks]), marker='x')
            axes[1, i].scatter(l_ticks[~var2_masks], np.log2(var2_data[~var2_masks]), color='red')
            axes[1, i].scatter(l_ticks[var2_masks], np.log2(var2_data[var2_masks]), marker='+', color='gray')
            axes[1, i].ticklabel_format(axis='y', style='plain', useOffset=False)

            axes[0, i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axes[1, i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            axes[2, i].plot(l_ticks, kurt_data)
            axes[3, i].plot(l_ticks, chk_data)

            mean_fit = np.polyfit(l_ticks[1:], np.log2(mean_data), 1)
            payoff_fit = np.polyfit(l_ticks, np.log2(payoff_data), 1)
            var_fit = np.polyfit(l_ticks[1:], np.log2(var_data), 1)
            var2_fit = np.polyfit(l_ticks, np.log2(var2_data), 1)

            axes[0, i].plot(l_ticks[1:], np.polyval(mean_fit, l_ticks[1:]), linestyle="-", label=rf'$P_\ell - P_{{\ell - 1}}$')
            axes[0, i].plot(l_ticks, np.polyval(payoff_fit, l_ticks), linestyle="--", label=rf'$P_\ell$', color='red')
            axes[1, i].plot(l_ticks[1:], np.polyval(var_fit, l_ticks[1:]), linestyle="-", label=rf'$P_\ell - P_{{\ell - 1}}$')
            axes[1, i].plot(l_ticks, np.polyval(var2_fit, l_ticks), linestyle="--", label=rf'$P_\ell$', color='red')
            axes[0, 0].legend()
            axes[1, 0].legend() #only show it on one

            #payoff_max = np.max(del2)*1.05
            #axes[0, i].set_ylim(-payoff_max*0.15, payoff_max)
            for k in range(4):
                axes[k, i].set_xticks(l_ticks)
                axes[k, i].grid(alpha=0.3)

        for k in range(len(nodes)):
            axes[-1, k].set_xlabel(r"Step Level $\ell$")
        fig.suptitle(f"{title_seg} MLMC Test: Statistics along y = {y_coord:.2f} (M={N_conv_test})", fontsize=14)
        mlmc_plot_path=os.path.join(folder_path, f"{plot_keyword}_regression_data_N_{N_conv_test}")
        plt.savefig(mlmc_plot_path, dpi=300)
        plt.show()    

    #Beam axis plot:
    ys = np.unique(Y_meshgrid.flatten())
    mid_y_beam = ceil(len(ys)/2)*side_len
    plot_keyword="beamline"
    plot_mlmc_test(mid_y_beam, plot_keyword)

    #Outliers plot (along far field y)
    #outlier_y = ceil(len(ys)/5)*side_len
    #plot_keyword ="outliers"
    #plot_mlmc_test(outlier_y, plot_keyword)

    def plot_negative_alpha_beta_nodes(max_nodes=8):
        """
        This is the same function as plot mlmc test but with different nodes selected
        update 
        """
        if SPATIAL_DIM != 2:
            raise NotImplementedError(f"Negative alpha/beta regression plots not implemented in {SPATIAL_DIM}D.")

        negative_mask = (alpha < 0) | (beta < 0)
        negative_nodes = np.flatnonzero(negative_mask)

        if len(negative_nodes) == 0:
            print("No nodes with negative alpha or beta.")
            return

        printf(fp, f"\nTotal number of nodes with negative alpha or beta: {len(negative_nodes)}\n\n")

        # Sort negative nodes spatially in x and select an approximately even spread
        negative_nodes = negative_nodes[np.argsort(nodes_array[negative_nodes, 0])]
        if len(negative_nodes) > max_nodes:
            selection = np.linspace(0, len(negative_nodes) - 1, max_nodes, dtype=int)
            nodes = negative_nodes[selection]
        else:
            nodes = negative_nodes

        l_ticks = np.arange(MLMC_LEVEL_OFFSET, L + 1 + MLMC_LEVEL_OFFSET)

        fig, axes = plt.subplots(4, len(nodes), figsize=(3.1 * len(nodes), 12), sharex=True,
                                constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 0.70, 0.70]})

        if len(nodes) == 1:
            axes = axes[:, np.newaxis]

        for i, node in enumerate(nodes):
            axes[0, i].set_ylabel(r"$\log_2 |mean|$")
            axes[1, i].set_ylabel(r"$\log_2 variance$")
            axes[2, i].set_ylabel("Kurtosis")
            axes[3, i].set_ylabel("Consistency Check")

            axes[0, i].set_title(rf"Node {nodes_array[node]}" + "\n" + rf"$\alpha={alpha[node]:.3f}$")
            axes[1, i].set_title(rf"$\beta={beta[node]:.3f}$")

            mean_data = np.abs(del1[node, 1:])
            payoff_data = np.abs(del2[node, :])
            var_data = var1[node, 1:]
            var2_data = var2[node, :]

            mean_masks = mean_data < trunc_tol
            payoff_masks = payoff_data < trunc_tol
            var_masks = var_data < trunc_tol
            var2_masks = var2_data < trunc_tol

            mean_data = np.maximum(mean_data, trunc_tol)
            payoff_data = np.maximum(payoff_data, trunc_tol)
            var_data = np.maximum(var_data, trunc_tol)
            var2_data = np.maximum(var2_data, trunc_tol)

            kurt_data = kur1[node, :]
            chk_data = chk1[node, :]

            axes[0, i].scatter(l_ticks[1:][~mean_masks], np.log2(mean_data[~mean_masks]))
            axes[0, i].scatter(l_ticks[1:][mean_masks], np.log2(mean_data[mean_masks]), marker='x')
            axes[0, i].scatter(l_ticks[~payoff_masks], np.log2(payoff_data[~payoff_masks]), color='red')
            axes[0, i].scatter(l_ticks[payoff_masks], np.log2(payoff_data[payoff_masks]), marker='+', color='gray')

            axes[1, i].scatter(l_ticks[1:][~var_masks], np.log2(var_data[~var_masks]))
            axes[1, i].scatter(l_ticks[1:][var_masks], np.log2(var_data[var_masks]), marker='x')
            axes[1, i].scatter(l_ticks[~var2_masks], np.log2(var2_data[~var2_masks]), color='red')
            axes[1, i].scatter(l_ticks[var2_masks], np.log2(var2_data[var2_masks]), marker='+', color='gray')

            axes[0, i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axes[1, i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            axes[2, i].plot(l_ticks, kurt_data)
            axes[3, i].plot(l_ticks, chk_data)

            mean_fit = np.polyfit(l_ticks[1:], np.log2(mean_data), 1)
            payoff_fit = np.polyfit(l_ticks, np.log2(payoff_data), 1)
            var_fit = np.polyfit(l_ticks[1:], np.log2(var_data), 1)
            var2_fit = np.polyfit(l_ticks, np.log2(var2_data), 1)

            axes[0, i].plot(l_ticks[1:], np.polyval(mean_fit, l_ticks[1:]), linestyle="-",
                            label=rf'$P_\ell-P_{{\ell-1}}$')
            axes[0, i].plot(l_ticks, np.polyval(payoff_fit, l_ticks), linestyle="--",
                            label=rf'$P_\ell$', color='red')
            axes[1, i].plot(l_ticks[1:], np.polyval(var_fit, l_ticks[1:]), linestyle="-",
                            label=rf'$P_\ell-P_{{\ell-1}}$')
            axes[1, i].plot(l_ticks, np.polyval(var2_fit, l_ticks), linestyle="--",
                            label=rf'$P_\ell$', color='red')

            for k in range(4):
                axes[k, i].set_xticks(l_ticks)
                axes[k, i].grid(alpha=0.3)

            axes[-1, i].set_xlabel(r"Step Level $\ell$")

        axes[0, 0].legend()
        axes[1, 0].legend()

        fig.suptitle(
            f"{title_seg} MLMC Test: Nodes with Negative Convergence Rates "
            f"(M={N_conv_test}, total negative={len(negative_nodes)})",
            fontsize=14
        )

        save_path = os.path.join(
            folder_path,
            f"negative_alpha_beta_regression_data_N_{N_conv_test}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.show()
    plot_negative_alpha_beta_nodes()

    def write_negative_nodes():
        """
        Finds all the negative Y nodes and write their data to a text file
        """
        mlmc_estimator = del2[:, 0] + np.sum(del1[:, 1:], axis=1)
        negative_nodes = np.flatnonzero(mlmc_estimator < 0)

        output_txt_path = os.path.join(folder_path, f"mlmc_test_negative_Y_data_N_{N_conv_test}.txt")
        file_ptr = open(output_txt_path, "a")

        if len(negative_nodes) == 0:
            printf(file_ptr, "No nodes with negative MLMC estimator found. Skipping negative payoff plot.", printout=False)
            file_ptr.close()
            return

        levels = np.arange(MLMC_LEVEL_OFFSET+1, MLMC_LEVEL_OFFSET+L+1, 1)
        for node in negative_nodes:
            printf(file_ptr,
                f"Node {node}, coord={nodes_array[node]}, "
                f"MLMC estimator={mlmc_estimator[node]:.6e}, Levels={levels}: Kurtosis={kur1[node, 1:]}, Consistency={chk1[node, 1:]}\n", printout=False)
        
        file_ptr.close()
        return 
    write_negative_nodes()

    npz_save_path = os.path.join(folder_path,f"mlmc_test_data_{dose_method}_N_{N}_L_{L}_offset_{MLMC_LEVEL_OFFSET}.npz")
    np.savez(npz_save_path,nodes_array=nodes_array,mean_diff=del1,mean_fine=del2, var_diff=var1, var_fine=var2, kurtosis_diff=kur1,
            consistency_check=chk1,alpha=alpha,beta=beta,gamma=gamma,cost=cost,N=N,L=L,Lmin=Lmin,Lmax=Lmax,MLMC_LEVEL_OFFSET=MLMC_LEVEL_OFFSET,base=base)
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
    ranking = np.argsort(beta)
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

    #Edit: unsure why these were originally set to max? Min makes more sense to me
    #alpha_max = max(np.max(alpha), 0.5)
    #beta_max = max(np.max(beta), 0.5)

    alpha_min = np.min(alpha)
    beta_min = np.min(beta)
    print(f"alpha_min: {alpha_min}, beta_min: {beta_min}")

    #Assign known values after testing
    alpha = 1.0
    beta = 1.0

    #Create a new folder for the mlmc data associated to this run - folder name must contain all key params 
    save_folder = f"{sampling_type}_{dose_method}_{method}_eps_{min(Eps)}_{max(Eps)}_offset_{MLMC_LEVEL_OFFSET}_l_{side_len}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_sigma_{SIGMA:.2f}_kappa_{KAPPA}_eps0_{EPS_0}_theta_{theta}_width_sdev_{width_sdev_factor:.2f}_EMIN_{EMIN}"
    folder_path = os.path.join(file_path, save_folder)
    os.makedirs(folder_path, exist_ok=True)

    #set up title labels
    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel"

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
        results = mlmcv(mlmc_parallel_l, N0, eps, Lmin, Lmax, alpha, beta, gamma, *args) #alpha_max, beta_max,

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
        np.savez(path_3D, dose_expected=mlmc_dose, hybrid_dose_expected=hybrid_dose, accuracy=eps, Nl = Nl, Cl = Cl, std_mlmc_cost=std_mlmc_cost, std_cost=std_cost, step_levels=step_levels, sim_num=sim_num, dose_method=dose_method, method=method, SPATIAL_DIM=SPATIAL_DIM, l=side_len, X_meshgrid=X_meshgrid, title_seg=title_seg, folder_path=folder_path, sigma=SIGMA, dose_shape=dose_shape, KAPPA=KAPPA, EPS_0=EPS_0, sampling_type=sampling_type, num_neg_nodes=num_neg_nodes, mc_replacements=mc_replacements)
        
        #We already checked that this is not a duplicate
        f = open(paths_file, "a")
        f.write(path_3D + "\n")
        f.close()
        
        print(f"MLMC data for eps={eps} saved at {path_3D}.")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    printf(fp, '\n===================================================================\n')
    printf(fp, '>>> MLMC and MLMC test successfully completed in %.2f seconds.\n', elapsed)
    printf(fp, '===================================================================\n\n')
    printf(fp, '\n')

    fp.close()
    return folder_path, title_seg, paths_file
