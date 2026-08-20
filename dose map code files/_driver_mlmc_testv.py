import sys
import time
import numpy as np
from datetime import datetime
from _driver_mlmcv import mlmcv  # Vectorized controller
from doseparams import dose_shape, nodes_array, MLMC_LEVEL_OFFSET
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import ceil, log
from doseparams import base, T, dose_method, L_conv_test, N_conv_test, file_path, SPATIAL_DIM, X_meshgrid, Y_meshgrid, dose_shape, nodes_array #do not import l side length 
import os
from doseparams import l as side_len
from dosemap1_shape_function_geoEM import storage_position_convention

def mlmc_testv(mlmc_parallel_l, N, L, N0, Eps, Lmin, Lmax, fp, *args):
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

    if N <= 0:
        raise ValueError("N must be a positive integer.")

    printf(fp, '\n')
    printf(fp, '**********************************************************\n')
    printf(fp, '*** MLMC file version 1.0     produced by              ***\n')
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

    #Create a plot folder:
    plot_folder = f"mlmc_test_{dose_method}_N_{N_conv_test}_lvls_{MLMC_LEVEL_OFFSET}_{L_conv_test+MLMC_LEVEL_OFFSET}_l_{side_len}"
    folder_path = os.path.join(file_path, plot_folder)
    os.makedirs(folder_path, exist_ok=True)

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
    #the SF method will be too chunky imo

    #for k in range(dose_shape):
    #    name = f"Node {k}, {nodes_array[k]}"
    #    printf(fp, '\n', printout=False)
    #    printf(fp, '**********************************************************\n', printout=False)
    #    printf(fp, '*** %s Convergence tests, kurtosis, telescoping sum ***\n', name, printout=False)
    #    printf(fp, '*** using N =%7d samples                            ***\n', N, printout=False)
    #    printf(fp, '**********************************************************\n', printout=False)
    #    printf(fp, '\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)  var(Pf)   kurtosis    check     cost\n', printout=False)
    #    printf(fp, '-------------------------------------------------------------------------------\n', printout=False)
    #    for l in range(L + 1):
    #        printf(fp, "%2d  %11.4e %11.4e  %.3e  %.3e  %.2e  %.2e  %.2e \n",
    #               l, del1[k, l], del2[k, l], var1[k, l], var2_trunc[k, l], kur1[k, l], chk1[k, l], cost[l], printout=False)


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

    #Use linear regression to estimate alpha, beta and gamma
    #edit: rlly should produce plots to make sure this is working as intended
    x = np.arange(1, L + 1)
    alpha = np.zeros(dose_shape)
    beta = np.zeros(dose_shape)
    pg = np.polyfit(x, np.log2(np.abs(cost[1:L+1])), 1)
    gamma = pg[0]

    for k in range(dose_shape):
        #Prevent log2(0) explosions on un-differentiable assets or zero variations

        del_filtered = np.where(np.abs(del1[k, 1:L+1]) > 1e-15, np.abs(del1[k, 1:L+1]), 1e-15)
        var_filtered = np.where(np.abs(var1[k, 1:L+1]) > 1e-15, np.abs(var1[k, 1:L+1]), 1e-15)
        pa = np.polyfit(x, np.log2(del_filtered), 1); alpha[k] = -pa[0]
        pb = np.polyfit(x, np.log2(var_filtered), 1); beta[k] = -pb[0]

    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function Dose"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel Dose"

    def plot_alpha_beta_map():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc_alpha = axes[0].scatter(nodes_array[:, 0],nodes_array[:, 1],c=alpha)
        axes[0].set_title(r"$\alpha$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_aspect("equal")
        fig.colorbar(sc_alpha, ax=axes[0], label=r"$\alpha$")

        sc_beta = axes[1].scatter(nodes_array[:, 0],nodes_array[:, 1],c=beta)
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
    plot_alpha_beta_map()
    
    def plot_mlmc(y_coord, plot_keyword):
        """
        Produce the standard report style mlmc plots. 
        Named nodes should be in the order you want it to come out from left to right 
        """
        if SPATIAL_DIM!=2:
            raise NotImplementedError(f"alpha/beta regression plots not implemented in {SPATIAL_DIM}D.")     

        beamline_mask = nodes_array[:,1] == y_coord
        beam_nodes = nodes_array[beamline_mask] 
        beam_nodes_numbers = np.flatnonzero(beamline_mask)
        payoff_fine = del2[beamline_mask,-1]

        #Select nodes
        node1 = beam_nodes_numbers[np.argmax(payoff_fine)]
        #valid = payoff_fine > 1
        #node2_beam_idx = np.flatnonzero(valid)[np.argmin(payoff_fine[valid])]
        #node2 = beam_nodes_numbers[node2_beam_idx]
        #node2 = beam_nodes_numbers[np.argmin(payoff_fine)]
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
        for k in range(1,8):
            node_coord = np.array([ceil(len(np.unique(X_meshgrid.flatten()))/8)*side_len*k, y_coord])
            node = storage_position_convention(node_coord)
            nodes.append(node)
            names.append(None)

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
            axes[-1, k].set_xlabel(r"Step level $\ell$")
        fig.suptitle(f"{title_seg} MLMC Test: Statistics along y = {y_coord:.2f} (M={N_conv_test})", fontsize=14)
        mlmc_plot_path=os.path.join(folder_path, f"{plot_keyword}_regression_data_N_{N_conv_test}")
        plt.savefig(mlmc_plot_path, dpi=300)
        plt.show()    

    #Beam axis plot:
    ys = np.unique(Y_meshgrid.flatten())
    mid_y_beam = ceil(len(ys)/2)*side_len
    plot_keyword="beamline"
    plot_mlmc(mid_y_beam, plot_keyword)

    #Outliers plot
    outlier_y = ceil(len(ys)/5)*side_len
    plot_keyword ="outliers"
    plot_mlmc(outlier_y, plot_keyword)

    def plot_neg_payoff():
        """
        Produce the standard report style mlmc plots. 
        Named nodes should be in the order you want it to come out from left to right 
        """
        if SPATIAL_DIM!=2:
            raise NotImplementedError(f"alpha/beta regression plots not implemented in {SPATIAL_DIM}D.")     

        mlmc_estimator = del2[:, 0] + np.sum(del1[:, 1:], axis=1)
        negative_nodes = np.flatnonzero(mlmc_estimator < 0)

        if len(negative_nodes) == 0:
            print("No nodes with negative MLMC estimator found. Skipping negative payoff plot.")
            return

        for node in negative_nodes:
            print(
                f"Node {node}, coord={nodes_array[node]}, "
                f"MLMC estimator={mlmc_estimator[node]:.6e}"
            )
        rng = np.random.default_rng(seed=42)
        n_select = min(8, len(negative_nodes))
        nodes = rng.choice(
            negative_nodes,
            size=n_select,
            replace=False
        ).tolist()
        names = [rf"Y={mlmc_estimator[node]:.2e}" for node in nodes]

        named_nodes = [(node_number,name) for node_number, name in zip(nodes, names)]
        #Sorting this in increasing x
        named_nodes.sort(key=lambda x: nodes_array[x[0]][0])  
        l_ticks = np.arange(MLMC_LEVEL_OFFSET,L + 1 + MLMC_LEVEL_OFFSET)

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
            axes[-1, k].set_xlabel(r"Step level $\ell$")
        fig.suptitle(f"{title_seg} MLMC Test: Negative MLMC Estimator Nodes (M={N_conv_test})", fontsize=14)
        mlmc_plot_path=os.path.join(folder_path, f"negative_estimator_regression_data_N_{N_conv_test}")
        plt.savefig(mlmc_plot_path, dpi=300)
        plt.show()   
    plot_neg_payoff()

    npz_save_path = os.path.join(folder_path,f"mlmc_test_data_{dose_method}_N_{N}_L_{L}_offset_{MLMC_LEVEL_OFFSET}.npz")
    np.savez(npz_save_path,nodes_array=nodes_array,mean_diff=del1,mean_fine=del2, var_diff=var1, var_fine=var2, kurtosis_diff=kur1,
            consistency_check=chk1,alpha=alpha,beta=beta,gamma=gamma,cost=cost,N=N,L=L,Lmin=Lmin,Lmax=Lmax,MLMC_LEVEL_OFFSET=MLMC_LEVEL_OFFSET,base=base)
    print(f"MLMC test data saved to {npz_save_path}")

    #This is a print to verify that MLMC is working as intended
    #Print out info from the maximal dose node 
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

    # Second, mlmc complexity tests
    #printf(fp, '\n', printout=False)
    #printf(fp, '***************************** \n', printout=False)
    #printf(fp, '*** MLMC complexity tests *** \n', printout=False)
    #printf(fp, '***************************** \n\n', printout=False)
    
    #Simply there are too many headers for one per node, reduce:    
    #headers = "  eps       mlmc_cost     std_cost    savings       N_l\n"
    #printf(fp, headers, printout=False)
    #printf(fp, "-" * len(headers) + "\n", printout=False)
    
    #headers = "  eps      "
    #for k in range(dose_shape):
    #    headers += f"{nodes_array[k]}".ljust(12)
    #headers += "mlmc_cost   std_cost  savings     N_l \n"
    #printf(fp, headers, printout=False)
    #printf(fp, "-" * (len(headers) + 15) + "\n", printout=False)
    
    # Reset random number generator for complexity tests
    #np.random.seed(None) -- seed is assigned in the user function


    #Rewrite===

    alpha_max = max(np.max(alpha), 0.5)
    beta_max = max(np.max(beta), 0.5)
    theta = 0.25

    #==========
    all_dose_estimates = []

    for eps in Eps:
        # Dynamic unpacking of arbitrary lengths via index filtering slices
        results = mlmcv(mlmc_parallel_l, N0, eps, Lmin, Lmax, alpha_max, beta_max, gamma, *args)
        P_estimates = results[:-2]
        all_dose_estimates.append(np.asarray(P_estimates))
        Nl = results[-2]
        Cl = results[-1]

        mlmc_cost = np.sum(Nl * Cl)
        
        #idx is the finest index 
        idx = min(len(cost) - 1, len(Nl) - 1)
        #finds the max variance across all outputs at this eps
        var2_max = max([var2[k, idx] for k in range(dose_shape)])
        std_cost = var2_max * Cl[-1] / ((1.0 - theta) * eps**2)

    #We'll make the cost + Nl plots here, globally for all nodes

    #            plt.subplot(3, 2, 5)
    #        if Nls.size:
    #            plt.semilogy(np.arange(Nls.shape[0]), Nls)
    #        plt.xlabel(r"level $\ell$")
    #        plt.ylabel(r"$N_\ell$")
    #        if len(Eps):
    #            plt.legend([str(eps) for eps in Eps], loc="upper right")
    #
    #        plt.subplot(3, 2, 6)
    #        if len(Eps):
    #            plt.loglog(Eps, Eps**2 * std_cost, "-*", label="Std MC")
    #            plt.loglog(Eps, Eps**2 * mlmc_cost, ":*", label="MLMC")
    #        plt.xlabel(r"accuracy $\varepsilon$")
    #        plt.ylabel(r"$\varepsilon^2$ Cost")
    #        if len(Eps):
    #            plt.legend()

        # Print outputs -- too many outputs
        #printf(fp, "%.3e ", eps, printout=False)
        #for p in P_estimates:
        #    printf(fp, "%11.3e ", p, printout=False)
        #    
        #printf(fp, " %.3e  %.3e  %7.2f ", mlmc_cost, std_cost, std_cost / mlmc_cost, printout=False)
        # 
        #for n in Nl:
        #    printf(fp, "%10d ", n, printout=False)
        #printf(fp, "\n", printout=False)

        #printf(fp,"%.3e   %.3e   %.3e   %7.2f    ",eps,mlmc_cost,std_cost,std_cost / mlmc_cost, printout=False)
        #for n in Nl:
        #    printf(fp, "%10d ", n, printout=False)
        #printf(fp, "\n", printout=False)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    printf(fp, '\n=================================================================\n')
    printf(fp, '>>> MLMC evaluation successfully completed in %.2f seconds.\n', elapsed)
    printf(fp, '=================================================================\n\n')

    printf(fp, '\n')

    all_dose_estimates = np.array(all_dose_estimates)

    return all_dose_estimates, Eps, Nl, Cl, folder_path
