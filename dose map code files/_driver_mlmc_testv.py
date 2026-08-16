import sys
import time
import numpy as np
from datetime import datetime
from _driver_mlmcv import mlmcv  # Vectorized controller
from doseparams import dose_shape, nodes_array, MLMC_LEVEL_OFFSET
import matplotlib.pyplot as plt
from math import ceil
from doseparams import base, T, dose_method, L_conv_test, N_conv_test, file_path, SPATIAL_DIM #do not import l side length 
import os

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
    def printf(file_ptr, fmt, *pargs):
        """Helper to print to both stdout and a file pointer simultaneously."""
        text = fmt % pargs if pargs else fmt
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

    num_q = dose_shape  # Number of nodes, global parameter

    #Statistics storage
    del1 = np.zeros((num_q, L + 1), dtype=float)
    del2 = np.zeros((num_q, L + 1), dtype=float)
    var1 = np.zeros((num_q, L + 1), dtype=float)
    var2 = np.zeros((num_q, L + 1), dtype=float)
    kur1 = np.zeros((num_q, L + 1), dtype=float)
    chk1 = np.zeros((num_q, L + 1), dtype=float)
    cost = np.zeros(L + 1, dtype=float) #store the per level cost 

    cent2 = np.zeros((num_q, L + 1))
    cent4 = np.zeros((num_q, L + 1))

    for l in range(L + 1):
        print(f'Starting parameter testing for mlmc level {l}, stepsize level {l+MLMC_LEVEL_OFFSET}')

        stats_sum_l, cst_sum_l = mlmc_parallel_l(l, N, *args)
        if stats_sum_l.shape != (dose_shape,6):
            raise ValueError(f"shape mismatch between dose_mlmc output {stats_sum_l.shape} and dose_shape stats (global) {(dose_shape, 6)}.") 

        #Calculate mc estimator at level 1 (/N -> expected value)
        E_stats_sums_l = np.array(stats_sum_l, dtype=float) / N
        E_cost_l = cst_sum_l / N
        cost[l] = E_cost_l

        for k in range(num_q):
            s0, s1, s2, s3, s4, s5 = E_stats_sums_l[k, :]
            
            del1[k, l] = s0
            del2[k, l] = s4
            cent2[k, l] = (s1 - s0**2)**2 #second non centerd moment
            cent4[k, l] = s3 - 4*s2*s0 + 6*s1*(s0**2) - 3*(s0**4) #fourth 
            var1[k, l] = s1 - s0**2
            var2[k, l] = max(s5 - s4**2, 1e-10) #excludes variance of 0

            if l == 0:
                kur1[k, l] = 0.0
                chk1[k, l] = 0.0
            else:
                # Protect against division-by-zero errors in case variance hits zero floor
                denom = max(s1 - s0**2, 1e-12)
                kur1[k, l] = (s3 - 4*s2*s0 + 6*s1*(s0**2) - 3*(s0**4)) / (denom**2)
                
                chk1[k, l] = abs(del1[k, l] + del2[k, l-1] - del2[k, l]) / \
                             (3.0 * (np.sqrt(var1[k, l]) + np.sqrt(var2[k, l-1]) + np.sqrt(var2[k, l])) / np.sqrt(N)) #div by sampling error 

    #------------------------------------------------------------                                                                                                                  #i.e. can the diff be explained by sampling error 
    #PLots! L panels, shows how it changes over level (all mlmc levels except 0)
    
    #1. Kurtosis heat map

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
        kurt_save_path = os.path.join(file_path, f"{'log_' if log_kurt else ''}kurtosis_map_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(kurt_save_path, dpi=300)
        plt.show()
    #plot_kurtosis_heatmap()
    #plot_kurtosis_heatmap(log_kurt=True)

    #2. Fine Payoff Check - excluding sampling error, should match original MLMC 

    def plot_fine_payoff():
        mc_level = 9 #for fair comparison
        print(f"Plotting fine payoff from mlmc test with {N} samples at step size level {mc_level}")
        mlmc_level = mc_level - MLMC_LEVEL_OFFSET
        fine_payoff = del2[:, mlmc_level]
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(nodes_array[:, 0],nodes_array[:, 1],c=fine_payoff)
        plt.colorbar(sc, label=r"$\mathbb{E}[P_\ell]$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(rf"{title_seg} MLMC Test: Fine Payoff, $h_\ell = T \cdot {base}^{{-{mc_level}}}$ (M={N_conv_test})")
        plt.gca().set_aspect("equal")

        print("Maximum fine payoff:", np.max(fine_payoff))
        print("Node of max payoff:", nodes_array[np.argmax(fine_payoff)])
        fine_P_save_path = os.path.join(file_path, f"fine_payoff_mlmc_test_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
        plt.savefig(fine_P_save_path, dpi=300)
        plt.show()
    #plot_fine_payoff()
    #the SF method will be too chunky imo

    def plot_moments_2_4():

        for mlmc_level in [L]:

            m2 = cent2[:, mlmc_level]
            m4 = cent4[:, mlmc_level]

            ratio = np.full(num_q, np.nan)
            mask = m2 > 0
            ratio[mask] = (m4[mask]/ m2[mask])

            fig, axes = plt.subplots(1,3,figsize=(17, 5),sharex=True,sharey=True)
            sc0 = axes[0].scatter(nodes_array[:, 0],nodes_array[:, 1],c=m2)
            axes[0].set_title(r"$\mathbb{E}[(Y_\ell-\mathbb{E}(Y_\ell))^2]^2$")
            fig.colorbar(sc0,ax=axes[0])
            sc1 = axes[1].scatter(nodes_array[:, 0],nodes_array[:, 1],c=m4)
            axes[1].set_title(r"$\mathbb{E}[(Y_\ell - \mathbb{E}(Y_\ell))^4]$")
            fig.colorbar(sc1,ax=axes[1])

            sc2 = axes[2].scatter(nodes_array[:, 0],nodes_array[:, 1],c=ratio)
            axes[2].set_title(r"$\mathbb{E}[(Y_\ell - \mathbb{E}(Y_\ell))^4]/\mathbb{E}[(Y_\ell-\mathbb{E}(Y_\ell))^2]^2$")
            fig.colorbar(sc2,ax=axes[2])
            for ax in axes:
                ax.set_xlabel("x")
                ax.set_aspect("equal")
            axes[0].set_ylabel("y")
            fig.suptitle(rf"MLMC Moments, $h_\ell=T \cdot 2^{{-{mlmc_level+MLMC_LEVEL_OFFSET}}}$, (N={N})")
            plt.tight_layout()
            plt.show()

    #plot_moments_2_4()

    #-------------------------------------------------------------

    # Dynamically output Convergence Tables for all parameters
    #for k in range(num_q):
    #    name = f"NODE {k}"
    #    printf(fp, '\n')
    #    printf(fp, '**********************************************************\n')
    #    printf(fp, '*** %s Convergence tests, kurtosis, telescoping sum ***\n', name)
    #    printf(fp, '*** using N =%7d samples                            ***\n', N)
    #    printf(fp, '**********************************************************\n')
    #    printf(fp, '\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)  var(Pf)   kurtosis    check     cost\n')
    #    printf(fp, '-------------------------------------------------------------------------------\n')
    #    for l in range(L + 1):
    #        printf(fp, "%2d  %11.4e %11.4e  %.3e  %.3e  %.2e  %.2e  %.2e \n",
    #               l, del1[k, l], del2[k, l], var1[k, l], var2[k, l], kur1[k, l], chk1[k, l], cost[l])

    # Dynamic system warning evaluations
    for k in range(num_q):
        name = f"NODE {k, nodes_array[k]}"
        #for the nodes high kurtosis at the edges is inevitable - replace this with a heatmap
        #if kur1[k, -1] > 100.0:
        #    printf(fp, '\n WARNING: kurtosis on finest level for %s = %f \n', name, kur1[k, -1])
        #    printf(fp, ' indicates MLMC correction dominated by a few rare paths. \n')
        if max(chk1[k, :]) > 1.0: #i.e. the error in the tele sum >> sampling error
            printf(fp, '\n WARNING: maximum consistency error for %s = %f \n', name, max(chk1[k, :]))
            printf(fp, ' indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied. \n')

    # Use linear regression to estimate alpha, beta and gamma parameters dynamically
    x = np.arange(1, L + 1)
    alpha = np.zeros(num_q)
    beta = np.zeros(num_q)
    
    pg = np.polyfit(x, np.log2(np.abs(cost[1:L+1])), 1)
    gamma = pg[0]

    for k in range(num_q):
        #Prevent log2(0) explosions on un-differentiable assets or zero variations
        del_filtered = np.where(np.abs(del1[k, 1:L+1]) > 1e-15, np.abs(del1[k, 1:L+1]), 1e-15)
        var_filtered = np.where(np.abs(var1[k, 1:L+1]) > 1e-15, np.abs(var1[k, 1:L+1]), 1e-15)
        
        pa = np.polyfit(x, np.log2(del_filtered), 1); alpha[k] = -pa[0]
        pb = np.polyfit(x, np.log2(var_filtered), 1); beta[k] = -pb[0]

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
        ab_save_path = os.path.join(file_path,
        f"alpha_beta_map_{dose_method}"
        f"_maxlvl_{L_conv_test}"
        f"_offset_{MLMC_LEVEL_OFFSET}"
        f"_mcsims_{N_conv_test}.png"
        )

        plt.savefig(ab_save_path, dpi=300, bbox_inches="tight")
        plt.show()
    #plot_alpha_beta_map()

    
    def plot_alpha_beta_diagnostics(n_nodes=5, beam_y=None):
        """
        Plot:
        1. alpha/beta regression data for representative nodes along beam axis
        2. calculated alpha/beta along the beam axis
        """

        if beam_y is None:
            beam_y = np.median(nodes_array[:, 1])

        unique_y = np.unique(nodes_array[:, 1])
        beam_y_mesh = unique_y[np.argmin(np.abs(unique_y - beam_y))]

        beam_mask = np.isclose(nodes_array[:, 1], beam_y_mesh)
        beam_indices = np.where(beam_mask)[0]
        beam_indices = beam_indices[np.argsort(nodes_array[beam_indices, 0])]
        beam_x = nodes_array[beam_indices, 0]

        print(f"Using beam-axis mesh row y = {beam_y_mesh:.6f}")

        # Select representative active nodes along beam
        finest_fine_payoff = np.abs(del2[:, -1])
        active_beam_indices = beam_indices[finest_fine_payoff[beam_indices] > 0]

        if len(active_beam_indices) < n_nodes:
            raise ValueError(f"Only {len(active_beam_indices)} nonzero beam-axis nodes available, but n_nodes={n_nodes}.")

        selection_positions = np.linspace(0, len(active_beam_indices) - 1, n_nodes).astype(int)
        selected_nodes = active_beam_indices[selection_positions]

        levels = np.arange(1, L + 1)

        # ============================================================
        # FIGURE 1: INDIVIDUAL REGRESSIONS
        # ============================================================

        fig, axes = plt.subplots(n_nodes, 2, figsize=(11, 3*n_nodes), sharex=True)
        axes = np.atleast_2d(axes)

        for row, node in enumerate(selected_nodes):

            # ---------------- ALPHA ----------------

            mean_diff = np.abs(del1[node, 1:L+1])
            mean_filtered = np.where(mean_diff > 1e-15, mean_diff, 1e-15)
            log_mean = np.log2(mean_filtered)

            pa = np.polyfit(levels, log_mean, 1)
            fitted_mean = np.polyval(pa, levels)

            ax = axes[row, 0]
            ax.plot(levels, log_mean, "o", markersize=4)
            ax.plot(levels, fitted_mean, "-", linewidth=1)
            ax.set_ylabel(r"$\log_2|\mathbb{E}[Y_\ell]|$")
            ax.set_title(f"Node {node}, {nodes_array[node]}\n" rf"$\alpha={alpha[node]:.3f}$")

            floored = mean_diff <= 1e-15
            if np.any(floored):
                ax.scatter(levels[floored], log_mean[floored], marker="x", s=60, label=r"floored to $10^{-15}$")
                ax.legend(fontsize=7)

            # ---------------- BETA ----------------

            variance = np.abs(var1[node, 1:L+1])
            variance_filtered = np.where(variance > 1e-15, variance, 1e-15)
            log_variance = np.log2(variance_filtered)

            pb = np.polyfit(levels, log_variance, 1)
            fitted_variance = np.polyval(pb, levels)

            ax = axes[row, 1]
            ax.plot(levels, log_variance, "o", markersize=4)
            ax.plot(levels, fitted_variance, "-", linewidth=1)
            ax.set_ylabel(r"$\log_2(\mathrm{Var}[Y_\ell])$")
            ax.set_title(f"Node {node}, {nodes_array[node]}\n" rf"$\beta={beta[node]:.3f}$")

            floored = variance <= 1e-15
            if np.any(floored):
                ax.scatter(levels[floored], log_variance[floored], marker="x", s=60, label=r"floored to $10^{-15}$")
                ax.legend(fontsize=7)

        axes[-1, 0].set_xlabel(r"MLMC level $\ell$")
        axes[-1, 1].set_xlabel(r"MLMC level $\ell$")
        fig.suptitle("Node-wise MLMC convergence regressions\n" r"left: $\alpha$, right: $\beta$")
        plt.tight_layout()
        plt.show()

        # ============================================================
        # FIGURE 2: ALPHA/BETA ALONG BEAM AXIS
        # ============================================================

        alpha_beam = alpha[beam_indices]
        beta_beam = beta[beam_indices]

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        axes[0].plot(beam_x, alpha_beam, ".-", linewidth=0.8, markersize=3)
        axes[0].axhline(0, linewidth=0.6, color="black")
        axes[0].scatter(nodes_array[selected_nodes, 0], alpha[selected_nodes], marker="x", s=50)
        axes[0].set_ylabel(r"$\alpha$")
        axes[0].set_title(rf"MLMC convergence rates along $y={beam_y_mesh:.3f}$")

        axes[1].plot(beam_x, beta_beam, ".-", linewidth=0.8, markersize=3)
        axes[1].axhline(0, linewidth=0.6, color="black")
        axes[1].scatter(nodes_array[selected_nodes, 0], beta[selected_nodes], marker="x", s=50)
        axes[1].set_ylabel(r"$\beta$")
        axes[1].set_xlabel("x")

        plt.tight_layout()
        plt.show()

        print("\nRepresentative nodes:")
        print("node        position              alpha       beta")

        for node in selected_nodes:
            print(f"{node:5d}   {str(nodes_array[node]):18s}   {alpha[node]:9.4f}   {beta[node]:9.4f}")

        return selected_nodes
    selected_nodes = plot_alpha_beta_diagnostics(n_nodes=6, beam_y=1.6)

    #Save the data
    npz_save_path = os.path.join(file_path,f"mlmc_test_data_{dose_method}_N_{N}_L_{L}_offset_{MLMC_LEVEL_OFFSET}.npz")
    np.savez(npz_save_path,nodes_array=nodes_array,mean_diff=del1,mean_fine=del2,var_diff=var1, var_fine=var2, kurtosis_diff=kur1,
            consistency_check=chk1,alpha=alpha,beta=beta,gamma=gamma,cost=cost,N=N,L=L,Lmin=Lmin,Lmax=Lmax,MLMC_LEVEL_OFFSET=MLMC_LEVEL_OFFSET,base=base)
    print(f"MLMC test data saved to {npz_save_path}")

    #This is a print to verify that MLMC is working as intended
    #Print out info from the maximal dose node 
    node = np.argmax(np.abs(del2[:, -1]))
    print("\n--- HIGH-DOSE NODE CONVERGENCE DIAGNOSTIC ---")
    print("node:", node, nodes_array[node])
    print("E[Pf] on finest level:", del2[node, -1]) #should be in the realm of 4-20 with > about 4k sims
    print("E[Pf-Pc] by level:", del1[node, :])
    print("Var(Pf-Pc) by level:", var1[node, :]) #should be reducing with level
    print("E[Pf] by level:", del2[node, :])

    printf(fp, '\n******************************************************\n')
    printf(fp, '*** Linear regression estimates of MLMC parameters ***\n')
    printf(fp, '******************************************************\n')
    printf(fp, ' gamma = %f  (exponent for MLMC cost) \n', gamma)
    
    #for k in range(num_q):
    #    name = f"NODE {k}, nodes_array[k]"
    #    printf(fp, '\n --- %s ---\n', name)
    #    printf(fp, ' alpha = %f  (exponent for MLMC weak convergence)\n', alpha[k])
    #    printf(fp, ' beta  = %f  (exponent for MLMC variance) \n', beta[k])

    #Only print the best+worst n_show nodes

    n_show = 5
    ranking = np.argsort(beta)
    #small beta = slowest variance decay
    worst_nodes = ranking[:n_show]
    #large beta = fastest variance decay 
    best_nodes = ranking[-n_show:][::-1]

    printf(fp, '\n')
    printf(fp, '******************************************************\n')
    printf(fp, '*** Best/worst MLMC nodes by beta                  ***\n')
    printf(fp, '******************************************************\n')

    printf(fp, '\nWorst nodes (smallest beta):\n')
    printf(fp, ' node       alpha        beta\n')

    for k in worst_nodes:
        printf(fp, '%5d   %11.4e  %11.4e\n', k, alpha[k], beta[k])

    printf(fp, '\nBest nodes (largest beta):\n')
    printf(fp, ' node       alpha        beta\n')

    for k in best_nodes:
        printf(fp, '%5d   %11.4e  %11.4e\n', k, alpha[k], beta[k])

    # Second, mlmc complexity tests
    printf(fp, '\n')
    printf(fp, '***************************** \n')
    printf(fp, '*** MLMC complexity tests *** \n')
    printf(fp, '***************************** \n\n')
    
    #Simply there are too many headers for one per node, reduce:    
    headers = "  eps       mlmc_cost     std_cost    savings       N_l\n"
    printf(fp, headers)
    printf(fp, "-" * len(headers) + "\n")
    
    #headers = "  eps      "
    #for k in range(num_q):
    #    headers += f"P_{'primal' if k==0 else f'greek{k}'}".ljust(12)
    #headers += "mlmc_cost   std_cost  savings     N_l \n"
    #printf(fp, headers)
    #printf(fp, "-" * (len(headers) + 15) + "\n")
    # Reset random number generator for complexity tests
    #np.random.seed(None) -- seed is assigned in the user function

    alpha_max = max(np.max(alpha), 0.5)
    beta_max = max(np.max(beta), 0.5)
    theta = 0.25

    for eps in Eps:
        # Dynamic unpacking of arbitrary lengths via index filtering slices
        results = mlmcv(mlmc_parallel_l, N0, eps, Lmin, Lmax, alpha_max, beta_max, gamma, *args)
        P_estimates = results[:-2]
        Nl = results[-2]
        Cl = results[-1]
        
        mlmc_cost = np.sum(Nl * Cl)
        
        idx = min(len(cost) - 1, len(Nl) - 1)
        var2_max = max([var2[k, idx] for k in range(num_q)])
        std_cost = var2_max * Cl[-1] / ((1.0 - theta) * eps**2)

        # Print outputs -- too many outputs
        #printf(fp, "%.3e ", eps)
        #for p in P_estimates:
        #    printf(fp, "%11.3e ", p)
        #    
        #printf(fp, " %.3e  %.3e  %7.2f ", mlmc_cost, std_cost, std_cost / mlmc_cost)
        # 
        #for n in Nl:
        #    printf(fp, "%10d ", n)
        #printf(fp, "\n")

        printf(fp,"%.3e   %.3e   %.3e   %7.2f    ",eps,mlmc_cost,std_cost,std_cost / mlmc_cost)
        for n in Nl:
            printf(fp, "%10d ", n)
        printf(fp, "\n")
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    printf(fp, '\n=================================================================\n')
    printf(fp, '>>> MLMC evaluation successfully completed in %.2f seconds.\n', elapsed)
    printf(fp, '=================================================================\n\n')

    printf(fp, '\n')