
#August 2026
#Plotting functions for mlmc_test

import os
import sys
import numpy as np
from doseparams import MLMC_LEVEL_OFFSET, base, T, dose_method, L_conv_test, N_conv_test, file_path, SPATIAL_DIM, X_meshgrid, Y_meshgrid, dose_shape, nodes_array, l_reciprocal, EPS_0, KAPPA, method, E0, SIGMA, sampling_type, theta, width_sdev_factor, EMIN, title_seg #do not import l side length 
from doseparams import l as side_len
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from math import ceil, log
from dosemap1_shape_function_geoEM import storage_position_convention

def printf(file_ptr, fmt, *pargs, printout=True):
    """Helper to print to both stdout and a file pointer simultaneously."""
    text = fmt % pargs if pargs else fmt
    if printout:
        sys.stdout.write(text)
        sys.stdout.flush()
    if file_ptr is not None:
        file_ptr.write(text)
        file_ptr.flush()

def plot_kurtosis_variance_heatmap(L, kur1, var1, del1, del2, folder_path, log_kurt=False, level=None):
    print("Plotting kurtosis and beam axis statistics for MLMC estimator...")

    if level is None:
        raise ValueError("A single level must be specified.")
    if level < 1 or level > L:
        raise ValueError(f"level must be between 1 and {L}")

    kurt_all = kur1[:, 1:]
    var_all = var1[:, 1:]

    if log_kurt:
        kurt_plot = np.log10(1 + np.maximum(kurt_all, 0))
    else:
        kurt_plot = kurt_all

    kurt_vmin = np.nanmin(kurt_plot[:, level - 1])
    kurt_vmax = np.nanmax(kurt_plot[:, level - 1])

    ys = np.unique(Y_meshgrid.flatten())
    y_center = ceil(len(ys) / 2) * side_len
    center_mask = np.isclose(nodes_array[:, 1], y_center)
    x_center = nodes_array[center_mask, 0]
    order = np.argsort(x_center)

    kurt_line = kurt_all[center_mask, level - 1][order]
    var_line = var_all[center_mask, level - 1][order]
    payoff_diff_line = np.abs(del1[center_mask, level][order])
    payoff_line = del2[center_mask, level][order]

    kurt_norm = kurt_line / np.nanmax(kurt_line)
    var_norm = var_line / np.nanmax(var_line)
    payoff_diff_norm = payoff_diff_line / np.nanmax(payoff_diff_line)
    payoff_norm = payoff_line / np.nanmax(payoff_line)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    smallsize=14
    medsize=16
    bigsize=18

    sc_kurt = axes[0].scatter(nodes_array[:, 0], nodes_array[:, 1], c=kurt_plot[:, level - 1], vmin=kurt_vmin, vmax=kurt_vmax)
    axes[0].axhline(y_center, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[0].set_title("Estimated Kurtosis", fontsize=medsize)
    axes[0].set_xlabel(r"Depth $x$, cm", fontsize=medsize)
    axes[0].set_ylabel(r"$y$, cm", fontsize=medsize)
    axes[0].set_aspect("equal")
    axes[0].tick_params(axis="both", which="major", labelsize=smallsize)
    cbar = fig.colorbar(sc_kurt, ax=axes[0])
    cbar.ax.tick_params(labelsize=smallsize)
    #cbar.set_label("log10(1 + kurtosis)" if log_kurt else "Kurtosis", fontsize=8)

    axes[1].plot(x_center[order], kurt_norm, label=rf"Kurtosis, $\kappa_\ell$", color="C0", alpha=0.8)
    axes[1].plot(x_center[order], var_norm, label=rf"Variance, $\mathbb{{V}}[Q_{{\ell}}]$", color="C1", alpha=0.8)
    axes[1].plot(x_center[order], payoff_diff_norm, label=r"$|E[P^f_\ell-P^c_{\ell-1}]|$", color="C2", alpha=0.8)
    axes[1].plot(x_center[order], payoff_norm, label=r"$E[P^f_\ell]$", color="C3", alpha=0.8)
    axes[1].set_title(rf"Estimated Beam Axis Node Statistics ($y$={y_center:.2f})", fontsize=medsize)
    axes[1].set_xlabel(r"Depth $x$, cm", fontsize=medsize)
    axes[1].set_ylabel("Relative Value", fontsize=medsize)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axes[1].grid(True, which="major", axis="both", alpha=0.3)
    axes[1].grid(True, which="minor", axis="both", alpha=0.15)
    axes[1].tick_params(axis="both", which="major", labelsize=smallsize)
    axes[1].set_xlim(0, 4.5)
    axes[1].legend(loc="upper left", fontsize=smallsize)

    fig.canvas.draw()
    left = axes[0].get_position().x0
    right = axes[1].get_position().x1
    title_center = (left + right) / 2

    fig.suptitle(rf"{title_seg} MLMC Test Statistics ($\ell={level}$, M={N_conv_test})", fontsize=bigsize, x=title_center, ha="center")

    save_path = os.path.join(folder_path, f"{'log_' if log_kurt else ''}kurtosis_beamline_compare_level_{level}_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_consistency_error_heatmap(chk1, L, folder_path, log_error=False):
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
    fig.colorbar(sc, ax=axes[:L].tolist(), label=label, shrink=0.8)
    fig.suptitle(f"{title_seg} MLMC Test: {'Log ' if log_error else ''}Consistency Error Heatmap (M={N_conv_test})")
    save_path = os.path.join(folder_path, f"{'log_' if log_error else ''}consistency_error_map_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_fine_payoff(del2, N, folder_path):
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


def plot_alpha_beta_map(alpha, beta, folder_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    vmin = min(np.nanmin(alpha), np.nanmin(beta))
    vmax = max(np.nanmax(alpha), np.nanmax(beta))
    
    axes[0].scatter(nodes_array[:, 0],nodes_array[:, 1],c=alpha, vmin=vmin, vmax=vmax)
    axes[0].set_title(r"$\alpha$", fontsize=14)
    axes[0].set_xlabel(r"Depth $x$, cm", fontsize=12)
    axes[0].set_ylabel(r"$y$, cm", fontsize=12)
    axes[0].set_aspect("equal")
    axes[0].tick_params(axis="both", which="major", labelsize=10)
    #fig.colorbar(sc_alpha, ax=axes[0], label=r"$\alpha$")

    sc_beta = axes[1].scatter(nodes_array[:, 0],nodes_array[:, 1],c=beta, vmin=vmin, vmax=vmax)
    axes[1].set_title(r"$\beta$", fontsize=13)
    axes[1].set_xlabel(r"Depth $x$, cm", fontsize=12)
    axes[1].tick_params(axis="both", which="major", labelsize=10)
    #axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    #Shared colour bar
    cbar = fig.colorbar(sc_beta, ax=axes, label=r"Estimated Value", shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Convergence Rate", fontsize=10)
    fig.suptitle(rf"{title_seg} MLMC Test: $\alpha$, $\beta$ Estimation (M={N_conv_test})", y=0.97, fontsize=14)
    ab_save_path = os.path.join(folder_path,f"alpha_beta_map_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
    plt.savefig(ab_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    #Beam axis plot showing how they vary 
    ys = np.unique(Y_meshgrid.flatten())
    y_center = ceil(len(ys)/2)*side_len
    center_mask = np.isclose(nodes_array[:, 1], y_center)
    x_center = nodes_array[center_mask, 0]
    order = np.argsort(x_center)

    fig_line, ax_line = plt.subplots(figsize=(8, 3))
    ax_line.plot(x_center[order], alpha[center_mask][order], label=r"$\alpha$", alpha=0.7)
    ax_line.plot(x_center[order], beta[center_mask][order], label=r"$\beta$", alpha=0.7)
    ax_line.set_xlabel(r"Depth $x$, cm", fontsize=10)
    ax_line.set_ylabel("Estimated Value", fontsize=10)
    ax_line.set_title(rf"{title_seg} MLMC Test: Beam Axis Convergence Estimates (M={N_conv_test}, $y$={y_center})", fontsize=12)
    ax_line.legend(fontsize=11)
    #Integer ticks + gridlines
    ax_line.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_line.yaxis.set_major_locator(MultipleLocator(1))
    ax_line.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_line.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax_line.grid(True, which="major", axis='both', alpha=0.3)
    ax_line.grid(True, which="minor", axis='both', alpha=0.3)
    ax_line.tick_params("both", which="major", labelsize=10)
    plt.tight_layout()

    ab_line_save_path = os.path.join(folder_path, f"alpha_beta_centerline_{dose_method}_maxlvl_{L_conv_test}_offset_{MLMC_LEVEL_OFFSET}_mcsims_{N_conv_test}.png")
    plt.savefig(ab_line_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_mlmc_test(named_nodes, plot_keyword, L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path, y_coord=None):
    """
    Produce the standard report style mlmc plots. 
    Named nodes should be in the order you want it to come out from left to right 
    """
    if SPATIAL_DIM!=2:
        raise NotImplementedError(f"alpha/beta regression plots not implemented in {SPATIAL_DIM}D.")     

    #Sort in increasing x
    named_nodes.sort(key=lambda x: nodes_array[x[0]][0]) 
    nodes = [x[0] for x in named_nodes]

    l_ticks = np.arange(0, L + 1)
    if y_coord is None:
        y_coord=""
    else:
        y_coord = rf"$y$={y_coord:.2f} "

    #Print the raw data into the file 
    output_txt_path = os.path.join(folder_path, f"plot_mlmc_raw_regression_data_N_{N_conv_test}.txt")
    file_ptr = open(output_txt_path, "a")

    printf(file_ptr, "\n" + "=" * 156 + "\n", printout=False)
    printf(file_ptr, "RAW MLMC DATA AT SELECTED NODES%s\n", y_coord, printout=False)
    printf(file_ptr, "=" * 156 + "\n", printout=False)

    printf(file_ptr, "\n" + "=" * 156 + "\n", printout=False)
    printf(file_ptr, "%s\n", plot_keyword.upper() + " NODES", printout=False)
    printf(file_ptr, "=" * 156 + "\n", printout=False)

    for node, name in named_nodes:
        printf(file_ptr,"\nNode %d: %s  %s\n", node, str(nodes_array[node]), f"({name})" if name is not None else "", printout=False)
        printf(file_ptr, "alpha = %.12e\n", alpha[node], printout=False)
        printf(file_ptr, "beta  = %.12e\n", beta[node], printout=False)
        printf(file_ptr, "%8s %20s %20s %25s %20s %25s %20s\n", "level", "P_l - P_{l-1}", "P_l", "Var(P_l - P_{l-1})", "Var(P_l)", "Kurt(P_l - P_{l-1})", "Consistency", printout=False)
        printf(file_ptr, "-" * 156 + "\n", printout=False)

        for j, level in enumerate(l_ticks):
            printf(file_ptr, "%8d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e\n", level, del1[node, j], del2[node, j], var1[node, j], var2[node, j], kur1[node, j], chk1[node, j], printout=False)
    file_ptr.close()

    fig, axes = plt.subplots(6, len(nodes), figsize=(2.5 * len(nodes),10),sharex=True, constrained_layout=True, gridspec_kw={'height_ratios': [0.5, 0.5, 0.8, 0.8, 0.5, 0.5]})

    axes[0, 0].set_ylabel(r"$P^f_\ell$", fontsize=10)
    axes[1, 0].set_ylabel(r"$P^f_\ell - P^c_{\ell-1}$", fontsize=10)
    axes[2, 0].set_ylabel(r"$\log_2 |mean|$", fontsize=10)
    axes[3, 0].set_ylabel(r"$\log_2 variance$", fontsize=10)
    axes[4, 0].set_ylabel("Kurtosis", fontsize=10)
    axes[5, 0].set_ylabel("Consistency Check", fontsize=10)
    
    for i, (node, name) in enumerate(named_nodes):
        #Show which node is being plotted + why
        name_label=f" ({name})" if name != None else ""
        axes[0, i].set_title(f"Node {nodes_array[node]}\n{name_label}")
        axes[2, i].set_title(rf"$\alpha={alpha[node]:.3f}$")
        axes[3, i].set_title(rf"$\beta={beta[node]:.3f}$") 

        mean_data = np.abs(del1[node, 1:])
        payoff_diff = del1[node, 1:]
        raw_payoff = del2[node, :]
        payoff_data = np.abs(raw_payoff)
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

        kurt_data = kur1[node, 1:] 
        chk_data = chk1[node, 1:]

        axes[2, i].scatter(l_ticks[1:][~mean_masks], np.log2(mean_data[~mean_masks]))
        axes[2, i].scatter(l_ticks[1:][mean_masks], np.log2(mean_data[mean_masks]), marker='x')
        axes[2, i].scatter(l_ticks[~payoff_masks], np.log2(payoff_data[~payoff_masks]), color='red')
        axes[2, i].scatter(l_ticks[payoff_masks], np.log2(payoff_data[payoff_masks]), marker='+', color='gray')
        axes[2, i].ticklabel_format(axis='y', style='plain', useOffset=False)

        axes[3, i].scatter(l_ticks[1:][~var_masks], np.log2(var_data[~var_masks]))
        axes[3, i].scatter(l_ticks[1:][var_masks], np.log2(var_data[var_masks]), marker='x')
        axes[3, i].scatter(l_ticks[~var2_masks], np.log2(var2_data[~var2_masks]), color='red')
        axes[3, i].scatter(l_ticks[var2_masks], np.log2(var2_data[var2_masks]), marker='+', color='gray')
        axes[3, i].ticklabel_format(axis='y', style='plain', useOffset=False)

        axes[2, i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axes[3, i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        axes[4, i].plot(l_ticks[1:], kurt_data, color='black')
        axes[5, i].plot(l_ticks[1:], chk_data, color='black')
        axes[1, i].plot(l_ticks[1:], payoff_diff, color='black')
        axes[0, i].plot(l_ticks, raw_payoff, color='black')

        #Draw on the regression
        if np.count_nonzero(~mean_masks) >= 3:
            mean_fit = np.polyfit(l_ticks[1:][~mean_masks], np.log2(mean_data[~mean_masks]), 1)
            axes[2, i].plot(l_ticks[1:], np.polyval(mean_fit, l_ticks[1:]), linestyle="-", label=rf'$P^f_\ell - P^c_{{\ell-1}}$')
        if np.count_nonzero(~var_masks) >= 3:
            var_fit = np.polyfit(l_ticks[1:][~var_masks], np.log2(var_data[~var_masks]), 1)
            axes[3, i].plot(l_ticks[1:], np.polyval(var_fit, l_ticks[1:]), linestyle="-", label=rf'$P^f_\ell - P^c_{{\ell-1}}$')

        payoff_fit = np.polyfit(l_ticks, np.log2(payoff_data), 1)
        var2_fit = np.polyfit(l_ticks, np.log2(var2_data), 1)

        axes[2, i].plot(l_ticks, np.polyval(payoff_fit, l_ticks), linestyle="--", label=rf'$P^f_\ell$', color='red')
        axes[3, i].plot(l_ticks, np.polyval(var2_fit, l_ticks), linestyle="--", label=rf'$P^f_\ell$', color='red')
        axes[2, 0].legend()
        axes[3, 0].legend()

        payoff_diffmin = np.min(del1[nodes, 1:])
        payoff_diffmax = np.max(del1[nodes, 1:])
        axes[1, i].set_ylim(payoff_diffmin, payoff_diffmax)

        payoff_max = np.max(del2)*1.05
        axes[0, i].set_ylim(-payoff_max*0.15, payoff_max)
        for k in range(6):
            axes[k, i].set_xticks(l_ticks)
            axes[k, i].grid(alpha=0.3)

    for k in range(len(nodes)):
        axes[-1, k].set_xlabel(r"MLMC Level $\ell$", fontsize=10)
    fig.suptitle(f"{title_seg} MLMC Test: {plot_keyword.title()} Statistics {y_coord}(M={N_conv_test})", fontsize=16)
    mlmc_plot_path=os.path.join(folder_path, f"{plot_keyword}_regression_data_N_{N_conv_test}")
    plt.savefig(mlmc_plot_path, dpi=300)
    plt.show()  
    return  

#Beam axis plot:
def plot_beam_axis_nodes(L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path):
    """
    Use the plot mlmc test function above, make sets of nodes here
    """
    ys = np.unique(Y_meshgrid.flatten())
    y_coord = ceil(len(ys)/2)*side_len

    beamline_mask = nodes_array[:,1] == y_coord
    beam_nodes = nodes_array[beamline_mask] 
    beam_nodes_numbers = np.flatnonzero(beamline_mask) #gives you the node numbers on the beamline
    payoff_fine = del2[beamline_mask,-1]
    
    #Select the beta node
    #At each level, find the beamline node with maximum V_l
    var_payoff_diff = var1[beamline_mask, 1:]
    max_var_node_indices = np.argmax(var_payoff_diff, axis=0)
    max_var_nodes = beam_nodes_numbers[max_var_node_indices]
    candidate_nodes = np.unique(max_var_nodes)
    print(f"Beamline beta candidate nodes: {nodes_array[candidate_nodes]}")
    node2 = candidate_nodes[np.nanargmin(beta[candidate_nodes])]

    #Select nodes
    node1 = beam_nodes_numbers[np.argmax(payoff_fine)]
    node3 = beam_nodes_numbers[np.nanargmax(alpha[beamline_mask])]
    node4 = beam_nodes_numbers[np.nanargmax(beta[beamline_mask])]

    same_node=False
    if node4 == node3: #likely this could happen
        nodes=[node1, node2, node3] 
        names=[r"Max $P_L$", r"$\beta$, Beamline" ,r"Max $\alpha_i$, $\beta_i$"] #r"Min $P_\ell$",
        same_node=True
    else:
        nodes=[node1,node2,node3,node4] #node2,
        names=[r"Max $P_L$", r"$\beta$, Beamline" ,r"Max $\alpha_i$", r"Max $\beta_i$"] #r"Min $P_\ell$",
    for k in range(1,6):
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
        print(f"\nNo nodes found along y = {y_coord} with negative estimator")
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

    #Remove repeats:
    seen = set()
    unique_nodes = []
    unique_names = []
    for node, name in zip(nodes, names):
        if node not in seen:
            seen.add(node)
            unique_nodes.append(node)
            unique_names.append(name)
    nodes = unique_nodes
    names = unique_names

    named_nodes = [(node_number,name) for node_number, name in zip(nodes, names)]
    plot_keyword="beamline"
    plot_mlmc_test(named_nodes, plot_keyword, L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path, y_coord)

def plot_negative_alpha_beta_nodes(L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path, max_nodes=8):
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
    print(f"\nTotal number of nodes with negative alpha or beta: {len(negative_nodes)}\n\n")

    # Sort negative nodes spatially in x and select an approximately even spread
    negative_nodes = negative_nodes[np.argsort(nodes_array[negative_nodes, 0])]
    if len(negative_nodes) > max_nodes:
        selection = np.linspace(0, len(negative_nodes) - 1, max_nodes, dtype=int)
        nodes = negative_nodes[selection]
    else:
        nodes = negative_nodes
    named_nodes = [(node_number,None) for node_number in nodes]

    plot_keyword="negative convergence rate"
    plot_mlmc_test(named_nodes, plot_keyword, L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path, None)

def plot_negative_estimator_nodes(L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path, max_nodes=8):

    mlmc_estimator = del2[:, 0] + np.sum(del1[:, 1:], axis=1)
    negative_nodes = np.flatnonzero(mlmc_estimator < 0)
    if len(negative_nodes) == 0:
        print(f"\nNo nodes found with negative estimator")
        return
    
    order = negative_nodes[np.argsort(mlmc_estimator[negative_nodes])]
    if len(order) <= max_nodes:
        nodes = order
    else:
        n = max_nodes // 2
        nodes = np.concatenate((order[:n], order[-n:]))

    names=[f"Y={mlmc_estimator[node_idx]:.2f}" for node_idx in nodes]
    named_nodes=[(node_number,name) for node_number, name in zip(nodes, names)]
    plot_keyword="negative estimator"
    plot_mlmc_test(named_nodes, plot_keyword, L, alpha, beta, del1, del2, var1, var2, kur1, chk1, trunc_tol, folder_path, None)
