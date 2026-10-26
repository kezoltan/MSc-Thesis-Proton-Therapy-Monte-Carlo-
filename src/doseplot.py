
#July 2026
#Code to create dose maps 

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import PowerNorm 
from dosesetup import *
from doseparams import NUM_CPUS, Y_meshgrid #subject to change -- should save ymeshgrid
from math import ceil
import os
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from concurrent.futures import ProcessPoolExecutor

#edit-- this code is inefficient + needs updating, parallelisation could be improved

#The number of interpolation points for the SF method, careful of increasing this 
num_points = ceil(l_reciprocal * max(Y_meshgrid.flatten())*2.5)  

def dose_gy_convert(dose_expected):
    """
    Convert expected dose output from Monte Carlo to Gy, measured for one gigaproton.  
    """
    if type(dose_expected) != np.ndarray:
        raise TypeError("dose_expected must be an array.")
    giga_proton=1e9
    constant = 1.6*1e-10*giga_proton #Directly from V's code 
    return dose_expected * constant

#Function to help plot a Bragg curve:
#   we would like to plot the dose along the central line    

def depth_dose_beam_axis(fine_dose_field, x_vals, y_vals, relative=True):
    """
    Extract the dose along the central y-line of a 2D dose fine_dose_field.

    fine_dose_field is already in Gy
    """
    y0 = 0.5 * (y_vals[0] + y_vals[-1]) #halfway point 
    y_idx = np.argmin(np.abs(y_vals - y0)) #locate points closest 
    y_coord = y_vals[y_idx] #coordinate along which we do the beamline
    dose_line = fine_dose_field[:, y_idx]
    if relative: #relative dose
        peak = np.max(dose_line)
        if peak > 0:
            dose_line = dose_line / peak
    return x_vals, dose_line, y_coord

#Vectorised Phi for SF method:

def evaluate_phi(args):
    X, coeffs, node_coords, l = args
    phis = Phi_vectorised(l, X, node_coords)
    return np.dot(coeffs, phis)

def compute_U(Xg, Yg, coeffs, node_coords, l, max_workers=None):
    if max_workers is None:
        max_workers = ceil(NUM_CPUS / 2)
    
    points = [np.array([Xg[i, j], Yg[i, j]], dtype=float) for i in range(Xg.shape[0]) for j in range(Xg.shape[1])]
    args = [(X, coeffs, node_coords, l) for X in points]

    #edit: Parallelisation may not be needed 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        values = executor.map(evaluate_phi, args)

    return np.asarray(list(values), dtype=float).reshape(Xg.shape)

#2D plotter 

def dose_map_2D(path_3D, n_points = num_points, hybrid=False):
    """
    Loads all the data for one path for one plot
    """
    #Load the data 
    full_file_path = path_3D
    data = np.load(full_file_path, allow_pickle=True)
    dose = data["dose_expected"]
    sim_num = data["sim_num"]
    method = data["method"]
    dose_method = data["dose_method"]
    l = data["l"]
    X_meshgrid = data["X_meshgrid"]
    #Y_meshgrid = data["Y_meshgrid"]
    #title_seg = data["title_seg"]
    #set up title labels
    SPATIAL_DIM = data["SPATIAL_DIM"]
    if dose_method=='SF':
        title_seg = f"{'Bilinear' if SPATIAL_DIM==2 else 'Trilinear'} Basis Function"
    elif dose_method=='SK':
        title_seg = "Spatial Kernel"
    sampling_type = data["sampling_type"]
    dose_shape = data["dose_shape"]

    if SPATIAL_DIM != 2:
        raise ValueError("dose_plot_2D requires a 2D dose map.")

    print(f"Number of nodes: {dose_shape}")
    X_MIN, X_MAX = X_meshgrid[0, 0], X_meshgrid[-1, -1] #bounds
    Y_MIN, Y_MAX = Y_meshgrid[0, 0], Y_meshgrid[-1, -1]

    if sampling_type=='mlmc':
        eps= data['accuracy'] #stricted eps is saved here
        step_lvls = data['step_levels']
        min_step_lvl = min(step_lvls) # =offset i think
        max_step_lvl = max(step_lvls)
        if hybrid:
            hybrid_dose=data['hybrid_dose_expected']
            num_neg_nodes=data["num_neg_nodes"]
            mc_replacements=data["mc_replacements"]
            print(f"MC replacements made: {mc_replacements}/{num_neg_nodes}.")

    if dose_method == "SF":

        dose_title = title_seg
        node_coords = np.column_stack([arr.ravel() for arr in (X_meshgrid, Y_meshgrid)])
        
        #Generate a fine(r) grid over which to plot the heat map 
        x_vals = np.linspace(X_MIN, X_MAX, n_points)
        y_vals = np.linspace(Y_MIN, Y_MAX, n_points)
        Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")
        U = compute_U(Xg, Yg, dose, node_coords, l)
        #Convert to Grays + store in 2D
        fine_dose_field = np.asarray(dose_gy_convert(U), dtype=float)
        x_curve, depth_dose_curve, y_beam = depth_dose_beam_axis(fine_dose_field, x_vals, y_vals, sampling_type)
        plot_X = Xg
        plot_Y = Yg
        if sampling_type=='mlmc' and hybrid: #store the hybrid data + plot too
            U_hybrid=compute_U(Xg, Yg, hybrid_dose, node_coords, l)
            hybrid_fine_dose_field = np.asarray(dose_gy_convert(U_hybrid), dtype=float)
            hybrid_x_curve, hybrid_depth_dose_curve, y_beam = depth_dose_beam_axis(hybrid_fine_dose_field, x_vals, y_vals)

    if dose_method=="SK":

        dose_title = title_seg
        fine_dose_field = np.asarray(dose_gy_convert(dose), dtype=float).reshape(X_meshgrid.shape)
        x_vals = X_meshgrid[:, 0]
        y_vals = Y_meshgrid[0, :]
        #This is to plot the Bragg curve
        x_curve, depth_dose_curve, y_beam = depth_dose_beam_axis(fine_dose_field=fine_dose_field,x_vals=x_vals,y_vals=y_vals)
        plot_X = X_meshgrid
        plot_Y = Y_meshgrid
        if sampling_type=='mlmc' and hybrid: #store the hybrid data + plot too
            hybrid_fine_dose_field = np.asarray(dose_gy_convert(hybrid_dose), dtype=float).reshape(X_meshgrid.shape)
            hybrid_x_curve, hybrid_depth_dose_curve, y_beam = depth_dose_beam_axis(fine_dose_field=hybrid_fine_dose_field,x_vals=x_vals,y_vals=y_vals)

    dose_data = {
        "x_curve": x_curve,
        "depth_dose_curve": depth_dose_curve,
        "fine_dose_field": fine_dose_field,
        "name": "",
        "plot_X": plot_X,
        "plot_Y": plot_Y,
        "y_beam": y_beam,

        # spatial bounds
        "X_MIN": X_MIN,
        "X_MAX": X_MAX,
        "Y_MIN": Y_MIN,
        "Y_MAX": Y_MAX,

        # metadata
        "sampling_type": sampling_type,
        "sim_num": sim_num,
        "method": method,
        "dose_method": dose_method,
        "dose_title": dose_title,

        # MLMC metadata -- None for MC
        "eps": eps if sampling_type == "mlmc" else None,
        "step_lvls": step_lvls if sampling_type == "mlmc" else None,
        "min_step_lvl": min_step_lvl if sampling_type == "mlmc" else None,
        "max_step_lvl": max_step_lvl if sampling_type == "mlmc" else None, 
        "mc_replacements": mc_replacements if hybrid and sampling_type == "mlmc" else None,
        "num_neg_nodes": num_neg_nodes if hybrid and sampling_type == "mlmc" else None}

    if sampling_type == "mlmc" and hybrid:
        dose_data["x_curve"] = hybrid_x_curve
        dose_data["depth_dose_curve"] = hybrid_depth_dose_curve
        dose_data["fine_dose_field"] = hybrid_fine_dose_field
        dose_data["name"] = "Hybrid" 
    return dose_data


def dose_plot_2D(path_3D_list, hybrid=False):
    """
    2D dose plot with dose map + Bragg curve along beam axis 
    """
    full_dose_data = []
    for path_3D in path_3D_list:
        plot_data = dose_map_2D(path_3D)

        if plot_data["sampling_type"] == "mlmc":
            full_dose_data.append(plot_data)
            full_dose_data.append(dose_map_2D(path_3D, hybrid=True))
        else:
            full_dose_data.append(plot_data)

    #Construct the figure
    num_cols=2
    num_plots = len(full_dose_data)
    num_cols = min(num_cols, num_plots)
    num_rows = ceil(num_plots / num_cols)

    vmin = min(np.nanmin(plot_data["fine_dose_field"]) for plot_data in full_dose_data)
    vmax = max(np.nanmax(plot_data["fine_dose_field"])for plot_data in full_dose_data)

    fig = plt.figure(figsize=(15*num_cols, 12* num_rows))
    height_ratios = [0.2] + [2.5, 0.9, 0.2] * (num_rows)
    gs = fig.add_gridspec(3*num_rows+1,num_cols,height_ratios=height_ratios,wspace=0.05,hspace=0.08)

    big_size = 27
    med_size = 24
    small_size = 22

    for i, plot_data in enumerate(full_dose_data):
        x_curve = plot_data["x_curve"]
        depth_dose_curve = plot_data["depth_dose_curve"]
        fine_dose_field = plot_data["fine_dose_field"]
        plot_X = plot_data["plot_X"]
        plot_Y = plot_data["plot_Y"]
        sampling_type = plot_data["sampling_type"]
        sim_num = plot_data["sim_num"]
        method = plot_data["method"]
        dose_title = plot_data["dose_title"]
        name = plot_data["name"]
        y_beam = plot_data["y_beam"]
        X_MIN = plot_data["X_MIN"]
        X_MAX = min(4.5, plot_data["X_MAX"]) #hardcode cut off for visual clarity
        Y_MIN = plot_data["Y_MIN"]
        Y_MAX = plot_data["Y_MAX"]
        if sampling_type == "mlmc":
            eps = plot_data["eps"]
            step_lvls = plot_data["step_lvls"]
            max_step_lvl = plot_data["max_step_lvl"]  
            min_step_lvl = plot_data["min_step_lvl"] 
            L_final = max_step_lvl - min_step_lvl
            num_neg_nodes = plot_data["num_neg_nodes"]
            mc_replacements = plot_data["mc_replacements"]

        #Resizing between the two plots 
        row = i // num_cols
        col = i % num_cols

        gs_row=3*row+1
        ax_top = fig.add_subplot(gs[gs_row, col])
        ax_bottom = fig.add_subplot(gs[gs_row + 1, col], sharex=ax_top)
        levels_lines = np.linspace(0.5, vmax, 15)
        levels_filled = np.linspace(0, vmax, 100)

        #Deal with any negatives - show as scatter not dose field
        negative_mask = fine_dose_field < 0
        dose_for_contour = np.maximum(0.0, fine_dose_field)

        #colornorm = PowerNorm(gamma=0.8, vmin=vmin, vmax=vmax)
        cf = ax_top.contourf(plot_X, plot_Y, dose_for_contour, levels=levels_filled, cmap='inferno') #, norm=colornorm
        ax_top.contour(plot_X, plot_Y, dose_for_contour,levels=levels_lines,colors="w",linewidths=0.8,alpha=1) #0.9

        if sampling_type=='mlmc':
            if np.count_nonzero(negative_mask) >= 1:
                print(f"Maximum magnitude of negative dose: {np.max(np.abs(fine_dose_field[negative_mask])):.3f}")
            ax_top.scatter(plot_X[negative_mask], plot_Y[negative_mask], color="#4C78A8", s=5, zorder=10, label=f"Negative Dose")
            ax_top.legend(fontsize=small_size-2, markerscale=2.5, framealpha=0.9, loc="upper left")

        sampling_name = "Monte Carlo" if sampling_type=="mc" else "MLMC and Hybrid MC-MLMC"
        sim_num_label = "M" if sampling_type=="mc" else r"$\sum M_\ell$"
        accuracy = "" if sampling_type=='mc' else f"RMSE {eps:.2f}, "
        step_lvls = "" if sampling_type=='mc' else f", L={L_final}"

        if method == "V":
            title = "Track Length Model"
        elif method == "KZ":
            title = "Energy Model"
        #Dose map
        if sampling_type=="mc":
            ax_top.set_title(f"{title}, {dose_title}", fontsize=med_size+2, pad=10)
            fig.suptitle(rf"Estimated Proton Dose: {sampling_name} ({sim_num_label}={sim_num:.0f}), Beamline: $y$={y_beam:.2f}cm", y=0.90, fontsize=big_size+2)
        elif sampling_type=="mlmc" and col % 2 == 0:
            ax_top.set_title(f"{dose_title}, {sim_num_label}={sim_num:.0f}{step_lvls}", fontsize=med_size+2, pad=10)
            fig.suptitle(rf"Estimated Proton Dose: {sampling_name}, {accuracy}Beamline: $y$={y_beam:.2f}cm", y=0.90, fontsize=big_size+2)
        elif sampling_type=="mlmc" and col % 2 != 0:
            ax_top.set_title(f"{name}, Successful Replacements: {mc_replacements}/{num_neg_nodes}", fontsize=med_size+2, pad=10)
            fig.suptitle(rf"Estimated Proton Dose: {sampling_name}, {accuracy}Beamline: $y$={y_beam:.2f}cm", y=0.90, fontsize=big_size+2)
        ax_top.tick_params(labelbottom=False)
        ax_top.set_xlim(X_MIN, X_MAX)
        ax_top.set_ylim(Y_MIN, Y_MAX)

        #Bragg curve
        ax_bottom.plot(x_curve, depth_dose_curve, linewidth=2, color='black')
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_xlim(X_MIN, X_MAX)

        if col == 0:
            ax_top.set_ylabel(r"$y$, cm", labelpad=22, fontsize=med_size)
            ax_bottom.set_ylabel("Relative dose", fontsize=med_size)
            ax_top.tick_params(axis="y", labelleft=True, labelsize=small_size)
            ax_bottom.tick_params(axis="y", labelleft=True, labelsize=small_size)
        else:
            ax_top.set_ylabel("")
            ax_bottom.set_ylabel("")
            ax_top.tick_params(axis="y", left=True, labelleft=False, labelsize=small_size)
            ax_bottom.tick_params(axis="y", left=True, labelleft=False, labelsize=small_size)

        # Keep x tick marks everywhere, but labels only on the bottom pair
        ax_top.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
        if row == num_rows - 1:
            ax_bottom.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize=small_size)
            ax_bottom.set_xlabel(r"Depth $x$, cm", fontsize=med_size)
            ax_bottom.xaxis.set_minor_locator(MultipleLocator(0.2))
            ax_bottom.tick_params(axis="x", which="minor")
        else:
            ax_bottom.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
            ax_bottom.set_xlabel("")
                
    plot_name=""
    if name=='Hybrid ' and hybrid:
        plot_name="_hybrid"

    colorbar = fig.colorbar(cf, ax=fig.axes, label="Dose (1 gigaproton, Gy)", pad=0.02)
    colorbar.ax.tick_params(labelsize=big_size)
    colorbar.set_label("Dose (1 gigaproton, Gy)", fontsize=big_size, labelpad=10)
    dose_ticks = np.linspace(max(0.0, vmin), vmax, 10)
    colorbar.set_ticks(dose_ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    if num_plots==1:
        file_save_path = path_3D[:-4] + plot_name + ".png"
        print(f'Plot is trying to save at location: {file_save_path}')
        fig.savefig(file_save_path, dpi=300, bbox_inches="tight")
    else:
        file_save_path = path_3D[:-4] + "_REPORT_PAIR" + plot_name + ".png"
        print(f'Plot is trying to save at location: {file_save_path}')
        fig.savefig(file_save_path, dpi=500, bbox_inches="tight")

    return fine_dose_field, x_curve, depth_dose_curve, file_save_path

#Needs update to combine this with the above, not implemented 
def dose_plot_3D(method, dose_method, path_3D, n_points=num_points, av_width=0.1, curve_factor=8):
    raise NotImplementedError("dose_plot_3D is incomplete and not vectorised, please edit before use.")
    
    if SPATIAL_DIM != 3:
        raise ValueError("dose_plot_3D requires SPATIAL_DIM == 3.")
    if method == "V":
        title = "Track Length Model"
    elif method == "KZ":
        title = "Energy Model"

    grid_shape = X_meshgrid.shape
    print(f"Number of nodes: {dose_shape}")
    print(f"Grid shape: {grid_shape}")

    x_nodes = np.asarray(X_meshgrid[:, 0, 0], dtype=float)
    y_nodes = np.asarray(Y_meshgrid[0, :, 0], dtype=float)
    z_nodes = np.asarray(Z_meshgrid[0, 0, :], dtype=float)

    X_MIN, X_MAX = np.min(x_nodes), np.max(x_nodes)
    Y_MIN, Y_MAX = np.min(y_nodes), np.max(y_nodes)
    Z_MIN, Z_MAX = np.min(z_nodes), np.max(z_nodes)

    Y_BEAM = 0.5 * (Y_MIN + Y_MAX)
    Z_BEAM = 0.5 * (Z_MIN + Z_MAX)

    print(f"Beam axis: y = {Y_BEAM:.6g} cm, z = {Z_BEAM:.6g} cm")

    file_path_full = path_3D
    print(f"Loading: {file_path_full}")
    data = np.load(file_path_full)

    if dose_method == "SF":
        dose_title = "Trilinear Basis Function Method"

        coeffs = np.asarray(data["coeffs_expected"], dtype=float)
        sim_num = data["sim_num"]
        node_coords = np.column_stack([X_meshgrid.ravel(), Y_meshgrid.ravel(), Z_meshgrid.ravel()])

        x_vals = np.linspace(X_MIN, X_MAX, n_points)
        y_vals = np.linspace(Y_MIN, Y_MAX, n_points)
        Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")

        U = compute_U(Xg, Yg, coeffs, node_coords, l, n_points)

        #U = np.zeros_like(Xg, dtype=float)
        #for i in range(n_points):
        #    for j in range(n_points):
        #        X_eval = np.array([Xg[i, j], Yg[i, j], Z_BEAM], dtype=float)
        #        val = 0.0
        #        for c, node in zip(coeffs, node_coords):
        #            val += c * Phi(l, X_eval, node)
        #        U[i, j] = val

        fine_dose_field = np.asarray(dose_gy_convert(U), dtype=float)
        plot_X = Xg
        plot_Y = Yg

        n_curve = max(curve_factor * n_points, 300)
        x_curve = np.linspace(X_MIN, X_MAX, n_curve)
        n_y_average = 31
        y_lower = max(Y_MIN, Y_BEAM - av_width)
        y_upper = min(Y_MAX, Y_BEAM + av_width)
        y_average_vals = np.linspace(y_lower, y_upper, n_y_average)
        dose_curve_raw = np.zeros(n_curve, dtype=float)

        for ix, x in enumerate(x_curve):
            dose_across_y = np.zeros(n_y_average, dtype=float)
            for iy, y in enumerate(y_average_vals):
                X_eval = np.array([x, y, Z_BEAM], dtype=float)
                val = 0.0
                for c, node in zip(coeffs, node_coords):
                    val += c * Phi(l, X_eval, node)
                dose_across_y[iy] = val
            dose_curve_raw[ix] = np.mean(dose_across_y)

        depth_dose_curve = np.asarray(dose_gy_convert(dose_curve_raw), dtype=float)
        peak = np.nanmax(depth_dose_curve)

        if np.isfinite(peak) and peak > 0:
            depth_dose_curve = depth_dose_curve / peak

        z_slice = Z_BEAM

    else:
        dose_title = "Spatial Kernel Method"
        dose = np.asarray(data["dose_expected"])
        sim_num = data["sim_num"]

        field_3d = np.asarray(dose_gy_convert(dose), dtype=float).reshape(grid_shape)
        exact_z = np.where(np.isclose(z_nodes, Z_BEAM, rtol=1e-10, atol=1e-12))[0]

        if exact_z.size > 0:
            z_idx = exact_z[0]
            fine_dose_field = field_3d[:, :, z_idx]
            z_slice = z_nodes[z_idx]
            print(f"Using exact z plane: z = {z_slice:.6g} cm")
        else:
            order = np.argsort(z_nodes)
            z_sorted = z_nodes[order]
            field_sorted = field_3d[:, :, order]
            upper_position = np.searchsorted(z_sorted, Z_BEAM)

            if upper_position == 0 or upper_position == len(z_sorted):
                raise ValueError("Beam-axis z coordinate lies outside the z-node range.")

            lower_position = upper_position - 1
            z0 = z_sorted[lower_position]
            z1 = z_sorted[upper_position]
            field0 = field_sorted[:, :, lower_position]
            field1 = field_sorted[:, :, upper_position]
            weight = (Z_BEAM - z0) / (z1 - z0)
            fine_dose_field = (1.0 - weight) * field0 + weight * field1
            z_slice = Z_BEAM

            print(f"Interpolating between z = {z0:.6g} cm and z = {z1:.6g} cm")

        plot_X = X_meshgrid[:, :, 0]
        plot_Y = Y_meshgrid[:, :, 0]

        y_idx = np.where(np.abs(y_nodes - Y_BEAM) <= av_width)[0]

        if y_idx.size == 0:
            y_idx = np.array([np.argmin(np.abs(y_nodes - Y_BEAM))])

        coarse_line = np.mean(fine_dose_field[:, y_idx], axis=1)
        n_curve = max(curve_factor * len(x_nodes), 300)
        x_curve = np.linspace(X_MIN, X_MAX, n_curve)
        depth_dose_curve = np.interp(x_curve, x_nodes, coarse_line)
        peak = np.nanmax(depth_dose_curve)

        if np.isfinite(peak) and peak > 0:
            depth_dose_curve = depth_dose_curve / peak

    finite_vals = fine_dose_field[np.isfinite(fine_dose_field)]

    if finite_vals.size == 0:
        raise ValueError("Dose fine_dose_field contains no finite values.")

    vmin = np.nanmin(fine_dose_field)
    vmax = np.nanmax(fine_dose_field)

    print(f"Maximum dose reads as {vmax}.")

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[3.0, 1.2], wspace=0.08, hspace=0.10)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[0, 1])

    cf = ax_top.contourf(plot_X, plot_Y, fine_dose_field, levels=80, cmap="plasma")

    if vmax > vmin:
        levels_lines = np.linspace(vmin, vmax, 12)
        ax_top.contour(plot_X, plot_Y, fine_dose_field, levels=levels_lines, colors="k", linewidths=0.6, alpha=0.35)

    ax_top.axhline(Y_BEAM, color="white", linestyle="--", linewidth=0.9, alpha=0.8)

    colorbar = fig.colorbar(cf, cax=cax, label="Dose (1 gigaproton, Gy)")
    colorbar.set_ticks(np.linspace(vmin, vmax, 10))
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

    ax_top.set_ylabel("y (cm)")
    ax_top.set_title(f"Estimated Dose: {title}\n{dose_title}, z = {z_slice:.3g} cm (n={sim_num})")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(X_MIN, X_MAX)
    ax_top.set_ylim(Y_MIN, Y_MAX)

    ax_bottom.plot(x_curve, depth_dose_curve, linewidth=2)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("Depth x (cm)")
    ax_bottom.set_ylabel("Relative dose")
    ax_bottom.set_xlim(X_MIN, X_MAX)
    ax_bottom.set_ylim(bottom=0)

    file_save_path = path_3D[:-3] + "png"
    print(f'Plot is trying to save at location: {file_save_path}')
    fig.savefig(file_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fine_dose_field, x_curve, depth_dose_curve, file_save_path

if __name__ == '__main__':

    #Set up file with names of data files inside 

    file_names = [os.path.join(file_path, "mlmc_results.txt")] #os.path.join(file_path, "mc_results.txt") 
    for file in file_names:
        with open(file, "r") as f:
            strings = [line.strip() for line in f]
            data_path=strings[0]
        path_3d_list = [os.path.join(data_path, path) for path in strings[1:]]
        dose_plot_2D(path_3d_list)
    

    


    



