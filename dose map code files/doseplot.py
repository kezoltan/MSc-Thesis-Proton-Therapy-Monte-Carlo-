
#This file will plot and store the results from the dose MC


import numpy as np
from math import prod
import matplotlib.pyplot as plt 
from dosesetup import *
from doseparams import *
import os
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor

def dose_gy_convert(dose_expected):
    """
    Convert expected dose output from Monte Carlo to Gy, measured for one gigaproton.  
    """
    if type(dose_expected) != np.ndarray:
        raise TypeError("dose_expected must be an array.")
    
    giga_proton=1e9
    constant = 1.6*1e-10*giga_proton #Directly from V's code 
    return dose_expected * constant

def _beam_axis_band_indices_2d(y_vals, y0=None, halfwidth_cm=0.1):
    """
    Return indices of y-values within +/- halfwidth_cm of y0.
    """
    if y0 is None:
        y0 = 0.5 * (y_vals[0] + y_vals[-1])

    idx = np.where(np.abs(y_vals - y0) <= halfwidth_cm)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(y_vals - y0)))], dtype=int)

    return idx

def _depth_dose_shape_function_fine(
    coeffs,
    node_coords,
    l,
    x_min,
    x_max,
    y_vals,
    av_width=0.1,
    n_x=600,
    n_y=31,
    relative=True,
):
    """
    Fine depth-dose curve for the shape_function method by evaluating the basis
    directly on a finer x grid.
    """
    y0 = 0.5 * (y_vals[0] + y_vals[-1])
    y_band = np.linspace(y0 - av_width, y0 + av_width, n_y)
    x_fine = np.linspace(x_min, x_max, n_x)

    proxy_line = np.zeros_like(x_fine, dtype=float)

    for i, xv in enumerate(x_fine):
        vals = np.zeros(n_y, dtype=float)
        for j, yv in enumerate(y_band):
            X = np.array([xv, yv], dtype=float)
            val = 0.0
            for c, n_i in zip(coeffs, node_coords):
                val += c * Phi(l, X, n_i)
            vals[j] = val
        proxy_line[i] = np.mean(vals)

    dose_line = np.asarray(dose_gy_convert(proxy_line), dtype=float)

    if relative:
        peak = np.max(dose_line)
        if peak > 0:
            dose_line = dose_line / peak

    return x_fine, dose_line

def _eval_phi_point(args):
    X, coeffs, node_coords, l = args
    val = 0.0
    for c, n_i in zip(coeffs, node_coords):
        val += c * Phi(l, X, n_i)
    return val

def compute_U(Xg, Yg, coeffs, node_coords, l, max_workers=None):
    points = [np.array([Xg[i, j], Yg[i, j]], dtype=float) for i in range(Xg.shape[0]) for j in range(Xg.shape[1])]
    args = [(X, coeffs, node_coords, l) for X in points]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        values = executor.map(_eval_phi_point, args)

    return np.asarray(list(values), dtype=float).reshape(Xg.shape)

def _depth_dose_spatial_kernel_fine(field,x_vals,y_vals,av_width=0.1,n_x=600,relative=True):
    """
    Fine depth-dose curve for the spatial_kernel method by averaging on the coarse grid, then interpolating to a finer x grid.
    """
    y_idx = _beam_axis_band_indices_2d(y_vals, halfwidth_cm=av_width)
    coarse_line = np.mean(field[:, y_idx], axis=1)

    x_fine = np.linspace(x_vals.min(), x_vals.max(), n_x)
    dose_fine = np.interp(x_fine, x_vals, coarse_line)

    if relative:
        peak = np.max(dose_fine)
        if peak > 0:
            dose_fine = dose_fine / peak

    return x_fine, dose_fine

def dose_plot_2D(method, dose_method, path_3D, n_points=50, av_width=0.1, save=True, curve_factor=8):
    """
    2D dose plot with dose map + Bragg curve along beam axis 
    """
    if SPATIAL_DIM != 2:
        raise ValueError("dose_plot_2D requires a 2D dose map.")
    if method == "V":
        title = "Track Length Model"
    elif method == "KZ":
        title = "Energy Model"

    print(f"Number of nodes: {dose_shape}")
    X_MIN, X_MAX = X_meshgrid[0, 0], X_meshgrid[-1, -1]
    Y_MIN, Y_MAX = Y_meshgrid[0, 0], Y_meshgrid[-1, -1]

    full_file_path = path_3D
    data = np.load(full_file_path)

    if dose_method == "SF":

        dose_title = "Bilinear Basis Function Method"
        coeffs = data["coeffs_expected"]
        sim_num = data["sim_num"]
        node_coords = np.column_stack([arr.ravel() for arr in (X_meshgrid, Y_meshgrid)])
        x_vals = np.linspace(X_MIN, X_MAX, n_points)
        y_vals = np.linspace(Y_MIN, Y_MAX, n_points)
        Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")

        U = compute_U(Xg, Yg, coeffs, node_coords, l, n_points)

        #U = np.zeros_like(Xg, dtype=float)
        #for i in range(n_points):
        #    for j in range(n_points):
        #        X = np.array([Xg[i, j], Yg[i, j]], dtype=float)
        #        val = 0.0
        #        for c, n_i in zip(coeffs, node_coords):
        #            val += c * Phi(l, X, n_i)
        #        U[i, j] = val

        field = np.asarray(dose_gy_convert(U), dtype=float)

        n_curve = max(curve_factor * n_points, 300)
        x_curve, dose_curve = _depth_dose_shape_function_fine(coeffs=coeffs,node_coords=node_coords,l=l,x_min=X_MIN,x_max=X_MAX,y_vals=y_vals,av_width=av_width,n_x=n_curve,n_y=31,relative=True)
        plot_X = Xg
        plot_Y = Yg
        
    if dose_method=="SK":

        dose_title = "Spatial Kernel Method"
        dose = data["dose_expected"]
        sim_num = data["sim_num"]
        field = np.asarray(dose_gy_convert(dose), dtype=float).reshape(X_meshgrid.shape)
        x_vals = X_meshgrid[:, 0]
        y_vals = Y_meshgrid[0, :]
        n_curve = max(curve_factor * len(x_vals), 300)
        x_curve, dose_curve = _depth_dose_spatial_kernel_fine(field=field,x_vals=x_vals,y_vals=y_vals,av_width=av_width,n_x=n_curve,relative=True)
        plot_X = X_meshgrid
        plot_Y = Y_meshgrid
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 2,width_ratios=[20, 1],height_ratios=[3.0, 1.2],wspace=0.08,hspace=0.10)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[0, 1])

    finite_vals = field[np.isfinite(field)]
    if finite_vals.size == 0:
        raise ValueError("Dose field contains no finite values.")

    #Retrieve the extreme values
    vmin = np.nanmin(field)
    vmax = np.nanmax(field) #np.nanpercentile(field, 99.5)
    print(f"Maximum dose reads as {vmax}.")

    #if not np.isfinite(vmax) or vmax <= vmin:
    #    vmax = np.nanmax(field)

    #levels_filled = np.linspace(vmin, vmax, 60)
    levels_lines = np.linspace(vmin, vmax, 12)
    levels_filled = 80
    cf = ax_top.contourf(plot_X, plot_Y, field, levels=levels_filled, cmap='plasma')
    ax_top.contour(plot_X, plot_Y, field,levels=levels_lines,colors="k",linewidths=0.6,alpha=0.5)

    colorbar = fig.colorbar(cf, cax=cax, label="Dose (1 gigaproton, Gy)")
    dose_ticks = np.linspace(vmin, vmax, 10)
    colorbar.set_ticks(dose_ticks)
    #Report the labels
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))

    ax_top.set_ylabel("y")
    ax_top.set_title(f"Estimated Dose: {title}\n{dose_title}, (n={sim_num})")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(X_MIN, X_MAX)
    ax_top.set_ylim(Y_MIN, Y_MAX)

    ax_bottom.plot(x_curve, dose_curve, linewidth=2)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("Depth x (cm)")
    ax_bottom.set_ylabel("Relative dose")
    ax_bottom.set_xlim(X_MIN, X_MAX)

    file_save_path = path_3D[:-3] + "png"
    print(f'Plot is trying to save at location: {file_save_path}')
    fig.savefig(file_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return field, x_curve, dose_curve, file_save_path

def dose_plot_3D(method, dose_method, path_3D, n_points=50, av_width=0.1, curve_factor=8):
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

        field = np.asarray(dose_gy_convert(U), dtype=float)
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

        dose_curve = np.asarray(dose_gy_convert(dose_curve_raw), dtype=float)
        peak = np.nanmax(dose_curve)

        if np.isfinite(peak) and peak > 0:
            dose_curve = dose_curve / peak

        z_slice = Z_BEAM

    else:
        dose_title = "Spatial Kernel Method"
        dose = np.asarray(data["dose_expected"])
        sim_num = data["sim_num"]

        field_3d = np.asarray(dose_gy_convert(dose), dtype=float).reshape(grid_shape)
        exact_z = np.where(np.isclose(z_nodes, Z_BEAM, rtol=1e-10, atol=1e-12))[0]

        if exact_z.size > 0:
            z_idx = exact_z[0]
            field = field_3d[:, :, z_idx]
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
            field = (1.0 - weight) * field0 + weight * field1
            z_slice = Z_BEAM

            print(f"Interpolating between z = {z0:.6g} cm and z = {z1:.6g} cm")

        plot_X = X_meshgrid[:, :, 0]
        plot_Y = Y_meshgrid[:, :, 0]

        y_idx = np.where(np.abs(y_nodes - Y_BEAM) <= av_width)[0]

        if y_idx.size == 0:
            y_idx = np.array([np.argmin(np.abs(y_nodes - Y_BEAM))])

        coarse_line = np.mean(field[:, y_idx], axis=1)
        n_curve = max(curve_factor * len(x_nodes), 300)
        x_curve = np.linspace(X_MIN, X_MAX, n_curve)
        dose_curve = np.interp(x_curve, x_nodes, coarse_line)
        peak = np.nanmax(dose_curve)

        if np.isfinite(peak) and peak > 0:
            dose_curve = dose_curve / peak

    finite_vals = field[np.isfinite(field)]

    if finite_vals.size == 0:
        raise ValueError("Dose field contains no finite values.")

    vmin = np.nanmin(field)
    vmax = np.nanmax(field)

    print(f"Maximum dose reads as {vmax}.")

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[3.0, 1.2], wspace=0.08, hspace=0.10)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[0, 1])

    cf = ax_top.contourf(plot_X, plot_Y, field, levels=80, cmap="plasma")

    if vmax > vmin:
        levels_lines = np.linspace(vmin, vmax, 12)
        ax_top.contour(plot_X, plot_Y, field, levels=levels_lines, colors="k", linewidths=0.6, alpha=0.35)

    ax_top.axhline(Y_BEAM, color="white", linestyle="--", linewidth=0.9, alpha=0.8)

    colorbar = fig.colorbar(cf, cax=cax, label="Dose (1 gigaproton, Gy)")
    colorbar.set_ticks(np.linspace(vmin, vmax, 10))
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

    ax_top.set_ylabel("y (cm)")
    ax_top.set_title(f"Estimated Dose: {title}\n{dose_title}, z = {z_slice:.3g} cm (n={sim_num})")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(X_MIN, X_MAX)
    ax_top.set_ylim(Y_MIN, Y_MAX)

    ax_bottom.plot(x_curve, dose_curve, linewidth=2)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("Depth x (cm)")
    ax_bottom.set_ylabel("Relative dose")
    ax_bottom.set_xlim(X_MIN, X_MAX)
    ax_bottom.set_ylim(bottom=0)

    file_save_path = path_3D[:-3] + "png"
    print(f'Plot is trying to save at location: {file_save_path}')
    fig.savefig(file_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return field, x_curve, dose_curve, file_save_path

#for this to work it requires you load the params you want

#if __name__ == "__main__": 
    #sampling_type='mc'
    #if method == "V":
    #    h = abs(ds)
    #elif method == "KZ":
    #    h = abs(dE)
    #path_3D = os.path.join(file_path, f"{sampling_type}_{dose_method}_{method}_{h}_{SPATIAL_DIM}D_shape_{dose_shape}_E0_{E0}_l_{l}_N_{SIMS_PER_CPU*NUM_CPUS}.npz") 
    #if SPATIAL_DIM==2:
    #    plot = dose_plot_2D(method, dose_method, path_3D)
    #if SPATIAL_DIM==3:
    #    plot = dose_plot_3D(method, dose_method, path_3D)


    


    



