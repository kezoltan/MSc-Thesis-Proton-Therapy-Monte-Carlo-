
#This file will plot and store the results from the dose MC


import numpy as np
from math import prod
import matplotlib.pyplot as plt 
from dosesetup import *
import doseparams as dp
import os
import matplotlib.ticker as mticker

#We require the dose method (and params) to load the correct file 
dose_method = "spatial_kernel"
#dose_method = "shape_function"

#Do not overwrite
method = dp.METHOD
ds = dp.ds
dE = dp.dE
spatial_dim = dp.SPATIAL_DIM
E0 = dp.E0
sims_per_CPU = dp.SIMS_PER_CPU
num_cpus = dp.NUM_CPUs
KAPPA = dp.KAPPA

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


def _depth_dose_spatial_kernel_fine(
    field,
    x_vals,
    y_vals,
    av_width=0.1,
    n_x=600,
    relative=True,
):
    """
    Fine depth-dose curve for the spatial_kernel method by averaging on the
    coarse grid, then interpolating to a finer x grid.
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


def dose_plot_2D(method, dose_method, n_points=50, av_width=0.1, save=True, curve_factor=8):
    """
    2D dose plot with:
      - top panel: heatmap + contour lines
      - bottom panel: fine depth-dose curve
      - no padding
      - aligned x-axes
      - colorbar only as tall as the top panel
    """
    if spatial_dim != 2:
        raise ValueError("dose_plot_2D requires a 2D dose map.")

    if method == "V":
        h = abs(ds)
        title = "Path Length Scheme"
    elif method == "KZ":
        h = abs(dE)
        title = "Energy Scheme"
    else:
        raise ValueError("method must be 'KZ' or 'V'.")

    if dose_method == "spatial_kernel":
        l = dp.l
        SIGMA = dp.SIGMA
    elif dose_method == "shape_function":
        l = dp.l
    else:
        raise ValueError("dose_method must be 'spatial_kernel' or 'shape_function'.")

    X_meshgrid, Y_meshgrid = nodes(l)
    shape = prod(X_meshgrid.shape)
    print(f"Number of nodes: {shape}")

    l_rounded = round(l, 3)
    h_rounded = round(h, 3)

    X_MIN, X_MAX = X_meshgrid[0, 0], X_meshgrid[-1, -1]
    Y_MIN, Y_MAX = Y_meshgrid[0, 0], Y_meshgrid[-1, -1]

    folder_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map results"
    file_path = os.path.join(
        folder_path,
        f"{dose_method}_{method}_{h_rounded}_{spatial_dim}D_shape_{shape}_E0_{E0}_l_{l_rounded}.npz"
    )
    data = np.load(file_path)

    if dose_method == "shape_function":
        dose_title = "Bilinear Basis Function Method"

        coeffs = data["coeffs_expected"]
        sim_num = data["sim_num"]
        get_l = data["l"]
        get_h = data["absolute_h"]
        get_method = data["method"]

        if get_l != l_rounded or get_h != h_rounded or get_method != method:
            print(f"l: {l}, {get_l} \nh: {h}, {get_h} \nmethod: {method}, {get_method}")
            raise ValueError("retrieved parameters do not match file name parameters.")

        node_coords = np.column_stack([arr.ravel() for arr in (X_meshgrid, Y_meshgrid)])
        if len(node_coords) != len(coeffs):
            raise ValueError(
                f"Length mismatch: coeffs has length {len(coeffs)}, "
                f"but meshgrid has {len(node_coords)} nodes."
            )

        x_vals = np.linspace(X_MIN, X_MAX, n_points)
        y_vals = np.linspace(Y_MIN, Y_MAX, n_points)
        Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")

        U = np.zeros_like(Xg, dtype=float)
        for i in range(n_points):
            for j in range(n_points):
                X = np.array([Xg[i, j], Yg[i, j]], dtype=float)
                val = 0.0
                for c, n_i in zip(coeffs, node_coords):
                    val += c * Phi(l, X, n_i)
                U[i, j] = val

        field = np.asarray(dose_gy_convert(U), dtype=float)

        n_curve = max(curve_factor * n_points, 300)
        x_curve, dose_curve = _depth_dose_shape_function_fine(
            coeffs=coeffs,
            node_coords=node_coords,
            l=l,
            x_min=X_MIN,
            x_max=X_MAX,
            y_vals=y_vals,
            av_width=av_width,
            n_x=n_curve,
            n_y=31,
            relative=True,
        )

        plot_X = Xg
        plot_Y = Yg

    else:
        dose_title = "Spatial Kernel Method"

        dose = data["dose_expected"]
        sim_num = data["sim_num"]
        get_l = data["l"]
        get_h = data["absolute_h"]
        get_method = data["method"]
        get_sigma = data["sigma"]

        if get_l != l or get_h != h or get_method != method or get_sigma != SIGMA:
            print(
                f"l: {l}, {get_l} \nh: {h}, {get_h} \nmethod: {method}, {get_method}\n"
                f"sigma: {SIGMA}, {get_sigma}"
            )
            raise ValueError("retrieved parameters do not match file name parameters.")

        field = np.asarray(dose_gy_convert(dose), dtype=float).reshape(X_meshgrid.shape)
        x_vals = X_meshgrid[:, 0]
        y_vals = Y_meshgrid[0, :]

        n_curve = max(curve_factor * len(x_vals), 300)
        x_curve, dose_curve = _depth_dose_spatial_kernel_fine(
            field=field,
            x_vals=x_vals,
            y_vals=y_vals,
            av_width=av_width,
            n_x=n_curve,
            relative=True,
        )

        plot_X = X_meshgrid
        plot_Y = Y_meshgrid

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[20, 1],
        height_ratios=[3.0, 1.2],
        wspace=0.08,
        hspace=0.10
    )

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
    cf = ax_top.contourf(plot_X, plot_Y, field, levels=levels_filled)
    ax_top.contour(
        plot_X, plot_Y, field,
        levels=levels_lines,
        colors="k",
        linewidths=0.6,
        alpha=0.35
    )

    colorbar = fig.colorbar(cf, cax=cax, label="Dose (1 gigaproton, Gy)")
    dose_ticks = np.linspace(vmin, vmax, 10)
    colorbar.set_ticks(dose_ticks)
    #Report the labels
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))

    ax_top.set_ylabel("y")
    ax_top.set_title(f"Estimated Dose: {title}\n{dose_title}, (n={sim_num}, $\\kappa$ = {KAPPA})")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(X_MIN, X_MAX)
    ax_top.set_ylim(Y_MIN, Y_MAX)

    ax_bottom.plot(x_curve, dose_curve, linewidth=2)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("Depth x (cm)")
    ax_bottom.set_ylabel("Relative dose")
    ax_bottom.set_xlim(X_MIN, X_MAX)

    folder_save_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map graphs"
    file_save_path = os.path.join(
        folder_save_path,
        f"{dose_method}_{method}_dose_and_depthdose_MC_{sim_num}_sims_E0_{E0}_h_{h}_l_{l}.png"
    )

    fig.savefig(file_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return field, x_curve, dose_curve, file_save_path

# def dose_plot_2D(method, dose_method, n_points=50):
#     """
#     Provide the dose method and (scheme) method, return and save a plot in 2D only. 
#     Dose method is either "spatial_kernel" or "shape_function"
#     Method is either "KZ" or "V"
#     """
#     if spatial_dim != 2:
#         raise ValueError("dose_plot_2D requires a 2D dose map.")

#     if method == 'V':
#         h= abs(ds)
#         title = "Path Length Scheme"
#     if method =='KZ':
#         h= abs(dE) 
#         title = "Energy Scheme"

#     if dose_method=="spatial_kernel":
#         l = dp.l
#         SIGMA = dp.SIGMA
#     if dose_method=="shape_function":
#         l=dp.l #Come back to this!!!!
#         #l = choose_l()

#     X_meshgrid, Y_meshgrid = nodes(l) 
#     shape = prod(X_meshgrid.shape)
#     print(f"Number of nodes: {shape}")
#     l_rounded=round(l,3)
#     h_rounded=round(h,3)
#     X_MIN, X_MAX = X_meshgrid[0,0], X_meshgrid[-1,-1]
#     Y_MIN, Y_MAX = Y_meshgrid[0,0], Y_meshgrid[-1,-1]

#     folder_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map results"
#     file_path = os.path.join(folder_path,f"{dose_method}_{method}_{h_rounded}_{spatial_dim}D_shape_{shape}_E0_{E0}_l_{l_rounded}.npz") 
#     data = np.load(file_path)

#     if dose_method == "shape_function":
#         node_coords = node_coords = np.column_stack([arr.ravel() for arr in (X_meshgrid, Y_meshgrid)])
#         dose_title="Bilinear Basis Function Method"

#         coeffs = data['coeffs_expected']
#         sim_num = data['sim_num']
#         get_l = data['l']
#         get_h = data['absolute_h']
#         get_method = data['method']

#         if get_l != l_rounded or get_h != h_rounded or get_method != method:
#             print(f"l: {l}, {get_l} \nh: {h}, {get_h} \nmethod: {method}, {get_method}")
#             raise ValueError("retrieved parameters do not match file name parameters.")

#         if len(node_coords) != len(coeffs):
#                 raise ValueError(
#                     f"Length mismatch: coeffs has length {len(coeffs)}, "
#                     f"but meshgrid has {len(node_coords)} nodes."
#                 )
#         x = np.linspace(X_MIN, X_MAX, n_points)
#         y = np.linspace(Y_MIN, Y_MAX, n_points)

#         #New meshgrid of all the x,y points, finer than before  
#         Xg, Yg = np.meshgrid(x, y, indexing="ij")
#         U = np.zeros_like(Xg, dtype=float)

#         for i in range(n_points):
#             for j in range(n_points):
#                 X = np.array([Xg[i, j], Yg[i, j]], dtype=float)
#                 val = 0.0
#                 for c, n_i in zip(coeffs, node_coords):
#                     val += c * Phi(l,X, n_i)
#                 U[i, j] = val

#         #I think! This just converts from 1 proton in MeV /g ->gigaprton in Gy
#         dose_plot = dose_gy_convert(U)

#         plt.figure(figsize=(7, 5))
#         #levels = np.linspace(0, np.percentile(dose_plot, 99.5), 80)
#         plt.contourf(Xg, Yg, dose_plot, levels=80)
#         plt.colorbar(label="u(x, y)")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.title(f"Estimated Dose: {title} \n{dose_title}, (n={sim_num})")
#         plt.tight_layout()
#         folder_save_path = folder_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map graphs"
#         file_save_path = os.path.join(folder_save_path, f"{dose_method}_{method}_dose_MC_{sim_num}_sims_E0_{E0}_h_{h}_l_{l}.png")
#         plt.savefig(file_save_path, dpi=300)
#         plt.show()

#         #This is plotting the Bragg curve

#         x_vals, dose_vals, axis_nodes = depth_dose_curve_from_file(
#                     file_path=file_path,
#                     dose_method=dose_method,
#                     method=method,
#                     l=l,
#                     X_meshgrid=X_meshgrid,
#                     Y_meshgrid=Y_meshgrid,
#                 )

#         return U
        
#     elif dose_method == "spatial_kernel":
#         dose_title = "Spatial Kernel Method"
#         node_coords = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)   #"list" of nodes

#         dose = data['dose_expected']
#         sim_num = data['sim_num']
#         get_l = data['l']
#         get_h = data['absolute_h']
#         get_method = data['method']
#         get_sigma = data['sigma']

#         if get_l != l or get_h != h or get_method != method or get_sigma != SIGMA:
#             print(f"l: {l}, {get_l} \nh: {h}, {get_h} \nmethod: {method}, {get_method}\nsigma: {SIGMA}, {get_sigma}")
#             raise ValueError("retrieved parameters do not match file name parameters.")

#         dose_plot = dose_gy_convert(dose)
#         U = np.asarray(dose_plot, dtype=float).reshape(X_meshgrid.shape)

#         plt.figure(figsize=(7, 5))
#         plt.contourf(X_meshgrid, Y_meshgrid, U, levels=50)
#         plt.colorbar(label="Smoothed dose")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.title(f"Estimated Dose {title} \n{dose_title}, (n={sim_num})")
#         plt.tight_layout()
#         folder_save_path = folder_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map graphs"
#         file_save_path = os.path.join(folder_save_path, f"{dose_method}_{method}_dose_MC_{sim_num}_sims_E0_{E0}_h_{h}_l_{l}.png")
#         plt.savefig(file_save_path, dpi=300)
#         plt.show()

#         x_vals, dose_vals, axis_nodes = depth_dose_curve_from_file(
#                     file_path=file_path,
#                     dose_method=dose_method,
#                     method=method,
#                     l=l,
#                     X_meshgrid=X_meshgrid,
#                     Y_meshgrid=Y_meshgrid,
#                     av_width=0.1,
#                     relative=True,
#                 )

#         return U

# #------------CHAT WROTE THIS 

# def _beam_axis_band_indices_2d(Y_meshgrid, y0=None, halfwidth_cm=0.1):
#     y_vals = Y_meshgrid[0, :]
#     if y0 is None:
#         y0 = 0.5 * (y_vals[0] + y_vals[-1])

#     idx = np.where(np.abs(y_vals - y0) <= halfwidth_cm)[0]
#     if idx.size == 0:
#         idx = np.array([int(np.argmin(np.abs(y_vals - y0)))], dtype=int)

#     return idx, y_vals[idx]


# def depth_dose_curve_from_file(
#     file_path,
#     dose_method,
#     method,
#     l,
#     X_meshgrid,
#     Y_meshgrid,
#     av_width=0.1,
#     relative=True,
#     title=None,
#     save_path=None,
# ):
#     data = np.load(file_path)

#     if dose_method == "shape_function":
#         field = np.asarray(data["coeffs_expected"], dtype=float).reshape(X_meshgrid.shape)
#         ylabel = "Relative dose" if relative else "Coefficient / dose proxy"
#     elif dose_method == "spatial_kernel":
#         field = np.asarray(dose_gy_convert(data["dose_expected"]), dtype=float).reshape(X_meshgrid.shape)
#         ylabel = "Relative dose" if relative else "Dose (Gy)"
#     else:
#         raise ValueError("dose_method must be 'shape_function' or 'spatial_kernel'.")

#     if spatial_dim != 2:
#         raise ValueError("This version is written for the 2D case.")

#     y_idx, y_band = _beam_axis_band_indices_2d(
#         Y_meshgrid,
#         halfwidth_cm=av_width
#     )

#     dose_line = np.mean(np.take(field, y_idx, axis=1), axis=1)

#     if relative:
#         peak = np.max(dose_line)
#         if peak > 0:
#             dose_line = dose_line / peak

#     x_vals = X_meshgrid[:, 0]

#     plt.figure(figsize=(7, 4))
#     plt.plot(x_vals, dose_line, linewidth=2)
#     plt.xlabel("Depth x (cm)")
#     plt.ylabel(ylabel)
#     if title is None:
#         title = f"Depth-dose curve ({method}, {dose_method})"
#     plt.title(title + "(averaged locally)")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)

#     plt.show()
#     return x_vals, dose_line, y_idx


if __name__ == "__main__":
    plot = dose_plot_2D(method, dose_method)

    


    



