
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
dose_max=8.5

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
        title = "Track Length Model"
    elif method == "KZ" and dose_method=="shape_function":
        h = abs(dE)
        title = "Energy Model"
    elif method == "KZ" and dose_method=="spatial_kernel":
        h = abs(-ds * stopping_power(E0)/2)
        title = "Energy Model"
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

    folder_path = dp.file_path
    file_path = os.path.join(
        folder_path,
        f"TEST_{dose_method}_{method}_{h_rounded}_{spatial_dim}D_shape_{shape}_E0_{E0}_l_{l_rounded}_N_{sims_per_CPU*num_CPUs}.npz"
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
            print(f"l: {l_rounded}, {get_l} \nh: {h_rounded}, {get_h} \nmethod: {method}, {get_method}")
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

        if get_l != l or get_h != h_rounded or get_method != method or get_sigma != SIGMA:
            print(
                f"l: {l}, {get_l} \nh: {h_rounded}, {get_h} \nmethod: {method}, {get_method}\n"
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
    cf = ax_top.contourf(plot_X, plot_Y, field, levels=levels_filled, cmap='plasma')
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
    ax_top.set_title(f"Estimated Dose: {title}\n{dose_title}, (n={sim_num})")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(X_MIN, X_MAX)
    ax_top.set_ylim(Y_MIN, Y_MAX)

    ax_bottom.plot(x_curve, dose_curve, linewidth=2)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.set_xlabel("Depth x (cm)")
    ax_bottom.set_ylabel("Relative dose")
    ax_bottom.set_xlim(X_MIN, X_MAX)

    folder_save_path = dp.file_path
    file_save_path = os.path.join(
        folder_save_path,
        f"{dose_method}_{method}_{sim_num}_sims_E0_{E0}_h_{h}_l_{l}_N_{sims_per_CPU*num_CPUs}.png"
    )

    fig.savefig(file_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return field, x_curve, dose_curve, file_save_path

def dose_plot_3D(
    method,
    dose_method="spatial_kernel",
    av_width=0.1,
    save=True,
    curve_factor=8,
):
    """
    3D dose plotting function for the spatial-kernel method.

    Assumes:
        - beam direction is parallel to +x
        - beam axis lies at the midpoint of the y bounds
        - beam axis lies at the midpoint of the z bounds

    Produces:
        - top panel: x-y planar dose slice through the beam axis
                     at z = midpoint of z bounds
        - bottom panel: relative depth-dose curve along x, obtained by
                        averaging over a narrow y-band around the beam axis
        - colourbar matching the height of the top panel

    Returns
    -------
    field_slice : ndarray
        2D x-y dose slice through the beam axis.

    x_curve : ndarray
        Fine x coordinates for depth-dose curve.

    dose_curve : ndarray
        Relative depth-dose curve.

    file_save_path : str
        Path used for saving the figure.
    """

    if spatial_dim != 3:
        raise ValueError("dose_plot_3D requires a 3D dose map.")

    if dose_method != "spatial_kernel":
        raise ValueError(
            "This dose_plot_3D function currently supports "
            "dose_method='spatial_kernel' only."
        )

    # ------------------------------------------------------------
    # Choose discretisation step and title
    # ------------------------------------------------------------

    if method == "V":
        h = abs(ds)
        title = "Track Length Model"

    elif method == "KZ":
        h = abs(-ds * stopping_power(E0) / 2)
        title = "Energy Model"

    else:
        raise ValueError("method must be 'KZ' or 'V'.")

    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------

    l = dp.l
    SIGMA = dp.SIGMA

    l_rounded = round(l, 3)
    h_rounded = round(h, 3)

    dose_title = "Spatial Kernel Method"

    # ------------------------------------------------------------
    # Generate the 3D node grid
    # ------------------------------------------------------------

    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(l)

    shape = prod(X_meshgrid.shape)

    print(f"Number of nodes: {shape}")
    print(f"Grid shape: {X_meshgrid.shape}")

    # Coordinate vectors.
    #
    # Assumes nodes(l) uses meshgrid indexing consistent with:
    #
    #     field[ix, iy, iz]
    #
    x_vals = X_meshgrid[:, 0, 0]
    y_vals = Y_meshgrid[0, :, 0]
    z_vals = Z_meshgrid[0, 0, :]

    X_MIN = np.min(x_vals)
    X_MAX = np.max(x_vals)

    Y_MIN = np.min(y_vals)
    Y_MAX = np.max(y_vals)

    Z_MIN = np.min(z_vals)
    Z_MAX = np.max(z_vals)

    # ------------------------------------------------------------
    # Beam-axis position
    # ------------------------------------------------------------

    Y_BEAM = 0.5 * (Y_MIN + Y_MAX)
    Z_BEAM = 0.5 * (Z_MIN + Z_MAX)

    print(
        f"Beam axis:\n"
        f"    y = {Y_BEAM:.6g} cm\n"
        f"    z = {Z_BEAM:.6g} cm"
    )

    # ------------------------------------------------------------
    # Load dose result
    # ------------------------------------------------------------

    folder_path = dp.file_path

    file_path = os.path.join(
        folder_path,
        (
            f"TEST_{dose_method}_{method}_{h_rounded}_"
            f"{spatial_dim}D_shape_{shape}_"
            f"E0_{E0}_l_{l_rounded}_"
            f"N_{sims_per_CPU*num_CPUs}.npz"
        )
    )

    print(f"Loading:\n{file_path}")

    data = np.load(file_path)

    dose = data["dose_expected"]
    sim_num = data["sim_num"]
    get_l = data["l"]
    get_h = data["absolute_h"]
    get_method = data["method"]
    get_sigma = data["sigma"]

    # ------------------------------------------------------------
    # Check saved parameters
    # ------------------------------------------------------------

    if (
        get_l != l_rounded
        or get_h != h_rounded
        or get_method != method
        or get_sigma != SIGMA
    ):
        print(
            f"l:      expected {l_rounded}, loaded {get_l}\n"
            f"h:      expected {h_rounded}, loaded {get_h}\n"
            f"method: expected {method}, loaded {get_method}\n"
            f"sigma:  expected {SIGMA}, loaded {get_sigma}"
        )

        raise ValueError(
            "Retrieved parameters do not match file-name parameters."
        )

    # ------------------------------------------------------------
    # Convert and reshape dose to 3D field
    # ------------------------------------------------------------

    field_3d = np.asarray(
        dose_gy_convert(dose),
        dtype=float
    ).reshape(X_meshgrid.shape)

    # ------------------------------------------------------------
    # Find the z plane containing the beam axis
    # ------------------------------------------------------------

    z_idx = np.argmin(
        np.abs(z_vals - Z_BEAM)
    )

    z_slice = z_vals[z_idx]

    print(
        f"Beam-axis z coordinate = {Z_BEAM:.6g} cm\n"
        f"Using grid plane z     = {z_slice:.6g} cm "
        f"(index {z_idx})"
    )

    # Extract x-y slice
    field_slice = field_3d[:, :, z_idx]

    # Mesh coordinates for plotting this plane
    plot_X = X_meshgrid[:, :, z_idx]
    plot_Y = Y_meshgrid[:, :, z_idx]

    # ------------------------------------------------------------
    # Construct depth-dose curve
    # ------------------------------------------------------------

    # Find all y nodes within av_width of the beam axis.
    y_idx = np.where(
        np.abs(y_vals - Y_BEAM) <= av_width
    )[0]

    # If grid spacing is larger than av_width, use nearest node.
    if y_idx.size == 0:
        nearest_y_idx = np.argmin(
            np.abs(y_vals - Y_BEAM)
        )

        y_idx = np.array([nearest_y_idx])

    print(
        f"Depth-dose averaging over {len(y_idx)} y node(s), "
        f"within {av_width} cm of y = {Y_BEAM:.6g} cm."
    )

    # Average dose around beam axis
    coarse_line = np.mean(
        field_slice[:, y_idx],
        axis=1
    )

    # Fine x grid
    n_curve = max(
        curve_factor * len(x_vals),
        300
    )

    x_curve = np.linspace(
        X_MIN,
        X_MAX,
        n_curve
    )

    dose_curve = np.interp(
        x_curve,
        x_vals,
        coarse_line
    )

    # Normalise to maximum dose
    peak = np.nanmax(dose_curve)

    if np.isfinite(peak) and peak > 0:
        dose_curve = dose_curve / peak

    # ------------------------------------------------------------
    # Check field
    # ------------------------------------------------------------

    finite_vals = field_slice[
        np.isfinite(field_slice)
    ]

    if finite_vals.size == 0:
        raise ValueError(
            "Selected dose slice contains no finite values."
        )

    vmin = np.nanmin(field_slice)
    vmax = np.nanmax(field_slice)

    print(f"Maximum dose in slice: {vmax}")

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(9, 8)
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[20, 1],
        height_ratios=[3.0, 1.2],
        wspace=0.08,
        hspace=0.10
    )

    ax_top = fig.add_subplot(
        gs[0, 0]
    )

    ax_bottom = fig.add_subplot(
        gs[1, 0],
        sharex=ax_top
    )

    cax = fig.add_subplot(
        gs[0, 1]
    )

    # ------------------------------------------------------------
    # Filled dose contours
    # ------------------------------------------------------------

    levels_filled = 80

    cf = ax_top.contourf(
        plot_X,
        plot_Y,
        field_slice,
        levels=levels_filled,
        cmap="plasma"
    )

    # ------------------------------------------------------------
    # Contour lines
    # ------------------------------------------------------------

    if vmax > vmin:

        levels_lines = np.linspace(
            vmin,
            vmax,
            12
        )

        ax_top.contour(
            plot_X,
            plot_Y,
            field_slice,
            levels=levels_lines,
            colors="k",
            linewidths=0.6,
            alpha=0.35
        )

    # ------------------------------------------------------------
    # Beam axis
    # ------------------------------------------------------------

    ax_top.axhline(
        Y_BEAM,
        color="white",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )

    # ------------------------------------------------------------
    # Colourbar
    # ------------------------------------------------------------

    colorbar = fig.colorbar(
        cf,
        cax=cax,
        label="Dose (1 gigaproton, Gy)"
    )

    dose_ticks = np.linspace(
        vmin,
        vmax,
        10
    )

    colorbar.set_ticks(
        dose_ticks
    )

    colorbar.ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.2g")
    )

    # ------------------------------------------------------------
    # Top-panel formatting
    # ------------------------------------------------------------

    ax_top.set_ylabel(
        "y (cm)"
    )

    ax_top.set_title(
        f"Estimated Dose: {title}\n"
        f"{dose_title}, "
        f"z = {z_slice:.3g} cm, "
        f"(n={sim_num})"
    )

    ax_top.tick_params(
        labelbottom=False
    )

    ax_top.set_xlim(
        X_MIN,
        X_MAX
    )

    ax_top.set_ylim(
        Y_MIN,
        Y_MAX
    )

    # ------------------------------------------------------------
    # Bottom depth-dose plot
    # ------------------------------------------------------------

    ax_bottom.plot(
        x_curve,
        dose_curve,
        linewidth=2
    )

    ax_bottom.grid(
        True,
        alpha=0.3
    )

    ax_bottom.set_xlabel(
        "Depth x (cm)"
    )

    ax_bottom.set_ylabel(
        "Relative dose"
    )

    ax_bottom.set_xlim(
        X_MIN,
        X_MAX
    )

    ax_bottom.set_ylim(
        bottom=0
    )

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------

    folder_save_path = dp.file_path

    file_save_path = os.path.join(
        folder_save_path,
        (
            f"{dose_method}_{method}_{sim_num}_sims_"
            f"E0_{E0}_h_{h_rounded}_l_{l_rounded}_"
            f"N_{sims_per_CPU*num_CPUs}_"
            f"central_z_slice.png"
        )
    )

    if save:
        fig.savefig(
            file_save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return (
        field_slice,
        x_curve,
        dose_curve,
        file_save_path
    )

if __name__ == "__main__":
    if spatial_dim==2:
        plot = dose_plot_2D(method, dose_method)
    if spatial_dim==3:
        plot = dose_plot_3D(method, dose_method)


    


    



