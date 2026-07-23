
#This file will plot and store the results from the dose MC


import numpy as np
from math import prod
import matplotlib.pyplot as plt 
from dosesetup import *
import doseparams as dp
from dosemap2_spatial_kernel_parallel_EM_scheme import l, SIGMA

#We require the parameters to load the correct file 

dose_method = "spatial_kernel"

method = dp.METHOD
ds = dp.ds
dE = dp.dE
spatial_dim = dp.SPATIAL_DIM
E0 = dp.E0
sims_per_CPU = dp.SIMS_PER_CPU
num_cpus = dp.NUM_CPUs

def dose_gy_convert(dose_expected):
    """
    Convert expected dose output from Monte Carlo to Gy, measured for one gigaproton.  
    """
    if type(dose_expected) != np.ndarray:
        raise TypeError("dose_expected must be an array.")
    
    giga_proton=1e9
    constant = 1.6*1e-10*giga_proton #Directly from V's code 
    return dose_expected * constant

def dose_plot_2D(method, dose_method, n_points=120):
    """
    Provide the dose method and (scheme) method, return and save a plot in 2D only. 
    Dose method is either "spatial_kernel" or "shape_function"
    Method is either "KZ" or "V"
    """
    if spatial_dim != 2:
        raise ValueError("dose_plot_2D requires a 2D dose map.")

    if method == 'V':
        h= abs(ds)
        title = "Path Length Scheme"
    if method =='KZ':
        h= abs(dE) 
        #Overwrite l
        l = choose_l()
        title = "Energy Scheme"

    X_meshgrid, Y_meshgrid = nodes(l)
    node_coords = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)   #"list" of nodes 
    shape = prod(X_meshgrid.shape)
    l=round(l,3)
    h=round(h,3)
    X_MIN, X_MAX = X_meshgrid[0,0], X_meshgrid[-1,-1]
    Y_MIN, Y_MAX = Y_meshgrid[0,0], Y_meshgrid[-1,-1]

    data = np.load(f"{dose_method}_{method}_{h}_{spatial_dim}D_shape_{shape}_E0_{E0}_l_{l}.npz")

    if dose_method == "spatial_kernel":
        dose_title="Trilinear Basis Function Method"

        coeffs = data['coeffs_expected']
        sim_num = data['sim_num']
        get_l = data['l']
        get_h = data['absolute_h']
        get_method = data['method']

        if get_l != l or get_h != h or get_method != method:
            print(f"l: {l}, {get_l} \nh: {h}, {get_h} \nmethod: {method}, {get_method}")
            raise ValueError("retrieved parameters do not match file name parameters.")

        if len(node_coords) != len(coeffs):
                raise ValueError(
                    f"Length mismatch: coeffs has length {len(coeffs)}, "
                    f"but meshgrid has {len(node_coords)} nodes."
                )
        x = np.linspace(X_MIN, X_MAX, n_points)
        y = np.linspace(Y_MIN, Y_MAX, n_points)
        Xg, Yg = np.meshgrid(x, y, indexing="ij")

        U = np.zeros_like(Xg, dtype=float)

        for i in range(n_points):
            for j in range(n_points):
                X = np.array([Xg[i, j], Yg[i, j]], dtype=float)
                val = 0.0
                for c, n_i in zip(coeffs, node_coords):
                    val += c * Phi(X, n_i)
                U[i, j] = val

        plt.figure(figsize=(7, 5))
        plt.contourf(Xg, Yg, U, levels=50)
        plt.colorbar(label="u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Estimated Dose: {title} \n{dose_title}, (n={sim_num})")
        plt.tight_layout()
        plt.savefig(f"{dose_method}_{method}_dose_MC_{sim_num}_sims_E0_{E0}_h_{h}_l_{l}.png", dpi=300)
        plt.show()
        return U
        
    elif dose_method == "shape_function":
        dose_title = "Spatial Kernel Method"

        dose = data['dose_expected']
        sim_num = data['sim_num']
        get_l = data['l']
        get_h = data['absolute_h']
        get_method = data['method']
        get_sigma = data['sigma']

        if get_l != l or get_h != h or get_method != method or get_sigma != SIGMA:
            print(f"l: {l}, {get_l} \nh: {h}, {get_h} \nmethod: {method}, {get_method}\nsigma: {SIGMA}, {get_sigma}")
            raise ValueError("retrieved parameters do not match file name parameters.")

        dose_plot = dose_gy_convert(dose)
        U = np.asarray(dose_plot, dtype=float).reshape(X_meshgrid.shape)

        plt.figure(figsize=(7, 5))
        plt.contourf(X_meshgrid, Y_meshgrid, U, levels=50)
        plt.colorbar(label="Smoothed dose")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Estimated Dose {title} \n{dose_title}, (n={sim_num})")
        plt.tight_layout()
        plt.savefig(f"{dose_method}_{method}_dose_MC_{sim_num}_sims_E0_{E0}_h_{h}_l_{l}.png", dpi=300)
        plt.show()
        return U


if __name__ == "__main__":
    plot = dose_plot_2D(method, dose_method)


    



