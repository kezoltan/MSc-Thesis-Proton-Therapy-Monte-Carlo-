
#July 2026
#Spatial Kernel Dose Method Code

#This file is structured as follows
#   1. Parameters
#   2. Basic functions and other global object set up (+ load the mass matrix)
#   3. Dose function calculation with EM schemes 
#   4. Worker functions (for parallel)

#----------------IMPORTS-----------------------------------------------------

import numpy as np 
import os
from math import prod, exp, log, pi
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import doseparams as dp
from dosesetup import *
from scipy.special import ndtr 
from scipy.stats import chi2
from numpy.random import default_rng
from doseplot import dose_plot_2D

#--------------PARAMETERS----------------------------------------------------


#Specific params:
l=dp.l
SIGMA=dp.SIGMA
l_reciprocal = 1/l

#Imported/fixed params

method = dp.METHOD
spatial_dim = dp.SPATIAL_DIM #2D or 3D, both include the beam axis (x)
sims_per_CPU = dp.SIMS_PER_CPU #Update M if you change this!!!
num_CPUs = dp.NUM_CPUs
E0 = dp.E0
eps_0 = dp.EPS_0
alpha = dp.ALPHA
p = dp.P
dE = dp.dE
E_min = dp.EMIN
ds = dp.ds
rho = dp.RHO
sims_per_CPU = dp.SIMS_PER_CPU
KAPPA = dp.KAPPA
range_allowance = dp.range_allowance
y_scaling = dp.y_scaling
origin = dp.origin
master_seed_seq = dp.master_seed_seq
Omega0 = choose_Omega_0()

if spatial_dim==3:
    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(l)
    meshgrid = X_meshgrid, Y_meshgrid, Z_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel(), Z_meshgrid.ravel()], axis=-1)
if spatial_dim==2:
    X_meshgrid, Y_meshgrid= nodes(l)
    meshgrid = X_meshgrid, Y_meshgrid
    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)

if rho <= 0:
    raise ValueError("density must be positive.")

_, domain_lower_bounds, domain_upper_bounds = choose_X0(meshgrid)

dose_shape = prod(X_meshgrid.shape) #More accurately this is a length but it works as a 1D shape
max_steps = sum(X_meshgrid.shape) #All boundaries, just needs to be large enough but relatively small

#------------------------------SPATIAL KERNEL----------------------------------------------

#Precompute these 
KERNEL_PREFACTOR = 1/rho * 1/((2*pi)**(spatial_dim/2)*SIGMA**spatial_dim)
allowed_sd = 4 #For the gaussian truncation 
MASS_RESCALING = chi2.cdf(allowed_sd**2, df=spatial_dim)

def spatial_kernel_vectorized(X_i, nodes_array, truncate=True, allowed_sd=4):
    """
    Gaussian d-dimensional (d=2 or 3) spatial kernel, vectorised for efficiency to operate on all nodes simulataneously. 
    Truncated by default to within min(allowed_sd, min_distance) standard deviations to ensure conservation
    the min_distance is the smallest distance to the furthest node over all directions
        this is needed for the boundary cases 
        i.e. the gaussian truncates to ensure all deposit is within the boundary 

    Inputs:
        X_i: np.ndarray, the current position of the EM scheme 
        nodes_array: array containing all node coords, in order aligned with meshgrid

    Returns:
        values: Given X_i, contains the value of the dose deposit contribution at all nodes
    """
    diff = nodes_array - X_i #Find all diffs at once 
    norm_sqrd = np.einsum('ij,ij->i', diff, diff) #Claude's suggestion, for speed via vectorisation 
                                                  #We compute all the sqrd norms for all nodes 
    nodal_values = np.zeros(norm_sqrd.shape[0]) #Shape is number of nodes 
    radius = allowed_sd *SIGMA
    mass_rescaling = MASS_RESCALING

    #If we are near to a boundary, we need a different truncation 
    #   If we exactly touch a boundary, we get no deposit due to rescaling. This is an issue at X0. 
    #   Non-physical to just cut off energy dispersion at the wall 
    #   Instead I move the wall backwards for he dispersion only (the actual scheme is still absorbed if it goes backwards)

    if spatial_dim==2:    
        xi,yi = X_i[0], X_i[1]
        x_lo, y_lo = domain_lower_bounds
        x_hi, y_hi = domain_upper_bounds
        x_lo -= allowed_sd*SIGMA #artificially move the line at zero by one sd, else we get no deposit here at all (due to rescaling)

        #Shortest distances to boundaries:
        boundary_dist = np.array([abs(xi-x_lo), abs(xi-x_hi), abs(yi-y_lo), abs(yi-y_hi)])

    elif spatial_dim==3:    
        xi,yi,zi = X_i[0], X_i[1], X_i[2]
        x_lo, y_lo, z_lo = domain_lower_bounds
        x_hi, y_hi, z_hi = domain_upper_bounds
        x_lo -= allowed_sd * SIGMA

        #Shortest distances to boundaries:
        boundary_dist = np.array([abs(xi-x_lo), abs(xi-x_hi), abs(yi-y_lo), abs(yi-y_hi), abs(zi-z_lo), abs(zi-z_hi)])

    if np.any(boundary_dist < allowed_sd*SIGMA):
        radius = np.min(boundary_dist) #Choose the nearest one 
        mass_rescaling = chi2.cdf((radius/SIGMA)**2, df=spatial_dim)

    if truncate: #We need to rescale if we truncate the Gaussian to a smaller region 
        mask = norm_sqrd <= (radius) ** 2 #If we are inside the ball

        #Rescaling + evaluating: 
        nodal_values[mask] = (KERNEL_PREFACTOR
                         * np.exp(-norm_sqrd[mask] / (2 * SIGMA ** 2))
                         / mass_rescaling)
    else:
        nodal_values = KERNEL_PREFACTOR * np.exp(-norm_sqrd / (2 * SIGMA ** 2))
    return nodal_values

#-----------------------DOSE CALCULATION W/ EM SCHEMES-----------------------------------

def one_step_dose_contribution(X_i, constant):
    return constant * spatial_kernel_vectorized(X_i, nodes_array)

def one_path_dose_contribution(method, rng, E0=E0, Omega0=Omega0, dE = dE, ds=ds, s0 = 0, kappa=KAPPA, a=alpha, p=p):
    """
    Estimates the dose due to one proton path using a Gaussian smoothing kernel, not truncated.
    Assumes a composite trapezium rule is used for path integral approximation.  
    
    Inputs:
        method = "KZ" or "V" to denote which scheme we are using

    Returns: 
        dose_contribution, np.ndarray, shape according to the number of nodes 
    """ 
    dose_contribution = np.zeros(dose_shape)
    
    s = float(s0)
    X = sample_initial_position(meshgrid, rng, l)
    Omega = Omega0
    E = initial_energy_spread()

    #If V method:
    Y = log(E)
  
    if method == "KZ": #Runs the scheme in independent energy
        base_h = dE
        n = ceil((E - E_min) / abs(base_h)) 
        Es = np.linspace(E, E_min, n + 1)
        h = Es[1] - Es[0]
        #print(f"base_h is reading as {base_h:.4f}, actual h is reading as {h}") 
        num_steps = len(Es) - 1 #Minus 1 because we don't need to repeat the IC 

        coeff_prefactor = np.sqrt(2*eps_0 * a*p) 
        sqrt = np.sqrt(-h) 
        constant = 1 #This is for the comp trap rule, does not change 

        #Generate Brownians ahead of time because known endpoints (+ no reflection)
        dB1 = sqrt * rng.standard_normal(num_steps) 
        dB2_3D = sqrt * rng.standard_normal((num_steps, spatial_dim))

        step_counter = 0
        domain_check = True

        for k in range(len(dB1)): #Known start and end points 
            
            #We will include step 0 in the comp trap
            one_step = one_step_dose_contribution(X, constant)
            if step_counter == 0:
                dose_contribution += abs(h)/2 * one_step 
            else:
                dose_contribution += abs(h)*one_step
            
            S_inv = reciprocal_stopping_power(E) 
            coeff = coeff_prefactor*E**(p/2 - 0.5)  
            Omega_n = Omega #need this for the exp map 
            instance_dB2 = dB2_3D[k] 
            other_noise = instance_dB2 - Omega_n * np.dot(Omega_n, instance_dB2)
            
            #Exp map
            y = coeff * other_noise
            Omega = exponential_map_sphere(Omega_n, y)

            #Update variables without storage
            s_n = s
            s = s_n - S_inv * h + n_KZ_path_length_diffusion(E) * dB1[k]
            path_travelled = s - s_n
            X = X + Omega * (path_travelled)
            E += h
            step_counter+= 1

            if np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds): #If we left the domain 
                domain_check = False    
                break

        #Do the final step contribution as well, assuming we were still in the domain:
        if domain_check:    
            one_step = one_step_dose_contribution(X, constant)
            dose_contribution += abs(h)/2 * one_step
        
    elif method == "V":
        h = ds
        sqrt = np.sqrt(h) #for constant step size (change if adaptive)
        coeff = np.sqrt(2*eps_0)
        constant = stopping_power(E) #For the comp trap

        domain_check = True
        step_counter = 0

        E_start = E
        X_start = X

        while E >= E_min: #Unknown end point 

            one_step = one_step_dose_contribution(X, constant)

            if step_counter == 0:
                dose_contribution += abs(h)/2 * one_step
            else: 
                dose_contribution += abs(h) * one_step
            
            #Sample each Gaussian
            dB1 = sqrt * rng.standard_normal() 
            dB2_3D = sqrt * rng.standard_normal(spatial_dim)
            Omega_n = Omega 
            other_noise = dB2_3D - Omega_n * np.dot(Omega_n, dB2_3D) # =(1-OmegaOmega)dB
            
            #Exp map
            y = coeff * other_noise
            Omega = exponential_map_sphere(Omega_n, y)

            #Update the other variables
            X = X + Omega * h
            Y = Y + h * n_V_log_energy_drift(E) + n_V_log_energy_diffusion(E) * dB1
            E = exp(Y)
            s += h
            step_counter+=1 

            E_end = E
            X_end = X

            constant = (-E_end+E_start)/h
            
            if np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds): #If we left the domain 
                domain_check = False
                break

            if E <= E_min: #Check this condition again after E has updated
                #Interpolate the end point up to Emin

                t_min = (E_min - E_start)/(E_end - E_start) #in 0,1
                X_end = X_start + (X_end-X_start)*t_min 
                E_end = E_min 

                h_final = np.linalg.norm(X_end - X_start)
                constant = (-E_end+ E_start)/h_final
                one_step = one_step_dose_contribution(X_start, constant)

                #The final dose contribution
                dose_contribution += h_final/2 * one_step
                break

            X_start = X
            E_start = E

        if not(domain_check): #If we exited the domain 
            #Go back to the last full step that was inside D 
            dose_contribution -= abs(h)/2 * one_step #Update the quadrature so that this was the endpoint  

    return dose_contribution


#-------------------PARALLEL FUNCTIONS--------------------------------------------------
    
def worker(method, path_seed_chunk, sims_per_CPU = sims_per_CPU):
    """
    This is the worker function telling each CPU what to do 
    Each CPU will run sims_per_CPU independent simulations, and add up the load vectors.

    Returns:
        sum of load vectors, one per sim: np.ndarray, 1D: (load_shape,)
    """
    worker_dose = np.zeros(dose_shape)
    for path_seed in path_seed_chunk:
        rng=default_rng(path_seed)
        dose_contribution = one_path_dose_contribution(method, rng=rng)
        worker_dose += dose_contribution
    return worker_dose

def expected_dose(method, sims_per_CPU = sims_per_CPU, num_CPUs = num_CPUs):
    """
    Retrieves the load vector sums from all CPUs, adds them up, takes expectation.
    Returns the coefficients c_i for the function approximation. 

    Returns:
        expected load vector, np.ndarray, 1D
    """
    sim_num_list = [sims_per_CPU] * num_CPUs
    total_sims = sims_per_CPU * num_CPUs
    total_dose = None 
    sim_num = sims_per_CPU*num_CPUs

    #Setting seed
    path_seeds = master_seed_seq.spawn(sim_num) #Make enough for all paths 
    
    #Separate into sets for each worker
    path_seed_chunks = [path_seeds[i * sims_per_CPU : (i + 1) * sims_per_CPU]
            for i in range(num_CPUs)]

    #Add contributions one by one  
    with ProcessPoolExecutor(max_workers=num_CPUs) as ex: 
        for worker_contribution in ex.map(worker, repeat(method), path_seed_chunks):
            if total_dose is None: 
                total_dose = np.array(worker_contribution, copy=True)
            else:
                total_dose += worker_contribution
    dose = total_dose / total_sims

    return dose

if __name__ == "__main__":
    sim_num = num_CPUs * sims_per_CPU
    
    dose_method = "spatial_kernel"
    if method == 'KZ':
        h = round(abs(dE), 3)
    if method == 'V':
        h = round(abs(ds), 3)
    print(f"dose method {dose_method}, method {method}, {h} h, EO {E0}, l {l}, N {sims_per_CPU*num_CPUs}")

    folder_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map results"
    dose_expected = expected_dose(method)

    l=round(l,3)
    
    path_3D = os.path.join(folder_path, f"{dose_method}_{method}_{h}_{spatial_dim}D_shape_{dose_shape}_E0_{E0}_l_{l}_N_{sims_per_CPU*num_CPUs}.npz") 
    np.savez(path_3D, dose_expected=dose_expected, sim_num=sim_num, method=method, absolute_h=h, spatial_dim=spatial_dim, l=l, X_meshgrid = X_meshgrid, sigma=SIGMA)

    plot = dose_plot_2D(method, "spatial_kernel")

