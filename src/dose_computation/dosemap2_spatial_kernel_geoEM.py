
#July 2026
#Spatial Kernel Dose Method Code

import numpy as np 
from math import exp, log, pi
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from doseparams import *
from dosesetup import *
from scipy.stats import chi2

#Precompute these 
KERNEL_PREFACTOR = 1/RHO * 1/((2*pi)**(SPATIAL_DIM/2)*SIGMA**SPATIAL_DIM)
allowed_sd = 6 #For the gaussian truncation, must be tiny  
MASS_RESCALING = chi2.cdf(allowed_sd**2, df=SPATIAL_DIM)

def spatial_kernel_vectorized(X_i, nodes_array, truncate=True, allowed_sd=4):
    """
    Gaussian d-dimensional (d=2 or 3) spatial kernel, vectorised for efficiency to operate on all nodes simulataneously. 
    Truncated by default to within min(allowed_sd, min_distance) standard deviations to ensure conservation
    the min_distance is the smallest distance to the furthest node over all directions
        this is needed for the boundary cases 
        i.e. the gaussian truncates to ensure all deposit is within the boundary 

    Inputs:
        X_i: np.ndarray, the current position of the EM scheme (i.e. where the dose is "dropped") 
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

    if SPATIAL_DIM==2:    
        xi,yi = X_i[0], X_i[1]
        x_lo, y_lo = domain_lower_bounds
        x_hi, y_hi = domain_upper_bounds
        x_lo -= allowed_sd*SIGMA #artificially move the line at zero by one sd, else we get no deposit here at all (due to rescaling)

        #Shortest distances to boundaries:
        boundary_dist = np.array([abs(xi-x_lo), abs(xi-x_hi), abs(yi-y_lo), abs(yi-y_hi)])

    elif SPATIAL_DIM==3:    
        xi,yi,zi = X_i[0], X_i[1], X_i[2]
        x_lo, y_lo, z_lo = domain_lower_bounds
        x_hi, y_hi, z_hi = domain_upper_bounds
        x_lo -= allowed_sd * SIGMA
        boundary_dist = np.array([abs(xi-x_lo), abs(xi-x_hi), abs(yi-y_lo), abs(yi-y_hi), abs(zi-z_lo), abs(zi-z_hi)])

    if np.any(boundary_dist < allowed_sd*SIGMA):
        radius = np.min(boundary_dist) #Choose the nearest one 
        mass_rescaling = chi2.cdf((radius/SIGMA)**2, df=SPATIAL_DIM)

    if truncate: #We need to rescale if we truncate the Gaussian to a smaller region 
        mask = norm_sqrd <= (radius) ** 2 #If we are inside the ball
        nodal_values[mask] = (KERNEL_PREFACTOR
                         * np.exp(-norm_sqrd[mask] / (2 * SIGMA ** 2))
                         / mass_rescaling)
    else:
        nodal_values = KERNEL_PREFACTOR * np.exp(-norm_sqrd / (2 * SIGMA ** 2))
    return nodal_values


#-----------------------DOSE CALCULATION W/ EM SCHEMES-----------------------------------

def one_step_dose_contribution(h, misc_coeff, dB1_onestep, dB2_3D_onestep, E, Omega, s, X, Y):
    """
    Compute the next step + compute the dose
    the misc_coeff is different for each scheme 
    """
    if method=='KZ':
        E_n1, Omega_n1, s_n1, X_n1 = one_step_KZ_g_euler(h, dB1_onestep, dB2_3D_onestep, misc_coeff, E, Omega, s, X)
        return spatial_kernel_vectorized(X_n1, nodes_array), E_n1, Omega_n1, s_n1, X_n1
    elif method=='V':
        E_n1, Omega_n1, s_n1, X_n1, Y_n1 = one_step_V_g_euler(h, dB1_onestep, dB2_3D_onestep, misc_coeff, E, Omega, s, X, Y)
        constant = (-E_n1+E)/h
        return constant * spatial_kernel_vectorized(X_n1, nodes_array), E_n1, Omega_n1, s_n1, X_n1, Y_n1

def KZ_one_path_dose_contribution(method, brownian_paths, setup_tuple):
    """
    Estimates the dose due to one proton path using a Gaussian smoothing kernel, not truncated.
    Assumes a composite trapezium rule is used for path integral approximation.  
    
    Inputs:
        method = "KZ" or "V" to denote which scheme we are using
        brownian paths = dB1 and dB2 (2D/3D), MUST BE BROWNIAN not standard normal 

    Returns: 
        dose_contribution, np.ndarray, shape according to the number of nodes 
    """ 
    dose_contribution = np.zeros(dose_shape)
    s, X, Omega, E, Y, h, coeff_prefactor, num_steps = setup_tuple
    dB1, dB2_3D = brownian_paths
    if method != "KZ": 
        raise ValueError("must run KZ dose contribution with method KZ.")
    if dose_method != 'SK':
        raise ValueError("must not run dose contribution functions if dose_method not SK.")
    current_step = 0
    domain_check = True

    if len(dB1) != num_steps:
        raise ValueError(f'Length of dZ1 is {len(dB1)}, should be {num_steps}.')

    for k in range(num_steps): #Known start and end points 

        #We include step 0 in the comp trap
        if current_step == 0:
            dose_contribution += abs(h)/2 * spatial_kernel_vectorized(X, nodes_array)
        else: 
            dB1_onestep = dB1[k] 
            dB2_3D_onestep = dB2_3D[k]
            X_start = X
            one_step, E, Omega, s, X = one_step_dose_contribution(h, coeff_prefactor, dB1_onestep, dB2_3D_onestep, E, Omega, s, X, Y) #Y will not be used or returned
            if domain_exit_check(X): #If we left the domain 
                print(f"Track exited the domain between {X_start} and {X}.")
                domain_check = False    
                break
            dose_contribution += abs(h)*one_step
        current_step+= 1

    #Do the final step contribution as well, assuming we were still in the domain:
    if domain_check:    
        #Use the final step 
        dose_contribution += abs(h)/2 * spatial_kernel_vectorized(X, nodes_array)

    return dose_contribution
    
def V_one_path_dose_contribution(method, rng, level):
    """
    Estimates the dose due to one proton path using a Gaussian smoothing kernel, not truncated.
    Assumes a composite trapezium rule is used for path integral approximation.  
    
    Inputs:
        method = "KZ" or "V" to denote which scheme we are using

    Returns: 
        dose_contribution, np.ndarray, shape according to the number of nodes 
    """ 
    dose_contribution = np.zeros(dose_shape)
    s, X, Omega, E, Y = sample_gauss_beam(rng)
    _, h = choose_dE_ds(level)
    coeff = np.sqrt(2*EPS_0)
    initial_constant = stopping_power(E) #For the comp trap
    sqrt = np.sqrt(abs(h)) 
    if method != "V": 
        raise ValueError("must run V dose contribution with method V.")
    if dose_method != 'SK':
        raise ValueError("must not run dose contribution functions if dose_method not SK.")
    
    domain_check = True
    current_step = 0
    E_start = E
    X_start = X

    while E > EMIN: #Unknown end point 

        if current_step == 0:
            dose_contribution += abs(h)/2 * initial_constant * spatial_kernel_vectorized(X, nodes_array)
        else: 
            dB1_onestep = sqrt * rng.standard_normal() 
            dB2_3D_onestep = sqrt * rng.standard_normal(SPATIAL_DIM)
            one_step, E, Omega, s, X, Y = one_step_dose_contribution(h, coeff, dB1_onestep, dB2_3D_onestep, E, Omega, s, X, Y)
            E_end = E
            X_end = X
            dose_contribution += abs(h) * one_step
        current_step+=1 
    
        if domain_exit_check(X): #If we left the domain 
            domain_check = False
            break

        if E <= EMIN: #Check this condition again after E has updated
            #Interpolate the end point up to Emin

            t_min = (EMIN - E_start)/(E_end - E_start) #in 0,1
            X_end = X_start + (X_end-X_start)*t_min 
            E_end = EMIN 

            h_final = np.linalg.norm(X_end - X_start)
            constant = (-E_end+ E_start)/h_final
            one_step = constant * spatial_kernel_vectorized(X_end, nodes_array)

            #The final dose contribution
            dose_contribution += h_final/2 * one_step
            break

        X_start = X
        E_start = E

    if not(domain_check): #If we exited the domain 
        #Go back to the last full step that was inside D 
        dose_contribution -= abs(h)/2 * one_step #Update the quadrature so that this was the endpoint  

    return dose_contribution
