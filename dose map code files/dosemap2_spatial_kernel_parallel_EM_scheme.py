
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
from math import ceil, prod, exp, log, cos, sin, pi
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import doseparams as dp
from dosesetup import *

#--------------PARAMETERS----------------------------------------------------


#Specific params:

l = 0.02
l_reciprocal = 1/l
SIGMA = l/2

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
Omega0 = choose_Omega_0()
if spatial_dim==3:
    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(l)
if spatial_dim==2:
    X_meshgrid, Y_meshgrid= nodes(l)

if rho <= 0:
    raise ValueError("density must be positive.")

#------------BASIC FUNCTIONS-------------------------------------------------

if spatial_dim == 3: 
    X_meshgrid,Y_meshgrid,Z_meshgrid = nodes()
    
    upper_X = X_meshgrid[-1,-1,-1]
    upper_Y = Y_meshgrid[-1,-1,-1]
    lower_Y = Y_meshgrid[0,0,0] #This file relies on this being 0
    lower_Z = Z_meshgrid[0,0,0]
    upper_Z = Z_meshgrid[-1,-1,-1] #hi

    if lower_Y != 0 or lower_Z != 0:
        raise ValueError("lower domain bounds should be 0.")

    y0 = (lower_Y + upper_Y)*1/2
    z0 = (lower_Z + upper_Z)*1/2
    X0 = np.array([0.0,y0,z0])

    domain_upper_bounds = np.array([upper_X, upper_Y, upper_Z])

    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel(), Z_meshgrid.ravel()], axis=-1)   # shape (n_nodes, 3)

elif spatial_dim ==2:
    X_meshgrid, Y_meshgrid = nodes()

    upper_X = X_meshgrid[-1,-1]
    upper_Y = Y_meshgrid[-1,-1]
    lower_Y = Y_meshgrid[0,0] #This file relies on this being 0

    if lower_Y != 0:
        raise ValueError("lower domain bounds should be 0.")

    y0 = (lower_Y + upper_Y)*1/2
    X0 = np.array([0.0,y0])

    domain_upper_bounds = np.array([upper_X, upper_Y])

    nodes_array = np.stack([X_meshgrid.ravel(), Y_meshgrid.ravel()], axis=-1)   # shape (n_nodes, 2)

domain_lower_bounds = np.zeros(spatial_dim)
origin = np.zeros(spatial_dim)

dose_shape = prod(X_meshgrid.shape) #More accurately this is a length but it works as a 1D shape
max_steps = sum(X_meshgrid.shape) #All boundaries, just needs to be large enough but relatively small

#------------------------------SPATIAL KERNEL----------------------------------------------


def spatial_kernel(X_i, node):
    """
    Gaussian spatial kernel scaled by 1/rho, 2D or 3D. 
    No truncation has been used i.e. there will be some energy loss due to smoothing.
    Centered at X_i, sigma standard deviation.  

    Inputs:
        X_i, np.ndarray: current position of the proton in the scheme 
        node, np.ndarray, same shape: any node within the domain meshgrid 
        sigma, positive float: the sd of the Gaussian, smoothing factor. To be manually tuned.

    Returns:
        phi, float: the value of the smoothing kernel at X_i-node
    """ 
    if type(X_i)!= np.ndarray:
        raise TypeError("position must be np.ndarray.")
    if type(node)!= np.ndarray:
        raise TypeError("nodes must be np.ndarrays.")
    
    norm_sqrd = np.linalg.norm(X_i - node)**2 #defaults to 2-norm
    nodal_value = 1/rho * 1/((2*pi)**(spatial_dim/2)*SIGMA**spatial_dim) * exp(- norm_sqrd / (2*SIGMA**2))
    return nodal_value


#-----------------------DOSE CALCULATION W/ EM SCHEMES-----------------------------------


def one_step_dose_contribution(X_i, constant):
    """
    Calculates the one_step contribution to the dose calculation for the composite trapezium rule. 
    
    Inputs: 
        X_i, np.ndarray: current position in either 2D or 3D at step i of the scheme
        constant is either 1 or the stopping power at Ei depending on the method used 

    Returns:
        one_step, np.ndarray, 1D, same shape as F_current: one linear segment contribution to the dose
    """
    one_step = np.zeros(dose_shape) #Load shape must be defined globally for this
    nodes_shape = nodes_array.shape[0] 
    if dose_shape != nodes_shape:
        raise ValueError("dose_shape and nodes_shape must match.")

    #For each node (out of all of them), calculate the dose
    #We'll iterate over the indices, retrieve the node from the meshgrid, then calculate

    for i in range(nodes_shape): #defined globally list of nodes, i is the storage position
        node = nodes_array[i] 
        contrib = spatial_kernel(X_i, node) * constant
        one_step[i] += contrib 

    return one_step #Each worker will be adding all these up for one path

def one_path_dose_contribution(method, E0=E0, Omega0=Omega0, dE = dE, ds=ds, s0 = 0, kappa=KAPPA, a=alpha, p=p):
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
    X = X0
    Omega = Omega0
    E = E0

    #If V method:
    Y = log(E)
  
    if method == "KZ": #Runs the scheme in independent energy
        h = dE
        Es = np.arange(E0, E_min, dE)
        num_steps = len(Es) - 1 #Minus 1 because we don't need to repeat the IC 
        coeff_prefactor = np.sqrt(2*eps_0 * a*p) 
        sqrt = np.sqrt(-h) 
        constant = 1 #This is for the comp trap rule, does not change 

        #Generate Brownians ahead of time because known endpoints (+ no reflection)
        dB1 = sqrt * np.random.standard_normal(num_steps) 
        dB2_3D = sqrt * np.random.standard_normal((num_steps, spatial_dim))

        for k in range(len(dB1)): #Known start and end points 
            
            #We will include step 0 in the comp trap
            one_step = one_step_dose_contribution(X, constant)
            dose_contribution += one_step
            
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

            if np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds): #If we left the domain 
                break
        
    elif method == "V":
        h = ds
        Es = [E0]
        sqrt = np.sqrt(h) #for constant step size (change if adaptive)
        coeff = np.sqrt(2*eps_0)
        constant = stopping_power(E) #For the comp trap

        while E >= E_min: #Unknown end point 
            
            one_step = one_step_dose_contribution(X, constant)
            dose_contribution += one_step
            
            #Sample each Gaussian
            dB1 = sqrt * np.random.standard_normal() 
            dB2_3D = sqrt * np.random.standard_normal(spatial_dim)
            Omega_n = Omega 
            other_noise = dB2_3D - Omega_n * np.dot(Omega_n, dB2_3D) # =(1-OmegaOmega)dB
            
            #Exp map
            y = coeff * other_noise
            Omega = exponential_map_sphere(Omega_n, y)

            #Update the other variables
            X = X + Omega * h
            Y = Y + h * n_V_log_energy_drift(E) + n_V_log_energy_diffusion(E) * np.sqrt(h) * dB1
            E = exp(Y)
            s += h
            constant = stopping_power(E)
            
            if np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds): #If we left the domain 
                break

    dose_contribution *= abs(h)/2 #check this abs 

    return dose_contribution


#-------------------PARALLEL FUNCTIONS--------------------------------------------------
    
def worker(method, sims_per_CPU = sims_per_CPU):
    """
    This is the worker function telling each CPU what to do 
    Each CPU will run sims_per_CPU independent simulations, and add up the load vectors.

    Returns:
        sum of load vectors, one per sim: np.ndarray, 1D: (load_shape,)
    """
    worker_dose = np.zeros(dose_shape)
    for _ in range(sims_per_CPU):
        dose_contribution = one_path_dose_contribution(method)
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

    #Add contributions one by one  
    with ProcessPoolExecutor(max_workers=num_CPUs) as ex: 
        for worker_contribution in ex.map(worker, repeat(method), sim_num_list):
            if total_dose is None: 
                total_load = np.array(worker_contribution, copy=True)
            else:
                total_load += worker_contribution
    dose = total_load / total_sims

    return dose



if __name__ == "__main__":

    dose_method = "spatial_kernel"
    folder_path = r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map results"
    dose_expected = expected_dose(method)
    sim_num = num_CPUs * sims_per_CPU

    if method == 'KZ':
        h = round(abs(dE), 3)
    if method == 'V':
        h = round(abs(ds), 3)

    l=round(l,3)
    
    path_3D = os.path.join(folder_path, f"{dose_method}_{method}_{h}_{spatial_dim}D_shape_{dose_shape}_E0_{E0}_l_{l}.npz") 
    np.savez(path_3D, dose_expected=dose_expected, sim_num=sim_num, method=method, absolute_h=h, spatial_dim=spatial_dim, l=l, X_meshgrid = X_meshgrid, sigma=SIGMA)

    

