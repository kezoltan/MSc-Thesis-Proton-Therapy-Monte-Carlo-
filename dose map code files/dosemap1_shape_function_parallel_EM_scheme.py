
#July 2026
#Finite Element Dose Method Code

#This file is structured as follows
#   1. Dose function calculation by finite element method
#   2. Euler Maruyama schemes for proton path generation 
#   3. Worker functions (for parallel)

#READ: if there is a mistake anywhere, it's the storage_position_convention function

#----------------IMPORTS-----------------------------------------------------

import numpy as np 
from math import ceil, prod, floor, exp, log
from itertools import repeat
from numba import njit
import os
from concurrent.futures import ProcessPoolExecutor
from dosesetup import *
import doseparams as dp
from doseplot import dose_plot_2D
from numpy.random import default_rng

#--------------PARAMETERS-----------------------------------------------------

#Units:
#length = cm 
#mass = g
#Beam axis: x

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

#For the finite element dose calculation
t1 = 1/np.sqrt(3)
quad_nodes = [-t1, t1]

#Select l using the given function for this method
#l = choose_l()
l=dp.l #For now let's fix it 
l_reciprocal = 1/l

#These require l to be chosen so we'll do these here:

if spatial_dim==3:
    X_meshgrid, Y_meshgrid, Z_meshgrid = nodes(l)
    meshgrid = X_meshgrid, Y_meshgrid, Z_meshgrid
if spatial_dim==2:
    X_meshgrid, Y_meshgrid= nodes(l)
    meshgrid = X_meshgrid, Y_meshgrid

if rho <= 0:
    raise ValueError("density must be positive.")

_, domain_lower_bounds, domain_upper_bounds = choose_X0(meshgrid)

#------------BASIC FUNCTIONS AND OTHER GLOBAL OBJECTS------------------------

def storage_position_convention(node_coordinate):
    """
    Converts node coordinate to the position in the lookup matrix.

    Note: we could possibly remove this function in favour of:
        flattening the meshgrid into an array of node coords
        and then indexing the required node directly
    """
    if spatial_dim == 3:
        i = round((node_coordinate[0] - origin[0]) * l_reciprocal)
        j = round((node_coordinate[1] - origin[1]) * l_reciprocal)
        k = round((node_coordinate[2] - origin[2]) * l_reciprocal)

        #Floating point error may put us outside the domain
        #The proton would be absorbed if this had actually happened

        nx, ny, nz = lookup_matrix.shape
        i = max(0, min(i, nx - 1))
        j = max(0, min(j, ny - 1))
        k = max(0, min(k, nz - 1))

        return lookup_matrix[i, j, k]

    elif spatial_dim == 2:
        i = round((node_coordinate[0] - origin[0]) * l_reciprocal)
        j = round((node_coordinate[1] - origin[1]) * l_reciprocal)

        nx, ny = lookup_matrix.shape
        i = max(0, min(i, nx - 1))
        j = max(0, min(j, ny - 1))

        return lookup_matrix[i, j]

    else:
        raise ValueError("spatial_dim must be 2 or 3.")

lookup_matrix = position_lookup_matrix(X_meshgrid)
lookup_shape = lookup_matrix.shape
load_shape = prod(lookup_shape) #More accurately this is a length but it works as a 1D shape
max_steps = sum(X_meshgrid.shape) #All boundaries, just needs to be large enough but relatively small

    
#------------------------LOAD VECTOR CALCULATION-----------------------------------

def linear_path_parameterise(start, end, points=quad_nodes):
    """
    Parameterises the line segment from start to end using s in [-1, 1].
    
    Inputs:
        start, np.array() with shape (number of dimensions,)
        end, "" similar
        points, np.array: contains all the points at which you want to know the value of the path
                          defaults to the two point gauss quadrature nodes

    Returns:
        output points, array with rows containing coords of the line at points 
    """

    # if not isinstance(start, np.ndarray) or not isinstance(end, np.ndarray):
    #     raise TypeError("start and end must be NumPy arrays.")

    points = np.asarray(points)

    # if np.any((points < -1) | (points > 1)):
    #     raise ValueError("All parameter values must lie in [-1, 1].")

    return start + 0.5 * (points[:, None] + 1) * (end - start)

max_steps = sum(nodes(l)[0].shape) #All boundaries, just needs to be large enough but relatively small

def traversal_alg(start, end, rounding_tol = 8, max_steps = max_steps, corner_tol = 10**(-4)):
    """
    Runs all validation checks for the traversal algorithm below before running the alg with Numba
    Call this function to run the algorithm fully
    """
    if type(start) != np.ndarray or type(end) != np.ndarray:
        raise TypeError("start and end points must be np.ndarrays arrays.")
    if start.shape != (spatial_dim,):
        raise ValueError("start and end must have shape (n,) for n spatial dimensions.")
    if start.shape != end.shape:
        raise ValueError("start and end must be np.ndarrays of the same shape (n,) for n spatial dimensions.")
    
    return traversal_alg_numba(start, end, rounding_tol, max_steps, corner_tol)

@njit
def traversal_alg_numba(start, end, rounding_tol, max_steps, corner_tol):
    """
    Fast voxel/cell traversal algorithm (based on Amanatides and Woo) for identifying where a straight line proton path (the ray) crosses boundaries on a Cartesian grid
    Requires that the proton belong to a uniform voxel grid of side length l 
    To be executed once per linear segment of a single proton path 

    Inputs:
        start/end: np.ndarrays, must be the same shape, start and end coordinates of a linear path segment

    Returns: 
        array of coordinates (np.ndarrays) in order they are crossed, including start and end  
            i.e. the straight line proton path between start and end split into subsegments 
    """
    direction = end - start
    t_upper = 1.0 #Bound on parameter t, by def t=1 if at end    

    #----Initialise the algorithm-----
    tmaxs = np.empty((spatial_dim))
    tdeltas = np.empty((spatial_dim))
    ts = np.empty((max_steps)) #Arbitrarily long for Numba implementation 

    #1. [NOT NEEDED: StepX, ... : StepX is the length and direction of the step between two adjacent X boundaries, if dir is negative in X then StepX is negative] 
    #2. tMaxX, ... : from start where is the next boundary in X direction - relative to start NOT current position
    #                is np.inf if there is no remaining uncrossed boundary in that direction bounded to along line segment                  
    #3. tDeltaX,... : fixed parameters - once we are on an X boundary, how far in t along that coordinate do we go to get to the next X boundary (imagine this in 1D) 

    for i in range(spatial_dim):
        
        start_i = start[i]
        dir_i = direction[i]

        if dir_i > 0:
            tdelta = l/dir_i
            tdeltas[i] = tdelta

            #Locate nearest boundary along direction from start
            nearest_bdry = ceil(round(start_i * l_reciprocal, rounding_tol)) * l
            t = (nearest_bdry - start_i) / dir_i #If further than end[i], will be > 1           
            if t >= t_upper: #If the nearest boundary overshoots end or is end
                tmaxs[i] = np.inf  
            elif t < corner_tol and t>= 0: #We start on boundary 
                tmaxs[i] = t + tdelta #Push to the next boundary 
                if tmaxs[i] >= t_upper: #If the next boundary is not strictly on the segment
                    tmaxs[i] = np.inf 
            else:
                tmaxs[i] = t

        elif dir_i < 0:
            tdelta = -l/dir_i
            tdeltas[i] = tdelta

            nearest_bdry = floor(round(start_i * l_reciprocal, rounding_tol)) * l
            t = (nearest_bdry - start_i) / dir_i 
            if t >= t_upper: 
                tmaxs[i] = np.inf
            elif t < corner_tol and t>= 0: #We start on boundary 
                tmaxs[i] = t + tdelta #Push to the next boundary 
                if tmaxs[i] >= t_upper:
                    tmaxs[i] = np.inf 
            else:
                tmaxs[i] = t
        else: 
            tmaxs[i] = np.inf 
            tdeltas[i] = np.inf  
    
    #----Run the algorithm (sweeping)----

    #The idea is to store a sequence to t values which define positions along the line 
    #t is always relative to the start point, not current position 
    #tmaxs ALWAYS contains the t value of the next boundary in that direction, or inf if none left

    counter = 1 #An index to store results at 
    while True: #Runs until below condition:  
        
        idx = np.argmin(tmaxs) #Dimension index for next boundary 
        next_t = tmaxs[idx]

        #This condition will end the loop
        if next_t >= t_upper:
            break
        
        ts[counter] = next_t #Store boundary position in t
        tmaxs[idx] += tdeltas[idx] #Next boundary is known length away in t 

        #Corner check, we will get duplicates if we pass through a corner/face 
        for i in range(spatial_dim):
            if i!= idx and abs(tmaxs[i] - next_t) <= corner_tol: #If another coordinate is close enough to the same boundary, presume we crossed both
                tmaxs[i] += tdeltas[i] #Update this one as well
                if tmaxs[i] >= t_upper:
                    tmaxs[i] = np.inf #Inf will never be the min 

        if tmaxs[idx] >= t_upper: 
            tmaxs[idx] = np.inf #Next boundary is strictly between endpoints 

        counter+=1 
        #Check that there is a slot remaining to store results in ts before storing 
        if counter >= max_steps:
            raise RuntimeError("max_steps insufficient for number of crossings.")

    #Add the endpoints back on since they have been omitted

    ts[0] = 0
    ts[counter] = 1 #Counter is updated after it is used 
    ts_len = counter+1
    ts = ts[:ts_len]

    #Finally retrieve all the points:
    
    points = start + ts[:, None] * direction #Right most term puts each elt of ts into its own row and then * direction for each row in ts, gives one big matrix 

    return points

@njit
def subsegment_voxel_vertices_numba(start, end, rounding_tol = 8):
    """
    Given the start and end coordinates of a linear subsegment in ONE voxel/cell
    return the vertices of that voxel/cell that will receive a contribution from that proton path segment. 
        NOT the same as finding all voxel vertices:
        e.g. if in 3D but segment remains on a plane then the function returns 4 coords, not 8

    Returns:
        vertices, np.ndarray: each row is one relevant vertex
    """
    #We need the start and end in case one is on a boundary

    direction = end - start
    bounds = np.empty((spatial_dim, 2)) #Store a pair of bounds in each direction
    num_vertices = 2**np.count_nonzero(direction)
    vertices = np.empty((num_vertices, spatial_dim))

    varying_axes = []

    for i in range(spatial_dim):

        start_i = start[i]
        end_i = end[i]
        lo = floor(round(start_i * l_reciprocal, rounding_tol))
        hi = ceil(round(start_i * l_reciprocal, rounding_tol)) 

        lo_end = floor(round(end_i * l_reciprocal, rounding_tol))
        hi_end = ceil(round(end_i * l_reciprocal, rounding_tol)) 
        if lo_end < lo:
            lo = lo_end
        if hi_end > hi:
            hi = hi_end

        if lo == hi: #If equal then we lose this direction 
            val = lo*l
            bounds[i, 0] = val
            bounds[i, 1] = val 
            vertices[:, i] = val
        else:
            bounds[i,0] = lo*l 
            bounds[i,1] = hi*l
            varying_axes.append(i) #Store the axis index if we are moving in this direction

    #Fill out the vertices by mixing x,y,z in the same way as np.meshgrid 
    for axis_idx, i in enumerate(varying_axes):
        chunk_len = num_vertices // (2 ** (axis_idx + 1))
        step = chunk_len * 2

        for k in range(0, num_vertices, step):
            vertices[k:k + chunk_len, i] = bounds[i, 0]
            vertices[k + chunk_len:k + step, i] = bounds[i, 1]

    return vertices

def one_step_F_contribution(start, end, E_start, E_end):
    """
    Add the contribution from one linear step ds of the proton path simulation to the load vector.
    Designed to be called once each step of the proton path and recursively construct F. 
    Assumes 2 point quadrature is being used, everything must be np.ndarrays! 
    Requires the lookup matrix and the phis matrix to be defined globally. 
    
    Inputs: 
        F_current, np.ndarray, 1D: the current load vector from the end of the previous segment 
        start, end: arrays for the start and end point of the full linear segment 
        dE, ds: floats for the energy lost + path length covered between start and end 

    Returns:
        one_step, np.ndarray, 1D, same shape as F_current: one linear segment contribution to the dose
    """
    one_step = np.zeros(load_shape) #Load shape must be defined globally for this

    if len(quad_nodes) != 2:
        raise ValueError("load vector calculation assumes two point Gauss quadrature is used, please update quad_nodes.")

    #First separate the linear segment into subsegments:
    subsegment_endpts = traversal_alg(start, end)
    num_subsegments = len(subsegment_endpts) - 1 

    direction = end - start
    direction_sq = np.dot(direction, direction)

    if direction_sq == 0:
        raise ValueError("segment direction vector should not be 0.")
    if E_start - E_end <= 0:
        if method=="KZ": #Energy should be controlled - this shouldn't happen
            print(f"{start, end}, {E_start, E_end}")
            raise ValueError("full segment energy must be strictly decreasing.")
        #V method is checked for this in EM code

        #elif method=='V': #Can produce spurious non physical energy changes
        #    return True

    #Every pair of adjacent points is one subsegment
    #For each, need (1) the list of nodes 
    #               (2) the value of the parameterised subsegment at the quad nodes 
    #These combined give the phi inputs 

    for i in range(num_subsegments): #For each subsegment
        sub_start = subsegment_endpts[i] #these are coordinates 
        sub_end = subsegment_endpts[i+1]

        #If the subsegment length is effectively 0, move on
        #This is an extra guardrail 
        if np.allclose(sub_start, sub_end, atol=1e-8, rtol=0):
            continue #end this iteration 

        #Estimate the energy loss per subsegment
        t0 = np.dot(sub_start - start, direction) / direction_sq
        t1 = np.dot(sub_end   - start, direction) / direction_sq 
        sub_E_start = E_start + t0 * (E_end - E_start)
        sub_E_end   = E_start + t1 * (E_end - E_start)

        if sub_E_start - sub_E_end <= 0:
            if method=='KZ':
                print(f"{sub_start, sub_end}: {sub_E_start, sub_E_end} at {t0, t1}")
                raise ValueError("subsegment energy must be strictly decreasing.")

        energy_length = sub_E_start - sub_E_end

        #We have a slightly different integral depending on the method choice  
        if method=="V": 
            #length = np.linalg.norm(sub_end - sub_start) #Length of the subsegment
            gamma_t1_point, gamma_t2_point = linear_path_parameterise(sub_start, sub_end) 

            #Retrieve the stopping power values at the nodes
            E_t1, E_t2 = linear_path_parameterise(sub_E_start, sub_E_end).flatten()  # same v(t) as KZ uses
            #S_t1, S_t2 = stopping_power(E_t1), stopping_power(E_t2)

        elif method=="KZ":
            #This is the value of E(t1) and E(t2), not X
            E_t1_point, E_t2_point = linear_path_parameterise(sub_E_start, sub_E_end).flatten() 

            #Use a linear path approximation to get X(E1) and X(E2)
            gamma_t1_point, gamma_t2_point = [sub_start + (sub_E_start - E)/(sub_E_start - sub_E_end) * (sub_end - sub_start) for E in [E_t1_point, E_t2_point]]

        nodes_touched = subsegment_voxel_vertices_numba(sub_start, sub_end)   

        #Where to store for each node:
        for node in nodes_touched:
            position = storage_position_convention(node)

            phi1 = Phi(l, gamma_t1_point, node)
            phi2 = Phi(l, gamma_t2_point, node)

            if method=='KZ':
                node_contribution = (energy_length / 2) * (phi1 + phi2) / rho
            elif method=='V':
                node_contribution = (energy_length / 2) * (phi1 + phi2) / rho
                #node_contribution = (energy_length / 2) * (S_t1 * phi1 + S_t2 * phi2) / rho

            #node_contribution = integral_factor * (Phi(l,gamma_t1_point, node) + Phi(l,gamma_t2_point, node)) / rho

            one_step[position] += node_contribution

    return one_step #Each worker will be adding all these up for one path


#--------------------------EULER-MARUYAMA SCHEMES----------------------------------------


#These generate the proton paths - the full path should NOT be stored

def one_path_sim_load_vector(method, rng, E0=E0, Omega0=Omega0, dE = dE, ds=ds, s0 = 0, kappa=KAPPA, a=alpha, p=p):
    """
    Generates the load vector contribution for one full proton path simulation.
    Does not store the path - at each step, run one_step_F_contribution. These are totalled up and returned.
    Does not reflect the Gaussian increment is ds becomes negative.   
    
    Inputs:
        method = "KZ" or "V" to denote which scheme we are using

    Returns: 
        total load vector contribution from this simulation, 1D np.ndarray 

    """ 
    #Load vector empty
    one_path_load_contribution = np.zeros(load_shape)
    
    s = float(s0)

    #Sample the Gaussian beam 
    X = sample_initial_position(meshgrid, rng, l)
    Omega = Omega0
    E = initial_energy_spread() 

    #If V method:
    Y = log(E)
    X_start = X #For the path segment calculation  
    E_start = E
    Omega_start = Omega
  
    if method == "KZ": #Runs the scheme in independent energy
        base_h = dE

        #Es should include Emin to be consistent with the other scheme
        n = ceil((E - E_min) / abs(base_h)) 
        Es = np.linspace(E, E_min, n + 1)

        #h may have changed slightly from the base
        h = Es[1] - Es[0]
        #print(f"base_h is reading as {base_h:.4f}, actual h is reading as {h}") 

        num_steps = len(Es) - 1 #Minus 1 because we don't need to repeat the IC 
        coeff_prefactor = np.sqrt(2*eps_0 * a*p) 
        sqrt = np.sqrt(abs(h)) 

        #Generate Brownians ahead of time because known endpoints (+ no reflection)
        dB1 = sqrt * rng.standard_normal(num_steps) 
        dB2_3D = sqrt * rng.standard_normal((num_steps, spatial_dim))
        #negative_path_total = 0

        for k in range(num_steps): #Known start and end points 
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
            path_travelled = s - s_n #Might be negative 
            X = X + Omega * (path_travelled)
            E += h
            E_end = E

            if E < 0:
                raise ValueError(f"E became negative at step {k}: {E}")

            if np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds): #If we left the domain 
                break

            #The end point is now known 
            X_end = X 
            one_step = one_step_F_contribution(X_start, X_end, E_start, E_end)
            one_path_load_contribution += one_step

            #Update for next loop
            X_start = X
            E_start = E

    elif method == "V":
        h = ds
        sqrt = np.sqrt(h) #for constant step size (change if adaptive)
        coeff = np.sqrt(2*eps_0)
        Y_start = log(E_start)

        while E > E_min: #Unknown end point 
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

            if E - E_start > 0: #Energy spuriously increased in this step
                #Rerun this step
                print("Spurious energy increase, rerunning step.")
                E = E_start
                Y = Y_start
                Omega = Omega_start
                X = X_start
                continue #Redo the loop
                
            E_end = E
            #s += h
            X_end = X
            
            if np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds): #If we left the domain 
                break

            if E <= E_min: #Check this condition again after E has updated
                #Interpolate the end point up to Emin

                t_min = (E_min - E_start)/(E_end - E_start) #in 0,1
                X_end = X_start + (X_end-X_start)*t_min 
                E_end = E_min 
                one_step = one_step_F_contribution(X_start, X_end, E_start, E_end)
                one_path_load_contribution += one_step
                return one_path_load_contribution     

            one_step = one_step_F_contribution(X_start, X_end, E_start, E_end)
            one_path_load_contribution += one_step

            #Update for next loop, given an accepted step
            X_start = X
            E_start = E
            Y_start = Y
            Omega_start = Omega

    # else: 
    #     raise NameError("incorrect input for 'method' argument, available methods: KZ, V (as strings)") 

    return one_path_load_contribution

#-------------------PARALLEL FUNCTIONS--------------------------------------------------
    
def worker(method, path_seed_chunk, sims_per_CPU = sims_per_CPU):
    """
    This is the worker function telling each CPU what to do 
    Each CPU will run sims_per_CPU independent simulations, and add up the load vectors.

    Returns:
        sum of load vectors, one per sim: np.ndarray, 1D: (load_shape,)
    """
    #rng = default_rng()  
    worker_load_vector = np.zeros(load_shape)
    for path_seed in path_seed_chunk:
        rng=default_rng(path_seed)
        one_path_load_contribution = one_path_sim_load_vector(method, rng)
        worker_load_vector += one_path_load_contribution
    return worker_load_vector

def expected_coefficients_vector(method, sims_per_CPU = sims_per_CPU, num_CPUs = num_CPUs):
    """
    Retrieves the load vector sums from all CPUs, adds them up, takes expectation.
    Returns the coefficients c_i for the function approximation. 

    Returns:
        expected load vector, np.ndarray, 1D
    """
    total_sims = sims_per_CPU * num_CPUs
    total_load = None 

    #Setting seed
    path_seeds = master_seed_seq.spawn(sim_num) #Make enough for all paths 

    #Separate into sets for each worker
    path_seed_chunks = [
        path_seeds[i * sims_per_CPU : (i + 1) * sims_per_CPU]
        for i in range(num_CPUs)
    ]

    #Add contributions one by one  
    with ProcessPoolExecutor(max_workers=num_CPUs) as ex: 
        for worker_contribution in ex.map(worker, repeat(method), path_seed_chunks):
            if total_load is None: 
                total_load = np.array(worker_contribution, copy=True)
            else:
                total_load += worker_contribution

    expected_F = total_load / total_sims
    c_vector = expected_F / M_diag #elementwise divide
    return c_vector

#------------------------------FINAL CALCULATION-------------------------------------------

if __name__ == "__main__":
    
    sim_num = num_CPUs * sims_per_CPU
    
    dose_method = "shape_function"
    if method == 'KZ':
        h = round(abs(dE), 3)
    if method == 'V':
        h = round(abs(ds), 3)
    print(f"dose method {dose_method}, method {method}, {h} h, EO {E0}, l {l}, N = {sim_num}, KAPPA {KAPPA}")

    M_diag = linear_lumped_mass_matrix(X_meshgrid,l)

    folder_path= r"C:\Users\kathe\OneDrive - Zolution Technologies\Oxford\Dissertation\Code\Dose Map Code\dose map results"

    if spatial_dim==2:
        M_path = os.path.join(folder_path, "2D_lumped_mass_matrix.npz")
        np.savez(M_path, M_diag=M_diag, l = l, X = X_meshgrid, Y = Y_meshgrid)
    if spatial_dim ==3:
        M_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D_lumped_mass_matrix.npz")
        np.savez(M_path, M_diag=M_diag, l = l, X = X_meshgrid, Y =Y_meshgrid, Z = Z_meshgrid)

    c_expected = expected_coefficients_vector(method)
    coeffs_shape = c_expected.shape

    l=round(l,3)

    path_3D = os.path.join(folder_path, f"{dose_method}_{method}_{h}_{spatial_dim}D_shape_{coeffs_shape[0]}_E0_{E0}_l_{l}.npz") 
    np.savez(path_3D, coeffs_expected=c_expected, sim_num=sim_num, method=method, absolute_h=h, spatial_dim=spatial_dim,  l = l, X = X_meshgrid)

    print("Now plotting...")
    plot = dose_plot_2D(method, dose_method)