
#July 2026
#Finite Element Dose Method Code

import numpy as np 
from math import ceil, prod, floor, exp, log
from numba import njit
from dosesetup import *
from doseparams import *

def storage_position_convention(node_coordinate):
    """
    Converts node coordinate to the vector index.

    NB: for mlmc, very important that floating pt error doesn't push data onto the wrong nodes

    Note: we could possibly remove this function in favour of:
        flattening the meshgrid into an array of node coords
        and then indexing the required node directly
    """
    #the nodes_array should all be distinct 
    return np.where(np.all(nodes_array == node_coordinate, axis=1))[0][0]

    #if SPATIAL_DIM == 3:
    #    i = round((node_coordinate[0] - origin[0]) * l_reciprocal)
    #    j = round((node_coordinate[1] - origin[1]) * l_reciprocal)
    #    k = round((node_coordinate[2] - origin[2]) * l_reciprocal)
    #    nx, ny, nz = lookup_matrix.shape
    #    i = max(0, min(i, nx - 1))
    #    j = max(0, min(j, ny - 1))
    #    k = max(0, min(k, nz - 1))
    #    return lookup_matrix[i, j, k]
    #elif SPATIAL_DIM == 2:
    #    i = round((node_coordinate[0] - origin[0]) * l_reciprocal)
    #    j = round((node_coordinate[1] - origin[1]) * l_reciprocal)
    #    nx, ny = lookup_matrix.shape
    #    i = max(0, min(i, nx - 1))
    #    j = max(0, min(j, ny - 1))
    #    return lookup_matrix[i, j]
    #else:
    #    raise ValueError("SPATIAL_DIM must be 2 or 3.")

lookup_matrix = position_lookup_matrix(X_meshgrid)
lookup_shape = lookup_matrix.shape
load_shape = prod(lookup_shape) #More accurately this is a length but it works as a 1D shape
if load_shape != dose_shape:
    raise ValueError("dose_shape must equal load_shape.")

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
    if start.shape != (SPATIAL_DIM,):
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
    tmaxs = np.empty((SPATIAL_DIM))
    tdeltas = np.empty((SPATIAL_DIM))
    ts = np.empty((max_steps)) #Arbitrarily long for Numba implementation 

    #1. [NOT NEEDED: StepX, ... : StepX is the length and direction of the step between two adjacent X boundaries, if dir is negative in X then StepX is negative] 
    #2. tMaxX, ... : from start where is the next boundary in X direction - relative to start NOT current position
    #                is np.inf if there is no remaining uncrossed boundary in that direction bounded to along line segment                  
    #3. tDeltaX,... : fixed parameters - once we are on an X boundary, how far in t along that coordinate do we go to get to the next X boundary (imagine this in 1D) 

    for i in range(SPATIAL_DIM):
        
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
        for i in range(SPATIAL_DIM):
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
    bounds = np.empty((SPATIAL_DIM, 2)) #Store a pair of bounds in each direction

    #Outdated code (though caused no bugs) 
    #num_vertices = 2**np.count_nonzero(direction)
    #vertices = np.empty((num_vertices, SPATIAL_DIM))

    varying_axes = []
    idx_vals = []

    for i in range(SPATIAL_DIM):

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
            idx_vals.append((i,val))
            #vertices[:, i] = val
        else:
            bounds[i,0] = lo*l 
            bounds[i,1] = hi*l
            varying_axes.append(i) #Store the axis index if we are moving in this direction

        num_vertices = 2**len(varying_axes)
        vertices = np.empty((num_vertices, SPATIAL_DIM))
        for pair in idx_vals:
            i, val = pair
            vertices[:,i] = val

    #Fill out the vertices by mixing x,y,z in the same way as np.meshgrid 
    for axis_idx, i in enumerate(varying_axes):
        chunk_len = num_vertices // (2 ** (axis_idx + 1))
        step = chunk_len * 2

        for k in range(0, num_vertices, step):
            vertices[k:k + chunk_len, i] = bounds[i, 0]
            vertices[k + chunk_len:k + step, i] = bounds[i, 1]

    return vertices

def one_step_F_contribution(h, misc_coeff, dB1_onestep, dB2_3D_onestep, E, Omega, s, X, Y):
    """
    Computes on step of proton path 
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
    #This takes care of the last step for the V method
    final_V_step = type(E) == tuple and type(X) == tuple and method=='V'
    if final_V_step: 
        start, end = X
        E_start, E_end = E
        one_step = np.zeros(dose_shape)
    else:
        if method=='KZ':
            E_n1, Omega_n1, s_n1, X_n1 = one_step_KZ_g_euler(h, dB1_onestep, dB2_3D_onestep, misc_coeff, E, Omega, s, X)
        elif method=='V':
            E_n1, Omega_n1, s_n1, X_n1, Y_n1 = one_step_V_g_euler(h, dB1_onestep, dB2_3D_onestep, misc_coeff, E, Omega, s, X, Y)
        if E_n1 < 0:
            raise ValueError(f"E became negative.")

        start=X
        end=X_n1
        E_start=E
        E_end=E_n1
        one_step = np.zeros(dose_shape) 
        if domain_exit_check(X_n1) and not(final_V_step): #If we left the domain during the simulation
            #print(f"Track exited the domain, contribution = {np.sum(one_step)}")
            if method=='KZ':
                return one_step, E_n1, Omega_n1, s_n1, X_n1
            elif method=='V':
                return one_step, E_n1, Omega_n1, s_n1, X_n1, Y_n1  

    if len(quad_nodes) != 2:
        raise ValueError("load vector calculation assumes two point Gauss quadrature is used, please update quad_nodes.")

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

    for i in range(num_subsegments): 
        sub_start = subsegment_endpts[i] #these are coordinates 
        sub_end = subsegment_endpts[i+1]

        #If the subsegment length is effectively 0, move on
        if np.allclose(sub_start, sub_end, atol=1e-8, rtol=0):
            continue #end this iteration 

        t0 = np.dot(sub_start - start, direction) / direction_sq
        t1 = np.dot(sub_end   - start, direction) / direction_sq 
        sub_E_start = E_start + t0 * (E_end - E_start)
        sub_E_end   = E_start + t1 * (E_end - E_start)

        if sub_E_start - sub_E_end <= 0:
            if method=='KZ':
                print(f"{sub_start, sub_end}: {sub_E_start, sub_E_end} at {t0, t1}")
                raise ValueError("subsegment energy must be strictly decreasing.")

        energy_length = sub_E_start - sub_E_end

        if method=="V": 
            gamma_t1_point, gamma_t2_point = linear_path_parameterise(sub_start, sub_end) 

            #E_t1, E_t2 = linear_path_parameterise(sub_E_start, sub_E_end).flatten()  # same v(t) as KZ uses
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

            node_contribution = (energy_length / 2) * (phi1 + phi2) / RHO
            one_step[position] += node_contribution

    if final_V_step:
        return one_step
    else:   
        if method=='KZ':
            return one_step, E_n1, Omega_n1, s_n1, X_n1 
        elif method=='V':
            return one_step, E_n1, Omega_n1, s_n1, X_n1, Y_n1 

def KZ_one_path_load_contribution(method, brownian_paths, setup_tuple):
    """
    Generates the load vector contribution for one full proton path simulation.
    Does not store the path - at each step, run one_step_F_contribution. These are totalled up and returned.
    Does not reflect the Gaussian increment is ds becomes negative.   
    
    Inputs:
        method = "KZ" or "V" to denote which scheme we are using
        brownian_paths, tuple: precomputed dB1 and dB2 for this scheme

    Returns: 
        total load vector contribution from this simulation, 1D np.ndarray 
    """ 
    if method != "KZ": #Runs the scheme in independent energy
        raise ValueError("must run KZ load contribution with method KZ.")
    if dose_method != 'SF':
        raise ValueError("must not run load contribution functions if dose_method not SF.")

    one_path_load_contribution = np.zeros(dose_shape)
    s, X, Omega, E, Y, h, coeff_prefactor, num_steps = setup_tuple
    dB1, dB2_3D = brownian_paths

    if len(dB1) != num_steps:
        raise ValueError(f'Length of dZ1 is {len(dB1)}, should be {num_steps}.')

    for k in range(num_steps):  
        dB1_onestep = dB1[k] 
        dB2_3D_onestep = dB2_3D[k]
        one_step, E, Omega, s, X = one_step_F_contribution(h, coeff_prefactor, dB1_onestep, dB2_3D_onestep, E, Omega, s, X, Y)
        one_path_load_contribution += one_step
    return one_path_load_contribution


def V_one_path_load_contribution(method, rng, level):
    """
    Generates the load vector contribution for one full proton path simulation.
    Does not store the path - at each step, run one_step_F_contribution. These are totalled up and returned.
    Does not reflect the Gaussian increment is ds becomes negative.   
    
    Inputs:
        method = "KZ" or "V" to denote which scheme we are using

    Returns: 
        total load vector contribution from this simulation, 1D np.ndarray 
    """ 
    one_path_load_contribution = np.zeros(dose_shape)
    s, X, Omega, E, Y = sample_gauss_beam(rng)

    _, h = choose_dE_ds(level)
    coeff = np.sqrt(2*EPS_0)
    X_start = X #For the path segment calculation  
    E_start = E
    Y_start = log(E_start)
    Omega_start = Omega
    s_start=s
    sqrt = np.sqrt(abs(h)) 
    if method != "V": #Runs the scheme in independent energy
        raise ValueError("must run V load contribution with method V.")
    if dose_method != 'SF':
        raise ValueError("must not run load contribution functions if dose_method not SF.")

    while E > EMIN: #Unknown end point 
        dB1_onestep = sqrt * rng.standard_normal() 
        dB2_3D_onestep = sqrt * rng.standard_normal(SPATIAL_DIM)
        one_step, E, Omega, s, X, Y = one_step_F_contribution(h, coeff, dB1_onestep, dB2_3D_onestep, E, Omega, s, X, Y)
        
        if E - E_start > 0: #Energy spuriously increased in this step
            #Rerun this step
            print("Spurious energy increase, rerunning step.")
            E = E_start
            Y = Y_start
            Omega = Omega_start
            X = X_start
            s=s_start
            continue #Redo the loop
        E_end = E
        X_end = X
        
        if E <= EMIN: #Check this condition again after E has updated
            #Interpolate the end point up to Emin

            t_min = (EMIN - E_start)/(E_end - E_start) #in 0,1
            X_end = X_start + (X_end-X_start)*t_min 
            E_end = EMIN 

            #Need to be tuples here for the function to work correctly
            one_step = one_step_F_contribution(h, coeff, dB1_onestep, dB2_3D_onestep, (E_start, E_end), Omega, s, (X_start, X_end), Y)
            one_path_load_contribution += one_step
            return one_path_load_contribution     

        one_path_load_contribution += one_step

        #Update for next loop, given an accepted step
        X_start = X
        E_start = E
        Y_start = Y
        Omega_start = Omega
        s_start=s
        
    return one_path_load_contribution