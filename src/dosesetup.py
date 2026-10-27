
#July 2026
#Dose basic functions 

import numpy as np 
from math import prod, comb, cos, sin, exp, log
from itertools import product
from .doseparams import *

def exponential_map_sphere(Omega_n, strat_proj_brownian):
    """Update a direction vector along a geodesic on the unit sphere.

    The projected Brownian increment defines the tangent direction and the
    geodesic distance for the update. It is the diffusion multiplied by the
    Brownian increment in Stratonovich form; do not convert it to Itô form.

    Parameters
    ----------
    Omega_n : numpy.ndarray, shape (d,)
        Current unit direction vector.
    strat_proj_brownian : numpy.ndarray, shape (d,)
        Brownian increment projected onto the tangent space at
        ``Omega_n``.

    Returns
    -------
    numpy.ndarray, shape (d,)
        Updated direction vector on the sphere. 
        If the increment is zero, this is equal to ``Omega_n``.
    """
    norm = np.linalg.norm(strat_proj_brownian)

    if norm == 0:
        return np.asarray(Omega_n).copy() # copy is used so Omega_n is not modified

    return cos(norm) * Omega_n + sin(norm) * (strat_proj_brownian / norm) 

def reciprocal_stopping_power(E, a=ALPHA, p=p): #this is the drift 
    return a*p*E**(p-1)

def T_energy_stragg(E, kappa = KAPPA):
    return kappa*E*stopping_power(E)

def reference_cube_mapping(side_len, center, coordinate):
    """
    Maps the global coordinates inside a given voxel onto the reference cube [-1, 1] 

    Inputs: as numpy arrays 
        center = the (global) center of the given voxel 
        coordinate = the (global) coordinate that we wish to transform (i.e. could be a node or a proton position)

    Returns:
        np.array, the transformed coordinate in [-1, 1]^3 
    """
    return 2*(coordinate - center) / side_len

def Phi(side_len, X, n_i):
    """
    The basis function Phi_i attached to the ith node, defined over the whole domain 

    Inputs: as numpy arrays 
        X = np.array(x,y,z), a 2D or 3D position in space (at which Phi(ni) is evaluated)
        n_i = 2D or 3D coordinates of the node to which Phi is attached 

    Returns: t
        The value of Phi_i at one point, X_3D 
    """

    if not np.all(np.abs(X - n_i) <= side_len):
        return 0.0

    scaled = X / side_len
    floors = np.floor(scaled)
    ceils = np.ceil(scaled)

    #Preliminary center (will be correct if we are not on a voxel boundary)
    center = (floors + ceils) * side_len / 2

    # Boundary corrections
    for d in range(len(X)): #Testing each coord in turn
        if floors[d] == ceils[d]:
            node_index = round(n_i[d] / side_len) 
            if floors[d] == node_index: #if the node and position are on the same side of the cube
                center[d] = n_i[d] - side_len/2 #convention: push the midpoint backwards into a voxel (i.e. this will set this node coord to 1)
            else:
                center[d] = (node_index + floors[d]) * side_len / 2

    #Map inputs and node ni onto the reference cube:
    X_ref = reference_cube_mapping(side_len,center, X)
    n_ref = reference_cube_mapping(side_len,center, n_i)

    phi = np.prod((1 + n_ref * X_ref) / 2)
    return phi

def Phi_vectorised(side_len, X, nodes):
    """
    Receives all nodes and evaluates all corresponding Phi simultaneously at X.  
    """
    X = np.asarray(X)
    nodes = np.asarray(nodes)

    active_nodes_mask = np.all(np.abs(X-nodes) <= side_len, axis=1) 
    phis = np.zeros(len(nodes), dtype=float)
    active_nodes = nodes[active_nodes_mask]

    if len(active_nodes)==0:
        return phis

    #X tells you which voxel you are in
    scaled = X / side_len
    floors = np.floor(scaled)
    ceils = np.ceil(scaled)
    center = (floors + ceils) * side_len / 2 #an array 

    #We need this repeated for all active nodes (copy is needed else array is read only)
    centers = np.broadcast_to(center, active_nodes.shape).copy()

    #Correct centres for boundary X values:
    X_boundaries = floors == ceils
    active_bdry_node_indx = np.rint(active_nodes[:, X_boundaries] / side_len) #rowwise mask

    #Node and position same side of cube:
    same_side = floors[X_boundaries] == active_bdry_node_indx

    #Replace based on this condition (i.e. same side, not same side)
    centers[:, X_boundaries] = np.where(same_side, active_nodes[:, X_boundaries] - side_len/2, (active_bdry_node_indx + floors[X_boundaries])*side_len/2)

    X_ref = reference_cube_mapping(side_len, centers, X)
    n_ref = reference_cube_mapping(side_len, centers, active_nodes)

    #Fill in all nonzero values:
    phis[active_nodes_mask] = np.prod((1 + n_ref * X_ref) / 2, axis=1)
    return phis 

def position_lookup_matrix(X_meshgrid):
    """
    Provide the first array of an np.meshgrid, return a 1D position lookup matrix. Compute once.
    
    How to use the output matrix:
        Given an ijk index for the node on the grid, returns a 1D index 0,1,2,.. position for storage in F.
    """
    grid_shape = X_meshgrid.shape
    return np.arange(0, prod(grid_shape)).reshape(grid_shape)

#----------------------------GAUSSIAN BEAM---------------------------------------

def initial_energy_spread():
    """
    Gaussian sample the initial proton energy with % energy spread 
    """
    return np.random.normal(E0, energy_sdev**2)

def sample_initial_position(meshgrid, rng, side_len): 
    """
    Gaussian sample the initial beam position given the X0 mean. 
    The spread (i.e. the standard dev) should be order side_len
    """
    X0_mean,_,_ = choose_X0(meshgrid)

    if SPATIAL_DIM == 2:
        #we only need to sample the back "wall" where the beam originates
        x0,y0 = X0_mean
        width_sdev = width_sdev_factor * side_len
        y = rng.normal(loc=y0, scale=width_sdev)
        return np.array([x0, y], dtype=float)

    elif SPATIAL_DIM == 3:
        x0,y0,z0 = X0_mean
        width_sdev = width_sdev_factor * side_len
        y = rng.normal(loc=y0, scale=width_sdev)
        z = rng.normal(loc=z0, scale=width_sdev)
        return np.array([x0, y, z], dtype=float)

# def retrieve_h(setup_tuple):
#     """
#     Retrieves the final stepsize (which will be simulated) from the setup tuple
#     This exists purely to avoid mistakes -- can update + make setup_tuple and setup_dict instead 
#     """
#     return setup_tuple[5]

def domain_exit_check(X) -> bool:
    """
    Returns True if the position X is outside the domain D
    """
    return np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds)

#----------------------------MLMC----------------------------------------------

def coarsen_step(brownian_incs):
    """
    Take an input list/array of Brownian increments dB over some energy step, say h, can be any dimension of Brownians  
    Generate path one level more coarse than the input (each set of base fine steps -> sum -> one coarse step) 

    Must be sqrt(h)*dZ = dB! Brownian increments  
    """
    brownian_incs = np.asarray(brownian_incs)
    b_shape = brownian_incs.shape
    level_shape = log(b_shape[0])/log(base)
    if not(abs(level_shape - round(level_shape)) < 1e-6):
        raise ValueError(f"input (fine) brownian increments do not have geometric step size: "\
                         f"num_steps: {b_shape[0]}, level reading as {level_shape}, base: {base}.") 
    if len(b_shape) == 1: #1D array
        return brownian_incs.reshape(-1, base).sum(axis=1)
    elif len(b_shape) == 2:
        dim = b_shape[1]
        incs_list = []
        for idx in range(dim): #2D, 3D
            coarsened_inc = brownian_incs[:,idx].reshape(-1, base).sum(axis=1) 
            incs_list.append(coarsened_inc)
        incs_tuple = tuple(incs_list)
        coarsed_brownian=np.column_stack(incs_tuple)
        level_shape = log(coarsed_brownian.shape[0])/log(base)
        if not(abs(level_shape - round(level_shape)) < 1e-6):
            raise ValueError(f"coarsened brownian increments do not have geometric step size, but fine brownian incremements did. "\
                             f"num_steps: {coarsed_brownian.shape[0]}, level reading as {level_shape}, base: {base}.") 
        return np.column_stack(incs_tuple)
    else:
        raise ValueError(f"brownian_incs shape is incompatible, should be (num_steps,) or (num_steps, dimension), is: {b_shape}")
        
#-----------------------MASS MATRIX--------------------------------------------

def linear_lumped_mass_matrix(X_meshgrid, side_len):
    """
    Calculates the diagonal of the lumped (row sum) mass matrix for (Q1) bi/trilinear basis functions over a regular cubic voxel mesh side length side_len
    Exploits the symmetry and mapping onto the reference cube. Works in 2D or 3D.
    
    Inputs:
        meshgrid: tuple, X,Y,Z
                         X,Y,Z = np.arrays from np.meshgrid indexed by i,j(,k)j

    Returns:
        M_diag, 1D np.array, shape = (number of nodes,) contains the diagonal elements of the diagonal matrix M 
    """
    Xs = X_meshgrid #This is the beam axis so is always there
    nodes_shape = Xs.shape 

    #Useful integral values (no prefactor)
    I1 = 4/3 #if different coordinates (1 and -1)
    I2 = 8/3 #if same coords (1 and 1, or -1 and -1)

    lower_pos = tuple([0]*SPATIAL_DIM)
    pos_1 = tuple([1] + [0]*(SPATIAL_DIM - 1))
    regular_side_length = Xs[pos_1] - Xs[lower_pos]
    print(f'Regular side length is reading as {regular_side_length:.8f}, should be {side_len}.')
    if not(np.isclose(regular_side_length, side_len)): #Only tests if they are almost the same 
        raise ValueError("Meshgrid sidelength must be side_len and it must be uniform.")

    #Store the values of all possible volume integrals: 
    integral_prefactor = (1/(2**SPATIAL_DIM))**2 * (side_len/2)**(SPATIAL_DIM) #From the chain rule and normalisation of basis functions
    integral_values = np.zeros(SPATIAL_DIM + 1) #Index 0: 0 differences (i.e. same node), 1: 1 coord different, ...
    for num_differences in range(SPATIAL_DIM + 1):
        integral_values[num_differences] = I2**(SPATIAL_DIM - num_differences) * I1**(num_differences)
    integral_values *= integral_prefactor 

    #If an interior point, the number of nodes of each type surrounding them is:
    interior_instance_num = [1]*(SPATIAL_DIM + 1) #Start with ni=nj
    for i in range(1, SPATIAL_DIM + 1):
        #in 3D this should be: [1, 6, 12, 8], in 2D this should be: [1, 4, 4] 
        interior_instance_num[i] = comb(SPATIAL_DIM, i) * 2**i
    interior_instance_num = np.array(interior_instance_num)
    
    #For each type of node, the intersection of supports also changes - must add over all available voxels as well
    #The number of voxels in the intersection of supports is the weight 
    #Exploit symmetry between ni and each node nj of the same type 

    interior_weights = [2**i for i in range(SPATIAL_DIM + 1)]
    interior_weights = interior_weights[::-1]

    #Obtain the total number of integrals of each type over ONE reference cube
    interior_instance_num = interior_instance_num * interior_weights

    #Finally total up the sum of all integrals relating to ni:
    interior_sum = np.dot(interior_instance_num, integral_values)

    #Set up the diagonal component of M - set all to the interior value 
    num_nodes = prod(nodes_shape)
    M_diag = np.ones(num_nodes) * interior_sum #1d array to contain the diagonal elements

    #For a node on a boundary, we need to replace the value of Mii from above

    lower_boundary_index = np.zeros(SPATIAL_DIM) #Index starts from 0 in all coords
    upper_boundary_index = np.array(nodes_shape) - 1 #Should give the total number of nodes in each dim, minus 1 since we start indexing at 0
    
    if np.any(lower_boundary_index == upper_boundary_index): #If any were equal then it's just a line
        raise ValueError("Upper and Lower boundary values are the same in at least one dimension.")

    #Make the weight dictionary dependent on how many boundaries ni/nj lie on 

    boundary_weighted_integrals = {}
    ni_boundary_masks = list(product([0, 1], repeat=SPATIAL_DIM)) #1 if ni touches that boundary
    nj_equal_masks = list(product([0, 1], repeat=SPATIAL_DIM)) #1 if nj and ni match that coordinate (will be used to check if nj is also on a boundary)

    #For each combination we will calculate the weight:
    #Boundary mask is over ni, equal mask is over nj 
    for boundary_mask, equal_mask in product(ni_boundary_masks, nj_equal_masks):

        #Calculate the weight of the node nj (i.e. num voxels)
        nj_num_boundaries = sum(b==1 and e==1 for b,e in zip(boundary_mask, equal_mask)) #Count the number of times nj is on a boundary
        nj_num_differences = SPATIAL_DIM - np.sum(equal_mask) 
        
        #The weight between ni and nj halves once for each boundary nj is also on 
        weight = interior_weights[nj_num_differences] * (1/2) ** (nj_num_boundaries)
        
        #Calculate the integral over a single voxel and node pair
        integral = integral_values[nj_num_differences]
        
        #Store the total contribution from this nj: 
        boundary_weighted_integrals[(boundary_mask, equal_mask)] = weight * integral

    #Update the values of Mii for the boundary nodes:
    counter = 0
    M_numbering = np.arange(0, num_nodes).reshape(nodes_shape) #numbers 0, 1, 2, ... assigned to the nodes at the same indicies as in Xs

    for index in product(*(range(n) for n in nodes_shape)): #Index should be (i,j,k) numbers (like meshgrid)
        index = np.asarray(index)
        lower_test = index == lower_boundary_index
        upper_test = index == upper_boundary_index

        
        if np.any(lower_test) or np.any(upper_test): #Is ni on the boundary
            boundary_mask = lower_test + upper_test #Is 1 wherever ni is on a boundary 

            Mii = 0
            #Run over all possible equal_masks for the points nj 
            for equal_mask in nj_equal_masks:
                #In the coordindates which are not fixed to ni, we have 1 way to move if on boundary, or 2 if not 

                equal_mask_arr = np.array(equal_mask)
                difference_mask = 1 - equal_mask_arr #Is 1 if nj is different from ni in this direction
                
                number_of_nodes = 1
                for coord in range(SPATIAL_DIM):
                    if difference_mask[coord] == 1 and boundary_mask[coord] == 0:
                        number_of_nodes*= 2            
                
                boundary_mask_tuple = tuple(boundary_mask)
                Mii += number_of_nodes * boundary_weighted_integrals[(boundary_mask_tuple, equal_mask)]
            
            #Update M - Retrieve the index i of Mii to be changed (this is convention!)
            position = M_numbering[tuple(index)]
            M_diag[position] = Mii
            counter+=1 
                      
    #Final check using the known number of interior nodes:
    num_interior_nodes = np.array(Xs.shape) - 2*np.ones(SPATIAL_DIM, dtype=int) #Subtract the number of boundary nodes 
    if np.any(num_interior_nodes <= 0):
        print("Note: there are no interior nodes.")
    else:
        num_interior_nodes = np.prod(num_interior_nodes)
        if num_nodes - num_interior_nodes != counter:
            print(f"Warning: boundary node values have not updates correctly. {counter} boundary nods were updated, but there should have been {num_nodes - num_interior_nodes}.") 

    return M_diag

def load_mass_matrix():
    M_path = os.path.join(file_path, f"{SPATIAL_DIM}D_{dose_shape}_lumped_mass_matrix.npz")
    if os.path.exists(M_path):
        mass_data = np.load(M_path)
        M_diag=mass_data['M_diag']
        mass_shape=M_diag.shape[0]
        if mass_shape != dose_shape:
            raise ValueError(f"Lumped mass matrix shape must equal {dose_shape}, is {mass_shape}.") 
        print("Mass matrix loaded successfully.")
    else:
        M_diag = linear_lumped_mass_matrix(X_meshgrid,side_len)
        np.savez(M_path, M_diag=M_diag, side_len = side_len, X = X_meshgrid, Y = Y_meshgrid)
    return M_diag
