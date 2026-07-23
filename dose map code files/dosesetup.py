
# Parameters and basic functions 

import numpy as np 
from math import ceil, prod, comb, cos, sin
from itertools import product
import doseparams as dp

spatial_dim = dp.SPATIAL_DIM
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

#----1. FUNCTIONS REQUIRED FOR SETUP

def choose_l():
    """
    Calculates l given the number of simulations N for the given dimension, for the FEM method.
    """
    N = sims_per_CPU * num_CPUs
    return N**(-1/(2*spatial_dim))

def choose_Omega_0():
    if spatial_dim == 3:
        Omega0 = np.array([1.0, 0.0, 0.0]) #Must be on the unit sphere
    elif spatial_dim == 2:
        Omega0 = np.array([1.0, 0.0]) #Beam axis is the x axis, must be on the unit sphere 
    return Omega0 

def exponential_map_sphere(Omega_n, strat_proj_brownian):
    """
    The exponential map Exp_{Omega}(y) on the sphere acts on the term y to generate each update.
    It follows the geodesic starting from the value of the scheme at the previous step (Omega_n)
    which is constructed by following the direction y for unit time. (see ref)
    y is the diffusion * brownian term in Strat form (i.e. drift is 0, do not convert to Ito form) 
    """
    norm = np.linalg.norm(strat_proj_brownian)
    return cos(norm) * Omega_n + sin(norm) * (strat_proj_brownian / norm) 

def stopping_power(E, a = alpha, p = p):
    return 1/(a*p) * E**(1-p)

def reciprocal_stopping_power(E, a=alpha, p=p): #this is the drift 
    return a*p*E**(p-1)

def T_energy_stragg(E, kappa = KAPPA):
    return kappa*E*stopping_power(E)

#Path length model:

def n_V_log_energy_drift(L):
    """
    L = exp(Yn) must be calculated in the main code
    """
    return -stopping_power(L)*L**(-1) - 0.5*T_energy_stragg(L)*L**(-2)

def n_V_log_energy_diffusion(L):
    return np.sqrt(T_energy_stragg(L))*L**(-1)

def n_V_angular_drift(Omega, eps_0=eps_0):
    return -2*eps_0*Omega #Make sure that you correct this

def n_V_angular_diffusion(Omega, eps_0=eps_0):
    return np.sqrt(2*eps_0)*(np.eye(3) - np.outer(Omega, Omega))

#Energy model:

def n_KZ_path_length_diffusion(E, a=alpha, p=p):
    return a*p*np.sqrt(KAPPA)*E**(p-0.5)

def n_KZ_angular_diffusion(E, Omega, eps_0=eps_0, a=alpha, p=p):
    return np.sqrt(2*eps_0 * a*p)*E**(p/2 - 0.5) * (np.eye(3) - np.outer(Omega, Omega))

#For the dose map:

def calculate_R0():
    """
    Analytical approx for the max range of protons 
    """
    return alpha*E0**p 

def nodes(l):
    """
    Defines the length dimensions of the cuboid D and gives the positions of all nodes in D 
    The origin is placed at the corner of the face of domain D where the beam originates, and is such that the coords within D are all positive 
    ALL nodes coordinates must be exactly divisible by the side length l

    Inputs: 
            range_allowance, float in (1, 2): R0*range_allowance \approx length of D along beam axis 
            y_scaling, float in (0, 1): length of D*y_scaling \approx the width and depth in the other two directions

    Returns: tuple X,Y,(Z): type(X) = np.array ; X.shape = (number of nodes along x axis, y sim, z sim)
                                                                including the endpoints! 
                          
    Notes: X[i,j,k] will retrieve the x coordinate of the ith,jth,kth node
           i,j,k=0,0,0 -> (x,y,z)=(0,-1,-1) 
           
    """
    R0 = calculate_R0()
    range_upper = R0*range_allowance
    
    #Make sure the lengths are divisible by l
    beam_axis_depth = ceil(range_upper / l)*l 
    height = ceil(range_upper*y_scaling / l)*l #height from the origin lower corner   
    
    #Calculate the total number of nodes in each direction 
    nx = int(round(beam_axis_depth / l)) + 1
    ny = int(round(height / l)) + 1 
    
    #Turn these into node positions with l
    xs = np.arange(nx) * l
    ys = np.arange(ny) * l  
    
    if spatial_dim==3:
        zs = ys #the y and z directions are the same 
        X,Y,Z=np.meshgrid(xs,ys,zs, indexing='ij')
        return X,Y,Z
    elif spatial_dim==2:
        X,Y=np.meshgrid(xs,ys, indexing='ij') 
        return X,Y
    else:
        raise ValueError("Dimension (dim) must be either 2D or 3D.")
    
def reference_cube_mapping(l,center, coordinate):
    """
    Maps the global coordinates inside a given voxel onto the reference cube [-1, 1] 

    Inputs: as numpy arrays 
        center = the (global) center of the given voxel 
        coordinate = the (global) coordinate that we wish to transform (i.e. could be a node or a proton position)

    Returns:
        np.array, the transformed coordinate in [-1, 1]^3 
    """
    return 2*(coordinate - center) / l

def Phi(l, X, n_i):
    """
    The basis function Phi_i attached to the ith node, defined over the whole domain 

    Inputs: as numpy arrays 
        X = np.array(x,y,z), a 2D or 3D position in space (at which Phi(ni) is evaluated)
        n_i = 2D or 3D coordinates of the node to which Phi is attached 

    Returns: t
        The value of Phi_i at one point, X_3D 
    """

    if not np.all(np.abs(X - n_i) <= l):
        return 0.0

    scaled = X / l
    floors = np.floor(scaled)
    ceils = np.ceil(scaled)

    #Preliminary center (will be correct if we are not on a voxel boundary)
    center = (floors + ceils) * l / 2

    # Boundary corrections
    for d in range(len(X)): #Testing each coord in turn
        if floors[d] == ceils[d]:
            node_index = round(n_i[d] / l) 
            if floors[d] == node_index: #if the node and position are on the same side of the cube
                center[d] = n_i[d] - l/2 #convention: push the midpoint backwards into a voxel (i.e. this will set this node coord to 1)
            else:
                center[d] = (node_index + floors[d]) * l / 2

    #Map inputs and node ni onto the reference cube:
    X_ref = reference_cube_mapping(center, X)
    n_ref = reference_cube_mapping(center, n_i)

    phi = np.prod((1 + n_ref * X_ref) / 2)
    return phi

def position_lookup_matrix(X_meshgrid):
    """
    Provide the first array of an np.meshgrid, return a 1D position lookup matrix. Compute once.
    
    How to use the output matrix:
        Given an ijk index for the node on the grid, returns a 1D index 0,1,2,.. position for storage in F.
    """
    grid_shape = X_meshgrid.shape
    return np.arange(0, prod(grid_shape)).reshape(grid_shape)

#----mass matrix

def linear_lumped_mass_matrix(X_meshgrid, l):
    """
    Calculates the diagonal of the lumped (row sum) mass matrix for (Q1) bi/trilinear basis functions over a regular cubic voxel mesh side length l
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

    lower_pos = tuple([0]*spatial_dim)
    pos_1 = tuple([1] + [0]*(spatial_dim - 1))
    regular_side_length = Xs[pos_1] - Xs[lower_pos]
    print(f'Regular side length is reading as {regular_side_length:.8f}, should be {l}.')
    if not(np.isclose(regular_side_length, l)): #Only tests if they are almost the same 
        raise ValueError("Meshgrid sidelength must be l and it must be uniform.")

    #Store the values of all possible volume integrals: 
    integral_prefactor = (1/(2**spatial_dim))**2 * (l/2)**(spatial_dim) #From the chain rule and normalisation of basis functions
    integral_values = np.zeros(spatial_dim + 1) #Index 0: 0 differences (i.e. same node), 1: 1 coord different, ...
    for num_differences in range(spatial_dim + 1):
        integral_values[num_differences] = I2**(spatial_dim - num_differences) * I1**(num_differences)
    integral_values *= integral_prefactor 

    #If an interior point, the number of nodes of each type surrounding them is:
    interior_instance_num = [1]*(spatial_dim + 1) #Start with ni=nj
    for i in range(1, spatial_dim + 1):
        #in 3D this should be: [1, 6, 12, 8], in 2D this should be: [1, 4, 4] 
        interior_instance_num[i] = comb(spatial_dim, i) * 2**i
    interior_instance_num = np.array(interior_instance_num)
    
    #For each type of node, the intersection of supports also changes - must add over all available voxels as well
    #The number of voxels in the intersection of supports is the weight 
    #Exploit symmetry between ni and each node nj of the same type 

    interior_weights = [2**i for i in range(spatial_dim + 1)]
    interior_weights = interior_weights[::-1]

    #Obtain the total number of integrals of each type over ONE reference cube
    interior_instance_num = interior_instance_num * interior_weights

    #Finally total up the sum of all integrals relating to ni:
    interior_sum = np.dot(interior_instance_num, integral_values)

    #Set up the diagonal component of M - set all to the interior value 
    num_nodes = prod(nodes_shape)
    M_diag = np.ones(num_nodes) * interior_sum #1d array to contain the diagonal elements

    #For a node on a boundary, we need to replace the value of Mii from above

    lower_boundary_index = np.zeros(spatial_dim) #Index starts from 0 in all coords
    upper_boundary_index = np.array(nodes_shape) - 1 #Should give the total number of nodes in each dim, minus 1 since we start indexing at 0
    
    if np.any(lower_boundary_index == upper_boundary_index): #If any were equal then it's just a line
        raise ValueError("Upper and Lower boundary values are the same in at least one dimension.")

    #Make the weight dictionary dependent on how many boundaries ni/nj lie on 

    boundary_weighted_integrals = {}
    ni_boundary_masks = list(product([0, 1], repeat=spatial_dim)) #1 if ni touches that boundary
    nj_equal_masks = list(product([0, 1], repeat=spatial_dim)) #1 if nj and ni match that coordinate (will be used to check if nj is also on a boundary)

    #For each combination we will calculate the weight:
    #Boundary mask is over ni, equal mask is over nj 
    for boundary_mask, equal_mask in product(ni_boundary_masks, nj_equal_masks):

        #Calculate the weight of the node nj (i.e. num voxels)
        nj_num_boundaries = sum(b==1 and e==1 for b,e in zip(boundary_mask, equal_mask)) #Count the number of times nj is on a boundary
        nj_num_differences = spatial_dim - np.sum(equal_mask) 
        
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
                for coord in range(spatial_dim):
                    if difference_mask[coord] == 1 and boundary_mask[coord] == 0:
                        number_of_nodes*= 2            
                
                boundary_mask_tuple = tuple(boundary_mask)
                Mii += number_of_nodes * boundary_weighted_integrals[(boundary_mask_tuple, equal_mask)]
            
            #Update M - Retrieve the index i of Mii to be changed (this is convention!)
            position = M_numbering[tuple(index)]
            M_diag[position] = Mii
            counter+=1 
                      
    #Final check using the known number of interior nodes:
    num_interior_nodes = np.array(Xs.shape) - 2*np.ones(spatial_dim, dtype=int) #Subtract the number of boundary nodes 
    if np.any(num_interior_nodes <= 0):
        print("Note: there are no interior nodes.")
    else:
        num_interior_nodes = np.prod(num_interior_nodes)
        if num_nodes - num_interior_nodes != counter:
            print(f"Warning: boundary node values have not updates correctly. {counter} boundary nods were updated, but there should have been {num_nodes - num_interior_nodes}.") 

    return M_diag

