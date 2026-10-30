# July 2026
# Dose basic functions

import numpy as np
from math import prod, comb, ceil
from itertools import product
from src.parameterisation._fixed_params import *
from src.parameterisation.core_params import *
import warnings

def calculate_R0():
    """
    Calculate the proton range using the Bragg Kleeman model.
    Parameters are imported and set to defaults in src/parameterisation.

    Returns
    -------
    float
        Value of the Bragg Kleeman proton range.
    """
    return ALPHA*E0**p

def nodes(side_len):
    """
    Generate node coordinate grids for closed cuboid domain D, assigned in src/parameterisation, including endpoints.
    Node coordinates are integer multiples of ``side_len``.

    Parameters
    ----------
    side_len : float

    Returns
    -------
    tuple of numpy.ndarray
        Coordinate grids ``(X, Y)`` in 2D or ``(X, Y, Z)`` in 3D.
        Meshgrid structure.
    """
    R0=calculate_R0()
    range_upper = R0*range_allowance
    lower_x_bd = -9 #needs to be large enough to get hit w ~0% chance
                    #if varying parameters -- this will need to be adjusted

    #Make sure the lengths are divisible by side_len
    beam_axis_depth = ceil(range_upper / side_len)*side_len
    height = ceil(range_upper*y_scaling / side_len)*side_len #height from the origin lower corner

    #Calculate the total number of nodes in each direction
    nx = int(round(beam_axis_depth / side_len)) + 1
    ny = int(round(height / side_len)) + 1

    #Turn these into node positions with side_len
    xs = np.arange(lower_x_bd, nx) * side_len
    ys = np.arange(ny) * side_len

    if SPATIAL_DIM==3:
        zs = ys #the y and z directions are the same
        X,Y,Z=np.meshgrid(xs,ys,zs, indexing='ij')
        return X,Y,Z
    elif SPATIAL_DIM==2:
        X,Y=np.meshgrid(xs,ys, indexing='ij')
        return X,Y
    else:
        raise ValueError("Dimension (dim) must be either 2D or 3D.")

def choose_X0(meshgrid):
    """
    Calculate length, depth and width of the cuboid domain. Assumes bounded below by zero in all dims.
    Then returns X0, the expected initial position of the Gaussian beam.

    meshgrid: tuple (X, Y, Z) indexing ij
    """
    domain_lower_bounds = np.zeros(SPATIAL_DIM)

    if SPATIAL_DIM==3:
        X_meshgrid, Y_meshgrid, Z_meshgrid = meshgrid
        upper_X = X_meshgrid[-1,-1,-1]
        upper_Y = Y_meshgrid[-1,-1,-1]
        lower_X = X_meshgrid[0,0,0]
        lower_Y = Y_meshgrid[0,0,0]
        lower_Z = Z_meshgrid[0,0,0]
        upper_Z = Z_meshgrid[-1,-1,-1] #hi

        domain_upper_bounds = np.array([upper_X, upper_Y, upper_Z])
        domain_lower_bounds = np.array([lower_X, lower_Y, lower_Z])

        y0 = (lower_Y + upper_Y)*1/2
        z0 = (lower_Z + upper_Z)*1/2
        X0 = np.array([0.0,y0,z0])

    elif SPATIAL_DIM ==2:
        X_meshgrid, Y_meshgrid = meshgrid
        upper_X = X_meshgrid[-1,-1]
        lower_X = X_meshgrid[0,0]
        upper_Y = Y_meshgrid[-1,-1]
        lower_Y = Y_meshgrid[0,0] #This file relies on this being 0

        domain_upper_bounds = np.array([upper_X, upper_Y])
        domain_lower_bounds = np.array([lower_X, lower_Y])

        y0 = (lower_Y + upper_Y)*1/2
        X0 = np.array([0.0,y0])

    return X0, domain_lower_bounds, domain_upper_bounds

def domain_exit_check(X, domain_upper_bounds, domain_lower_bounds) -> bool:
    """
    Returns True if the position X is outside the domain D
    """
    return np.any(X > domain_upper_bounds) or np.any(X < domain_lower_bounds)

def position_lookup_matrix(X_meshgrid):
    """Give each point in a mesh its position in a one-dimensional list.

    The returned array has the same shape as the mesh. Each entry contains
    the point's index in a flattened version of the mesh, starting at zero.

    Parameters
    ----------
    X_meshgrid : numpy.ndarray
        A coordinate array from the meshgrid. Only its shape is used.

    Returns
    -------
    numpy.ndarray
        Integer array with the same shape as ``X_meshgrid``. Its values are
        consecutive positions in NumPy's default flattened order.
    """
    grid_shape = X_meshgrid.shape
    return np.arange(0, prod(grid_shape)).reshape(grid_shape)

def linear_lumped_mass_matrix(X_meshgrid, side_len, spatial_dim=SPATIAL_DIM):
    """
    Calculate the lumped mass matrix diagonal for Q1 basis functions.

    Uses a regular square or cubic mesh with element side length ``side_len``.
    Assumes matching mass matrix and meshgrid node ordering.

    Parameters
    ----------
    meshgrid : tuple of numpy.ndarray
    side_len : float
    spatial_dim : int

    Returns
    -------
    M_diag : numpy.ndarray
        One-dimensional array of diagonal entries, one per node.
    """
    Xs = X_meshgrid  # This is the beam axis so is always there
    nodes_shape = Xs.shape

    print(Xs)
    print(Xs.shape)
    print(np.transpose(Xs)[0].shape)

    # Useful integral values (no prefactor)
    I1 = 4 / 3  # if different coordinates (1 and -1)
    I2 = 8 / 3  # if same coords (1 and 1, or -1 and -1)

    # Test the side length along X direction only
    if Xs.shape[0] == 1:
        warnings.warn("Only one X coordinate, side length not well defined.")
    elif not np.allclose(np.diff(Xs, axis=0), side_len):
        raise ValueError("Non-uniform X spacing and sidelength did not match, other dimensions not tested.")

    # Store the values of all possible volume integrals:
    # From the chain rule and normalisation of basis functions
    integral_prefactor = (1 / (2**spatial_dim)) ** 2 * (side_len / 2) ** (spatial_dim)
    integral_values = np.zeros(spatial_dim + 1)  # Index 0: 0 differences (i.e. same node), 1: 1 coord different, ...
    for num_differences in range(spatial_dim + 1):
        integral_values[num_differences] = I2 ** (spatial_dim - num_differences) * I1 ** (num_differences)
    integral_values *= integral_prefactor

    # If an interior point, the number of nodes of each type surrounding them is:
    interior_instance_num = [1] * (spatial_dim + 1)  # Start with ni=nj
    for i in range(1, spatial_dim + 1):
        # in 3D this should be: [1, 6, 12, 8], in 2D this should be: [1, 4, 4]
        interior_instance_num[i] = comb(spatial_dim, i) * 2**i
    interior_instance_num = np.array(interior_instance_num)

    # For each type of node, the intersection of supports also changes - must add over all available voxels as well
    # The number of voxels in the intersection of supports is the weight
    # Exploit symmetry between ni and each node nj of the same type

    interior_weights = [2**i for i in range(spatial_dim + 1)]
    interior_weights = interior_weights[::-1]

    # Obtain the total number of integrals of each type over ONE reference cube
    interior_instance_num = interior_instance_num * interior_weights

    # Finally total up the sum of all integrals relating to ni:
    interior_sum = np.dot(interior_instance_num, integral_values)

    # Set up the diagonal component of M - set all to the interior value
    num_nodes = prod(nodes_shape)
    M_diag = np.ones(num_nodes) * interior_sum  # 1d array to contain the diagonal elements

    # For a node on a boundary, we need to replace the value of Mii from above

    lower_boundary_index = np.zeros(spatial_dim)  # Index starts from 0 in all coords
    # Should give the total number of nodes in each dim, minus 1 since we start indexing at 0
    upper_boundary_index = np.array(nodes_shape) - 1

    if np.any(lower_boundary_index == upper_boundary_index):  # If any were equal then it's just a line
        raise ValueError("Upper and Lower boundary values are the same in at least one dimension.")

    # Make the weight dictionary dependent on how many boundaries ni/nj lie on

    boundary_weighted_integrals = {}
    ni_boundary_masks = list(product([0, 1], repeat=spatial_dim))  # 1 if ni touches that boundary
    # 1 if nj and ni match that coordinate (will be used to check if nj is also on a boundary)
    nj_equal_masks = list(product([0, 1], repeat=spatial_dim))

    # For each combination we will calculate the weight:
    # Boundary mask is over ni, equal mask is over nj
    for boundary_mask, equal_mask in product(ni_boundary_masks, nj_equal_masks):
        # Calculate the weight of the node nj (i.e. num voxels)
        # Count the number of times nj is on a boundary
        nj_num_boundaries = sum(b == 1 and e == 1 for b, e in zip(boundary_mask, equal_mask))
        nj_num_differences = spatial_dim - np.sum(equal_mask)

        # The weight between ni and nj halves once for each boundary nj is also on
        weight = interior_weights[nj_num_differences] * (1 / 2) ** (nj_num_boundaries)

        # Calculate the integral over a single voxel and node pair
        integral = integral_values[nj_num_differences]

        # Store the total contribution from this nj:
        boundary_weighted_integrals[(boundary_mask, equal_mask)] = weight * integral

    # Update the values of Mii for the boundary nodes:
    counter = 0
    # Count the number of times nj is on a boundary
    M_numbering = np.arange(0, num_nodes).reshape(nodes_shape)

    for index in product(*(range(n) for n in nodes_shape)):  # Index should be (i,j,k) numbers (like meshgrid)
        index = np.asarray(index)
        lower_test = index == lower_boundary_index
        upper_test = index == upper_boundary_index

        if np.any(lower_test) or np.any(upper_test):  # Is ni on the boundary
            boundary_mask = lower_test + upper_test  # Is 1 wherever ni is on a boundary

            Mii = 0
            # Run over all possible equal_masks for the points nj
            for equal_mask in nj_equal_masks:
                # In the coordindates which are not fixed to ni, we have 1 way to move if on boundary, or 2 if not

                equal_mask_arr = np.array(equal_mask)
                difference_mask = 1 - equal_mask_arr  # Is 1 if nj is different from ni in this direction

                number_of_nodes = 1
                for coord in range(spatial_dim):
                    if difference_mask[coord] == 1 and boundary_mask[coord] == 0:
                        number_of_nodes *= 2

                boundary_mask_tuple = tuple(boundary_mask)
                Mii += number_of_nodes * boundary_weighted_integrals[(boundary_mask_tuple, equal_mask)]

            # Update M - Retrieve the index i of Mii to be changed (this is convention!)
            position = M_numbering[tuple(index)]
            M_diag[position] = Mii
            counter += 1

    # Final check using the known number of interior nodes:
    num_interior_nodes = np.array(Xs.shape) - 2 * np.ones(
        spatial_dim, dtype=int
    )  # Subtract the number of boundary nodes
    if np.any(num_interior_nodes <= 0):
        print("Note: there are no interior nodes.")
    else:
        num_interior_nodes = np.prod(num_interior_nodes)
        if num_nodes - num_interior_nodes != counter:
            print(
                f"Warning: boundary node values have not updated correctly. "
                f"{counter} boundary nodes were updated, but there should have been "
                f"{num_nodes - num_interior_nodes}."
            )

    return M_diag

