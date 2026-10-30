# July 2026
# Define the shape functions

import numpy as np
from src.parameterisation import *

def reference_cube_mapping(side_len, center, coordinate):
    """Map a global coordinate in a voxel to the reference cube ``[-1, 1]^d``.

    Parameters
    ----------
    side_len : float
        Side length of the voxel.
    center : numpy.ndarray, shape (d,)
        Global coordinates of the voxel center.
    coordinate : numpy.ndarray, shape (d,)
        Global coordinates to map, such as a node or proton position.

    Returns
    -------
    numpy.ndarray, shape (d,)
        Coordinate mapped relative to the voxel center. The voxel center maps
        to zero and its corners map to coordinates with components ``-1`` or
        ``1``.
    """
    return 2 * (coordinate - center) / side_len


def Phi(side_len, X, n_i):
    """Evaluate the basis function ``Phi_i`` attached to node ``i``.

    The basis function is defined over the whole domain. ``X`` is the position
    at which ``Phi_i`` is evaluated, and ``n_i`` is the coordinate of the node
    to which the basis function is attached. The function is zero outside the
    node's support. Coordinates on voxel boundaries follow the mesh convention.

    Parameters
    ----------
    side_len : float
        Side length of a mesh voxel.
    X : numpy.ndarray, shape (d,)
        2D or 3D position at which to evaluate ``Phi_i``.
    n_i : numpy.ndarray, shape (d,)
        2D or 3D coordinates of the node to which ``Phi_i`` is attached.

    Returns
    -------
    float
        Basis function value at ``X``. Returns zero when ``X`` is outside
        the node's support.
    """

    if not np.all(np.abs(X - n_i) <= side_len):
        return 0.0

    scaled = X / side_len
    floors = np.floor(scaled)
    ceils = np.ceil(scaled)

    # Preliminary center (will be correct if we are not on a voxel boundary)
    center = (floors + ceils) * side_len / 2

    # Boundary corrections
    for d in range(len(X)):  # Testing each coord in turn
        if floors[d] == ceils[d]:
            node_index = round(n_i[d] / side_len)
            if floors[d] == node_index:  # if the node and position are on the same side of the cube
                center[d] = (
                    n_i[d] - side_len / 2
                )  # convention: push the midpoint backwards into a voxel (i.e. this will set this node coord to 1)
            else:
                center[d] = (node_index + floors[d]) * side_len / 2

    # Map inputs and node ni onto the reference cube:
    X_ref = reference_cube_mapping(side_len, center, X)
    n_ref = reference_cube_mapping(side_len, center, n_i)

    phi = np.prod((1 + n_ref * X_ref) / 2)
    return phi


def Phi_vectorised(side_len, X, nodes):
    """Evaluate basis functions for multiple nodes at one point.

    Computes the same values as calling :func:`Phi` once for each node, while
    evaluating all supplied nodes together. Nodes outside the point's support
    receive a value of zero. Voxel-boundary coordinates follow the same
    convention as :func:`Phi`.

    Parameters
    ----------
    side_len : float
        Side length of a mesh voxel.
    X : numpy.ndarray, shape (d,)
        2D or 3D point at which to evaluate the basis functions.
    nodes : numpy.ndarray, shape (n_nodes, d)
        Coordinates of the mesh nodes whose basis functions are evaluated.

    Returns
    -------
    numpy.ndarray, shape (n_nodes,)
        Basis function value for each row of ``nodes``, in the same order.
    """
    X = np.asarray(X)
    nodes = np.asarray(nodes)

    active_nodes_mask = np.all(np.abs(X - nodes) <= side_len, axis=1)
    phis = np.zeros(len(nodes), dtype=float)
    active_nodes = nodes[active_nodes_mask]

    if len(active_nodes) == 0:
        return phis

    # X tells you which voxel you are in
    scaled = X / side_len
    floors = np.floor(scaled)
    ceils = np.ceil(scaled)
    center = (floors + ceils) * side_len / 2  # an array

    # We need this repeated for all active nodes (copy is needed else array is read only)
    centers = np.broadcast_to(center, active_nodes.shape).copy()

    # Correct centres for boundary X values:
    X_boundaries = floors == ceils
    active_bdry_node_indx = np.rint(active_nodes[:, X_boundaries] / side_len)  # rowwise mask

    # Node and position same side of cube:
    same_side = floors[X_boundaries] == active_bdry_node_indx

    # Replace based on this condition (i.e. same side, not same side)
    centers[:, X_boundaries] = np.where(
        same_side,
        active_nodes[:, X_boundaries] - side_len / 2,
        (active_bdry_node_indx + floors[X_boundaries]) * side_len / 2,
    )

    X_ref = reference_cube_mapping(side_len, centers, X)
    n_ref = reference_cube_mapping(side_len, centers, active_nodes)

    # Fill in all nonzero values:
    phis[active_nodes_mask] = np.prod((1 + n_ref * X_ref) / 2, axis=1)
    return phis
