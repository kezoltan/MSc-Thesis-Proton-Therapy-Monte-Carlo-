"""Behavior-focused test names for ``src.dosesetup``.

Implement these cases incrementally, starting with the pure helper functions.
"""
import numpy as np 
from math import prod, comb, cos, sin, exp, log
from itertools import product
from src.doseparams import *
import pytest
from src.dosesetup import (
    exponential_map_sphere,
    reciprocal_stopping_power,
    T_energy_stragg,
    reference_cube_mapping,
    Phi,
    Phi_vectorised,
)

@pytest.mark.parametrize(
    ("omega", "increment", "expected"), # omega = initial direction
    [
        # start along x and turn towards y by 0.3 rad
        (np.array([1.0, 0.0]), np.array([0.0, 0.3]), np.array([np.cos(0.3), np.sin(0.3)])),
        # start along y and turn towards x by 0.3 rad
        (np.array([0.0, 1.0]), np.array([0.2, 0.0]), np.array([np.sin(0.2), np.cos(0.2)])),
        # in 3d start on x and turn towards z by 0.4 rad
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.4]),
            np.array([np.cos(0.4), 0.0, np.sin(0.4)]),
        ),
        # start along x and turn an increment larger than pi.
        (np.array([1.0, 0.0]), np.array([0.0, 4.0]), np.array([np.cos(4.0), np.sin(4.0)])),
    ],
)
def test_exponential_map_sphere_returns_unit_direction(omega, increment, expected):
    # call the function that is being tested - 
    result = exponential_map_sphere(omega, increment)

    # check that the expected result occured
    assert np.allclose(result, expected)
    # check that the updated direction still has length 1
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_exponential_map_sphere_handles_zero_increment():
    omega = np.array([1.0, 0.0])
    zero_increment = np.zeros_like(omega)

    result = exponential_map_sphere(omega, zero_increment)

    assert np.array_equal(result, omega)


@pytest.mark.parametrize(
    ("energy", "a", "exponent"),
    [
        (1.0, 0.5, 2.0), # todo : check with katie what valid inputs are
        (4.0, 0.25, 1.5),
        (10.0, 0.01, 0.75),
    ],
)
def test_reciprocal_stopping_power_matches_drift_formula(energy, a, exponent):
    # calculate expected drift from its formula: apE^(p - 1).
    expected = a * exponent * energy ** (exponent - 1)

    # call the function being tested with explicit parameters, independent of module defaults.
    result = reciprocal_stopping_power(energy, a=a, p=exponent)

    # check the floating-point values with a numerical tolerance.
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    ("energy", "kappa"),
    [
        (1.0, 0.0),
        (10.0, 1e-6),
        (62.0, 4e-5),
    ],
)
def test_energy_straggling_term_matches_formula(energy, kappa):
    # stopping_power(E) = E**(1-p) / (ALPHA*p), so multiply by kappa*E.
    # Use ALPHA and p from doseparams, matching the stopping_power  used by the function in doseparams.
    expected = kappa * energy ** (2 - p) / (ALPHA * p)

    # Pass kappa explicitly so the case does not depend on its default value.
    result = T_energy_stragg(energy, kappa=kappa)

    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    ("side_len", "center", "coordinate", "expected"),
    [
        (2.0, np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0])),
        (2.0, np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([-1.0, -1.0])),
        (2.0, np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([1.0, 1.0])),
        (4.0, np.array([2.0, 2.0, 2.0]), np.array([4.0, 0.0, 2.0]), np.array([1.0, -1.0, 0.0])),
    ],
)
def test_reference_cube_mapping_maps_voxel_center_and_corners(
    side_len, center, coordinate, expected
):
    # map the global coordinate into the reference cube.
    result = reference_cube_mapping(side_len, center, coordinate)

    # Check the center maps to zero and voxel corners map to +/-1.
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    ("node", "coordinate"),
    [
        (np.array([0.0, 0.0]), np.array([1.01, 0.0])),
        (np.array([0.0, 0.0]), np.array([0.0, -1.01])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.01])),
        (np.array([0.0, 0.0, 0.0]), np.array([-1.01, 0.0, 0.0])),
    ],
)
def test_phi_is_zero_outside_node_support(node, coordinate):
    # calling testing function
    # at least one of the coords are more than one side length away
    result = Phi(side_len=1.0, X=coordinate, n_i=node)

    assert result == 0.0


@pytest.mark.parametrize(
    ("node", "coordinate", "expected"),
    [
        # Bilinear basis at the lower-left node of a 2D voxel.
        (np.array([0.0, 0.0]), np.array([0.25, 0.25]), 0.5625),
        # Trilinear basis at the lower corner of a 3D voxel.
        (np.array([0.0, 0.0, 0.0]), np.array([0.25, 0.25, 0.25]), 0.421875),
    ],
)
def test_phi_evaluates_basis_inside_voxel(node, coordinate, expected):
    # Evaluate the basis function with a unit voxel side length.
    result = Phi(side_len=1.0, X=coordinate, n_i=node)

    # Check the known basis value.
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    ("coordinate", "node_on_boundary", "node_across_boundary"),
    [
        # Boundary in the x coordinate.
        (np.array([1.0, 0.25]), np.array([1.0, 0.0]), np.array([0.0, 0.0])),
        # Boundary in the y coordinate.
        (np.array([0.25, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])),
    ],
)
def test_phi_applies_boundary_coordinate_convention(
    coordinate, node_on_boundary, node_across_boundary
):
    # Evaluate basis values for nodes on either side of the voxel boundary.
    boundary_value = Phi(side_len=1.0, X=coordinate, n_i=node_on_boundary)
    across_value = Phi(side_len=1.0, X=coordinate, n_i=node_across_boundary)

    # The convention assigns the boundary point to the voxel on its left.
    assert np.isclose(boundary_value, 0.75)
    assert across_value == 0.0


@pytest.mark.parametrize(
    ("coordinate", "nodes"),
    [
        # 2D point inside a voxel, with active and inactive nodes.
        (
            np.array([0.25, 0.25]),
            np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]),
        ),
        # 3D point inside a voxel.
        (
            np.array([0.25, 0.25, 0.25]),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]),
        ),
        # 2D point on a voxel boundary.
        (
            np.array([1.0, 0.25]),
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        ),
        # 3D point on a voxel boundary.
        (
            np.array([1.0, 0.25, 0.25]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ),
    ],
)
def test_vectorised_phi_matches_scalar_phi_for_nodes(coordinate, nodes):
    # Evaluate all node basis functions in one vectorised call.
    vectorised_result = Phi_vectorised(side_len=1.0, X=coordinate, nodes=nodes)

    # Evaluate the same basis functions individually for comparison.
    scalar_results = np.array(
        [Phi(side_len=1.0, X=coordinate, n_i=node) for node in nodes]
    )

    assert np.allclose(vectorised_result, scalar_results)


@pytest.mark.parametrize(
    ("coordinate", "nodes"),
    [
        (np.array([0.0, 0.0]), np.array([[2.0, 0.0], [0.0, -2.0]])),
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([[2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0]]),
        ),
    ],
)
def test_vectorised_phi_returns_zeros_when_no_nodes_are_active(coordinate, nodes):
    result = Phi_vectorised(side_len=1.0, X=coordinate, nodes=nodes)

    assert np.array_equal(result, np.zeros(len(nodes)))


def test_position_lookup_matrix_matches_mesh_shape_and_flat_order():
    pass


def test_initial_energy_spread_is_reproducible_when_random_state_is_seeded():
    pass


def test_sample_initial_position_is_centered_on_expected_beam_position():
    pass


def test_sample_initial_position_uses_expected_spread_in_2d_and_3d():
    pass


def test_domain_exit_check_is_false_inside_or_on_domain_boundary():
    pass


def test_domain_exit_check_is_true_outside_each_domain_boundary():
    pass


def test_coarsen_step_sums_groups_of_one_dimensional_increments():
    pass


def test_coarsen_step_sums_groups_of_multidimensional_increments():
    pass


def test_coarsen_step_rejects_non_geometric_step_count():
    pass


def test_coarsen_step_rejects_unsupported_array_dimensions():
    pass


def test_linear_lumped_mass_matrix_has_expected_shape_and_positive_entries():
    pass


def test_linear_lumped_mass_matrix_is_symmetric_across_regular_mesh():
    pass


def test_linear_lumped_mass_matrix_rejects_nonuniform_or_wrong_spacing():
    pass


def test_linear_lumped_mass_matrix_rejects_mesh_without_interior_extent():
    pass


def test_load_mass_matrix_loads_cached_matrix(tmp_path, monkeypatch):
    pass


def test_load_mass_matrix_computes_and_saves_when_cache_is_missing(tmp_path, monkeypatch):
    pass
