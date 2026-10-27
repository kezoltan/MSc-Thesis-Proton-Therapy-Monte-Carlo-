"""Behavior-focused test names for ``src.dosesetup``.

Implement these cases incrementally, starting with the pure helper functions.
"""
import numpy as np 
from math import prod, comb, cos, sin, exp, log
from itertools import product
from src.doseparams import *
import pytest
from src.dosesetup import exponential_map_sphere, reciprocal_stopping_power

@pytest.mark.parametrize(
    ("omega", "increment", "expected"), # omega = initial direction
    [
        # start along x and turn towards y by 0.3 rad
        ([1.0, 0.0], [0.0, 0.3], [np.cos(0.3), np.sin(0.3)]),
        # start along y and turn towards x by 0.3 rad
        ([0.0, 1.0], [0.2, 0.0], [np.sin(0.2), np.cos(0.2)]),
        # in 3d start on x and turn towards z by 0.4 rad
        (
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.4],
            [np.cos(0.4), 0.0, np.sin(0.4)],
        ),
        # start along x and turn an increment larger than pi.
        ([1.0, 0.0], [0.0, 4.0], [np.cos(4.0), np.sin(4.0)]),
    ],
)
def test_exponential_map_sphere_returns_unit_direction(omega, increment, expected):
    # convert the lists into Numpy arrays 
    omega = np.array(omega)
    increment = np.array(increment)

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


def test_energy_straggling_term_matches_formula():
    pass


def test_reference_cube_mapping_maps_voxel_center_and_corners():
    pass


def test_phi_is_zero_outside_node_support():
    pass


def test_phi_evaluates_basis_inside_voxel():
    pass


def test_phi_applies_boundary_coordinate_convention():
    pass


def test_vectorised_phi_matches_scalar_phi_for_nodes():
    pass


def test_vectorised_phi_returns_zeros_when_no_nodes_are_active():
    pass


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
