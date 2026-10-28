"""Behavior-focused test names for ``src.dosesetup``.

Implement these cases incrementally, starting with the pure helper functions.
"""
import numpy as np
from src.doseparams import *
import pytest
from src.dosesetup import (
    exponential_map_sphere,
    reciprocal_stopping_power,
    V_energy_loss_var_pul,
    reference_cube_mapping,
    Phi,
    Phi_vectorised,
    position_lookup_matrix,
    initial_energy_spread,
    domain_exit_check,
    coarsen_step,
    linear_lumped_mass_matrix,
    load_mass_matrix,
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
    result = V_energy_loss_var_pul(energy, kappa=kappa)

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


@pytest.mark.parametrize(
    "X_meshgrid",
    [
        np.zeros((2, 3)),
        np.zeros((2, 3, 4)),
    ],
)
def test_position_lookup_matrix_matches_mesh_shape_and_flat_order(X_meshgrid):
    result = position_lookup_matrix(X_meshgrid)
    expected = np.arange(X_meshgrid.size).reshape(X_meshgrid.shape)

    assert result.shape == X_meshgrid.shape
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_initial_energy_spread_is_reproducible_when_random_state_is_seeded(seed):
    # Save the current global random state so this test does not affect others.
    original_state = np.random.get_state()
    try:
        np.random.seed(seed)
        first_sample = initial_energy_spread()

        np.random.seed(seed)
        second_sample = initial_energy_spread()
    finally:
        np.random.set_state(original_state)

    assert first_sample == second_sample

def test_sample_initial_position_is_centered_on_expected_beam_position():
    pass


def test_sample_initial_position_uses_expected_spread_in_2d_and_3d():
    pass

@pytest.mark.parametrize(
    "point_type",
    ["inside", "lower_boundary", "upper_boundary"],
)
def test_domain_exit_check_is_false_inside_or_on_domain_boundary(point_type):
    if point_type == "inside":
        position = (domain_lower_bounds + domain_upper_bounds) / 2
    elif point_type == "lower_boundary":
        position = domain_lower_bounds.copy()
    else:
        position = domain_upper_bounds.copy()

    assert not bool(domain_exit_check(position))


@pytest.mark.parametrize(
    ("boundary", "axis"),
    [("lower", axis) for axis in range(SPATIAL_DIM)]
    + [("upper", axis) for axis in range(SPATIAL_DIM)],
)
def test_domain_exit_check_is_true_outside_each_domain_boundary(boundary, axis):
    epsilon = np.max(domain_upper_bounds - domain_lower_bounds) * 1e-6
    if boundary == "lower":
        position = domain_lower_bounds.copy()
        position[axis] -= epsilon
    else:
        position = domain_upper_bounds.copy()
        position[axis] += epsilon

    assert bool(domain_exit_check(position))


@pytest.mark.parametrize(
    ("increments", "expected"),
    [
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 7.0])),
        (np.arange(1.0, 9.0), np.array([3.0, 7.0, 11.0, 15.0])),
    ],
)
def test_coarsen_step_sums_groups_of_one_dimensional_increments(
    increments, expected
):
    assert np.array_equal(coarsen_step(increments), expected)


@pytest.mark.parametrize(
    "increments",
    [
        np.arange(8.0).reshape(4, 2),
        np.arange(24.0).reshape(8, 3),
    ],
)
def test_coarsen_step_sums_groups_of_multidimensional_increments(increments):
    expected = increments.reshape(-1, base, increments.shape[1]).sum(axis=1)

    assert np.array_equal(coarsen_step(increments), expected)


@pytest.mark.parametrize("num_steps", [3, 6])
def test_coarsen_step_rejects_non_geometric_step_count(num_steps):
    increments = np.ones(num_steps)

    with pytest.raises(ValueError, match="geometric step size"):
        coarsen_step(increments)


@pytest.mark.parametrize("shape", [(4, 2, 1), (8, 1, 2)])
def test_coarsen_step_rejects_unsupported_array_dimensions(shape):
    increments = np.ones(shape)

    with pytest.raises(ValueError, match="shape is incompatible"):
        coarsen_step(increments)


@pytest.mark.parametrize("dimension", [2, 3])
def test_linear_lumped_mass_matrix_has_expected_shape_and_positive_entries(
    dimension,
):
    shape = (3,) * dimension
    X_meshgrid = np.meshgrid(*[np.arange(3.0)] * dimension, indexing="ij")[0]

    result = linear_lumped_mass_matrix(X_meshgrid, side_len=1.0, spatial_dim=dimension)

    assert result.shape == (int(np.prod(shape)),)
    assert np.all(result > 0.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_linear_lumped_mass_matrix_is_symmetric_across_regular_mesh(
    dimension,
):
    shape = (3,) * dimension
    X_meshgrid = np.meshgrid(*[np.arange(3.0)] * dimension, indexing="ij")[0]
    mass_diagonal = linear_lumped_mass_matrix(
        X_meshgrid, side_len=1.0, spatial_dim=dimension
    ).reshape(shape)

    for axis in range(dimension):
        assert np.allclose(mass_diagonal, np.flip(mass_diagonal, axis=axis))


@pytest.mark.parametrize(
    ("dimension", "coordinates", "side_len"),
    [
        (2, np.array([0.0, 1.0, 2.0]), 0.5),
        (3, np.array([0.0, 1.5, 2.5]), 1.0),
    ],
)
def test_linear_lumped_mass_matrix_rejects_nonuniform_or_wrong_spacing(
    dimension, coordinates, side_len
):
    X_meshgrid = np.meshgrid(*[coordinates] * dimension, indexing="ij")[0]

    with pytest.raises(ValueError, match="sidelength"):
        linear_lumped_mass_matrix(
            X_meshgrid, side_len=side_len, spatial_dim=dimension
        )


@pytest.mark.parametrize("dimension", [2, 3])
def test_linear_lumped_mass_matrix_handles_mesh_without_interior_extent(
    dimension,
):
    shape = (2,) * dimension
    coordinates = [np.arange(size, dtype=float) for size in shape]
    X_meshgrid = np.meshgrid(*coordinates, indexing="ij")[0]

    result = linear_lumped_mass_matrix(
        X_meshgrid, side_len=1.0, spatial_dim=dimension
    )

    assert result.shape == (2**dimension,)
    assert np.all(result > 0.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_load_mass_matrix_loads_cached_matrix(tmp_path, dimension):
    mesh_shape = (3,) * dimension
    meshgrid = np.meshgrid(
        *[np.arange(size, dtype=float) for size in mesh_shape], indexing="ij"
    )
    dose_shape = int(np.prod(mesh_shape))
    expected = np.arange(1.0, dose_shape + 1)
    side_len = 1.0
    cache_path = tmp_path / f"{dimension}D_{expected.size}_lumped_mass_matrix.npz"
    np.savez(cache_path, M_diag=expected)

    result = load_mass_matrix(
        file_path=str(tmp_path),
        spatial_dim=dimension,
        dose_shape=dose_shape,
        X_meshgrid=meshgrid[0],
        Y_meshgrid=meshgrid[1],
        side_len=side_len,
    )

    assert np.array_equal(result, expected)


@pytest.mark.parametrize("dimension", [2, 3])
def test_load_mass_matrix_computes_and_saves_when_cache_is_missing(
    tmp_path, dimension
):
    mesh_shape = (3,) * dimension
    side_len = 0.5
    meshgrid = np.meshgrid(
        *[np.arange(size, dtype=float) * side_len for size in mesh_shape],
        indexing="ij",
    )
    dose_shape = int(np.prod(mesh_shape))
    expected = linear_lumped_mass_matrix(
        meshgrid[0], side_len, spatial_dim=dimension
    )

    result = load_mass_matrix(
        file_path=str(tmp_path),
        spatial_dim=dimension,
        dose_shape=dose_shape,
        X_meshgrid=meshgrid[0],
        Y_meshgrid=meshgrid[1],
        side_len=side_len,
    )
    cache_path = tmp_path / f"{dimension}D_{dose_shape}_lumped_mass_matrix.npz"

    assert np.array_equal(result, expected)
    assert cache_path.exists()
    with np.load(cache_path) as cache:
        assert np.array_equal(cache["M_diag"], expected)
