import numpy as np
import pytest

from diffusion_lab.visualization import Grid, evaluate_field, recover_potential


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_quadratic_potential_and_coordinate_order(dimension):
    axes = (np.linspace(-2, 3, 13), np.linspace(-1, 2, 9), np.linspace(-3, 1, 7))[:dimension]
    grid = Grid(axes)
    weights = np.arange(1, dimension + 1)
    field = -grid.points * weights
    result = recover_potential(grid, field)
    expected = grid.reshape(0.5 * np.sum(weights * grid.points**2, axis=-1))
    expected -= expected.flat[0]
    assert grid.shape == tuple(len(a) for a in axes[::-1])
    np.testing.assert_allclose(result.potential, expected, atol=2e-7)
    assert result.relative_residual < 1e-8
    assert result.curl_rms < 1e-12
    assert result.is_conservative


def test_rotational_field_is_not_reported_as_potential():
    grid = Grid.from_bounds([(-2, 2), (-1, 1)], (31, 19))
    x, y = grid.points.T
    result = recover_potential(grid, np.c_[-y, x])
    assert result.relative_residual > 0.5
    assert result.curl_rms == pytest.approx(2)
    assert not result.is_conservative
    assert "projection" in result.message


def test_masked_wall_has_independent_component_gauges():
    grid = Grid.from_bounds([(-2, 2), (-1, 1)], (13, 9))
    mask = np.ones(grid.shape, dtype=bool)
    mask[:, 6] = False
    field = -grid.reshape(grid.points)
    field[~mask] = np.nan
    result = recover_potential(grid, field, mask=mask)
    assert result.component_count == 2
    assert np.isnan(result.potential[:, 6]).all()
    assert result.potential[0, 0] == 0
    assert result.potential[0, 7] == 0
    assert result.relative_residual < 1e-8
    assert result.is_conservative


def test_one_dimensional_nonuniform_grid_and_gap():
    grid = Grid((np.array([-2.0, -1.0, 0.0, 0.3, 1.0, 2.0]),))
    mask = np.array([True, True, False, True, True, True])
    field = -2 * grid.points
    result = recover_potential(grid, field, mask=mask)
    np.testing.assert_allclose(result.potential[[0, 1]], [0, -3])
    np.testing.assert_allclose(result.potential[[3, 4, 5]], [0, 1 - 0.09, 4 - 0.09])
    assert result.component_count == 2
    assert result.is_conservative


def test_masked_hole_circulation_has_graph_residual():
    # The vortex is locally curl-free outside its excluded origin, but its
    # circulation around the hole prevents a single-valued global potential.
    grid = Grid.from_bounds([(-2, 2), (-2, 2)], (51, 51))
    x, y = grid.mesh
    radius_squared = x**2 + y**2
    mask = radius_squared > 0.6**2
    field = np.zeros(grid.shape + (2,))
    np.divide(-y, radius_squared, out=field[..., 0], where=mask)
    np.divide(x, radius_squared, out=field[..., 1], where=mask)
    result = recover_potential(grid, field, mask=mask)
    assert result.relative_residual > 0.5
    assert not result.is_conservative


def test_batched_field_and_mesh_coordinates():
    grid = Grid.from_bounds([(10, 12), (-4, -1), (0, 5)], (3, 4, 6))
    calls = []

    def field(points):
        calls.append(len(points))
        return 2 * points

    values = evaluate_field(field, grid, batch_size=7)
    assert max(calls) == 7
    np.testing.assert_allclose(values[5, 3, 2], [24, -2, 10])
    np.testing.assert_allclose(grid.points[0], [10, -4, 0])
    np.testing.assert_allclose(grid.points[1], [11, -4, 0])


def test_invalid_axes_or_field_fail_early():
    with pytest.raises(ValueError, match="increasing"):
        Grid((np.array([1, 0]),))
    grid = Grid.from_bounds([(-1, 1), (-1, 1)], (7, 9))
    with pytest.raises(ValueError, match="shape"):
        evaluate_field(lambda points: points[:, 0], grid)
    with pytest.raises(ValueError, match="mask"):
        recover_potential(grid, np.zeros(grid.shape + (2,)), mask=np.zeros(grid.shape, dtype=bool))


def test_three_dimensional_rotational_curl():
    grid = Grid.from_bounds([(-1, 1), (-2, 2), (0, 1)], (11, 9, 5))
    x, y, z = grid.points.T
    field = np.c_[-y, x, z * 0]
    result = recover_potential(grid, field)
    assert result.curl_rms == pytest.approx(2)
    assert result.relative_residual > 0.5
    assert not result.is_conservative
