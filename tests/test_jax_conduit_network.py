import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import (
    assert_approx_equal,
    assert_array_equal,
    assert_array_almost_equal,
)
import equinox as eqx
import pytest

from landlab import RasterModelGrid

from basis.components.jax_conduit_network import ConduitNetwork


@pytest.fixture
def grid():
    grid = RasterModelGrid((3, 4), 2.0)
    grid.add_zeros("bedrock_elevation", at="node")
    grid.add_zeros("ice_thickness", at="node")
    grid.add_zeros("ice_sliding_velocity", at="link")
    grid.add_zeros("meltwater_input", at="node")
    grid.add_zeros("water_pressure", at="node")

    grid.add_zeros("water_flux", at="link")
    grid.add_zeros("hydraulic_gradient", at="link")
    grid.add_zeros("conduit_area", at="link")
    grid.add_zeros("effective_pressure", at="link")

    return grid


def test_init_with_defaults(grid):
    model = ConduitNetwork(grid)
    assert_approx_equal(model.melt_constant, 3.26e-9, 3)
    assert_approx_equal(model.closure_constant, 4.44e-25, 3)
    assert_approx_equal(model.flow_constant, 0.33, 3)


def test_init_override_params(grid):
    model = ConduitNetwork(grid, ice_density=200)
    assert_approx_equal(model.melt_constant, 1.49e-8, 3)
    assert_approx_equal(model.closure_constant, 4.44e-25, 3)
    assert_approx_equal(model.flow_constant, 0.33, 3)


def test_raise_error_if_n_neq_3(grid):
    with pytest.raises(NotImplementedError):
        model = ConduitNetwork(grid, glens_n=4)


def test_if_valid_pytree(grid):
    model = ConduitNetwork(grid)
    assert eqx.tree_check(model) is None


def test_map_to_links(grid):
    model = ConduitNetwork(grid)
    result = model.map_to_links(model.water_pressure, model.grid)
    assert result.shape[0] == grid.number_of_links


def test_map_to_nodes(grid):
    model = ConduitNetwork(grid)
    result = model.map_to_nodes(model.conduit_area, model.grid)
    assert result.shape[0] == grid.number_of_nodes


def test_sum_at_nodes(grid):
    grid.at_link["ice_sliding_velocity"] = grid.xy_of_link[:, 0]
    model = ConduitNetwork(grid)
    result = model.sum_at_nodes(model.ice_sliding_velocity, model.grid)
    assert_array_equal(result, [-1, -4, -6, -1, -1, -2, -2, 5, -1, 0, 2, 11])


def test_calc_melt_opening(grid):
    grid.at_link["water_flux"] = grid.xy_of_link[:, 0]
    grid.at_link["hydraulic_gradient"] = grid.xy_of_link[:, 1]
    model = ConduitNetwork(grid)
    melt = model._calc_melt_opening()

    assert_array_almost_equal(
        melt,
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            6.51052263e-09,
            1.30210453e-08,
            1.95315679e-08,
            6.51052263e-09,
            1.95315679e-08,
            3.25526132e-08,
            0.00000000e00,
            1.95315679e-08,
            3.90631358e-08,
            5.85947037e-08,
            1.30210453e-08,
            3.90631358e-08,
            6.51052263e-08,
        ],
    )


def test_calc_gap_opening(grid):
    grid.at_link["ice_sliding_velocity"] = grid.xy_of_link[:, 0]
    model = ConduitNetwork(grid)
    gap = model._calc_gap_opening()

    assert_array_almost_equal(
        gap,
        [
            0.1,
            0.3,
            0.5,
            0.0,
            0.2,
            0.4,
            0.6,
            0.1,
            0.3,
            0.5,
            0.0,
            0.2,
            0.4,
            0.6,
            0.1,
            0.3,
            0.5,
        ],
    )


def test_calc_creep_closure(grid):
    grid.at_link["effective_pressure"] = grid.xy_of_link[:, 0] * 1e6
    grid.at_link["conduit_area"] = grid.xy_of_link[:, 1]
    model = ConduitNetwork(grid)
    closure = model._calc_creep_closure(model.conduit_area)

    assert_array_almost_equal(
        closure,
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            3.55555556e-06,
            2.84444444e-05,
            9.60000000e-05,
            8.88888889e-07,
            2.40000000e-05,
            1.11111111e-04,
            0.00000000e00,
            1.06666667e-05,
            8.53333333e-05,
            2.88000000e-04,
            1.77777778e-06,
            4.80000000e-05,
            2.22222222e-04,
        ],
    )


@pytest.mark.slow
def test_solve_for_conduit_area(grid):
    grid.at_link["conduit_area"][:] = 5.0
    grid.at_link["effective_pressure"][:] = 1e6
    grid.at_link["water_flux"][:8] = 1.15e-7
    grid.at_link["water_flux"][8:] = -1.15e-7
    grid.at_link["hydraulic_gradient"][:8] = -100
    grid.at_link["hydraulic_gradient"][8:] = 100
    model = ConduitNetwork(grid)

    result = model._solve_for_conduit_area(60 * 60 * 24)

    assert_array_almost_equal(result.ys[-1], jnp.full(grid.number_of_links, 4.81164))


def test_calc_flux_overflow(grid):
    grid.at_node["ice_thickness"][:] = 300
    grid.at_node["bedrock_elevation"][:] = grid.node_x * 0.01
    grid.at_link["conduit_area"][:] = 0.5
    model = ConduitNetwork(grid)

    residual = model._calc_flux_overflow(model.water_pressure, model.conduit_area)

    assert_approx_equal(residual, 33416454)


def test_solve_for_water_pressure(grid):
    grid.at_node["ice_thickness"][:] = 300
    grid.at_node["bedrock_elevation"][:] = grid.node_x * 0.01
    grid.at_link["conduit_area"][:] = 0.5
    model = ConduitNetwork(grid)

    solution = model._solve_for_water_pressure(60 * 60 * 24, model.conduit_area)

    assert_array_almost_equal(
        solution.params,
        [
            5.88975454e02,
            3.92775454e02,
            1.96575439e02,
            3.75426828e-01,
            5.88975454e02,
            3.92775454e02,
            1.96575439e02,
            3.75426828e-01,
            5.88975454e02,
            3.92775454e02,
            1.96575439e02,
            3.75426828e-01,
        ],
    )


@pytest.mark.slow
def test_run_one_step(grid):
    grid.at_node["ice_thickness"][:] = 300
    grid.at_node["bedrock_elevation"][:] = grid.node_x * 0.01
    grid.at_link["conduit_area"][:] = 0.5
    model = ConduitNetwork(grid)
    update = model.run_one_step(1.0)

    assert_array_almost_equal(update.conduit_area, np.full(grid.number_of_links, 0.5))

    assert_array_almost_equal(
        update.water_pressure,
        [
            5.88975454e02,
            3.92775454e02,
            1.96575439e02,
            3.75426828e-01,
            5.88975454e02,
            3.92775454e02,
            1.96575439e02,
            3.75426828e-01,
            5.88975454e02,
            3.92775454e02,
            1.96575439e02,
            3.75426828e-01,
        ],
    )

    assert_array_almost_equal(
        update.effective_pressure,
        [
            2698142.024546,
            2698338.224546,
            2698534.424561,
            2698730.624573,
            2698142.024546,
            2698338.224546,
            2698534.424561,
            2698730.624573,
            2698142.024546,
            2698338.224546,
            2698534.424561,
            2698730.624573,
        ],
    )

    assert_array_almost_equal(
        update.hydraulic_gradient,
        [
            9.31237309e-11,
            7.38850794e-06,
            6.24926763e-06,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            9.31237309e-11,
            7.38850794e-06,
            6.24926763e-06,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            9.31237309e-11,
            7.38850794e-06,
            6.24926763e-06,
        ],
    )

    assert_array_almost_equal(
        update.water_flux,
        [
            1.34197962e-06,
            3.78002009e-04,
            3.47640445e-04,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.34197962e-06,
            3.78002009e-04,
            3.47640445e-04,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.34197962e-06,
            3.78002009e-04,
            3.47640445e-04,
        ],
    )
