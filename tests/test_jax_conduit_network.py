import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_approx_equal, assert_array_equal
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
    model = ConduitNetwork(grid, ice_density = 200)
    assert_approx_equal(model.melt_constant, 1.49e-8, 3)
    assert_approx_equal(model.closure_constant, 4.44e-25, 3)
    assert_approx_equal(model.flow_constant, 0.33, 3)

def test_raise_error_if_n_neq_3(grid):
    with pytest.raises(NotImplementedError):
        model = ConduitNetwork(grid, glens_n = 4)

def test_if_valid_pytree(grid):
    model = ConduitNetwork(grid)
    assert eqx.tree_check(model) is None

def test_run_one_step(grid):
    model = ConduitNetwork(grid)
    update = model.run_one_step(1.0)
    assert_approx_equal(
        update.conduit_area[8],
        0.0
    )

def test_map_to_links(grid):
    model = ConduitNetwork(grid)
    result = model.map_to_links(model.water_pressure, model.grid)
    assert result.shape[0] == grid.number_of_links

def test_map_to_nodes(grid):
    model = ConduitNetwork(grid)
    result = model.map_to_nodes(model.conduit_area, model.grid)
    assert result.shape[0] == grid.number_of_nodes

def test_sum_at_nodes(grid):
    grid.at_link['ice_sliding_velocity'] = grid.xy_of_link[:, 0]
    model = ConduitNetwork(grid)
    result = model.sum_at_nodes(model.ice_sliding_velocity, model.grid)
    assert_array_equal(
        result,
        [-1, -4, -6, -1,
         -1, -2, -2, 5,
         -1, 0, 2, 11]
    )
    