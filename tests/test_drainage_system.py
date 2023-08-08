import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal
import matplotlib.pyplot as plt

import pytest

from landlab import RasterModelGrid

from basis.components.subglacial_drainage_system import (
    SubglacialDrainageSystem, SolutionTensor, BoundaryCondition
)

@pytest.fixture
def grid():
    grid = RasterModelGrid((3, 4), 2.0)
    grid.add_zeros("bedrock__elevation", at="node")
    grid.add_zeros("ice__thickness", at="node")
    grid.add_zeros("ice__sliding_velocity", at="link")
    grid.add_zeros("meltwater__input", at="node")

    grid.add_zeros("water__discharge", at="link")
    grid.add_zeros("hydraulic__gradient", at="link")
    grid.add_zeros("conduit__area", at="link")
    grid.add_zeros("effective_pressure", at="link")

    return grid


def test_init(grid):
    SDS = SubglacialDrainageSystem(grid)
    assert_approx_equal(SDS.params["melt_constant"], 3.26e-9, 3)
    assert_approx_equal(SDS.params["closure_constant"], 4.44e-25, 3)
    assert_approx_equal(SDS.params["flow_constant"], 0.33, 3)


def test_partition_meltwater(grid):
    grid.at_node["meltwater__input"][5] = 3.0
    grid.at_node["bedrock__elevation"][:6] = 10.0

    SDS = SubglacialDrainageSystem(grid)
    discharge = SDS._partition_meltwater()

    assert_array_equal(discharge, [0, 0, 0, 0, 3, 0, 0, 3, 3, 0, 0, 3, 0, 0, 0, 0, 0])


def test_calc_hydraulic_gradient(grid):
    grid.at_node["ice__thickness"][:6] = 100.0
    grid.at_node["bedrock__elevation"][:] = grid.node_x
    SDS = SubglacialDrainageSystem(grid)

    psi0 = SDS._calc_base_hydraulic_gradient()

    assert_array_equal(
        psi0,
        [
            -9810.0,
            -9810.0,
            -9810.0,
            -0.0,
            -0.0,
            449788.5,
            449788.5,
            -9810.0,
            439978.5,
            -9810.0,
            449788.5,
            449788.5,
            -0.0,
            -0.0,
            -9810.0,
            -9810.0,
            -9810.0,
        ],
    )

    # Invariant: if N = 0, psi = psi0
    psi = SDS._calc_hydraulic_gradient(np.zeros(grid.number_of_links))

    assert_array_equal(psi, psi0)

    # Invariant if grad(N) = 0, psi = psi0
    psi = SDS._calc_hydraulic_gradient(np.full(grid.number_of_links, 1e6))

    assert_array_equal(psi, psi0)

    # Now let N vary
    psi = SDS._calc_hydraulic_gradient(
        grid.map_mean_of_link_nodes_to_link(grid.at_node["ice__thickness"]) * 917 * 9.81
    )

    assert_array_equal(
        psi,
        [
            -9810.0,
            -84774.75,
            -47292.375,
            -74964.75,
            -112447.125,
            187411.875,
            187411.875,
            -47292.375,
            215084.25,
            -47292.375,
            187411.875,
            187411.875,
            -112447.125,
            -74964.75,
            -47292.375,
            -84774.75,
            -9810.0,
        ],
    )


def test_calc_melt_opening(grid):
    grid.at_link["ice__sliding_velocity"][8] = 50 / 31556926
    SDS = SubglacialDrainageSystem(grid)

    melt = SDS._calc_melt_opening(3.14e-7, -500)

    assert_approx_equal(melt, -5.11e-13, 3)


def test_calc_gap_opening(grid):
    grid.at_link["ice__sliding_velocity"][8] = 50 / 31556926
    SDS = SubglacialDrainageSystem(grid)

    gap = SDS._calc_gap_opening()

    assert_approx_equal(gap[8], 1.58e-7, 3)


def test_calc_closure(grid):
    SDS = SubglacialDrainageSystem(grid)

    closure = SDS._calc_closure(1e6, 1)

    assert_approx_equal(closure, 4.44e-7, 3)


def test_calc_pressure(grid):
    grid.at_link["water__discharge"][:] = 1.15e-7
    grid.at_link["hydraulic__gradient"][:6] = 500
    grid.at_link["hydraulic__gradient"][6:] = -500
    SDS = SubglacialDrainageSystem(grid)

    n_to_the_n = SDS._calc_pressure_to_the_n(
        grid.at_link["water__discharge"], grid.at_link["hydraulic__gradient"]
    )

    assert_approx_equal(np.cbrt(n_to_the_n)[8], 172e3, 3)


def test_calc_discharge(grid):
    grid.at_link["conduit__area"][8] = 1.0
    grid.at_link["hydraulic__gradient"][8] = 500.0
    SDS = SubglacialDrainageSystem(grid)

    discharge = SDS._calc_discharge(
        grid.at_link["conduit__area"], grid.at_link["hydraulic__gradient"]
    )

    assert_approx_equal(discharge[8], 7.40, 3)

def test_RHS(grid):
    grid.at_node['bedrock__elevation'][:6] = 10.
    grid.at_node['ice__thickness'][:] = 100. + 2 * grid.node_x
    grid.at_link['ice__sliding_velocity'][:] = 50 / 31556926
    grid.at_node['meltwater__input'][:] = 1.15e-7
    
    SDS = SubglacialDrainageSystem(grid)
    SDS.initialize(force = True)

    RHS = SDS._RHS(values=SDS._build_solution_tensor(to_array=True))

def test_iter_rk4(grid):
    grid.at_node['bedrock__elevation'][:6] = 10.
    grid.at_node['ice__thickness'][:] = 100. + 2 * grid.node_x
    grid.at_link['ice__sliding_velocity'][:] = 50 / 31556926
    grid.at_node['meltwater__input'][:] = 1.15e-7
    
    SDS = SubglacialDrainageSystem(grid)
    SDS.initialize(force = True)

    result = SDS._iter_RK4(
        SDS._RHS, 
        SDS._build_solution_tensor(to_array=True),
        step = 1e-3
    )

def test_run_one_step(grid):
    grid.at_node['bedrock__elevation'][:6] = 10.
    grid.at_node['ice__thickness'][:] = 100. + 2 * grid.node_x
    grid.at_link['ice__sliding_velocity'][:] = 50 / 31556926
    grid.at_node['meltwater__input'][:] = 1.15e-7
    
    grid.status_at_node[grid.boundary_nodes] = grid.BC_NODE_IS_CLOSED
    SDS = SubglacialDrainageSystem(grid)
    SDS.initialize(force = True)

    SDS.run_one_step(step = 1e-3)
    