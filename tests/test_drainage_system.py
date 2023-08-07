import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal
import matplotlib.pyplot as plt

import pytest

from landlab import RasterModelGrid

from basis.components.steady_state_drainage import (
    Mesh, HydrologicConstants, GlacierData
)

from basis.components.subglacial_drainage_system import (
    SubglacialDrainageSystem
)

@pytest.fixture
def grid():
    grid = RasterModelGrid((100, 100), 1.)

    bedrock = (0.2 * (grid.node_y - 50))**2 + 2 * grid.node_x
    ice_thickness = grid.node_y * 0.1 + 300 - bedrock
    node_velocity = 50 - (2 * grid.node_y - np.max(grid.node_y))
    sliding_velocity = grid.map_mean_of_link_nodes_to_link(node_velocity)
    melt = np.full(grid.shape, 1.15e-7) # 1 cm per day

    grid.add_field('bedrock__elevation', bedrock, at = 'node')

    grid.add_field('ice__thickness', ice_thickness, at = 'node')

    grid.add_field('ice__sliding_velocity', sliding_velocity, at = 'link')

    grid.add_field('meltwater__input', melt, at = 'node')

    return grid

@pytest.fixture
def SDS(grid):
    sds = SubglacialDrainageSystem(grid)
    return sds

def transform(grid, field: np.ndarray, at: str = 'node'):
    if at == 'link':
        field = grid.map_mean_of_links_to_node(field)

    reshaped = np.reshape(field, grid.shape)

    flipped = np.flip(reshaped, axis = 0)

    return flipped

def plot_field(grid, field: np.ndarray, at: str = 'node'):
    to_plot = transform(grid, field, at = at)
    im = plt.imshow(to_plot)
    plt.colorbar(im)
    plt.show()

def test_init(SDS):
    pass

def test_calc_base_hydraulic_gradient(SDS):
    psi0 = SDS._calc_base_hydraulic_gradient()

def test_partition_meltwater_input(SDS):
    SDS._partition_meltwater_input()