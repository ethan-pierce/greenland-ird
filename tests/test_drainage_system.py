import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal
import matplotlib.pyplot as plt

import pytest

from landlab import RasterModelGrid

from basis.components.subglacial_drainage_system import (
    SubglacialDrainageSystem
)

@pytest.fixture
def grid():
    grid = RasterModelGrid((3, 4), 2.)
    grid.add_zeros('bedrock__elevation', at = 'node')
    grid.add_zeros('ice__thickness', at = 'node')
    grid.add_zeros('ice__sliding_velocity', at = 'link')
    grid.add_zeros('meltwater__input', at = 'node')

    return grid

def test_init(grid):
    SDS = SubglacialDrainageSystem(grid)
    assert_approx_equal(SDS.params['melt_constant'], 3.26e-9, 3)
    assert_approx_equal(SDS.params['closure_constant'], 4.44e-25, 3)
    assert_approx_equal(SDS.params['flow_constant'], 0.33, 3)

def test_partition_meltwater(grid):
    grid.at_node['meltwater__input'][5] = 3.0
    SDS = SubglacialDrainageSystem(grid)
    discharge = SDS._partition_meltwater()

    assert_array_equal(
        discharge,
        [0, 0, 0,
         0, 4, 0, 0,
         4, 4, 0,
         0, 4, 0, 0,
         0, 0, 0]
    )