import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal
import pytest

from tests.fixtures.hydrology import (
    mesh, hydrologic_constants, glacier_data
)

def test_mesh(mesh):
    assert_array_equal(mesh.nodes, np.arange(100) * 10)
    assert mesh.dims == 1
    assert_array_equal(mesh.d[0], np.full(100, 10.0))

def test_hydrologic_constants(hydrologic_constants):
    constants = hydrologic_constants
    assert_approx_equal(constants.melt_opening_coeff, 3.26e-9, 3)
    assert_approx_equal(constants.closure_coeff, 2.22e-25, 3)
    assert_approx_equal(constants.flux_coeff, 0.331, 3)

def test_glacier(glacier_data):
    glacier = glacier_data
    assert glacier.ice_pressure[0] == 2698731.0
    assert_approx_equal(glacier.ice_pressure_gradient[0], 8.99, 3)
    assert glacier.elevation_gradient[0] == 0.01
    assert_approx_equal(glacier.base_hydraulic_gradient[0], -107.09, 3)