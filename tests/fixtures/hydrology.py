import numpy as np
import pytest

from basis.components.steady_state_drainage import (
    Mesh,
    HydrologicConstants,
    GlacierData,
)


@pytest.fixture
def mesh():
    nodes = np.arange(100) * 10
    mesh = Mesh(nodes)
    return mesh


@pytest.fixture
def hydrologic_constants():
    const = HydrologicConstants()
    return const


@pytest.fixture
def glacier_data(mesh):
    glacier = GlacierData(
        mesh=mesh,
        bed_elevation=np.arange(100) * 0.1,
        ice_thickness=np.arange(100) * 0.01 + 300,
        discharge=np.full(100, 0.001),
    )
    return glacier
