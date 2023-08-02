import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal
import pytest

from basis.components.steady_state_drainage import (
    Mesh, HydrologicConstants, GlacierData, DrainageSystem
)