"""Interpolate data between grids."""

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

def interpolate(
    source: np.ndarray, 
    values: np.ndarray,
    destination: np.ndarray,
    **kwargs 
) -> np.ndarray:
    """Interpolate values defined on a source grid to a destination grid."""
    interpolant = CloughTocher2DInterpolator(source, values, **kwargs)
    interp_values = interpolant(destination)
    return interp_values
