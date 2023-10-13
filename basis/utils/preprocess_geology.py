"""Pre-processes remote sensing data and writes to netCDF."""

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr

class RasterProcessor:
    """Pre-processes raster data."""

    def __init__(self, path_to_shapefile: str, **kwargs):
        """Initializes the pre-processor with a path to a shapefile."""
        self.basin = gpd.read_file(path_to_shapefile)

        self._rasters = {key: None for key in kwargs.keys()}
        for var, path in kwargs.items():
            self._rasters[var] = rxr.open_rasterio("netcdf:" + path + ":" + var).squeeze(dim = None)
