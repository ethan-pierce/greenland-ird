"""Clips raster data to a vector shapefile.

Given multiple input rasters and a vector shapefile of the regional boundary, 
this utility will clip each dataset to the catchment and resample the raster
data to the same (x, y) size. The utility provides a single netCDF file as
its output, with variables for each raster and the boundary shapefile.

Note: raster data is expected to be in netcdf format. 
Similarly, vector data is expected in geojson format.
"""

import numpy as np
import glob
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
from rasterio.enums import Resampling
from itertools import groupby

class ClipToCatchment:
    """Algorithm for clipping and resampling rasters to a basin shapefile.

    Attributes:
        self._basin: geopandas GeoDataFrame with the basin outline
        self._rasters: dict of variable name, xarray DataArray for input rasters
    """

    def __init__(
        self,
        path_to_shapefile: str,
        **kwargs
    ):
        """Initializes the instance with paths to input data files."""

        self._basin = gpd.read_file(path_to_shapefile)

        self._rasters = {key: None for key in kwargs.keys()}
        for var, path in kwargs.items():
            self._rasters[var] = rxr.open_rasterio("netcdf:" + path + ":" + var).squeeze(dim = None)

        self._results = {key: None for key in kwargs.keys()}
        self.dataset = None

    def calc_field(self, name: str, function, vars: list, output = False):
        """Calculates a new field based on a given function and existing rasters."""
        fields = []
        for var in vars:
            fields.append(self._rasters[var])

        result = function(*fields)
        self._rasters[name] = result

        if output:
            return result

    def clip(self, raster: xr.DataArray) -> xr.DataArray:
        """Clip an input raster to the catchment boundary."""
        geometry = self._basin.geometry.values
        crs = self._basin.crs
        clipped = raster.rio.clip(geometry, crs)

        return clipped

    def resample_to_shape(self, raster: xr.DataArray, shape: tuple[int, int]) -> xr.DataArray:
        """Resample an input raster to a given shape."""
        resampled = raster.rio.reproject(
            raster.rio.crs, 
            shape = shape, 
            resampling = Resampling.bilinear
        )

        return resampled

    def resample_to_resolution(self, raster: xr.DataArray, resolution: tuple[int, int]) -> xr.DataArray:
        """Resample an input raster to a given resolution."""
        resampled = raster.rio.reproject(
            raster.rio.crs,
            resolution = resolution,
            resampling = Resampling.bilinear
        )

        return resampled

    def interpolate(
        self, 
        raster: xr.DataArray, 
        coords: dict[str, xr.DataArray], 
        method: str = 'linear'
    ) -> xr.DataArray:
        """Interpolate a DataArray onto new coordinates."""
        coords_to_use = {key: val for key, val in coords.items() if key in ['x', 'y']}
        interp = raster.interp(coords_to_use, method)

        return interp

    def build_dataset(
        self, 
        shape: tuple[int, int], 
        coords: dict[int, xr.DataArray], 
        out = False
    ):
        """Build an xarray Dataset from the fields provided to this instance."""

        for var, raster in self._rasters.items():
            clipped = self.clip(raster)
            resampled = self.resample_to_shape(clipped, shape)
            interpolated = self.interpolate(resampled, coords)

            self._results[var] = interpolated

        self.dataset = xr.Dataset(self._results)

        if out:
            return self.dataset

    def write_netcdf(self, output_path: str, **kwargs):
        """Write output, optionally renaming variables."""
        ds = self.dataset.rename(kwargs)
        ds.to_netcdf(output_path)

    def plot_raster(self, raster: xr.DataArray) -> None:
        """Plot a raster with xr.plot.imshow()."""
        xr.plot.imshow(raster)
        plt.show()

def main():
    """Runs the ClipToCatchment algorithm for all files in data/basin-outlines."""
    shapefiles = glob.glob('data/basin-outlines/**/*.geojson')

    var_names = {
        'surface': 'usurfobs',
        'thickness': 'thkobs',
        'vx': 'uvelsurfobs',
        'vy': 'vvelsurfobs'
    }

    output_dir = 'data/basin-netcdfs/'

    for basin in shapefiles:
        basin_name = basin.split('/')[-1].replace('.geojson', '')

        CC = ClipToCatchment(
            basin,
            surface = 'data/ignore/BedMachineGreenland-v5.nc',
            thickness = 'data/ignore/BedMachineGreenland-v5.nc',
            vx = 'data/ignore/GRE_G0120_0000.nc',
            vy = 'data/ignore/GRE_G0120_0000.nc'
        )
        
        clipped = CC.clip(CC._rasters['thickness'])
        resampled = CC.resample_to_resolution(clipped, (100, 100))        
        shape = resampled.shape
        coords = resampled.coords

        CC.build_dataset(shape = shape, coords = coords)
        CC.write_netcdf(output_dir + basin_name + '.nc', **var_names)

        print('Finished processing input data for ' + basin_name.replace('-', ' ').title() + '.')
                
if __name__ == '__main__':
    main()