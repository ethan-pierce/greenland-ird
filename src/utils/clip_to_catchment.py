"""Clips raster data to a vector shapefile.

Given multiple input rasters and a vector shapefile of the regional boundary, 
this utility will clip each dataset to the catchment and resample the raster
data to the same (x, y) size. The utility provides a single netCDF file as
its output, with variables for each raster and the boundary shapefile.

Note: raster data is expected to be in netcdf format. 
Similarly, vector data is expected in geojson format.
"""

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
            self._rasters[var] = rxr.open_rasterio("netcdf:" + path + ":" + var)

        self.resample = {
            'shape': self.resample_to_shape,
            'resolution': self.resample_to_resolution
        }

        self.dataset = None # this attribute will be populated by build_dataset()

    def clip(self, var: str) -> xr.DataArray:
        """Clip an input raster to the catchment boundary."""
        data = self._rasters[var]
        geometry = self._basin.geometry.values
        crs = self._basin.crs
        clipped = data.rio.clip(geometry, crs)

        return clipped

    def resample_to_shape(self, var: str, shape: tuple[int, int]) -> xr.DataArray:
        """Resample an input raster to a given shape."""
        data = self._rasters[var]
        resampled = data.rio.reproject(
            data.rio.crs, 
            shape = shape, 
            resampling = Resampling.bilinear
        )

        return resampled

    def resample_to_resolution(self, var: str, resolution: tuple[int, int]) -> xr.DataArray:
        """Resample an input raster to a given resolution."""
        data = self._rasters[var]
        resampled = data.rio.reproject(
            data.rio.crs,
            resolution = resolution,
            resampling = Resampling.bilinear
        )

        return resampled

    def build_dataset(self, sampling: tuple[int, int], method = 'shape', out = False):
        """Build an xarray Dataset from the fields provided to this instance."""
        data_vars = {var: None for var in self._rasters.keys()}
        shapes = []

        for var in self._rasters.keys():
            clipped = self.clip(var)
            resampled = self.resample[method](var, sampling)
            data_vars[var] = resampled
            shapes.append(resampled.shape)

        group = groupby(shapes)
        if not (next(group, True) and not next(group, False)):
            raise ValueError("After resampling, not all rasters have the same shape.")

        ds = xr.Dataset(data_vars)

        self.dataset = ds

        if out:
            return ds

    def write_netcdf(self, output_path: str, **kwargs):
        """Write output, optionally renaming variables."""
        ds = self.dataset.rename(kwargs)
        ds.to_netcdf(output_path)

def main():
    """Runs the ClipToCatchment algorithm for all files in data/basin-outlines."""
    shapefiles = glob.glob('data/basin-outlines/**/*.geojson')

    var_names = {
        'bed': 'bedrock_elevation',
        'thickness': 'ice_thickness',
        'vx': 'surface_velocity_x',
        'vy': 'surface_velocity_y'
    }

    output_dir = 'data/basin-netcdfs/'

    for basin in shapefiles:
        basin_name = basin.split('/')[-1].replace('.geojson', '')

        CC = ClipToCatchment(
            basin,
            bed = 'data/ignore/BedMachineGreenland-v5.nc',
            thickness = 'data/ignore/BedMachineGreenland-v5.nc',
            vx = 'data/ignore/GRE_G0120_0000.nc',
            vy = 'data/ignore/GRE_G0120_0000.nc'
        )

        res_sample = CC.resample_to_resolution('bed', (100, 100))
        CC.build_dataset(sampling = res_sample.shape)
        CC.write_netcdf(output_dir + basin_name + '.nc', var_names)

        break

if __name__ == '__main__':
    main()