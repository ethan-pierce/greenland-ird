"""Generate a TriangleMeshGrid, add netCDF data, and pickle it."""

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import itertools
from scipy.interpolate import RBFInterpolator
from landlab import TriangleMeshGrid

class GridLoader:
    """Constructs a Landlab grid and adds gridded data to it."""

    def __init__(
        self, 
        shapefile: str, 
        quality: int = 30, 
        max_area: float = 1e6
    ):
        """Initialize a new GridLoader object."""
        quality_flag = 'q' + str(quality)
        area_flag = 'a' + str(max_area)
        triangle_opts = 'pDevjz' + quality_flag + area_flag

        self.grid = TriangleMeshGrid.from_shapefile(shapefile, triangle_opts = triangle_opts)
        self.geoseries = gpd.read_file(shapefile)
        self.crs = str(self.geoseries.crs)

    def _open_data(self, path: str, var: str, crs = None, no_data = None) -> xr.DataArray:
        """Read a netCDF file as an xarray Dataset."""
        ds = xr.open_dataset(path)
        da = ds.data_vars[var]

        if crs:
            da.rio.write_crs(crs, inplace=True)

        if no_data:
            da = da.where(da != no_data)
            da.rio.write_nodata(np.nan, inplace=True)

        return da

    def _clip(self, source: xr.DataArray) -> xr.DataArray:
        """Clip data to the shapefile bounds."""
        return source.rio.clip(geometries = self.geoseries.geometry, crs = self.crs, drop = True)

    def _reproject(self, source: xr.DataArray, dest: str = '') -> xr.DataArray:
        """Reproject data from source crs to destination crs."""
        if len(dest) == 0:
            dest = self.crs

        return source.rio.reproject(dst_crs = dest)

    def _interpolate(self, source: xr.DataArray, neighbors: int = 3, smoothing: float = 0.0) -> np.ndarray:
        """Interpolate a dataarray to the new grid coordinates."""
        stack = source.stack(z = ('x', 'y'))
        coords = np.vstack([stack.coords['x'], stack.coords['y']]).T
        values = source.values.flatten(order = 'C')

        destination = np.array(itertools.product(self.grid.node_x, self.grid.node_y))
    
        interpolator = RBFInterpolator(coords, values, neighbors=neighbors, smoothing=smoothing)
        result = interpolator(destination)

        return result

    def add_field(self):
        """Read a field from a netCDF file and add it to the grid."""
        pass
    

def main():
    """Generate a mesh and add netCDF data."""
    path = '/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson'
    gl = GridLoader(path)

    bedmachine = '/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc'

    da = gl._open_data(bedmachine, 'thickness', crs = 'epsg:3413', no_data = -9999.0)
    clip = gl._clip(da)
    proj = gl._reproject(clip)
    interp = gl._interpolate(proj, neighbors = 100)

    print(interp.shape)
    print(interp)



if __name__ == '__main__':
    main()