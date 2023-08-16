"""Generate a TriangleMeshGrid, add netCDF data, and pickle it."""

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
from scipy.interpolate import CloughTocher2DInterpolator
import argparse
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

    def load_netcdf(self, path: str, vars: list, names: list = []):
        """Load netCDF data and add it to the grid."""
        ds = xr.open_dataset(path)
        xs = ds.coords['x']
        ys = ds.coords['y']

        for i in range(len(vars)):
            da = ds.data_vars[vars[i]]
            field = da.interp(x = self.grid.node_x, y = self.grid.node_y, method = 'cubic')
            self.grid.add_field(names[i], field, at = 'node')

    def calc_raster(self, name: str, function, *args):
        """Apply a function to existing grid fields to create a new field."""
        field = function(*args)
        self.grid.add_field(name, field, at = 'node')

def main():
    """Generate a mesh and add netCDF data."""
    parser = argparse.ArgumentParser(description = 'Utility for loading gridded data.')
    parser.add_argument(
        '--file', 
        '-f', 
        metavar='shapefile', 
        dest='shapefile', 
        help='Shapefile of the catchment area.'
    )
    
    args = parser.parse_args()

    grid = GridLoader(args.shapefile)

    ncfile = args.shapefile.replace('basin-outlines', 'basin-netcdfs').replace('CW/', '').replace('.geojson', '.nc')
    grid.load_netcdf(
        ncfile, 
        vars = ['thkobs', 'usurfobs'],
        names = ['ice_thickness', 'surface_elevation']
    )
    grid.calc_raster(
        'bedrock_elevation', 
        lambda u, h: u - h,
        grid.grid.at_node['surface_elevation'][:],
        grid.grid.at_node['ice_thickness'][:]
    )


if __name__ == '__main__':
    main()