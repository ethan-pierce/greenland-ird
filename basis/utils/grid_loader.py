"""Generate a TriangleMeshGrid, add netCDF data, and pickle it."""

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
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
        self.geoseries = gpd.read_file(shapefile)
        self.crs = str(self.geoseries.crs)

    def _open_data(self, path: str, crs_key: str = '', crs: str = '') -> xr.Dataset:
        """Read a netCDF file as an xarray Dataset."""
        ds = xr.open_dataset(path)

        if len(crs_key) > 0:
            ds.rio.write_crs(ds.attrs[crs_key], inplace=True)
        elif len(crs) > 0:
            ds.rio.write_crs(crs, inplace=True)

        return ds

    def _reproject(self, source: xr.Dataset, dest: str) -> xr.Dataset:
        """Reproject a dataset from source crs to destination crs."""
        return source.rio.reproject(dst_crs = dest)

    def add_field(self):
        """Read a field from a netCDF file and add it to the grid."""
        pass
    

def main():
    """Generate a mesh and add netCDF data."""
    path = '/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson'
    gl = GridLoader(path)

    bedmachine = '/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc'

    ds = gl._open_data(bedmachine, crs = 'epsg:3413')
    plt.imshow(gl._reproject(ds, gl.crs))
    plt.show()


    # with xr.open_dataset(bedmachine) as ds:
    #     print(ds.attrs['proj4'])

if __name__ == '__main__':
    main()