"""Clips raster data to a vector shapefile.

Given multiple input rasters and a vector shapefile of the catchment boundary, 
this utility will clip each dataset to the catchment and resample the raster
data to the same (x, y) size. The utility provides a single netCDF file as
its output, with variables for each raster and the boundary shapefile.

"""

import os
import numpy as np

class ClipToCatchment:

    def __init__(
        self,
        shapefile: str,
        raster_files: dict[str: str]
    ):
        pass
        

    def clip_raster(self):
        """Clip an input raster to the catchment boundary."""
        pass

    def resample_raster(self):
        """Resample an input raster to a given shape."""
        pass

    def build_dataset(self):
        """Build an xarray Dataset from the fields provided to this instance."""
        pass

    def write_netcdf(self):
        """Write output."""
        pass

def main():
    """Runs the ClipToCatchment algorithm with user-specified inputs."""
    pass

if __name__ == '__main__':
    main()