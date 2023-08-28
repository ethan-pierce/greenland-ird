"""Generate a TriangleMeshGrid, add netCDF data, and pickle it."""

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import shapely
import itertools
from scipy.interpolate import RBFInterpolator
from landlab import TriangleMeshGrid


class GridLoader:
    """Constructs a Landlab grid and adds gridded data to it."""

    def __init__(self, shapefile: str, quality: int = 30, max_area: float = 1e6, buffer = 0.0):
        """Initialize a new GridLoader object."""
        quality_flag = "q" + str(quality)
        area_flag = "a" + str(max_area)
        triangle_opts = "pDevjz" + quality_flag + area_flag
        
        self.geoseries = gpd.read_file(shapefile)
        self.crs = str(self.geoseries.crs)

        self.polygon = self._smooth_boundary(self.geoseries.geometry, buffer = buffer)[0]
        nodes_y = np.array(self.polygon.exterior.xy[1])
        nodes_x = np.array(self.polygon.exterior.xy[0])
        holes = self.polygon.interiors

        self.grid = TriangleMeshGrid(
            (nodes_y, nodes_x), holes = holes, triangle_opts=triangle_opts
        )

    def _smooth_boundary(self, polygon, buffer: float) -> shapely.Polygon:
        """Smooth the exterior boundary of the input shapefile."""

        # Dilate by x, erode by 2x, dilate again by x
        new_poly = (
            polygon.buffer(buffer, join_style='round')
            .buffer(-2 * buffer, join_style='round')
            .buffer(buffer, join_style='round')
        )

        return new_poly

    def _open_data(self, path: str, var: str, crs=None, no_data=None) -> xr.DataArray:
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
        return source.rio.clip(
            geometries=[self.polygon], crs=self.crs, drop=True
        )

    def _reproject(self, source: xr.DataArray, dest: str = "") -> xr.DataArray:
        """Reproject data from source crs to destination crs."""
        if len(dest) == 0:
            dest = self.crs

        return source.rio.reproject(dst_crs=dest)

    def _interpolate_na(
        self, source: xr.DataArray, method: str = "nearest"
    ) -> xr.DataArray:
        """Interpolate missing data using scipy.interpolate.griddata."""
        return source.rio.interpolate_na(method=method)

    def _rescale(self, source: xr.DataArray, scalar: float) -> xr.DataArray:
        """Multiply a dataarray by a scalar."""
        return source * scalar

    def _interpolate(
        self, source: xr.DataArray, neighbors: int = 9, smoothing: float = 0.0
    ) -> np.ndarray:
        """Interpolate a dataarray to the new grid coordinates."""
        stack = source.stack(z=("x", "y"))
        coords = np.vstack([stack.coords["x"], stack.coords["y"]]).T
        values = source.values.flatten(order="F")

        destination = np.vstack([self.grid.node_x, self.grid.node_y]).T

        interp = RBFInterpolator(
            coords, values, neighbors=neighbors, smoothing=smoothing
        )
        result = interp(destination)

        return result

    def add_field(
        self,
        path: str,
        nc_name: str,
        ll_name: str = "",
        crs=None,
        no_data=None,
        neighbors=9,
        smoothing=0.0,
        scalar=1.0,
    ):
        """Read a field from a netCDF file and add it to the grid."""
        if len(ll_name) == 0:
            ll_name = nc_name

        opened = self._open_data(path, nc_name, crs=crs, no_data=no_data)
        clipped = self._clip(opened)
        projected = self._reproject(clipped)
        filled = self._interpolate_na(projected)
        rescaled = self._rescale(filled, scalar)
        gridded = self._interpolate(rescaled, neighbors=neighbors, smoothing=smoothing)

        self.grid.add_field(ll_name, gridded, at="node")


def main():
    """Generate a mesh and add netCDF data."""
    path = "/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson"
    gl = GridLoader(path, quality=30, max_area=400**2)
    print(
        "Generated grid for: ",
        path.split("/")[-1].replace("-", " ").replace(".geojson", "").capitalize(),
    )

    bedmachine = "/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"
    gl.add_field(
        bedmachine,
        "thickness",
        "ice_thickness",
        crs="epsg:3413",
        neighbors=100,
        no_data=-9999.0,
    )
    print("Added ice thickness to grid nodes.")

    im = plt.scatter(
        gl.grid.node_x, gl.grid.node_y, c=gl.grid.at_node["ice_thickness"], s=2
    )
    plt.colorbar(im)
    plt.show()

    gl.add_field(
        bedmachine,
        "bed",
        "bedrock_elevation",
        crs="epsg:3413",
        neighbors=100,
        no_data=-9999.0,
    )
    print("Added bedrock elevation to grid nodes.")

    velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
    gl.add_field(velocity, "v", "surface_velocity", crs="epsg:3413", neighbors=100)
    print("Added surface velocity to grid nodes.")


if __name__ == "__main__":
    main()
