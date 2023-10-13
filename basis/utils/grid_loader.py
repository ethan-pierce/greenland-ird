"""Generate a TriangleMeshGrid, add netCDF data, and pickle it."""

import matplotlib.pyplot as plt

import os
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

    def __init__(
        self, 
        shapefile: str, 
        quality: int = 30, 
        max_area: float = 1e6, 
        buffer: float = 0.0, 
        tolerance: float = 0.0,
        generate_grid: bool = True
    ):
        """Initialize a new GridLoader object."""
        quality_flag = "q" + str(quality)
        area_flag = "a" + str(max_area)
        triangle_opts = "pDevjz" + quality_flag + area_flag
        
        self.geoseries = gpd.read_file(shapefile)
        self.crs = str(self.geoseries.crs)

        boundary = self._smooth_boundary(self.geoseries.geometry, buffer = buffer)[0]
        self.polygon = shapely.simplify(boundary, tolerance)

        if generate_grid:
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

    def _clip_to_box(self, source: xr.DataArray) -> xr.DataArray:
        """Clip data to a bounding box around the shapefile."""
        return source.rio.clip_box(
            minx = self.geoseries.get_coordinates().x.min(),
            miny = self.geoseries.get_coordinates().y.min(),
            maxx = self.geoseries.get_coordinates().x.max(),
            maxy = self.geoseries.get_coordinates().y.max()
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

    def write_input_nc(
        self, 
        path_to_write: str,
        data_vars: list,
        nc_files: list,
        input_names: list,
        crs: list,
        no_data: list,
        scalars: list,
        add_igm_aux_vars: True,
        write_output = False,
        yield_output = False
    ):
        data_arrays = []

        for i in range(len(data_vars)):
            opened = self._open_data(nc_files[i], input_names[i], crs = crs[i], no_data = no_data[i])
            clipped = self._clip_to_box(opened)
            projected = self._reproject(clipped)
            filled = self._interpolate_na(projected)
            rescaled = self._rescale(filled, scalars[i])

            data_arrays.append(rescaled)

        if add_igm_aux_vars:
            thkidx = data_vars.index('thk')
            icemask = np.where(
                data_arrays[thkidx].values[:] > 0.1,
                1.0,
                0.0
            )
            data_vars.append('icemask')
            data_arrays.append((data_arrays[thkidx].dims, icemask))

            icemaskobs = icemask
            data_vars.append('icemaskobs')
            data_arrays.append((data_arrays[thkidx].dims, icemaskobs))

            data_vars.append('thkobs')
            data_arrays.append(data_arrays[thkidx])

            usurfidx = data_vars.index('usurf')
            data_vars.append('usurfobs')
            data_arrays.append(data_arrays[usurfidx])

            uvelsurfidx = data_vars.index('uvelsurf')
            data_vars.append('uvelsurfobs')
            data_arrays.append(data_arrays[uvelsurfidx])

            vvelsurfidx = data_vars.index('vvelsurf')
            data_vars.append('vvelsurfobs')
            data_arrays.append(data_arrays[vvelsurfidx])

        dataset = xr.Dataset(
            data_vars = {data_vars[i]: data_arrays[i] for i in range(len(data_vars))},
            coords = data_arrays[0].coords
        )

        if write_output:
            dataset.to_netcdf(path_to_write)

        if yield_output:
            return dataset, data_arrays


def main():
    """Generate a mesh and add netCDF data."""
    import warnings
    warnings.filterwarnings("ignore")

    bedmachine = "/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"
    velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
    shapefiles = "/home/egp/repos/greenland-ird/data/basin-outlines/"
    paths = []
    for i in os.listdir('/home/egp/repos/greenland-ird/data/basin-outlines/CE/'):
        paths.append('CE/' + i)
    for i in os.listdir('/home/egp/repos/greenland-ird/data/basin-outlines/CW/'):
        paths.append('CW/' + i)
    for i in os.listdir('/home/egp/repos/greenland-ird/data/basin-outlines/SW/'):
        paths.append('SW/' + i)

    for path in paths:
        glacier = path.split('/')[-1].replace('.geojson', '')

        loader = GridLoader(shapefiles + path, generate_grid = False)
        loader.write_input_nc(
            path_to_write = '/home/egp/repos/greenland-ird/data/igm-inputs/' + glacier + '.nc',
            data_vars = ['usurf', 'thk', 'uvelsurf', 'vvelsurf'],
            nc_files = [bedmachine, bedmachine, velocity, velocity],
            input_names = ['surface', 'thickness', 'vx', 'vy'],
            crs = ['epsg:3413', 'epsg:3413', 'epsg:3413', 'epsg:3413'],
            no_data = [-9999.0, -9999.0, None, None],
            scalars = [1.0, 1.0, 1.0, 1.0],
            add_igm_aux_vars = True,
            write_output = True,
            yield_output = False
        )

        print('Finished loading data for ' + glacier.replace('-', ' ').capitalize())
    
    quit()

    path = "/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson"
    bedmachine = "/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"
    velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
    gl = GridLoader(path)
    ds, da = gl.write_input_nc(
        None,
        data_vars = ['usurf', 'thk', 'uvelsurf', 'vvelsurf'],
        nc_files = [bedmachine, bedmachine, velocity, velocity],
        input_names = ['surface', 'thickness', 'vx', 'vy'],
        crs = ['epsg:3413', 'epsg:3413', 'epsg:3413', 'epsg:3413'],
        no_data = [-9999.0, -9999.0, None, None],
        scalars = [1.0, 1.0, 1.0, 1.0],
        add_igm_aux_vars = True,
        write_output = False,
        yield_output = True
    )
    
    print(ds.data_vars.keys())
    print(ds.coords)
    quit()

    path = "/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson"
    gl = GridLoader(path, quality=30, max_area=400**2)
    print(
        "Generated grid for: ",
        path.split("/")[-1].replace("-", " ").replace(".geojson", "").capitalize(),
    )

    ds = gl.write_input_nc(
        None,
        data_vars = ['thk'],
        nc_files = ["/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"],
        input_names = ['thickness'],
        crs = ['epsg:3413'],
        no_data = [-9999.0],
        scalars = [1.0]
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
