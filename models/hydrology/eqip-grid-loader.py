"""Python script to load data and prepare a grid for Eqip Sermia."""

import numpy as np
from basis.utils.grid_loader import GridLoader

path = "/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson"
gl = GridLoader(path, quality = 30, max_area = 500**2, buffer = 225.0, tolerance = 10.0, centered = False)
print(
    "Generated grid for: ",
    path.split("/")[-1].replace("-", " ").replace(".geojson", "").capitalize(),
)
print(
    "Grid consists of "
    + str(gl.grid.number_of_nodes)
    + " nodes and "
    + str(gl.grid.number_of_links)
    + " links."
)

bedmachine = "/home/egp/repos/greenland-ird/data/ignore/BedMachineGreenland-v5.nc"
gl.add_field(
    bedmachine,
    "thickness",
    "ice_thickness",
    crs="epsg:3413",
    neighbors=9,
    no_data=-9999.0,
)
h = gl.grid.at_node['ice_thickness']
h[h < 0.5] = 0.0
print("Added ice thickness to grid nodes.")

gl.add_field(
    bedmachine,
    "bed",
    "bedrock_elevation",
    crs="epsg:3413",
    neighbors=9,
    no_data=-9999.0,
)
print("Added bedrock elevation to grid nodes.")

gl.grid.add_field(
    "surface_elevation", 
    gl.grid.at_node['ice_thickness'] + gl.grid.at_node['bedrock_elevation'], 
    at = 'node'
)

geotherm = "data/ignore/geothermal_heat_flow_map_10km.nc"
gl.add_field(geotherm, "GHF", "geothermal_heat_flux", crs="epsg:3413", neighbors=100, no_data=np.nan, scalar=1e-3)
print("Added geothermal heat flux to grid nodes.")

velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
gl.add_field(velocity, "v", "surface_velocity", crs="epsg:3413", neighbors=100, no_data=-1, scalar=1/31556926)
print("Added surface velocity to grid nodes.")

velocity_links = gl.grid.map_mean_of_link_nodes_to_link("surface_velocity")
gl.grid.add_field("ice_sliding_velocity", velocity_links * 0.6, at="link")

# Step 2: Estimate meltwater input
lapse = 5e-3
t0 = 2
z0 = 400
kh = 24
rho = 917
L = 3.34e5

air_temperature = t0 - lapse * (
    gl.grid.at_node["ice_thickness"][:] + gl.grid.at_node["bedrock_elevation"][:] - z0
)

# This import statement is here because eventually the grid object should have its own area_at_cell
from basis.components.conduit_network import StaticGraph
mesh = StaticGraph.from_grid(gl.grid)

specific_melt = air_temperature * kh / (rho * L)
melt = specific_melt * mesh.area_at_node
melt = melt.at[melt < 0].set(0.0)

gl.grid.add_field("meltwater_input", melt, at="node")
print("Added meltwater input to grid nodes.")

# Step 3: Save the grid
gl.grid.save('/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid', clobber = True)