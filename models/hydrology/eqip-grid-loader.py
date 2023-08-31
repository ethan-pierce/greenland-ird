"""Python script to load data and prepare a grid for Eqip Sermia."""

import numpy as np
from basis.utils.grid_loader import GridLoader


path = "/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson"
gl = GridLoader(path, quality = 30, max_area = 200**2, buffer = 225.0, tolerance = 10.0)
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

specific_melt = air_temperature * kh / (rho * L)
melt = specific_melt * np.mean(gl.grid.area_of_patch)
melt[melt < 0] = 0.0

gl.grid.add_field("meltwater_input", melt, at="node")
print("Added meltwater input to grid nodes.")

# Step 3: Set up the ConduitNetwork fields
base_effective_pressure = gl.grid.at_node["ice_thickness"] * 917 * 9.81
gl.grid.add_field(
    "effective_pressure",
    gl.grid.map_mean_of_link_nodes_to_link(base_effective_pressure),
    at="link",
)

base_hydraulic_gradient = 1000 * 9.81 * gl.grid.map_mean_of_link_nodes_to_link(
    "bedrock_elevation"
) + gl.grid.calc_grad_at_link(base_effective_pressure)
gl.grid.add_field("hydraulic_gradient", base_hydraulic_gradient, at="link")

gl.grid.add_field("conduit_area", np.full(gl.grid.number_of_links, 0.1), at="link")

naive_flux = (
    gl.grid.map_mean_of_link_nodes_to_link('meltwater_input') 
    * np.sign(base_hydraulic_gradient)
)
gl.grid.add_field("water_flux", naive_flux, at = "link")

gl.grid.save('/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid', clobber = True)