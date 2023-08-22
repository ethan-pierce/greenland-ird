"""Models the subglacial drainage system at Eqip Sermia, CW Greenland."""

import numpy as np
import xarray as xr
import rioxarray as rxr
import jax
import matplotlib.pyplot as plt
import pickle

from basis.utils.grid_loader import GridLoader
from basis.components.jax_conduit_network import ConduitNetwork

# Step 1: Pre-process ice thickness, bedrock elevation, and ice velocity
path = "/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson"
gl = GridLoader(path, quality = 30, max_area = 400**2)
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
im = plt.scatter(gl.grid.node_x, gl.grid.node_y, s = 2)
plt.colorbar(im)
plt.show()

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
im = plt.scatter(gl.grid.node_x, gl.grid.node_y, c = gl.grid.at_node['ice_thickness'], s = 2)
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
im = plt.scatter(gl.grid.node_x, gl.grid.node_y, c = gl.grid.at_node['bedrock_elevation'], s = 2)
plt.colorbar(im)
plt.show()

velocity = "/home/egp/repos/greenland-ird/data/ignore/GRE_G0120_0000.nc"
gl.add_field(velocity, "v", "surface_velocity", crs="epsg:3413", neighbors=100, no_data=-1, scalar=1/31556926)
print("Added surface velocity to grid nodes.")
im = plt.scatter(gl.grid.node_x, gl.grid.node_y, c = gl.grid.at_node['surface_velocity'], s = 2)
plt.colorbar(im)
plt.show()

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

melt = air_temperature * kh / (rho * L)

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
gl.grid.add_field("water_flux", gl.grid.map_mean_of_link_nodes_to_link('meltwater_input'), at = "link")

gl.grid.save('/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid', clobber = True)
# ##########################################

with open('/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid', 'rb') as pf:
    grid = pickle.load(pf)

model = ConduitNetwork(grid)
print("Established conduit network.")

melt = model._calc_melt_opening()
gap = model._calc_gap_opening()
opening = melt + gap

im = plt.scatter(grid.node_x, grid.node_y, c = model.map_to_nodes(opening, model.grid), s = 2)
plt.colorbar(im)
plt.show()