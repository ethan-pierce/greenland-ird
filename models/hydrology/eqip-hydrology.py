"""Model the subglacial drainage system at Eqip Sermia, CW Greenland."""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)

from basis.utils.plotting import plot_triangle_mesh, plot_links
from basis.components.conduit_network import *

with open(
    "/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid", "rb"
) as pf:
    grid = pickle.load(pf)
print("Grid size: ", grid.number_of_nodes, " nodes; ", grid.number_of_links, " links.")
print("Grid has: ", grid.at_node.keys(), " at nodes.")
print("Grid has: ", grid.at_link.keys(), " at links.")

mesh = StaticGraph.from_grid(grid)

slope_at_nodes = grid.calc_slope_at_node(grid.at_node["surface_elevation"])

glacier = Glacier(
    mesh,
    grid.at_node["ice_thickness"],
    grid.at_node["bedrock_elevation"],
    grid.at_node["surface_elevation"],
    slope_at_nodes,
    grid.at_node["meltwater_input"],
    grid.at_node["geothermal_heat_flux"],
    grid.at_link["ice_sliding_velocity"],
)

h0 = glacier.bedrock_elevation
s0 = jnp.full(mesh.number_of_nodes, 1e-3)
Re0 = jnp.full(mesh.number_of_links, 1 / glacier.flow_regime_scalar)

RI = ReynoldsIteration(mesh, glacier, s0, h0, Re0)
Re = RI.update()
Q = jnp.abs(Re) / glacier.water_viscosity

plot_links(grid, Re, subplots_args={'figsize': (18, 6)})
plot_links(grid, Q, subplots_args={'figsize': (18, 6)})
# plot_triangle_mesh(grid, Re, at = 'patch', subplots_args={'figsize': (18, 6)})

# fig, ax = plt.subplots(figsize = (18, 6))
# im = ax.scatter(mesh.node_x, mesh.node_y, c = Re, cmap = 'jet', s = 2)
# plt.colorbar(im)
# plt.show()

# bc = mesh.node_is_boundary
# im = plt.scatter(mesh.node_x[bc], mesh.node_y[bc], c = glacier.boundary_types[bc], s = 2)
# plt.colorbar(im)
# plt.show()
