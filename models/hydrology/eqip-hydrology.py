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

# s0 = jnp.full(mesh.number_of_nodes, 0.1)
# h0 = glacier.bedrock_elevation
# Re0 = np.full(mesh.number_of_links, 1 / glacier.flow_regime_scalar)

# model = Conduits(mesh, glacier, s0, h0, Re0)

# print('Solving initial system...')
# for i in range(10):
#     model = model.run_one_step(dt = 0.1, n_iter = 1, tolerance = 1, verbose = True)

# reynolds, discharge, transmissivity = model.calc_flow_properties(
#     model.conduit_size, model.hydraulic_head, model.reynolds
# )

# plot_links(grid, grid.length_of_link, subplots_args={'figsize': (18, 6)})
# plot_triangle_mesh(grid, model.hydraulic_head, at = 'patch', subplots_args={'figsize': (18, 6)})

fig, ax = plt.subplots(figsize = (18, 6))
im = ax.scatter(mesh.node_x[mesh.node_is_boundary], mesh.node_y[mesh.node_is_boundary], c = glacier.boundary_types[mesh.node_is_boundary], cmap = 'jet', s = 2)
plt.colorbar(im)
plt.show()

# bc = mesh.node_is_boundary
# im = plt.scatter(mesh.node_x[bc], mesh.node_y[bc], c = glacier.boundary_types[bc], s = 2)
# plt.colorbar(im)
# plt.show()
