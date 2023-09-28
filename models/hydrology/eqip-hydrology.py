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

s0 = jnp.full(mesh.number_of_nodes, 0.01)
h0 = glacier.bedrock_elevation
omega = np.full(mesh.number_of_links, glacier.flow_regime_scalar)
RI = ReynoldsIteration(mesh, glacier, s0, h0, omega)
Re0 = RI.update()

K0 = (
    mesh.map_mean_of_link_nodes_to_link(s0)**3 * glacier.gravity /
    (12 * glacier.water_viscosity * (1 + glacier.flow_regime_scalar * Re0))
)
f0 = np.ones(mesh.number_of_nodes)
f0[mesh.node_is_boundary] = glacier.bedrock_elevation[mesh.node_is_boundary]

HP = HeadPDE(mesh, glacier, h0, f0, K0)

print('Solving linear system...')
sol = HP.update()

print(sol)

# plot_links(grid, Re, subplots_args={'figsize': (18, 6)})
# plot_links(grid, Q, subplots_args={'figsize': (18, 6)})
# plot_triangle_mesh(grid, sol.value, at = 'patch', subplots_args={'figsize': (18, 6)})

# fig, ax = plt.subplots(figsize = (18, 6))
# im = ax.scatter(mesh.node_x, mesh.node_y, c = Re, cmap = 'jet', s = 2)
# plt.colorbar(im)
# plt.show()

# bc = mesh.node_is_boundary
# im = plt.scatter(mesh.node_x[bc], mesh.node_y[bc], c = glacier.boundary_types[bc], s = 2)
# plt.colorbar(im)
# plt.show()
