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

h = grid.at_node['ice_thickness']
h[h < 0] = 0.0

m = grid.at_node['meltwater_input']
m[mesh.node_is_boundary] = 0.0

S = grid.at_link['conduit_area']
S[:] = 0.0
S[mesh.status_at_link != 0] = 0.0

glacier = Glacier(
    mesh,
    grid.at_node["ice_thickness"],
    grid.at_node["bedrock_elevation"],
    grid.at_node["meltwater_input"],
    grid.at_link["ice_sliding_velocity"],
)

# plot_links(grid, Q, subplots_args={'figsize': (18, 6)})
fig, ax = plot_triangle_mesh(grid, mesh.area_at_node, at = 'cell', subplots_args={'figsize': (18, 6)})
