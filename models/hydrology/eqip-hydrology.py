"""Model the subglacial drainage system at Eqip Sermia, CW Greenland."""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from jax import config
config.update("jax_enable_x64", True)

from basis.utils.plotting import plot_triangle_mesh
from basis.components.conduit_network import *

with open(
    "/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid", "rb"
) as pf:
    grid = pickle.load(pf)
print("Grid has: ", grid.at_node.keys(), " at nodes.")
print("Grid has: ", grid.at_link.keys(), " at links.")

h = grid.at_node['ice_thickness']
h[h < 0] = 0.0

mesh = StaticGraph.from_grid(grid)

glacier = Glacier(
    mesh,
    grid.at_node["ice_thickness"],
    grid.at_node["bedrock_elevation"],
    grid.at_node["meltwater_input"],
    grid.at_link["ice_sliding_velocity"],
)

pw0 = jnp.full(mesh.number_of_nodes, 100e3)
S0 = jnp.asarray(grid.at_link['conduit_area'][:])
conduits = Conduits(mesh, glacier, pw0, S0)

dt = 60
for i in range(10):
    conduits = conduits.run_one_step(dt)

plot_triangle_mesh(grid, conduits.init_water_pressure, subplots_args={'figsize': (18, 6)})

Snodes = mesh.map_mean_of_links_to_node(conduits.init_conduit_area)
plot_triangle_mesh(grid, Snodes, subplots_args={'figsize': (18, 6)})
