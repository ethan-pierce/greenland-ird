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
print("Grid has: ", grid.at_node.keys(), " at nodes.")
print("Grid has: ", grid.at_link.keys(), " at links.")

mesh = StaticGraph.from_grid(grid)

h = grid.at_node['ice_thickness']
h[h < 0] = 0.0

m = grid.at_node['meltwater_input']
m[mesh.node_is_boundary] = 0.0

glacier = Glacier(
    mesh,
    grid.at_node["ice_thickness"],
    grid.at_node["bedrock_elevation"],
    grid.at_node["meltwater_input"],
    grid.at_link["ice_sliding_velocity"],
)

pw0 = jnp.asarray(0.1 * glacier.overburden_pressure)
pw0 = pw0.at[mesh.node_is_boundary].set(0.0)

S0 = jnp.asarray(grid.at_link['conduit_area'][:])
conduits = Conduits(mesh, glacier, pw0, S0)
init_state = conduits._resolve_state(conduits.init_water_pressure, conduits.init_conduit_area)

S = conduits._evolve_conduits(60)
melt = glacier.melt_constant * grid.at_link['water_flux'] * grid.at_link['hydraulic_gradient']
gap = glacier.ice_sliding_velocity * glacier.step_height
closure = (
    glacier.closure_constant
    * jnp.power(init_state[1], glacier.glens_n)
    * conduits.init_conduit_area
)

plot_links(grid, melt + gap - closure, subplots_args={'figsize': (18, 6)})
# plot_triangle_mesh(grid, glacier.meltwater_input, subplots_args={'figsize': (18, 6)})