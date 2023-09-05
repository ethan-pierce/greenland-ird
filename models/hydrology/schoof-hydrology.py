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

pw0 = jnp.asarray(0.1 * glacier.overburden_pressure)
S0 = jnp.asarray(grid.at_link['conduit_area'][:])
conduits = Conduits(mesh, glacier, pw0, S0)
N, Nlink, psi, Q = conduits._resolve_state(conduits.init_water_pressure, conduits.init_conduit_area)
psi = psi.at[mesh.status_at_link != 0].set(0.0)

# def sum_discharge(psi):
#     psi = psi.at[mesh.status_at_link != 0].set(0.0)
#     Q0 = glacier.flow_constant * conduits.init_conduit_area**glacier.flow_exp
#     Q = jnp.where(
#         psi != 0,
#         Q0 * psi * jnp.abs(psi)**(-1/2),
#         0.0
#     )
#     return conduits._sum_discharge(Q, glacier.meltwater_input)

# jac = jax.jacrev(sum_discharge)(psi)
# damping_parameter = jnp.max(jnp.matmul(jnp.transpose(jac), jac))
# print(damping_parameter)

# solver = jaxopt.LevenbergMarquardt(
#     residual_fun=sum_discharge,
#     damping_parameter=damping_parameter,
#     verbose=True
# )
# solution = solver.run(psi)

# psi0 = glacier.base_gradient.at[mesh.status_at_link != 0].set(0.0)
# forcing = solution.params - psi0


# plot_links(grid, Q, subplots_args={'figsize': (18, 6)})
plot_triangle_mesh(grid, glacier.ice_thickness + glacier.bedrock_elevation, subplots_args={'figsize': (18, 6)})

# bc = mesh.node_is_boundary
# im = plt.scatter(mesh.node_x[bc], mesh.node_y[bc], c = glacier.boundary_ids[bc], s = 2)
# plt.colorbar(im)
# plt.show()
