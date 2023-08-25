"""Model the subglacial drainage system at Eqip Sermia, CW Greenland."""
# import os
# os.environ["JAX_ENABLE_X64"] = True

import numpy as np
import pickle
import matplotlib.pyplot as plt
from jax import config

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

conduits = Conduits(mesh)

pw0 = jnp.zeros(mesh.number_of_nodes)
S0 = jnp.asarray(grid.at_link['conduit_area'][:])
y0 = [pw0, S0]
S = S0
y = y0
dt = 1

for i in range(10):
    pw, dS = conduits(0.0, y, glacier)
    S = S + dS * dt
    S = S.at[S < 0].set(0.0)
    y = [pw, S]

    pw, S = conduits(0.0, y, glacier)
    plot_triangle_mesh(grid, pw, subplots_args={'figsize': (18, 6)})

    print(np.count_nonzero(np.isnan(pw)))
    print(np.count_nonzero(np.isnan(S)))

    print(f"Mean water pressure: {jnp.nanmean(y[0])} Pa.")
    print(f"Mean conduit area: {jnp.nanmean(y[1])} m^2.")

# plot_triangle_mesh(grid, pw, subplots_args={'figsize': (18, 6)})


# terms = diffrax.ODETerm(conduits)
# solver = diffrax.Tsit5()
# t0 = 0.0
# t1 = 60 * 60
# dt0 = 0.1
# pw0 = jnp.zeros(mesh.number_of_nodes)
# S0 = jnp.asarray(grid.at_link['conduit_area'][:])
# y0 = [pw0, S0]
# args = glacier
# saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, 10))
# stepsize_controller = diffrax.PIDController(rtol = 1e-3, atol = 1e-6)

# sol = diffrax.diffeqsolve(
#     terms, 
#     solver, 
#     t0, 
#     t1, 
#     dt0, 
#     y0, 
#     args = args, 
#     saveat = saveat,
#     stepsize_controller = stepsize_controller
# )







# tprev = t0
# tnext = t0 + dt0
# y = y0

# state = solver.init(terms, tprev, tnext, y0, args)

# while tprev < t1:
#     y, _, _, state, _ = solver.step(terms, tprev, tnext, y, args, state, made_jump=False)
#     pw, S = y
#     S = S.at[S < 0].set(0.0)
#     y = [pw, S]

#     # plot_triangle_mesh(grid, pw, subplots_args={'figsize': (18, 6)})

#     print(f"Mean water pressure: {jnp.nanmean(y[0])} Pa.")
#     print(f"Mean conduit area: {jnp.nanmean(y[1])} m^2.")

#     tprev = tnext
#     tnext = min(tprev + dt0, t1)
