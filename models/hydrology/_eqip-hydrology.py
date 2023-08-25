"""Models the subglacial drainage system at Eqip Sermia, CW Greenland."""

import numpy as np
import xarray as xr
import rioxarray as rxr
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import equinox as eqx
import diffrax

from basis.utils.grid_loader import GridLoader
from basis.components.jax_conduit_network import ConduitNetwork

with open('/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid', 'rb') as pf:
    grid = pickle.load(pf)

model = ConduitNetwork(grid)
print("Established conduit network.")

import matplotlib.colors
import matplotlib.patches
import matplotlib.collections

field = model.meltwater_input
# field = model.map_to_nodes(field, model.grid)

boundaries = np.where(
    model.grid.node_is_boundary(model.grid.nodes[:3820]),
    1,
    0
)


fig, ax = plt.subplots(figsize = (14, 4))

cmap = plt.cm.jet

values = model.grid.map_mean_of_patch_nodes_to_patch(field)

coords = []
for patch in range(model.grid.number_of_patches):
    nodes = []

    for node in model.grid.nodes_at_patch[patch]:
        nodes.append(
            [model.grid.node_x[node], model.grid.node_y[node]]
        )

    coords.append(nodes)

coords = np.array(coords)

polys = [plt.Polygon(i) for i in coords]

collection = matplotlib.collections.PatchCollection(polys, cmap=cmap)
collection.set_array(values)
im = ax.add_collection(collection)
ax.autoscale()

plt.colorbar(im)
plt.show()



# def update_conduits(t, conduit_area: jax.Array, args) -> jax.Array:
#     melt_opening, gap_opening, closure = args
#     return melt_opening + gap_opening - closure * conduit_area
    
# terms = diffrax.ODETerm(update_conduits)
# args = (
#     melt_opening,
#     gap_opening,
#     closure
# )
# y0 = conduit_area
# t0 = 0.0
# t1 = 60 * 60 * 24
# dt0 = 1
# solver = diffrax.Tsit5()
# saveat = diffrax.SaveAt(ts = np.linspace(t0, t1, 10))
# ctrl = diffrax.PIDController(
#     rtol = 1e-3,
#     atol = 1e-6,
#     pcoeff=0.3, 
#     icoeff=0.4, 
#     dcoeff=0
# )

# solution = diffrax.diffeqsolve(
#     terms = terms,
#     solver = solver,
#     t0 = t0,
#     t1 = t1,
#     dt0 = dt0,
#     y0 = y0,
#     args = args,
#     saveat = saveat,
#     stepsize_controller = ctrl,
#     max_steps = 1000
# )

# print("Converged in ", solution.stats['num_steps'], " steps.")
# print(solution.ys[-1])