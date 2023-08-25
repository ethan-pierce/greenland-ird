"""Utilities for plotting fields on unstructured meshes."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.collections


def plot_triangle_mesh(
    grid, 
    field, 
    cmap = plt.cm.jet, 
    subplots_args = None,
    show = True
):
    """Plot a field defined on an unstructured mesh."""
    if isinstance(field, str):
        if field in grid.at_node.keys():
            field = grid.at_node[field][:]
        elif field in grid.at_link.keys():
            field = grid.map_mean_of_links_to_node(field)
        else:
            raise ValueError(
                "Could not find " + field + " at grid nodes or links."
            )

    if hasattr(field, 'shape'):
        if len(np.ravel(field)) == grid.number_of_nodes:
            pass
        elif len(np.ravel(field)) == grid.number_of_links:
            field = grid.map_mean_of_links_to_node(field)
        else:
            raise ValueError(
                "Could not broadcast " + field + " to grid nodes or links."
            )

    values = grid.map_mean_of_patch_nodes_to_patch(field)

    if subplots_args is None:
        subplots_args = {'nrows': 1, 'ncols': 1}

    fig, ax = plt.subplots(**subplots_args)

    coords = []
    for patch in range(grid.number_of_patches):
        nodes = []

        for node in grid.nodes_at_patch[patch]:
            nodes.append(
                [grid.node_x[node], grid.node_y[node]]
            )

        coords.append(nodes)

    coords = np.array(coords) # seems to improve performance?

    polys = [plt.Polygon(i) for i in coords]

    collection = matplotlib.collections.PatchCollection(polys, cmap=cmap)
    collection.set_array(values)
    im = ax.add_collection(collection)
    ax.autoscale()

    plt.colorbar(im)

    if show:
        plt.show()
    
    return fig, ax

    