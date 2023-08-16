"""Models the subglacial drainage system at Eqip Sermia, CW Greenland."""

import numpy as np
import matplotlib.pyplot as plt

from basis.components.jax_conduit_network import ConduitNetwork
from landlab import TriangleMeshGrid

class HydrologicModel:
    """Set up boundary conditions, model subglacial hydrology, and plot results."""

    def __init__(self, shapefile: str, mesh_opts: str = 'pqDevjz'):
        """Initialize the model with a path to a shapefile."""
        self.grid = TriangleMeshGrid.from_shapefile(shapefile, triangle_opts = mesh_opts)


def main():
    """Plot the drainage system evolution over the model run."""
    model = HydrologicModel(
        '/home/egp/repos/greenland-ird/data/basin-outlines/CW/eqip-sermia.geojson',
        mesh_opts = 'pa40000q30Devjz'
    )

    fig = model.grid.plot_nodes_and_links(
        subplots_args = {'figsize': (12, 4)},
        nodes_args = {'s': 5, 'color': 'tab:orange'},
        links_args = {'lw': 1, 'linestyle': ':', 'color': 'tab:blue'}
    )
    plt.show()
    
    print('Grid size: ', model.grid.number_of_links, ' links')
    print('Grid size: ', model.grid.number_of_nodes, ' nodes')
    print('Mean element area: ', np.mean(
        model.grid.area_of_patch[model.grid.area_of_patch > 0]
    ), ' m^2.')
    print('Mean link length: ', np.mean(model.grid.length_of_link), ' m.')

if __name__ == '__main__':
    main()