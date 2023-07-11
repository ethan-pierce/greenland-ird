"""Generates a mesh from a shapefile.

Provides utility functions to translate a geojson polygon into a constrained
Delaunay triangulation using a Python wrapper for Shewchuk's Triangle algorithm.
The resulting triangulation can then be used to generate a VoronoiDelaunayGrid
in the Landlab modeling framework.
"""

import numpy as np
import argparse
import geopandas as gpd
import triangle
import shapely
import matplotlib.pyplot as plt

class VoronoiDelaunay:
    """Constructs the Voronoi-Delaunay dual graph.
    
    We need to supply the following:
    * xy_of_node
    * nodes_at_link
    * links_at_patch
    * xy_of_corner
    * corners_at_face
    * faces_at_cell
    * node_at_cell
    * nodes_at_face
    """

    def __init__(self):
        """Initialize the constructor with dictionaries of Voronoi and Delaunay points."""
        pass

class MeshGenerator:
    """Uses Shewchuk's Triangle to generate a mesh over a specified domain."""

    def __init__(self, path: str):
        """Initialize the utility with a path to an input file."""
        self._path = path
        self._shape = gpd.read_file(self._path).geometry
        self._poly = shapely.build_area(self._shape.geometry[0])
        self._exterior = self._poly.exterior
        self._vertices = shapely.get_coordinates(self._exterior)
        self._segments = self.segment(self._exterior)

        self.delaunay = None
        self.voronoi = None
        
    def segment(self, curve) -> list:
        """Given a LineString or LinearRing, return a list of line segments."""
        lines = list(map(shapely.LineString, zip(curve.coords[:-1], curve.coords[1:])))
        segments = []

        for line in lines:
            x1, y1 = line.coords[0]
            x2, y2 = line.coords[1]
            
            start_vertex = np.argwhere((self._vertices[:, 0] == x1) & (self._vertices[:, 1] == y1))[0]
            end_vertex = np.argwhere((self._vertices[:, 0] == x2) & (self._vertices[:, 1] == y2))[0]

            segments.append([int(start_vertex[0]), int(end_vertex[0])])

        return segments

    def triangulate(self, opts: str = ''):
        """Perform the triangulation over the domain."""
        geometry = {
            'vertices': self._vertices,
            'segments': self._segments
        }

        self.delaunay = triangle.triangulate(geometry, opts = opts)

        points, edges, ray_origin, ray_direction = triangle.voronoi(self.delaunay['vertices'])
        self.voronoi = {
            'vertices': points,
            'edges': edges
        }

    def generate_grid():
        """Generate a Landlab grid using the triangulated mesh."""
        pass

    def write_out():
        """Write out the generated geometry and/or Landlab grid."""
        pass

    def plot(self, vertices, edges = None, bounds = None, subplots_kwargs: dict = {}):
        """Given a list of vertices with coordinates, plot the mesh."""
        fig, ax = plt.subplots(**subplots_kwargs)

        ax.scatter(vertices[:,0], vertices[:,1], color = 'tab:blue', s = 2)

        if len(edges) > 0:
            for edge in edges:
                a, b = edge
                coords = [
                    [vertices[a][0], vertices[b][0], vertices[a][0]],
                    [vertices[a][1], vertices[b][1], vertices[a][1]]
                ]

                ax.plot(coords[0], coords[1], color = 'tab:orange', linewidth = 1)

        if bounds is not None:
            x0, y0, x1, y1 = bounds
            ax.set_xlim([x0, x1])
            ax.set_ylim([y0, y1])

        plt.show()

def main():
    parser = argparse.ArgumentParser(description = 'Mesh generation utility.')
    parser.add_argument(
        'input',
        metavar = 'f',
        type = str,
        nargs = 1,
        help = 'Input shapefile of the mesh domain.'
    )

    path = parser.parse_args().input[0]

    Mesh = MeshGenerator(path)

    Mesh.triangulate(opts = 'pq30a100000Dez')

    Mesh.plot(Mesh.delaunay['vertices'], edges = Mesh.delaunay['edges'], subplots_kwargs = {'figsize': (16, 6)})

    Mesh.plot(
        Mesh.voronoi['vertices'], 
        edges = Mesh.voronoi['edges'], 
        bounds = [
            np.min(Mesh.delaunay['vertices'][:,0]),
            np.min(Mesh.delaunay['vertices'][:,1]),
            np.max(Mesh.delaunay['vertices'][:,0]),
            np.max(Mesh.delaunay['vertices'][:,1]),
        ],
        subplots_kwargs = {'figsize': (16, 6)}
    )

if __name__ == '__main__':
    main()