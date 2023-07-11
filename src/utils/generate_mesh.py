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

        self.mesh = None
        
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

        self.mesh = triangle.triangulate(geometry, opts = opts)

    def generate_grid():
        """Generate a Landlab grid using the triangulated mesh."""
        pass

    def write_out():
        """Write out the generated geometry and/or Landlab grid."""
        pass


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

    print(Mesh.mesh.keys())
    print(len(Mesh.mesh['triangles']))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.scatter(Mesh.mesh['vertices'][:,0], Mesh.mesh['vertices'][:,1], s = 2)

    for triangle in Mesh.mesh['triangles']:
        a, b, c = triangle

        coords = [
            [Mesh.mesh['vertices'][a][0], Mesh.mesh['vertices'][b][0], Mesh.mesh['vertices'][c][0], Mesh.mesh['vertices'][a][0]],
            [Mesh.mesh['vertices'][a][1], Mesh.mesh['vertices'][b][1], Mesh.mesh['vertices'][c][1], Mesh.mesh['vertices'][a][1]]
        ]

        ax.plot(coords[0], coords[1], color = 'red', linewidth = 1)
            
    plt.show()

if __name__ == '__main__':
    main()