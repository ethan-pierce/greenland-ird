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
        self._vertices = None
        self._segments = None

        self.mesh = None
        
    def segment(self, curve) -> list:
        """Given a LineString or LinearRing, return a list of line segments."""
        pass

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

if __name__ == '__main__':
    main()