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
import xarray as xr

class VoronoiDelaunay:
    """Constructs the Voronoi-Delaunay dual graph.
    
    We translate Landlab grid elements as follows:
    * nodes: Delaunay vertices
    * links: Delaunay edges
    * patches: Delaunay triangles
    * corners: Voronoi vertices
    * faces: Voronoi edges
    * cells: Voronoi polygons
    """

    def __init__(self, delaunay: dict, voronoi: dict):
        """Initialize the constructor with dictionaries of Voronoi and Delaunay points."""
        self._delaunay = delaunay
        self._voronoi = voronoi

        self._mesh = xr.Dataset(
            {
                "node": xr.DataArray(
                    data = np.arange(len(self._delaunay['vertices'])),
                    coords = {
                        "x_of_node": xr.DataArray(self._delaunay['vertices'][:, 0], dims = ("node",)),
                        "y_of_node": xr.DataArray(self._delaunay['vertices'][:, 1], dims = ("node",)),
                    },
                    dims = ("node",),
                ),
                "corner": xr.DataArray(
                    data = np.arange(len(self._voronoi['vertices'])),
                    coords = {
                        "x_of_corner": xr.DataArray(self._voronoi['vertices'][:, 0], dims = ("corner",)),
                        "y_of_corner": xr.DataArray(self._voronoi['vertices'][:, 1], dims = ("corner",)),
                    },
                    dims = ("corner",),
                ),
            }
        )

        corners_at_cell = self._get_corners_at_cell()
        self._mesh.update(
            {
                "nodes_at_link": xr.DataArray(
                    self._delaunay['edges'], dims = ("link", "Two")
                ),
                "nodes_at_patch": xr.DataArray(
                    self._delaunay['triangles'], dims = ("patch", "Three")
                ),
                "corners_at_face": xr.DataArray(
                    self._voronoi['edges'], dims = ("face", "Two")
                ),
                "corners_at_cell": xr.DataArray(
                    corners_at_cell, dims = ("cell", "max_corners_per_cell")
                ), 
                "n_corners_at_cell": xr.DataArray(
                    [len(cell) for cell in corners_at_cell], dims = ("cell",)
                ),
                "nodes_at_face": xr.DataArray(
                    self._get_nodes_at_face(), dims = ("face", "Two")
                ),
                "cell_at_node": xr.DataArray(
                    np.arange(self.number_of_nodes), dims = ("node",)
                )
            }
        )

    def _get_corners_at_cell(self) -> np.ndarray:
        """Construct an array of size (n_cells, max_corners_at_cell) from the Voronoi graph."""

        # TODO can we do better than O(n^2) here?
        corners = [[] for node in np.arange(self.number_of_nodes)]
        max_corners_per_cell = 0

        for cell in np.arange(self.number_of_nodes):
            triangles = np.where(np.isin(self._delaunay['triangles'], cell))[0]
            corners[cell] = triangles.tolist()

            if len(triangles) > max_corners_per_cell:
                max_corners_per_cell = len(triangles)

        corners_at_cell = np.full((self.number_of_nodes, max_corners_per_cell), -1)
        for cell in np.arange(self.number_of_nodes):
            corners_at_cell[cell, :len(corners[cell])] = corners[cell]

        return corners_at_cell

    def _get_nodes_at_face(self) -> np.ndarray:
        """Construct an array of size (n_faces, 2) from the Voronoi graph."""
        nodes_at_face = np.full((len(self._voronoi['edges']), 2), -1)

        for face in np.arange(len(self._voronoi['edges'])):
            if face < len(self._delaunay['edges']):
                nodes_at_face[face] = self._delaunay['edges'][face]
            else:
                pass

        return nodes_at_face

    @property
    def perimiter_nodes(self):
        return [
            node for node in np.arange(self._delaunay['vertex_markers'].shape[0]) if
            self._delaunay['vertex_markers'][node] == 1
        ]

    @property
    def perimiter_links(self):
        return [
            link for link in np.arange(self._delaunay['edge_markers'].shape[0]) if
            self._delaunay['edge_markers'][link] == 1
        ]

    @property
    def number_of_nodes(self):
        return self._mesh.dims["node"]

    @property
    def number_of_links(self):
        return self._mesh.dims["link"]

    @property
    def number_of_patches(self):
        return self._mesh.dims["patch"]

    @property
    def number_of_corners(self):
        return self._mesh.dims["corner"]

    @property
    def number_of_faces(self):
        return self._mesh.dims["face"]

    @property
    def number_of_cells(self):
        return self._mesh.dims["cell"]

    @property
    def x_of_node(self):
        return self._mesh["x_of_node"].values

    @property
    def y_of_node(self):
        return self._mesh["y_of_node"].values

    @property
    def x_of_corner(self):
        return self._mesh["x_of_corner"].values

    @property
    def y_of_corner(self):
        return self._mesh["y_of_corner"].values

    @property
    def nodes_at_patch(self):
        return self._mesh["nodes_at_patch"].values

    @property
    def nodes_at_link(self):
        return self._mesh["nodes_at_link"].values

    @property
    def nodes_at_face(self):
        return self._mesh["nodes_at_face"].values

    @property
    def corners_at_face(self):
        return self._mesh["corners_at_face"].values

    @property
    def corners_at_cell(self):
        return self._mesh["corners_at_cell"].values

    @property
    def n_corners_at_cell(self):
        return self._mesh["n_corners_at_cell"].values

    @property
    def cell_at_node(self):
        return self._mesh["cell_at_node"].values

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

    def generate_graph(self):
        """Generate a Landlab graph structure using the triangulated mesh."""
        self._graph = VoronoiDelaunay(self.delaunay, self.voronoi)

    def write_out():
        """Write out the generated geometry and/or Landlab grid."""
        pass

    def plot(
        self, 
        vertices, 
        edges = None, 
        bounds = None, 
        subplots_kwargs: dict = {}
    ):
        """Given a list of vertices with coordinates, plot the mesh."""
        fig, ax = plt.subplots(**subplots_kwargs)

        ax.scatter(vertices[:,0], vertices[:,1], color = 'tab:blue', s = 2)

        if edges is not None:
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

def test_square_grid(n: int, opts: str = '', plot = True) -> dict:
    ids = np.arange(n**2)
    xs = np.ravel(np.array(np.meshgrid(np.arange(n), np.arange(n))[0]))
    ys = np.ravel(np.array(np.meshgrid(np.arange(n), np.arange(n))[1]))

    verts = [[xs[id], ys[id]] for id in ids]

    tri = triangle.triangulate({'vertices': verts}, opts = opts)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(tri['vertices'][:,0], tri['vertices'][:,1], color = 'tab:blue', s = 2)
        for edge in tri['edges']:
            a, b = edge
            coords = [
                [tri['vertices'][a][0], tri['vertices'][b][0], tri['vertices'][a][0]],
                [tri['vertices'][a][1], tri['vertices'][b][1], tri['vertices'][a][1]]
            ]

            ax.plot(coords[0], coords[1], color = 'tab:orange', linewidth = 1)
        plt.show()

    return tri

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

    Mesh.triangulate(opts = 'pq30a1000000Dze')

    Mesh.generate_graph()

    print(Mesh.delaunay.keys())
    print(Mesh.voronoi.keys())

    # Mesh.plot(
    #     Mesh.delaunay['vertices'],
    #     Mesh.delaunay['edges'],
    #     subplots_kwargs = {'figsize': (16, 6)}
    # )

if __name__ == '__main__':
    main()