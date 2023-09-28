"""Component to model an evolving network of subglacial conduits."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import equinox as eqx
import lineax as lx
import shapely
import jaxopt

from landlab import ModelGrid


class StaticGraph(eqx.Module):
    """Static graph associated with a Landlab grid."""

    number_of_nodes: int
    number_of_links: int
    number_of_cells: int
    node_x: jax.Array
    node_y: jax.Array
    length_of_link: jax.Array
    # length_of_face: jax.Array
    # face_at_link: jax.Array
    node_at_link_head: jax.Array
    node_at_link_tail: jax.Array
    links_at_node: jax.Array
    link_dirs_at_node: jax.Array
    cell_at_node: jax.Array
    corners_at_cell: jax.Array
    x_of_corner: jax.Array
    y_of_corner: jax.Array
    node_is_boundary: jax.Array
    status_at_node: jax.Array
    status_at_link: jax.Array
    adjacent_nodes_at_node: jax.Array
    active_adjacent_nodes_at_node: jax.Array
    links_at_adjacent_nodes: jax.Array = eqx.field(init=False)
    area_of_cell: jax.Array = eqx.field(init=False)
    area_at_node: jax.Array = eqx.field(init=False)

    def __post_init__(self):
        self.area_of_cell = self.calc_area_of_cell()
        self.area_at_node = jnp.where(
            self.node_is_boundary, 0.0, self.area_of_cell[self.cell_at_node]
        )
        self.links_at_adjacent_nodes = self.calc_adjacent_links()

    @classmethod
    def from_grid(cls, grid: ModelGrid):
        """Instantiate a StaticGraph from an existing grid object."""
        return cls(
            grid.number_of_nodes,
            grid.number_of_links,
            grid.number_of_cells,
            grid.node_x,
            grid.node_y,
            grid.length_of_link,
            # grid.length_of_face,
            # grid.face_at_link,
            grid.node_at_link_head,
            grid.node_at_link_tail,
            grid.links_at_node,
            grid.link_dirs_at_node,
            grid.cell_at_node,
            grid.corners_at_cell,
            grid.x_of_corner,
            grid.y_of_corner,
            grid.node_is_boundary(grid.nodes[: grid.number_of_nodes]),
            grid.status_at_node,
            grid.status_at_link,
            grid.adjacent_nodes_at_node,
            grid.active_adjacent_nodes_at_node,
        )

    def calc_grad_at_link(self, array):
        return jnp.divide(
            array[self.node_at_link_head] - array[self.node_at_link_tail],
            self.length_of_link,
        )

    def calc_div_at_node(self, array):
        return jnp.divide(self.sum_at_nodes(array), self.area_at_node)

    def map_mean_of_links_to_node(self, array):
        return jnp.mean(array[self.links_at_node], axis=1)

    def map_mean_of_link_nodes_to_link(self, array):
        return 0.5 * (array[self.node_at_link_head] + array[self.node_at_link_tail])

    def sum_at_nodes(self, array):
        return jnp.sum(self.link_dirs_at_node * array[self.links_at_node], axis=1)

    def calc_area_of_cell(self):
        area_of_cell = np.empty(self.number_of_cells)

        for cell in range(self.number_of_cells):
            coords = [
                (self.x_of_corner[c], self.y_of_corner[c])
                for c in self.corners_at_cell[cell]
                if c != -1
            ]
            area_of_cell[cell] = shapely.Polygon(coords).convex_hull.area

        return jnp.asarray(area_of_cell)

    def calc_adjacent_links(self):
        common_links = np.zeros_like(self.adjacent_nodes_at_node)

        for node in range(self.number_of_nodes):
            if not self.node_is_boundary[node]:
                for adj in range(len(self.adjacent_nodes_at_node[node])):
                    neighbor = self.adjacent_nodes_at_node[node, adj]
                    if neighbor != -1:
                        link = np.intersect1d(
                            self.links_at_node[node], self.links_at_node[neighbor]
                        )
                        link = int(link[link != -1])
                        common_links[node, adj] = link

        return jnp.asarray(common_links)


class Glacier(eqx.Module):
    """Stores glacier properties."""

    mesh: StaticGraph
    ice_thickness: jax.Array = eqx.field(converter=jnp.asarray)
    bedrock_elevation: jax.Array = eqx.field(converter=jnp.asarray)
    surface_elevation: jax.Array = eqx.field(converter=jnp.asarray)
    surface_slope: jax.Array = eqx.field(converter=jnp.asarray)
    meltwater_input: jax.Array = eqx.field(converter=jnp.asarray)
    geothermal_heat_flux: jax.Array = eqx.field(converter=jnp.asarray)
    ice_sliding_velocity: jax.Array = eqx.field(converter=jnp.asarray)

    overburden_pressure: jax.Array = eqx.field(converter=jnp.asarray, init=False)
    boundary_types: jax.Array = eqx.field(converter=jnp.asarray, init=False)

    gravity: float = 9.81
    ice_density: float = 917
    water_density: float = 1000
    water_viscosity: float = 1.787e-6
    flow_regime_scalar: float = 1e-3
    latent_heat: float = 3.34e5
    ice_fluidity: float = 6e-24
    glens_n: int = 3
    threshold_velocity: float = 50
    till_friction_angle: float = 32
    till_friction_coeff: float = 0.3

    def __post_init__(self):
        self.overburden_pressure = self.ice_density * self.gravity * self.ice_thickness
        self.boundary_types = self.label_boundaries()

    def label_boundaries(self):
        boundary_ids = np.full(self.mesh.number_of_nodes, -1)

        for node in np.arange(self.mesh.number_of_nodes):
            if self.mesh.node_is_boundary[node]:
                if self.bedrock_elevation[node] < 0:
                    boundary_ids[node] = 1
                else:
                    boundary_ids[node] = 0

        return jnp.asarray(boundary_ids)

@jax.jit
class Conduits(eqx.Module):
    """Model subglacial conduit size, hydraulic head, and discharge."""

    mesh: StaticGraph
    glacier: Glacier
    conduit_size: jax.Array  # defined on nodes
    hydraulic_head: jax.Array  # defined on nodes
    reynolds: jax.Array  # defined on links

    def run_one_step(self, dt: float, n_iter: int, tolerance: float, verbose = False):
        conduit_size = self.conduit_size
        hydraulic_head = self.hydraulic_head
        reynolds = self.reynolds

        converged = False
        for i in range(n_iter):
            previous_head = jnp.copy(hydraulic_head)

            hydraulic_head, reynolds = self.solve_for_hydraulic_head(
                conduit_size, hydraulic_head, reynolds
            )

            converged, residual = self._check_converged(
                previous_head, 
                hydraulic_head, 
                tolerance, 
            )

            if not converged:
                if verbose:
                    print('Residual norm = ', residual)
            else:
                if verbose:
                    print('Converged after ', i, ' iterations.')
                    break
        
        _, discharge, transmissivity = self.calc_flow_properties(conduit_size, hydraulic_head, reynolds)
        melt_rate = self.calc_melt_rate(hydraulic_head)
        water_pressure = self.calc_water_pressure(hydraulic_head)
        melt_term = melt_rate * (
            1 / self.glacier.water_density - 1 / self.glacier.ice_density
        )
        closure_term = (
            self.glacier.ice_fluidity
            * jnp.power(self.glacier.overburden_pressure - water_pressure, 2)
            * (self.glacier.overburden_pressure - water_pressure)
            * self.conduit_size
        )
        updated_conduit_size = self.evolve_conduits(
            dt, conduit_size, melt_term / self.glacier.ice_density, closure_term
        )

        return Conduits(
            self.mesh,
            self.glacier,
            updated_conduit_size,
            hydraulic_head,
            reynolds,
        )

    def evolve_conduits(
        self,
        dt: float,
        conduit_size: jax.Array,
        melt_forcing: jax.Array,
        creep_closure: jax.Array,
    ) -> jax.Array:
        """Advance conduit sizes by one step of size dt."""
        solver = ConduitSizeODE(
            self.mesh, self.glacier, conduit_size, melt_forcing, creep_closure
        )
        return solver.update(dt)

    def solve_for_hydraulic_head(
        self, 
        conduit_size: jax.Array, 
        hydraulic_head: jax.Array, 
        previous_reynolds: jax.Array, 
    ) -> jax.Array:
        """Solve the elliptic PDE for hydraulic head."""
        reynolds, discharge, transmissivity = self.calc_flow_properties(
            conduit_size, hydraulic_head, previous_reynolds
        )

        melt_rate = self.calc_melt_rate(hydraulic_head)
        water_pressure = self.calc_water_pressure(hydraulic_head)

        melt_term = melt_rate * (
            1 / self.glacier.water_density - 1 / self.glacier.ice_density
        )
        closure_term = (
            self.glacier.ice_fluidity
            * jnp.power(self.glacier.overburden_pressure - water_pressure, 2)
            * (self.glacier.overburden_pressure - water_pressure)
            * self.conduit_size
        )
        input_term = self.glacier.meltwater_input
        forcing = melt_term + closure_term + input_term
        forcing = jnp.where(
            self.mesh.node_is_boundary,
            self.glacier.bedrock_elevation,
            forcing
        )

        solver = HeadPDE(
            self.mesh, self.glacier, self.hydraulic_head, forcing, transmissivity
        )

        head = solver.update()
        head = jnp.where(
            head < self.glacier.bedrock_elevation,
            self.glacier.bedrock_elevation,
            head
        )
        head = jnp.where(
            head > self.glacier.surface_elevation,
            self.glacier.surface_elevation,
            head
        )

        return head, reynolds

    def calc_flow_properties(
        self, conduit_size: jax.Array, hydraulic_head: jax.Array, reynolds: jax.Array
    ) -> tuple:
        """Calculate the Reynolds number, discharge, and transmissivity at links."""
        solver = ReynoldsIteration(
            self.mesh,
            self.glacier,
            conduit_size,
            hydraulic_head,
            reynolds,
        )
        updated_reynolds = solver.update()
        discharge = solver.calc_discharge(updated_reynolds)
        transmissivity = solver.calc_transmissivity(updated_reynolds)

        return updated_reynolds, discharge, transmissivity

    def calc_melt_rate(self, hydraulic_head: jax.Array) -> jax.Array:
        """Calculate the local melt rate at nodes."""
        grad_at_nodes = self.mesh.map_mean_of_links_to_node(
            self.mesh.calc_grad_at_link(hydraulic_head)
        )
        geotherm = self.glacier.geothermal_heat_flux
        friction = jnp.abs(
            self.mesh.map_mean_of_links_to_node(self.glacier.ice_sliding_velocity) 
            * self.calc_shear_stress()
        )

        return (1 / self.glacier.latent_heat) * (geotherm + friction)

    def calc_shear_stress(self) -> jax.Array:
        """Calculate local shear stress at nodes."""
        N = (
            self.glacier.overburden_pressure 
            + self.glacier.water_density * self.glacier.gravity * self.glacier.bedrock_elevation
        )

        yield_term = jnp.tan(self.glacier.till_friction_angle) * (
            self.glacier.ice_sliding_velocity / 
            (self.glacier.ice_sliding_velocity + self.glacier.threshold_velocity)
        )**(1/5)

        return N * self.mesh.map_mean_of_links_to_node(yield_term)

    def calc_water_pressure(self, hydraulic_head: jax.Array) -> jax.Array:
        """Calculate effective pressure at nodes."""
        water_pressure = (
            self.glacier.water_density
            * self.glacier.gravity
            * (hydraulic_head - self.glacier.bedrock_elevation)
        )

        return water_pressure

    def _check_converged(self, array1: jax.Array, array2: jax.Array, tolerance: float):
        """Check if the L2-norm of the residual is less than a specified tolerance."""
        norm = jnp.linalg.norm(array1 - array2)
        if norm < tolerance:
            return True, norm
        else:
            return False, norm

class ConduitSizeODE(eqx.Module):
    """Evolves the ODE for conduit size."""

    mesh: StaticGraph
    glacier: Glacier
    conduit_size: jax.Array
    melt_forcing: jax.Array
    creep_closure: jax.Array

    def update(self, dt: float) -> jax.Array:
        """Advance the model one step of size dt."""
        k1 = self.melt_forcing - self.creep_closure * self.conduit_size
        k2 = self.melt_forcing - self.creep_closure * (self.conduit_size + k1 * dt / 2)
        k3 = self.melt_forcing - self.creep_closure * (self.conduit_size + k2 * dt / 2)
        k4 = self.melt_forcing - self.creep_closure * (self.conduit_size + k3 * dt)

        return self.conduit_size + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class HeadPDE(eqx.Module):
    """Evolves the elliptic PDE for hydraulic head."""

    mesh: StaticGraph
    glacier: Glacier
    hydraulic_head: jax.Array
    forcing: jax.Array
    transmissivity: jax.Array

    def update(self):
        """Solve for hydraulic head, given fixed transmissivity and source terms."""
        operator = lx.FunctionLinearOperator(
            self._matrix_product, jax.eval_shape(lambda: self.forcing)
        )
        solver = lx.AutoLinearSolver(well_posed=None)
        solution = lx.linear_solve(operator, self.forcing, solver)

        return solution.value

    def _matrix_product(self, vector: jax.Array) -> jax.Array:
        """Return the matrix-vector product of the input vector and the finite volume matrix."""
        flux_at_links = (
            -self.transmissivity
            * (
                vector[self.mesh.node_at_link_head]
                - vector[self.mesh.node_at_link_tail]
            )
            / self.mesh.length_of_link
            * jnp.mean(self.mesh.length_of_link)  # TODO replace this with the line below
            # * self.mesh.length_of_face[self.mesh.face_at_link]
        )

        flux_at_active_links = jnp.where(
            self.mesh.links_at_node != -1, flux_at_links[self.mesh.links_at_node], 0.0
        )

        flux_at_nodes = jnp.sum(flux_at_active_links, axis=1)

        product = jnp.where(
            self.mesh.node_is_boundary,
            vector,
            flux_at_nodes / self.mesh.area_of_cell[self.mesh.cell_at_node],
        )

        return product


class ReynoldsIteration(eqx.Module):
    """Solves a fixed-point iteration for discharge and Reynolds number."""

    mesh: StaticGraph
    glacier: Glacier
    conduit_size: jax.Array
    hydraulic_head: jax.Array
    reynolds: jax.Array

    conduit_size_links: jax.Array = eqx.field(init=False)
    head_gradient: jax.Array = eqx.field(init=False)

    def __post_init__(self):
        """Pre-calculate the head gradient and conduit size at links."""
        self.conduit_size_links = self.mesh.map_mean_of_link_nodes_to_link(
            self.conduit_size
        )
        self.head_gradient = self.mesh.calc_grad_at_link(self.hydraulic_head)

    def update(self) -> jax.Array:
        """Identify the fixed point of the local Reynolds equation."""
        solver = jaxopt.AndersonAcceleration(
            lambda Re: jnp.abs(self.calc_discharge(Re)) / self.glacier.water_viscosity
        )
        return solver.run(self.reynolds).params

    def calc_transmissivity(self, reynolds: jax.Array):
        """Calculate the transmissivity of the current drainage geometry."""
        numerator = jnp.power(self.conduit_size_links, 3) * self.glacier.gravity
        denominator = (
            12
            * self.glacier.water_viscosity
            * (1 + self.glacier.flow_regime_scalar * reynolds)
        )
        return jnp.divide(numerator, denominator)

    def calc_discharge(self, reynolds: jax.Array) -> jax.Array:
        """Calculate discharge for a given Reynolds number."""
        transmissivity = self.calc_transmissivity(reynolds)
        return transmissivity * self.head_gradient
