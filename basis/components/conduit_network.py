"""Component to model an evolving network of subglacial conduits."""

import numpy as np
import pickle
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import equinox as eqx
import shapely
import jaxopt
from functools import partial

from landlab import ModelGrid


class StaticGraph(eqx.Module):
    """Static graph associated with a Landlab grid."""

    number_of_nodes: int
    number_of_links: int
    number_of_cells: int
    node_x: jax.Array
    node_y: jax.Array
    length_of_link: jax.Array
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
    active_adjacent_nodes_at_node: jax.Array
    area_of_cell: jax.Array = eqx.field(init=False)
    area_at_node: jax.Array = eqx.field(init=False)

    def __post_init__(self):
        self.area_of_cell = self.calc_area_of_cell()
        self.area_at_node = jnp.where(
            self.node_is_boundary, 0.0, self.area_of_cell[self.cell_at_node]
        )

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
    """Evolves the relative pressure, fluxes, and geometry within the drainage system."""

    mesh: StaticGraph
    glacier: Glacier
    conduit_size: jax.Array
    hydraulic_head: jax.Array
    reynolds: jax.Array

    def run_one_step(self, dt: float):
        """Advance the model one step."""
        updated_conduit_size = self._evolve_conduits(dt)
        updated_hydraulic_head = self._solve_for_head(updated_conduit_size)

        return Conduits(
            self.mesh, self.glacier, updated_conduit_size, updated_hydraulic_head
        )

    def evolve_conduits(self, dt: float) -> jax.Array:
        """Evolve conduit size with an explicit forward solve."""
        return self.conduit_size

    def solve_for_head(self, conduit_size: jax.Array) -> jax.Array:
        """Solve for hydraulic head via fixed-point iteration."""
        return self.hydraulic_head

    def _head_equation(
        self, 
        conduit_size: jax.Array, 
        hydraulic_head: jax.Array, 
        reynolds: jax.Array
    ) -> jax.Array:
        """Return the fixed-point iteration F(s, h) + h = 0 for the given s, h."""
        discharge = self._calc_discharge(conduit_size, hydraulic_head, reynolds)
        transmissivity = self._calc_transmissivity(conduit_size, reynolds)
        melt_rate = self._calc_melt_rate(hydraulic_head, discharge)
        water_pressure, effective_pressure = self._calc_pressure(hydraulic_head)

        flux_term = self.mesh.calc_div_at_node(
            -transmissivity * self.mesh.calc_grad_at_link(hydraulic_head)
        )

        melt_term = -melt_rate * ((1 / self.glacier.water_density) - (1 / self.glacier.ice_density))

        closure_term = -self.glacier.ice_fluidity * jnp.power(effective_pressure, 3) * hydraulic_head

        input_term = self.glacier.meltwater_input

        return flux_term - melt_term - closure_term - input_term + hydraulic_head

    def _calc_transmissivity(
        self, conduit_size: jax.Array, reynolds: jax.Array
    ) -> jax.Array:
        """Calculate the hydraulic transmissivity."""
        return jnp.divide(
            jnp.power(conduit_size, 3) * self.glacier.gravity,
            12
            * self.glacier.water_viscosity
            * (1 + self.glacier.flow_regime_scalar * reynolds),
        )

    def _calc_discharge(
        self, conduit_size: jax.Array, hydraulic_head: jax.Array, reynolds: jax.Array
    ) -> jax.Array:
        """Calculate the local discharge."""
        return -self._calc_transmissivity(
            conduit_size, reynolds
        ) * self.mesh.calc_grad_at_link(hydraulic_head)

    def _calc_reynolds(self, discharge: jax.Array) -> jax.Array:
        """Calculate the local Reynolds number."""
        return jnp.divide(jnp.abs(discharge), self.glacier.water_viscosity)

    def _calc_melt_rate(self, hydraulic_head: jax.Array, discharge: jax.Array) -> jax.Array:
        """Calculate the local melt rate from geothermal, frictional, and mechanical heat fluxes."""
        geothermal = self.mesh.map_mean_of_link_nodes_to_link(self.glacier.geothermal_heat_flux)
        frictional = jnp.abs(
            self.mesh.map_mean_of_links_to_node(self.glacier.ice_sliding_velocity)
            * self.glacier.shear_stress
        )
        dissipation = (
            self.glacier.water_density 
            * self.glacier.gravity 
            * discharge 
            * self.mesh.map_mean_of_links_to_node(self.mesh.calc_grad_at_link(hydraulic_head))
        )

        return jnp.divide(
            geothermal + frictional - dissipation,
            self.glacier.latent_heat
        )

    def _calc_shear_stress(self, hydraulic_head: jax.Array) -> jax.Array:
        """Calculate shear stress using the Zoet and Iverson slip law."""
        effective_pressure = self._calc_pressure(hydraulic_head)[1]
        velocity_at_nodes = jnp.abs(self.mesh.map_mean_of_links_to_node(self.glacier.ice_sliding_velocity))
        return (
            effective_pressure
            * jnp.tan(jnp.deg2rad(self.glacier.till_friction_angle))
            * jnp.power(
                (velocity_at_nodes / (velocity_at_nodes + self.glacier.threshold_velocity)),
                1 / 5
            )
        )

    def _calc_pressure(self, hydraulic_head: jax.Array) -> jax.Array:
        """Calculate the water pressure and effective pressure."""
        water_pressure = (
            (hydraulic_head - self.glacier.bedrock_elevation) 
            * self.glacier.water_density 
            * self.glacier.gravity
        )
        water_pressure = jnp.where(
            water_pressure > glacier.overburden_pressure, 
            glacier.overburden_pressure, 
            water_pressure
        )
        effective_pressure = self.glacier.overburden_pressure - water_pressure
        return water_pressure, effective_pressure
