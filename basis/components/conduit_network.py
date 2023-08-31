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
            self.node_is_boundary,
            0.0,
            self.area_of_cell[self.cell_at_node]
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
            grid.node_is_boundary(grid.nodes[:grid.number_of_nodes]),
            grid.status_at_node,
            grid.status_at_link,
            grid.active_adjacent_nodes_at_node
        )

    def calc_grad_at_link(self, array):
        return jnp.divide(
            array[self.node_at_link_head] - array[self.node_at_link_tail],
            self.length_of_link,
        )

    def map_mean_of_links_to_node(self, array):
        return jnp.mean(array[self.links_at_node], axis=1)

    def map_mean_of_link_nodes_to_link(self, array):
        return 0.5 * (array[self.node_at_link_head] + array[self.node_at_link_tail])

    def sum_at_nodes(self, array):
        return jnp.sum(
            self.link_dirs_at_node * array[self.links_at_node], axis=1
        )

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
    meltwater_input: jax.Array = eqx.field(converter=jnp.asarray)
    ice_sliding_velocity: jax.Array = eqx.field(converter=jnp.asarray)

    melt_constant: float = eqx.field(init=False)
    closure_constant: float = eqx.field(init=False)
    flow_constant: float = eqx.field(init=False)

    overburden_pressure: jax.Array = eqx.field(converter=jnp.asarray, init=False)
    pressure_slope: jax.Array = eqx.field(converter=jnp.asarray, init=False)
    bedrock_slope: jax.Array = eqx.field(converter=jnp.asarray, init=False)
    base_gradient: jax.Array = eqx.field(converter=jnp.asarray, init=False)
    boundary_ids: jax.Array = eqx.field(converter=jnp.asarray, init=False)

    gravity: float = 9.81
    ice_density: float = 917
    water_density: float = 1000
    latent_heat: float = 3.35e5
    step_height: float = 0.1
    ice_fluidity: float = 6e-24
    glens_n: int = 3
    darcy_friction: float = 3.75e-2
    flow_exp: float = 5 / 4
    nonzero: float = 1e-12

    def __post_init__(self):
        self.melt_constant = 1 / (self.ice_density * self.latent_heat)
        self.closure_constant = 2 * self.ice_fluidity * self.glens_n ** (-self.glens_n)
        self.flow_constant = (
            2 ** (1 / 4)
            * np.sqrt(np.pi + 2)
            / (np.pi ** (1 / 4) * np.sqrt(self.water_density * self.darcy_friction))
        )

        self.overburden_pressure = self.ice_density * self.gravity * self.ice_thickness
        self.pressure_slope = self.mesh.calc_grad_at_link(self.overburden_pressure)
        self.bedrock_slope = self.mesh.calc_grad_at_link(self.bedrock_elevation)
        self.base_gradient = (
            -self.pressure_slope
            - self.water_density * self.gravity * self.bedrock_slope
        )
        self.base_gradient = self.base_gradient.at[self.mesh.status_at_link != 0].set(0.0)
        self.boundary_ids = self.label_boundaries()

    def label_boundaries(self):
        boundary_ids = np.full(self.mesh.number_of_nodes, -1)
        boundary_nodes = jnp.arange(self.mesh.number_of_nodes)[self.mesh.node_is_boundary]
        ordered_gradients = self.mesh.link_dirs_at_node * self.base_gradient[self.mesh.links_at_node]
        neighbors = self.mesh.active_adjacent_nodes_at_node
        neighbor_elevation = self.bedrock_elevation[neighbors]
        lowest_neighbor = jnp.min(neighbor_elevation, axis = 1)

        boundary_ids = (
            (self.bedrock_elevation < lowest_neighbor)
            * jnp.any(ordered_gradients > 0, axis = 1)
            * ((self.ice_thickness + self.bedrock_elevation) < 1000)
        )

        return jnp.asarray(boundary_ids)

@jax.jit
class Conduits(eqx.Module):
    """Evolves conduit size and water pressure."""

    mesh: StaticGraph
    glacier: Glacier

    init_water_pressure: jax.Array
    init_conduit_area: jax.Array

    def run_one_step(self, dt: float):
        """Run one forward pass of the model with step size dt (seconds)."""
        new_pressure, solver_state = self._evolve_pressure(self.init_conduit_area)
        new_conduits = self._evolve_conduits(dt)

        new_pressure = new_pressure.at[new_pressure < 0].set(0.0)
        new_conduits = new_conduits.at[new_conduits < 0].set(0.0)
        new_conduits = new_conduits.at[self.mesh.status_at_link != 0].set(0.0)

        return Conduits(self.mesh, self.glacier, new_pressure, new_conduits)

    def _evolve_pressure(self, conduit_area: jax.Array) -> tuple:
        """Find the water pressure that minimizes mass gain/loss in the system."""
        solver = jaxopt.NonlinearCG(
            fun = self._estimate_overflow, 
            max_stepsize=1e12,
            maxiter=1000,
            verbose = True
        )
        result = solver.run(self.init_water_pressure, conduit_area = conduit_area)
        return (result.params, result.state)

    def _evolve_conduits(self, dt: float) -> jax.Array:
        """Evolve conduit area using a fourth-order Runge-Kutta scheme."""
        k1 = self._calc_conduit_rate(self.init_water_pressure, self.init_conduit_area)
        k2 = self._calc_conduit_rate(
            self.init_water_pressure, self.init_conduit_area + k1 * dt / 2
        )
        k3 = self._calc_conduit_rate(
            self.init_water_pressure, self.init_conduit_area + k2 * dt / 2
        )
        k4 = self._calc_conduit_rate(
            self.init_water_pressure, self.init_conduit_area + k3 * dt
        )

        dSdt = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return self.init_conduit_area + dSdt

    def _calc_conduit_rate(
        self, water_pressure: jax.Array, conduit_area: jax.Array
    ) -> jax.Array:
        """Determine the growth/decay rate of conduit area."""
        state = self._resolve_state(water_pressure, conduit_area)
        effective_pressure, conduit_pressure, hydraulic_gradient, discharge = state

        melt_opening = self.glacier.melt_constant * discharge * hydraulic_gradient

        gap_opening = self.glacier.ice_sliding_velocity * self.glacier.step_height

        creep_closure = (
            self.glacier.closure_constant
            * jnp.power(conduit_pressure, self.glacier.glens_n)
            * conduit_area
        )

        rate = melt_opening + gap_opening - creep_closure
        rate = rate.at[self.mesh.status_at_link != 0].set(0.0)
        return rate

    def _calc_discharge(
        self, hydraulic_gradient: jax.Array, conduit_area: jax.Array
    ) -> jax.Array:
        """Calculate discharge in active conduits."""
        sign = jnp.where(hydraulic_gradient >= 0, 1, -1)

        nonzero_potential = jnp.where(
            jnp.abs(hydraulic_gradient) < self.glacier.nonzero,
            sign * self.glacier.nonzero,
            hydraulic_gradient,
        )

        return (
            self.glacier.flow_constant
            * jnp.power(conduit_area, self.glacier.flow_exp)
            * jnp.power(jnp.abs(nonzero_potential), -1 / 2)
            * hydraulic_gradient
        )

    def _resolve_state(
        self, water_pressure: jax.Array, conduit_area: jax.Array
    ) -> tuple:
        """Resolve the effective pressure, hydraulic gradient, and discharge in conduits."""
        effective_pressure = self.glacier.overburden_pressure - water_pressure

        conduit_pressure = self.mesh.map_mean_of_link_nodes_to_link(effective_pressure)

        pressure_gradient = self.mesh.calc_grad_at_link(effective_pressure)
        hydraulic_gradient = self.glacier.base_gradient + pressure_gradient

        discharge = self._calc_discharge(hydraulic_gradient, conduit_area)

        return (effective_pressure, conduit_pressure, hydraulic_gradient, discharge)

    def _estimate_overflow(
        self, water_pressure: jax.Array, conduit_area: jax.Array
    ) -> float:
        """Given water pressure, estimate the excess/lack of flux in the system."""
        state = self._resolve_state(water_pressure, conduit_area)
        _, _, _, discharge = state
        net_flux = self._sum_discharge(discharge, self.glacier.meltwater_input)

        zeros = jnp.zeros_like(net_flux)
        log_cosh = jnp.log(jnp.cosh(zeros - net_flux))
        overflow_loss = jnp.nansum(log_cosh) / self.mesh.number_of_nodes

        return net_flux

    def _sum_discharge(
        self, discharge: jax.Array, meltwater_input: jax.Array
    ) -> jax.Array:
        """Sum discharge at nodes and return an array of residuals."""
        net_flux = jnp.sum(
            self.mesh.link_dirs_at_node * discharge[self.mesh.links_at_node], axis=1
        )

        return net_flux - meltwater_input
