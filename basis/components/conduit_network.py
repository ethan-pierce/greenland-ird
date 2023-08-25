"""Component to model an evolving network of subglacial conduits."""

import numpy as np
import pickle
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import jaxopt
import optax
from functools import partial

from landlab import ModelGrid


class StaticGraph(eqx.Module):
    """Static graph associated with a Landlab grid."""

    number_of_nodes: int
    number_of_links: int
    node_x: jax.Array
    node_y: jax.Array
    length_of_link: jax.Array
    node_at_link_head: jax.Array
    node_at_link_tail: jax.Array
    links_at_node: jax.Array
    link_dirs_at_node: jax.Array

    @classmethod
    def from_grid(cls, grid: ModelGrid):
        """Instantiate a StaticGraph from an existing grid object."""
        return cls(
            grid.number_of_nodes,
            grid.number_of_links,
            grid.node_x,
            grid.node_y,
            grid.length_of_link,
            grid.node_at_link_head,
            grid.node_at_link_tail,
            grid.links_at_node,
            grid.link_dirs_at_node,
        )

    def calc_grad_at_link(self, array):
        return jnp.divide(
            array[self.node_at_link_head] - array[self.node_at_link_tail],
            self.length_of_link,
        )

    def map_mean_of_links_to_node(self, array):
        return jnp.mean(array[self.links_at_node], axis=1)

    def map_mean_of_link_nodes_to_link(self, array):
        return 0.5 * (array[self.node_at_link_head] - array[self.node_at_link_tail])


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

@jax.jit
class Conduits(eqx.Module):
    """Evolves conduit size and water pressure."""

    mesh: StaticGraph

    def __call__(self, t, y, glacier: Glacier):
        water_pressure, conduit_area = y

        effective_pressure = glacier.overburden_pressure - water_pressure
        conduit_pressure = 0.5 * (
            effective_pressure[self.mesh.node_at_link_head]
            + effective_pressure[self.mesh.node_at_link_tail]
        )
        pressure_gradient = self.mesh.calc_grad_at_link(effective_pressure)
        hydraulic_gradient = glacier.base_gradient + pressure_gradient
        discharge = self._calc_discharge(glacier, hydraulic_gradient, conduit_area)

        melt_opening = glacier.melt_constant * discharge * hydraulic_gradient

        gap_opening = glacier.ice_sliding_velocity * glacier.step_height

        creep_closure = (
            glacier.closure_constant
            * jnp.power(conduit_pressure, glacier.glens_n)
            * conduit_area
        )

        bounds = (jnp.zeros(self.mesh.number_of_nodes), glacier.overburden_pressure)
        d_pressure = self._evolve_pressure(glacier, discharge, water_pressure, bounds)
        d_conduits = melt_opening + gap_opening - creep_closure

        return [d_pressure, d_conduits]

    def _calc_discharge(
        self, glacier: Glacier, hydraulic_gradient: jax.Array, conduit_area: jax.Array
    ) -> jax.Array:
        """Calculate discharge in active conduits."""
        sign = jnp.where(hydraulic_gradient >= 0, 1, -1)

        nonzero_potential = jnp.where(
            jnp.abs(hydraulic_gradient) < glacier.nonzero,
            sign * glacier.nonzero,
            hydraulic_gradient,
        )

        return (
            glacier.flow_constant
            * jnp.power(conduit_area, glacier.flow_exp)
            * jnp.power(jnp.abs(nonzero_potential), -1 / 2)
            * hydraulic_gradient
        )

    def _evolve_pressure(
        self, glacier: Glacier, water_pressure: jax.Array, bounds: jax.Array
    ):
        """Find water pressure vector such that mass is (mostly) conserved."""
        pass