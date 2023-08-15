"""JAX model of an evolving network of subglacial conduits."""

import jax
import jax.numpy as jnp
import numpy as np
from landlab import ModelGrid
from landlab import Component
import equinox as eqx
import diffrax
from jaxtyping import ArrayLike
from functools import partial

# Use double precision
from jax import config

config.update("jax_enable_x64", True)


class ConduitNetwork(eqx.Module):
    """Evolves conduit size, effective pressure, and hydraulic gradients."""

    grid: ModelGrid = eqx.field(static=True)
    bedrock_elevation: ArrayLike
    ice_thickness: ArrayLike
    ice_sliding_velocity: ArrayLike
    meltwater_input: ArrayLike
    conduit_area: ArrayLike
    water_pressure: ArrayLike
    effective_pressure: ArrayLike
    hydraulic_gradient: ArrayLike
    water_flux: ArrayLike
    melt_constant: float
    closure_constant: float
    flow_constant: float
    gravity: float = 9.81
    ice_density: float = 917.0
    water_density: float = 1000.0
    latent_heat: float = 3.35e5
    step_height: float = 0.1
    ice_fluidity: float = 6e-24
    glens_n: int = 3
    darcy_friction: float = 3.75e-2
    flow_exp: float = 5 / 4
    nonzero: float = 1e-12
    rtol: float = 1e-6
    atol: float = 1e-6

    def __init__(self, grid: ModelGrid, **kwargs):
        """Initialize the model with a grid and a dict of non-default params."""
        self.grid = grid

        # Ensure that all required input fields exist on the grid
        for field in ["bedrock_elevation", "ice_thickness", "meltwater_input"]:
            if field not in self.grid.at_node.keys():
                raise AttributeError("Missing " + str(field) + " at grid nodes.")

        for field in ["ice_sliding_velocity"]:
            if field not in self.grid.at_link.keys():
                raise AttributeError("Missing " + str(field) + " at grid links.")

        # Add fields defined over grid nodes
        self.bedrock_elevation = jnp.asarray(self.grid.at_node["bedrock_elevation"][:])
        self.ice_thickness = jnp.asarray(self.grid.at_node["ice_thickness"][:])
        self.meltwater_input = jnp.asarray(self.grid.at_node["meltwater_input"][:])

        if 'water_pressure' in self.grid.at_node.keys():
            self.water_pressure = jnp.asarray(self.grid.at_node['water_pressure'][:])
        else:
            self.water_pressure = jnp.zeros(self.grid.number_of_nodes)

        # Add fields defined over grid links
        self.ice_sliding_velocity = jnp.asarray(
            self.grid.at_link["ice_sliding_velocity"][:]
        )

        if 'conduit_area' in self.grid.at_link.keys():
            self.conduit_area = jnp.asarray(self.grid.at_link['conduit_area'][:])
        else:
            self.conduit_area = jnp.zeros(self.grid.number_of_links)

        if 'effective_pressure' in self.grid.at_link.keys():
            self.effective_pressure = jnp.asarray(self.grid.at_link['effective_pressure'][:])
        else:
            self.effective_pressure = jnp.zeros(self.grid.number_of_links)

        if 'hydraulic_gradient' in self.grid.at_link.keys():
            self.hydraulic_gradient = jnp.asarray(self.grid.at_link['hydraulic_gradient'][:])
        else:
            self.hydraulic_gradient = jnp.zeros(self.grid.number_of_links)

        if 'water_flux' in self.grid.at_link.keys():
            self.water_flux = jnp.asarray(self.grid.at_link['water_flux'][:])
        else:
            self.water_flux = jnp.zeros(self.grid.number_of_links)

        # Override any parameters set by the user
        for key, val in kwargs.items():
            setattr(self, str(key), val)

        # Fill in second-order parameters
        self.melt_constant = 1 / (self.ice_density * self.latent_heat)
        self.closure_constant = 2 * self.ice_fluidity * self.glens_n ** (-self.glens_n)
        self.flow_constant = (
            2 ** (1 / 4)
            * np.sqrt(np.pi + 2)
            / (np.pi ** (1 / 4) * np.sqrt(self.water_density * self.darcy_friction))
        )

        # Ensure that n = 3
        if self.glens_n != 3:
            raise NotImplementedError(
                "This component does not (yet) support values for Glen's n other than 3."
                + "\nPlease feel free to contact the author for more information."
            )

    def run_one_step(self, dt: float):
        """Advance the model by one step of size dt."""
        return self

    @partial(jax.jit, static_argnums = 2)
    def map_to_links(self, field: jax.Array, grid: ModelGrid) -> jax.Array:
        if len(field) != grid.number_of_nodes:
            raise ValueError(
                "Input field must be defined on nodes in order to map to links."
            )

        return (
            0.5 * (field[grid.node_at_link_head] + field[grid.node_at_link_tail])
        )

    def map_to_nodes(self, field: jax.Array, grid: ModelGrid) -> jax.Array:
        if len(field) != grid.number_of_links:
            raise ValueError(
                "Input field must be defined on links in order to map to nodes."
            )

        return grid.map_mean_of_links_to_node(field)

    def sum_at_nodes(self, field: jax.Array, grid: ModelGrid) -> jax.Array:
        if len(field) != grid.number_of_links:
            raise ValueError(
                "Input field must be defined on links in order to map its sum to nodes."
            )

        return (
            grid.map_sum_of_inlinks_to_node(field)
            - grid.map_sum_of_outlinks_to_node(field)
        )

    @jax.jit
    def _solve_for_conduit_area(self, t_end: float) -> diffrax.Solution:
        """Update conduit area with an implicit solution."""
        terms = diffrax.ODETerm(self.conduit_evolution_eq)
        y0 = self.conduit_area
        solver = diffrax.Kvaerno3()
        step_ctrl = diffrax.PIDController(self.rtol, self.atol)
        solution = diffrax.diffeqsolve(
            terms,
            solver,
            0.0,
            t_end,
            None,
            y0,
            stepsize_controller = step_ctrl
        )

        return solution

    def conduit_evolution_eq(self, t: float, conduit_size: jax.Array, args) -> jax.Array:
        """Return the right-hand side of the ODE for conduit size evolution."""
        melt_opening = self._calc_melt_opening()
        gap_opening = self._calc_gap_opening()
        creep_closure = self._calc_creep_closure(conduit_size)

        return melt_opening + gap_opening - creep_closure
    
    def _calc_melt_opening(self) -> jax.Array:
        """Calculate the rate of conduit growth from melting side-walls."""
        return (
            self.melt_constant
            * self.water_flux
            * self.hydraulic_gradient
        )
        
    def _calc_gap_opening(self) -> jax.Array:
        """Calculate the rate of conduit growth from sliding over bedrock steps."""
        return (
            self.ice_sliding_velocity
            * self.step_height
        )
    
    def _calc_creep_closure(self, conduit_size: jax.Array) -> jax.Array:
        """Calculate the rate of conduit closure from viscous creep."""
        return (
            self.closure_constant
            * jnp.power(self.effective_pressure, self.glens_n)
            * conduit_size
        )

