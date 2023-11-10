"""Newton method with weighting."""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import scipy
import equinox as eqx
import lineax as lx

import sys
sys.path.append('/home/egp/repos/greenland-ird/')
from basis.utils.plotting import plot_triangle_mesh, plot_links
from basis.components.conduit_network import *

with open(
    "/home/egp/repos/greenland-ird/models/hydrology/eqip-sermia.grid", "rb"
) as pf:
    grid = pickle.load(pf)
print("Grid size: ", grid.number_of_nodes, " nodes; ", grid.number_of_links, " links.")
print("Grid has: ", grid.at_node.keys(), " at nodes.")
print("Grid has: ", grid.at_link.keys(), " at links.")
print(grid.at_node.keys())

grid.at_node["ice_thickness"][grid.at_node["ice_thickness"] < 10.0] = 10.0

mesh = StaticGraph.from_grid(grid)

slope_at_nodes = grid.calc_slope_at_node(grid.at_node["surface_elevation"])

glacier = Glacier(
    mesh,
    grid.at_node["ice_thickness"],
    grid.at_node["bedrock_elevation"],
    grid.at_node["surface_elevation"],
    slope_at_nodes,
    grid.at_node["meltwater_input"],
    grid.at_node["geothermal_heat_flux"],
    grid.at_link["ice_sliding_velocity"],
)

class NewtonIteration(eqx.Module):
    """Two-step Newton iteration with weighted update."""

    mesh: StaticGraph
    glacier: Glacier

    def first_newton_update(self, head, Re):
        vector = self.nonlinear_operator(head, Re)
        jacobian = lx.JacobianLinearOperator(self.nonlinear_operator, head, args = Re)
        solver = lx.NormalCG(rtol = 1e-6, atol = 1e-6)
        solution = lx.linear_solve(jacobian, vector, solver)

        return solution

    @jax.jit
    def nonlinear_operator(self, head, Re):
        head = self.enforce_bcs(head)
        grad_head = self.mesh.calc_grad_at_link(head)
        effective_pressure = self.calc_effective_pressure(head)
        melt_flux = self.calc_melt_flux(effective_pressure)
        conduit_size = self.calc_conduit_size(effective_pressure, melt_flux)
        transmissivity = self.calc_transmissivity(conduit_size, Re)

        melt_term = melt_flux * ((1 / self.glacier.water_density) - (1 / self.glacier.ice_density))
        interior_melt_term = jnp.where(self.mesh.node_is_boundary, 0.0, melt_term)

        closure_term = self.calc_closure_term(effective_pressure, conduit_size)
        interior_closure_term = jnp.where(self.mesh.node_is_boundary, 0.0, closure_term)
        
        net_flux = self.mesh.sum_at_nodes(-transmissivity * grad_head * self.mesh.length_of_link) # Should be length of face?
        interior_net_flux = jnp.where(self.mesh.node_is_boundary, 0.0, net_flux)

        interior_node_area = jnp.where(self.mesh.node_is_boundary, 1.0, self.mesh.area_at_node)
        elliptic_term = interior_net_flux / interior_node_area

        return elliptic_term - melt_term - closure_term

    def enforce_bcs(self, head):
        return jnp.where(mesh.node_is_boundary, self.glacier.bedrock_elevation, head)

    def calc_water_pressure(self, head):
        return self.glacier.water_density * self.glacier.gravity * (head - self.glacier.bedrock_elevation)

    def calc_effective_pressure(self, head):
        water_pressure = self.calc_water_pressure(head)
        effective_pressure = self.glacier.overburden_pressure - water_pressure
        effective_pressure = jnp.where(
            effective_pressure > glacier.overburden_pressure, 
            glacier.overburden_pressure, 
            effective_pressure
        )
        effective_pressure = jnp.where(
            effective_pressure < 1e4,
            1e4,
            effective_pressure
        )
        return effective_pressure

    def calc_melt_flux(self, effective_pressure):
        shear_stress = self.glacier.till_friction_coeff * effective_pressure
        friction = jnp.abs(
            self.mesh.map_mean_of_links_to_node(self.glacier.ice_sliding_velocity) * shear_stress
        )
        return (self.glacier.geothermal_heat_flux + friction) / self.glacier.latent_heat

    def calc_closure_term(self, effective_pressure, conduit_size):
        return self.glacier.ice_fluidity * jnp.power(effective_pressure, 3) * conduit_size

    def calc_conduit_size(self, effective_pressure, melt_flux):
        conduit_size = (melt_flux / self.glacier.ice_density) / (self.glacier.ice_fluidity * effective_pressure**3)
        return conduit_size

    def calc_transmissivity(self, conduit_size, Re):
        conduits_at_links = self.mesh.map_mean_of_link_nodes_to_link(conduit_size)
        numerator = jnp.power(conduits_at_links, 3) * self.glacier.gravity
        denominator = 12 * self.glacier.water_viscosity * (1 + self.glacier.flow_regime_scalar * Re)
        return numerator / denominator

    def calc_reynolds(self, transmissivity, grad_head):
        return jnp.abs(-transmissivity * grad_head) / glacier.water_viscosity

model = NewtonIteration(mesh, glacier)

h0 = glacier.bedrock_elevation
Re0 = jnp.full(mesh.number_of_links, 1 / glacier.flow_regime_scalar)

plot_triangle_mesh(grid, h0, at = 'patch', subplots_args={'figsize': (18, 6)}, set_clim = {'vmin': None, 'vmax': None})
# plot_triangle_mesh(grid, model.calc_effective_pressure(head), at = 'patch', subplots_args={'figsize': (18, 6)}, set_clim = {'vmin': None, 'vmax': None})
