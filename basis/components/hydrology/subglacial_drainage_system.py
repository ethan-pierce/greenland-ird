"""Component to evolve a subglacial drainage system."""

import numpy as np
from dataclasses import dataclass
from typing import Callable
from landlab import Component
import scipy

@dataclass
class SolutionTensor:
    water__discharge: np.ndarray
    hydraulic__gradient: np.ndarray
    conduit__area: np.ndarray
    effective_pressure: np.ndarray

    def to_tensor(self):
        return np.array(
            [
                self.water__discharge,
                self.hydraulic__gradient,
                self.conduit__area,
                self.effective_pressure,
            ]
        )


@dataclass
class BoundaryCondition:
    field: str
    at: str
    nodes: np.ndarray
    values: np.ndarray

    @classmethod
    def from_function(
        cls, field: str, at: str, nodes: np.ndarray, function: Callable, **kwargs
    ):
        values = function(**kwargs)
        return cls(edge, field, values)


class SubglacialDrainageSystem(Component):
    """Evolves conduit size, effective pressure, and hydraulic gradients."""

    _name = "SubglacialDrainageSystem"

    _unit_agnostic = True

    _info = {
        "bedrock__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Elevation of the bedrock surface",
        },
        "ice__thickness": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Thickness of glacier ice",
        },
        "ice__sliding_velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m / s",
            "mapping": "link",
            "doc": "Ice velocity at the ice-bed interface",
        },
        "water__pressure": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "Pa",
            "mapping": "node",
            "doc": "Water pressure at nodes in the subglacial drainage network",
        },
        "meltwater__input": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m / s",
            "mapping": "node",
            "doc": "Specific melt (length per time) at each node",
        },
        "water__discharge": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m^3 / s",
            "mapping": "link",
            "doc": "Volume of water per unit time moving through each element",
        },
        "hydraulic__gradient": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "Pa / m",
            "mapping": "link",
            "doc": "Gradient in hydraulic pressure across conduits",
        },
        "conduit__area": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m^2",
            "mapping": "link",
            "doc": "Cross-sectional area of drainage conduits",
        },
        "effective_pressure": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "Pa",
            "mapping": "link",
            "doc": "Difference between ice overburden and water pressure",
        },
    }

    def __init__(
        self,
        grid,
        gravity=9.81,
        ice_density=917,
        water_density=1000,
        latent_heat=3.35e5,
        step_height=0.1,
        ice_fluidity=6e-24,
        glens_n=3,
        darcy_friction=3.75e-2,
        flow_exp=5 / 4,
        nonzero=1e-12,
        max_conduit_size=10
    ):
        super().__init__(grid)

        if glens_n != 3:
            raise NotImplementedError(
                "This component does not (yet) support values for Glen's n other than 3."
                + "\nPlease feel free to contact the author for more information."
            )

        self.params = {
            "gravity": gravity,
            "ice_density": ice_density,
            "water_density": water_density,
            "latent_heat": latent_heat,
            "melt_constant": 1 / (ice_density * latent_heat),
            "step_height": step_height,
            "ice_fluidity": ice_fluidity,
            "glens_n": glens_n,
            "darcy_friction": darcy_friction,
            "flow_exp": flow_exp,
            "closure_constant": 2 * ice_fluidity * glens_n ** (-glens_n),
            "flow_constant": (
                2 ** (1 / 4)
                * np.sqrt(np.pi + 2)
                / (np.pi ** (1 / 4) * np.sqrt(water_density * darcy_friction))
            ),
            "nonzero": nonzero,
            "max_conduit_size": max_conduit_size
        }

    def initialize(self, force=False):
        """Initialize the output variables with default values."""
        clobber = force

        if "water__discharge" not in self.grid.at_link.keys() or force:
            discharge = self._partition_meltwater()
            self.grid.add_field(
                "water__discharge", discharge, at="link", clobber=clobber
            )

        if "effective_pressure" not in self.grid.at_link.keys() or force:
            pressure = (
                self.params["ice_density"]
                * self.params["gravity"]
                * self.grid.map_mean_of_link_nodes_to_link("ice__thickness")
            )
            self.grid.add_field(
                "effective_pressure", pressure, at="link", clobber=clobber
            )

        if "hydraulic__gradient" not in self.grid.at_link.keys() or force:
            gradient = self._calc_hydraulic_gradient(
                self.grid.at_link["effective_pressure"]
            )
            self.grid.add_field(
                "hydraulic__gradient", gradient, at="link", clobber=clobber
            )

        if "conduit__area" not in self.grid.at_link.keys() or force:
            self.grid.add_zeros("conduit__area", at="link", clobber=clobber)
            self.grid.at_link["conduit__area"][:] = 0.1

    def _remap(self, field: np.ndarray, to: str) -> np.ndarray:
        """Map a field from links to nodes or nodes to links."""
        if to == 'node':
            return self.grid.map_mean_of_links_to_node(field)
        elif to == 'link':
            return self.grid.map_mean_of_link_nodes_to_link(field)

    def _partition_meltwater(self) -> np.ndarray:
        """Partition meltwater input at nodes to each downslope link."""
        specific_melt = self.grid.at_node["meltwater__input"][:]
        contributing_area = self.grid.cell_area_at_node[:]
        melt_flux = specific_melt * contributing_area

        n_links = np.sum([self.grid.links_at_node != -1], axis=2)
        discharge_per_adjacent_link = melt_flux / n_links

        discharge = np.take(discharge_per_adjacent_link, self.grid.nodes_at_link).sum(
            axis=1
        )

        return discharge

    def _calc_pressure(self, water_pressure) -> np.ndarray:
        """Given water pressure, calculate effective pressure at nodes."""
        overburden_pressure = (
            self.params['ice_density']
            * self.params['gravity']
            * self.grid.at_node['ice__thickness']
        )

        effective_pressure = overburden_pressure - water_pressure

        return effective_pressure


    def _calc_base_hydraulic_gradient(self) -> np.ndarray:
        """Compute the baseline hydraulic gradient from ice and bedrock geometry."""
        ice_pressure = (
            self.params["ice_density"]
            * self.params["gravity"]
            * self.grid.at_node["ice__thickness"][:]
        )

        ice_pressure_gradient = self.grid.calc_grad_at_link(ice_pressure)

        bedrock_gradient = self.grid.calc_grad_at_link("bedrock__elevation")

        potential_gradient = (
            self.params["water_density"] * self.params["gravity"] * bedrock_gradient
        )

        base_hydraulic_gradient = -ice_pressure_gradient - potential_gradient

        return base_hydraulic_gradient

    def _calc_hydraulic_gradient(self, effective_pressure, pressure_at = 'link') -> np.ndarray:
        """Calculate the hydraulic gradient."""
        psi0 = self._calc_base_hydraulic_gradient()

        if pressure_at == 'link':
            pressure_at_nodes = self._remap(effective_pressure, to = 'node')
        elif pressure_at == 'node':
            pressure_at_nodes = effective_pressure

        pressure_gradient = self.grid.calc_grad_at_link(pressure_at_nodes)

        return psi0 + pressure_gradient

    def _calc_melt_opening(self, discharge, hydraulic_gradient) -> np.ndarray:
        """Compute the channel opening rate from dissipation."""
        return self.params["melt_constant"] * discharge * hydraulic_gradient

    def _calc_gap_opening(self) -> np.ndarray:
        """Compute the cavity opening rate from sliding over bedrock steps."""
        return (
            self.grid.at_link["ice__sliding_velocity"][:] * self.params["step_height"]
        )

    def _calc_closure(self, effective_pressure, conduit_area) -> np.ndarray:
        """Compute the closure rate from viscous creep."""
        return (
            self.params["closure_constant"]
            * effective_pressure ** self.params["glens_n"]
            * conduit_area
        )

    def _calc_pressure_to_the_n(self, discharge, hydraulic_gradient) -> np.ndarray:
        """Compute the steady-state effective pressure (raised to Glen's n) at links."""
        numerator = (
            self._calc_melt_opening(discharge, hydraulic_gradient)
            + self._calc_gap_opening()
        )

        psi = hydraulic_gradient[:]

        nonzero_psi = np.where(psi == 0, self.params["nonzero"], psi)

        nonzero_discharge = np.where(discharge == 0, self.params["nonzero"], discharge)

        denominator = (
            self.params["closure_constant"]
            * self.params["flow_constant"] ** (-1 / self.params["flow_exp"])
            * np.sign(nonzero_discharge)
            * np.power(np.abs(nonzero_discharge), 1 / self.params["flow_exp"])
            * np.power(np.abs(nonzero_psi), 1 / (2 * self.params["flow_exp"]))
        )

        wet_pressure = np.abs(numerator / denominator)
        dry_pressure = np.power(
            (
                self.params["ice_density"]
                * self.params["gravity"]
                * self.grid.map_mean_of_link_nodes_to_link("ice__thickness")
            ),
            3,
        )

        pressure = np.where(discharge != 0, wet_pressure, dry_pressure)

        return pressure

    def _calc_discharge(self, conduit_area, hydraulic_gradient) -> np.ndarray:
        """Compute discharge as a function of conduit area and hydraulic gradient."""
        psi = hydraulic_gradient[:]
        nonzero_psi = np.where(psi == 0, self.params["nonzero"], psi)
        nonzero_conduits = np.where(
            conduit_area == 0,
            0.0,
            np.power(conduit_area, self.params["flow_exp"])
        )

        return (
            self.params["flow_constant"]
            * nonzero_conduits
            * np.power(np.abs(nonzero_psi), -1 / 2)
            * hydraulic_gradient
        )

    def _sum_discharge(self, discharge: np.ndarray) -> np.ndarray:
        """Sum the discharge entering and leaving each node."""
        total_discharge = (
            self.grid.map_sum_of_outlinks_to_node(discharge) 
            + self.grid.map_sum_of_inlinks_to_node(discharge)
        )
        return total_discharge

    def _build_solution_tensor(self, to_array=True) -> SolutionTensor:
        """Construct the SolutionTensor as either a dataclass or an array."""
        ST = SolutionTensor(
            water__discharge=np.array(self.grid.at_link["water__discharge"][:]),
            hydraulic__gradient=np.array(self.grid.at_link["hydraulic__gradient"][:]),
            conduit__area=np.array(self.grid.at_link["conduit__area"][:]),
            effective_pressure=np.array(self.grid.at_link["effective_pressure"][:]),
        )

        if to_array:
            return ST.to_tensor()
        else:
            return ST

    def _RHS(self, values: np.ndarray) -> np.ndarray:
        """Solve the right-hand side of the system."""
        v = {
            "water__discharge": 0,
            "hydraulic__gradient": 1,
            "conduit__area": 2,
            "effective_pressure": 3,
        }

        melt_opening = self._calc_melt_opening(
            values[v["water__discharge"]], values[v["hydraulic__gradient"]]
        )
        gap_opening = self._calc_gap_opening()
        closure = self._calc_closure(
            values[v["effective_pressure"]], values[v["conduit__area"]]
        )

        Qt = self._calc_discharge(
            values[v["conduit__area"]], values[v["hydraulic__gradient"]]
        )
        Pt = self._calc_hydraulic_gradient(values[v["effective_pressure"]])
        St = values[v["conduit__area"]] + (melt_opening + gap_opening - closure)
        Nt = np.cbrt(
            self._calc_pressure_to_the_n(
                values[v["water__discharge"]], values[v["hydraulic__gradient"]]
            )
        )

        return np.array([Qt, Pt, St, Nt])


    def _iter_RK4(self, function: Callable, values: np.ndarray, step: float) -> np.ndarray:
        """Perform one iteration, using a fourth-order Runge-Kutta scheme."""
        k1 = step * function(values)
        k2 = step * function(values + 0.5 * k1)
        k3 = step * function(values + 0.5 * k2)
        k4 = step * function(values + k3)

        return values + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _conserve_mass_at_nodes(self, water_pressure: np.ndarray) -> np.ndarray:
        """Given water pressure at nodes, return the sum of discharge at each link adjacent to nodes."""
        fixed_conduit_area = self.grid.at_link['conduit__area'][:]
        pressure = self._calc_pressure(water_pressure)
        potential = self._calc_hydraulic_gradient(pressure, pressure_at='node')
        discharge = self._calc_discharge(fixed_conduit_area, potential)
        net_discharge = self._sum_discharge(discharge)
        water_input = self.grid.at_node['meltwater__input'][:]

        return np.max(np.abs(water_input - net_discharge))

    def _find_root(
        self, 
        function: Callable, 
        initial_guess: np.ndarray
    ):
        """Find the root of a function."""
        bounds = np.array([np.zeros_like(initial_guess), self._remap(self.grid.at_link['effective_pressure'], to='node')]).T

        solved = scipy.optimize.minimize(function, initial_guess, bounds = bounds)

        return solved

    def _solve_conduit_area(self, step: float, values: np.ndarray) -> np.ndarray:
        """Given current state, solve a backward euler step for updated conduit area."""

        # Make sure to map variables correctly
        v = {
            "water__discharge": 0,
            "hydraulic__gradient": 1,
            "conduit__area": 2,
            "effective_pressure": 3
        }

        A = np.zeros((self.grid.number_of_links, self.grid.number_of_links))

        for link in np.arange(self.grid.number_of_links):
            A[link, link] = (
                step 
                * self.params['closure_constant'] 
                * values[v['effective_pressure']][link]**self.params['glens_n']
                + 1
            )

        melt_opening = self._calc_melt_opening(values[v['water__discharge']], values[v['hydraulic__gradient']])
        gap_opening = self._calc_gap_opening()
        B = (
            values[v['conduit__area']]
            + step * melt_opening
            + step * gap_opening
        )

        # Solve by minimizing ||Ax - B||
        fx = lambda x: np.linalg.norm(np.dot(A, x) - B)
        bounds = np.array([np.zeros_like(B), np.full_like(B, self.params['max_conduit_size'])]).T
        solution = scipy.optimize.minimize(fx, values[v['conduit__area']], bounds=bounds)

        return solution
        


    def run_one_step(self, step: float, boundary_conditions: dict = None):
        """Advance the model one step."""
        new_conduit_area = self._solve_conduit_area(
            step,
            self._build_solution_tensor(to_array=True)
        ).x

        self.grid.at_link['conduit__area'][:] = new_conduit_area.copy()

        initial_guess = self.grid.at_node['water__pressure'][:]

        new_water_pressure = self._find_root(
            self._conserve_mass_at_nodes,
            initial_guess
        ).x

        self.grid.at_node['water__pressure'][:] = new_water_pressure

        self.grid.at_link['effective_pressure'][:] = self._remap(
            self._calc_pressure(self.grid.at_node['water__pressure'][:]), to = 'link'
        )

        self.grid.at_link['hydraulic__gradient'][:] = self._calc_hydraulic_gradient(
            self.grid.at_link['effective_pressure'][:]
        )

        self.grid.at_link['water__discharge'][:] = self._calc_discharge(
            self.grid.at_link['conduit__area'][:], self.grid.at_link['hydraulic__gradient'][:]
        )
