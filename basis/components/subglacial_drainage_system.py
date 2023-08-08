"""Component to evolve a subglacial drainage system."""

import numpy as np
from landlab import Component


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
        nonzero=1e-12
    ):
        super().__init__(grid)

        if glens_n != 3:
            raise NotImplementedError(
                "This component does not (yet) support values for Glen's n other than 3." +
                "\nPlease feel free to contact the author for more information."
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
            "nonzero": nonzero
        }

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

    def _pressure_at_nodes(self, pressure_at_links) -> np.ndarray:
        return self.grid.map_mean_of_links_to_node(pressure_at_links)

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

    def _calc_hydraulic_gradient(self, effective_pressure) -> np.ndarray:
        """Calculate the hydraulic gradient."""
        psi0 = self._calc_base_hydraulic_gradient()
        pressure_at_nodes = self._pressure_at_nodes(effective_pressure)
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

        nonzero_psi = np.where(psi == 0, self.params['nonzero'], psi)

        nonzero_discharge = np.where(
            discharge == 0, self.params['nonzero'], discharge
        )

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
                self.params['ice_density']
                * self.params['gravity']
                * self.grid.map_mean_of_link_nodes_to_link('ice__thickness')
            ),
            3
        )

        pressure = np.where(
            discharge != 0,
            wet_pressure,
            dry_pressure
        )

        return pressure

    def _calc_discharge(self, conduit_area, hydraulic_gradient) -> np.ndarray:
        """Compute discharge as a function of conduit area and hydraulic gradient."""
        psi = hydraulic_gradient[:]
        nonzero_psi = np.where(psi == 0, self.params['nonzero'], psi)

        return (
            self.params["flow_constant"]
            * np.power(conduit_area, self.params["flow_exp"])
            * np.power(np.abs(nonzero_psi), -1 / 2)
            * hydraulic_gradient
        )

    def _RHS(
        self, discharge, hydraulic_gradient, conduit_area, effective_pressure
    ) -> np.ndarray:
        """Solve the right-hand side of the system."""
        melt_opening = self._calc_melt_opening(discharge, hydraulic_gradient)
        gap_opening = self._calc_gap_opening()
        closure = self._calc_closure(effective_pressure, conduit_area)
        return melt_opening + gap_opening - closure

    def _iterate(self, step: float, initial_values: dict, current_values: dict) -> dict:
        """Run one fixed-point iteration."""
        RHS = self._RHS(
            current_values["water__discharge"],
            current_values["hydraulic__gradient"],
            current_values["conduit__area"],
            current_values["effective_pressure"],
        )

        new_conduit_area = initial_values["conduit__area"] + step * RHS

        new_discharge = self._calc_discharge(
            new_conduit_area, current_values["hydraulic__gradient"]
        )

        new_pressure = np.cbrt(
            self._calc_pressure_to_the_n(
                new_discharge, current_values["hydraulic__gradient"]
            )
        )

        new_gradient = self._calc_hydraulic_gradient(new_pressure)

        return {
            "water__discharge": new_discharge,
            "hydraulic__gradient": new_gradient,
            "conduit__area": new_conduit_area,
            "effective_pressure": new_pressure,
        }

    def _max_diff(self, array1, array2) -> float:
        """Returns the absolute value of the maximum difference between two arrays."""
        return np.abs(np.max(array1 - array2))

    def _convergence_metric(self, old_values, new_values, check = ['conduit__area']) -> dict:
        """Calculate the convergence metric for each of the four iteration variables."""
        return {
            key: self._max_diff(old_values[key], new_values[key])
            for key in old_values.keys()
            if key in check
        }

    def initialize(self, force = False):
        """Initialize the output variables with default values."""
        clobber = force

        if 'water__discharge' not in self.grid.at_link.keys() or force:
            discharge = self._partition_meltwater()
            self.grid.add_field('water__discharge', discharge, at = 'link', clobber = clobber)

        if 'effective_pressure' not in self.grid.at_link.keys() or force:
            pressure = (
                self.params['ice_density']
                * self.params['gravity']
                * self.grid.map_mean_of_link_nodes_to_link('ice__thickness')
            )
            self.grid.add_field('effective_pressure', pressure, at = 'link', clobber = clobber)

        if 'hydraulic__gradient' not in self.grid.at_link.keys() or force:
            gradient = self._calc_hydraulic_gradient(self.grid.at_link['effective_pressure'])
            self.grid.add_field('hydraulic__gradient', gradient, at = 'link', clobber = clobber)

        if 'conduit__area' not in self.grid.at_link.keys() or force:
            self.grid.add_zeros('conduit__area', at = 'link', clobber = clobber)
            self.grid.at_link['conduit__area'][:] = 0.1

    def run_one_step(
        self, 
        step: float, 
        tolerance: float, 
        print_interval: int = None,
        max_iter: int = None,
        check = ['conduit__area'],
        save_metrics = False
    ):
        """Advance the model one step."""
        for required in [
            'water__discharge', 
            'hydraulic__gradient',
            'conduit__area',
            'effective_pressure'
        ]:
            if required not in self.grid.at_link.keys():
                self.initialize()

        initial_values = {
            "water__discharge": self.grid.at_link["water__discharge"],
            "hydraulic__gradient": self.grid.at_link["hydraulic__gradient"],
            "conduit__area": self.grid.at_link["conduit__area"],
            "effective_pressure": self.grid.at_link["effective_pressure"],
        }

        current_iteration = self._iterate(step, initial_values, initial_values)
        next_iteration = dict(current_iteration)
        tol = {key: tolerance + 1 for key in initial_values}
        count = 1
        tols_history = []

        while tol[max(tol)] > tolerance:
            count += 1
            next_iteration = self._iterate(step, initial_values, current_iteration)
            tol = self._convergence_metric(current_iteration, next_iteration, check = check)

            current_iteration = dict(next_iteration)

            if print_interval is not None:
                if count % print_interval == 0:
                    print(
                        "Iteration #" + str(count),
                        "\nMaximum difference: " + str(max(tol)),
                        "\nAll diffs: " + str(tol),
                    )

            if save_metrics:
                tols_history.append(tol)

            if max_iter:
                if count >= max_iter:
                    break
        
        if save_metrics:
            return initial_values, next_iteration, tols_history
        else:
            return initial_values, next_iteration