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
            "mapping": "node",
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
        flow_exp=5/4
    ):
        super().__init__(grid)

        self.params = {
            'gravity': gravity,
            'ice_density': ice_density,
            'water_density': water_density,
            'latent_heat': latent_heat,
            'melt_constant': 1 / (ice_density * latent_heat),
            'step_height': step_height,
            'ice_fluidity': ice_fluidity,
            'glens_n': glens_n,
            'darcy_friction': darcy_friction,
            'flow_exp': flow_exp,
            'closure_constant': 2 * ice_fluidity * glens_n**(-glens_n),
            'flow_constant': (
                2**(1/4) * np.sqrt(np.pi + 2) / 
                (np.pi**(1/4) * np.sqrt(water_density * darcy_friction))
            )
        }

    def _calc_base_hydraulic_gradient(self) -> np.ndarray:
        """Compute the baseline hydraulic gradient from ice and bedrock geometry."""
        ice_pressure = (
            self.params['ice_density'] 
            * self.params['gravity'] 
            * self.grid.at_node['ice__thickness'][:]
        )

        ice_pressure_gradient = self.grid.calc_grad_at_link(ice_pressure)

        bedrock_gradient = self.grid.calc_grad_at_link('bedrock__elevation')

        potential_gradient = (
            self.params['water_density']
            * self.params['gravity']
            * bedrock_gradient
        )

        base_hydraulic_gradient = -ice_pressure_gradient - potential_gradient

        return base_hydraulic_gradient

    def _calc_hydraulic_gradient(self) -> np.ndarray:
        """Calculate the hydraulic gradient."""
        psi0 = self._calc_base_hydraulic_gradient()
        pressure_gradient = self.grid.calc_grad_at_link('effective_pressure')

        return psi0 + pressure_gradient

    def _partition_meltwater(self) -> np.ndarray:
        """Partition meltwater input at nodes to each downslope link."""
        specific_melt = self.grid.at_node['meltwater__input'][self.grid.core_nodes]
        contributing_area = self.grid.cell_area_at_node
        melt_flux = specific_melt * contributing_area
        discharge = self.grid.map_mean_of_link_nodes_to_link(melt_flux)

        return discharge

    def _calc_melt_opening(self) -> np.ndarray:
        """Compute the channel opening rate from dissipation."""
        return (
            self.params['melt_constant'] 
            * self.grid.at_link['water__discharge'][:] 
            * self.grid.at_link['hydraulic__gradient'][:]
        )

    def _calc_gap_opening(self) -> np.ndarray:
        """Compute the cavity opening rate from sliding over bedrock steps."""
        return (
            self.grid.at_link['ice__sliding_velocity'][:] * self.params['step_height']
        )

    def _calc_closure(self) -> np.ndarray:
        """Compute the closure rate from viscous creep."""
        effective_pressure = self._calc_pressure_to_the_n()

        return (
            self.params['closure_constant']
            * effective_pressure
            * self.grid.at_link['conduit__area'][:]
        )

    def _calc_pressure_to_the_n(self) -> np.ndarray:
        """Compute the steady-state effective pressure (raised to Glen's n) at links."""
        numerator = self._calc_melt_opening() + self._calc_gap_opening()

        psi = self.grid.at_link['hydraulic__gradient'][:]

        nonzero_psi = np.where(
            psi == 0, 
            np.min(np.abs(psi[psi != 0])),
            psi
        )

        denominator = (
            self.params['closure_constant']
            * self.params['flow_constant']**(-1 / self.params['flow_exp'])
            * np.power(self.grid.at_link['water__discharge'][:], 1 / self.params['flow_exp'])
            * -1 * np.power(nonzero_psi, 1 / (2 * self.params['flow_exp']))
        )

        return numerator / denominator

    def _calc_discharge(self) -> np.ndarray:
        """Compute discharge as a function of conduit area and hydraulic gradient."""
        return (
            self.params['flow_constant']
            * np.power(self.grid.at_link['conduit__area'][:], self.params['flow_exp'])
            * np.power(
                np.abs(self.grid.at_link['hydraulic__gradient'][:]), -1/2
            )
            * self.grid.at_link['hydraulic__gradient'][:]
        )

    def _iterate(self, step: float, tolerance: float) -> np.ndarray:
        """Run one fixed-point iteration."""
        current = {
            'discharge': self.grid.at_link['water__discharge'][:],
            'gradient': self.grid.at_link['hydraulic__gradient'][:],
            'area': self.grid.at_link['conduit__area'][:],
            'pressure': self.grid.at_link['effective_pressure'][:]
        }

        