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
        water_density=1000
    ):
        super().__init__(grid)

        self.params = {
            'gravity': gravity,
            'ice_density': ice_density,
            'water_density': water_density
        }

    def _calc_base_hydraulic_gradient(self):
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

    def _partition_meltwater_input(self):
        """Partition meltwater input at nodes to each downslope link."""
        pass