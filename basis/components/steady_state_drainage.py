"""Component to model steady-state effective pressure and hydraulic gradients."""

import numpy as np
from dataclasses import dataclass, field

@dataclass
class Mesh:
    """Stores data about the model mesh."""
    nodes: np.ndarray
    dims: int = field(init = False)
    d: dict = field(init = False)

    def __post_init__(self):
        self.shape = self.nodes.shape
        self.dims = len(self.shape)
        self.d = {i: np.gradient(self.nodes, axis = i) for i in range(self.dims)}

@dataclass
class Parameters:
    """Stores the values of physical constants that are used widely."""
    gravity: float = 9.81
    ice_density: float = 917
    water_density: float = 1000

@dataclass
class HydrologicConstants(Parameters):
    """Stores the values of physical constants that are relevant to hydrology."""
    ice_fluidity: float = 6e-24
    glens_exp: int = 3
    latent_heat: float = 3.35e5
    darcy_friction: float = 3.75e-2
    darcy_exp: float = 5/4

    melt_opening_coeff: float = field(init = False)
    closure_coeff: float = field(init = False)
    flux_coeff: float = field(init = False)

    def __post_init__(self):
        self.melt_opening_coeff = 1 / (self.ice_density * self.latent_heat)
        self.closure_coeff = self.ice_fluidity * self.glens_exp**(-self.glens_exp)
        self.flux_coeff = (
            2**(1/4) * np.sqrt(np.pi + 2) / 
            (np.pi**(1/4) * np.sqrt(self.water_density * self.darcy_friction))
        )

@dataclass(kw_only = True)
class GlacierData(Parameters):
    """Stores data from the ice-covered region of the domain."""
    mesh: Mesh
    bed_elevation: np.ndarray
    ice_thickness: np.ndarray
    discharge: np.ndarray

    ice_pressure: np.ndarray = field(init = False)
    ice_pressure_gradient: np.ndarray = field(init = False)
    elevation_gradient: np.ndarray = field(init = False)
    base_hydraulic_gradient: np.ndarray = field(init = False)

    def __post_init__(self):
        for var in ['bed_elevation', 'ice_thickness', 'discharge']:
            if getattr(self, var).shape != self.mesh.shape:
                raise ValueError(
                    "Variable: " + var + " has shape " + str(getattr(self, var).shape) +
                    "\nbut should have shape " + str(self.mesh.shape) + " instead."
                )

        self.update_fields()

    def update_fields(self):
        self.ice_pressure = self.gravity * self.ice_density * self.ice_thickness
        self.ice_pressure_gradient = self._update_gradient(self.ice_pressure)
        self.elevation_gradient = self._update_gradient(self.bed_elevation)
        self.base_hydraulic_gradient = (
            -self.ice_pressure_gradient - (
                self.water_density * self.gravity * self.elevation_gradient
            )
        )
        
    def _update_gradient(self, field: np.ndarray):
        gradient = np.zeros_like(field)

        for i in range(self.mesh.dims):
            gradient += np.gradient(field, axis = i) / self.mesh.d[i]

        return gradient
        
class DrainageSystem:
    """Models the co-evolution of effective pressure and hydraulic gradients."""
    def __init__(self):
        pass