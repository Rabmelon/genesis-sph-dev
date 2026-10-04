import math
from typing import Annotated, Any

import quadrants as qd
from pydantic import Field

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


@qd.data_oriented
class Elastic(Base):
    """
    The linear elastic material class for SPH.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa). Higher values make the material stiffer, which tightens the elastic-wave timestep bound
        and raises the stress reached at a given strain. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    stress_diffusion : float, optional
        Coefficient of the stress diffusion stabilizer. The diffusion smooths the local stress differences that
        otherwise feed back into the momentum equation as noise, keeping a column stable at the cost of a softer
        stress response near free surfaces and walls. Zero disables it. Default is 0.1.
    """

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    stress_diffusion: NonNegativeFloat = 0.1

    # Derived material parameters, set in model_post_init.
    K: ValidFloat = Field(default=0.0, exclude=True)
    G: ValidFloat = Field(default=0.0, exclude=True)
    c_p: ValidFloat = Field(default=0.0, exclude=True)

    needs_stress: bool = Field(default=True, exclude=True, repr=False)

    @property
    def is_solid(self) -> bool:
        return True

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.c_p = math.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho)
        self.K0 = self.nu / (1.0 - self.nu)
        self.update_stress = self._update_stress_elastic

    @qd.func
    def _elastic_trial(
        self,
        dt: float,
        support_radius: float,
        sigma: qd.types.matrix(3, 3),
        L: qd.types.matrix(3, 3),
        stress_diffusion_sum: qd.types.matrix(3, 3),
    ):
        eye = qd.Matrix.identity(gs.qd_float, 3)
        strain_rate = 0.5 * (L + L.transpose())
        spin = 0.5 * (L - L.transpose())
        tr_strain_rate = strain_rate.trace()

        # Hooke's law plus the Jaumann rate, which keeps the stress objective under rigid rotation
        sigma_trial = (
            sigma
            + dt * (self.K * tr_strain_rate * eye + 2.0 * self.G * (strain_rate - tr_strain_rate / 3.0 * eye))
            + dt * (spin @ sigma - sigma @ spin)
        )

        # Stress diffusion (Feng): 2 * zeta * h * c_p * sum_j [Psi_ij (x_ij . grad W_ij) / (|x_ij|^2 + eta^2)] V_j,
        # where the smoothing length h is the support radius (see the diffusion accumulation in sph_solver.py)
        return sigma_trial + 2.0 * self.stress_diffusion * support_radius * self.c_p * dt * stress_diffusion_sum

    @qd.func
    def _update_stress_elastic(
        self,
        dt: float,
        support_radius: float,
        sigma: qd.types.matrix(3, 3),
        L: qd.types.matrix(3, 3),
        stress_diffusion_sum: qd.types.matrix(3, 3),
        eps_p: float,
    ):
        return self._elastic_trial(dt, support_radius, sigma, L, stress_diffusion_sum), eps_p
