import math
from typing import Annotated, Any

import quadrants as qd
from pydantic import Field

import genesis as gs
from genesis.typing import NonNegativeFloat, ValidFloat

from .elastic import Elastic


@qd.data_oriented
class DruckerPrager(Elastic):
    """
    The Drucker-Prager elasto-plastic material class for SPH.

    The material is ideal plastic (no hardening or softening) and follows the plane-strain cone of the
    Drucker-Prager surface, with cohesion and friction carried by the yield function `sqrt(J2) + alpha_phi * I1 - k_c`.
    The return mapping scales the deviatoric stress onto the surface, and a stress state past the apex is projected
    onto it, which leaves a cohesionless material with no tensile strength.

    Parameters
    ----------
    friction_angle : float, optional
        Internal friction angle in degrees, strictly between 0 and 90. Higher values make the material carry steeper
        slopes and spread less, and they tighten the apex condition under which the material loses all tensile
        strength. Default is 30.0.
    cohesion : float, optional
        Cohesion (Pa). Higher values give the material a tensile strength proportional to the cohesion, which lets
        it hold a slope or an unsupported face that dry sand would let go. Default is 0.0.
    dilation_angle : float, optional
        Dilation angle in degrees, from 0 to 90. A positive value makes the material expand while it yields, which
        raises the pressure of a confined material, and it commonly destabilizes a long-running flow. Zero keeps the
        volume constant during plastic flow. Default is 0.0.
    """

    friction_angle: Annotated[ValidFloat, Field(gt=0.0, lt=90.0)] = 30.0
    cohesion: NonNegativeFloat = 0.0
    dilation_angle: Annotated[ValidFloat, Field(ge=0.0, lt=90.0)] = 0.0

    # Derived yield-surface parameters, set in model_post_init.
    alpha_phi: ValidFloat = Field(default=0.0, exclude=True)
    alpha_psi: ValidFloat = Field(default=0.0, exclude=True)
    k_c: ValidFloat = Field(default=0.0, exclude=True)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        friction_angle = math.radians(self.friction_angle)
        tan_phi = math.tan(friction_angle)
        cone_denominator = math.sqrt(9.0 + 12.0 * tan_phi**2)
        self.alpha_phi = tan_phi / cone_denominator
        self.k_c = 3.0 * self.cohesion / cone_denominator
        # The dilation denominator carries the friction angle, like the friction term does
        self.alpha_psi = math.tan(math.radians(self.dilation_angle)) / cone_denominator
        # Lateral stress coefficient of the stress-diffusion gravity correction, from the Jaky relation
        self.K0 = 1.0 - math.sin(friction_angle)
        self.update_stress = self._update_stress_drucker_prager

    @qd.func
    def _update_stress_drucker_prager(
        self,
        dt: float,
        support_radius: float,
        sigma: qd.types.matrix(3, 3),
        L: qd.types.matrix(3, 3),
        stress_diffusion_sum: qd.types.matrix(3, 3),
        eps_p: float,
    ):
        eye = qd.Matrix.identity(gs.qd_float, 3)
        sigma_trial = self._elastic_trial(dt, support_radius, sigma, L, stress_diffusion_sum)

        mean_stress = sigma_trial.trace() / 3.0
        deviator = sigma_trial - mean_stress * eye
        sqrt_j2 = deviator.norm() / qd.sqrt(2.0)
        yield_function = sqrt_j2 + 3.0 * self.alpha_phi * mean_stress - self.k_c

        sigma_new = sigma_trial
        if yield_function > 0.0:
            if yield_function >= sqrt_j2:
                # Apex branch: the mean stress drops to the cone apex and the deviatoric part is dropped, which is
                # the no-tension condition of a cohesionless material
                if self.alpha_phi > 0.0:
                    sigma_new = (self.k_c / (3.0 * self.alpha_phi)) * eye
                else:
                    sigma_new = mean_stress * eye
            else:
                if sqrt_j2 > 1e-8:
                    # Plastic flow: the consistency parameter carries the dilation, and its denominator follows from
                    # the unit trace potential. A zero dilation angle reduces the projection to a radial scaling of
                    # the deviator, the mean stress staying untouched.
                    consistency = yield_function / (self.G + 9.0 * self.K * self.alpha_phi * self.alpha_psi)
                    radius = qd.max(1.0 - consistency * self.G / sqrt_j2, 0.0)
                    sigma_new = radius * deviator + (mean_stress - 3.0 * self.K * self.alpha_psi * consistency) * eye
                else:
                    sigma_new = mean_stress * eye

        # Accumulated equivalent plastic strain, from the deviatoric stress drop over the elastic shear stiffness
        plastic_stress = sigma_trial - sigma_new
        plastic_deviator = plastic_stress - plastic_stress.trace() / 3.0 * eye
        eps_p_new = eps_p + qd.sqrt(2.0 / 3.0 * (plastic_deviator * plastic_deviator).sum()) / (2.0 * self.G)
        return sigma_new, eps_p_new
