from typing import TYPE_CHECKING, Any, Literal

import quadrants as qd
from pydantic import Field

from genesis.typing import NonNegativeFloat, PositiveFloat, StrictInt, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.sph_entity import SPHEntity

SamplerType = Literal["pbs", "random", "regular"]


@qd.data_oriented
class Base(Material["SPHEntity"]):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler : str, optional
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    rho : float, optional
        The rest density (kg/m^3). Default is 1000.0.
    stiffness : float, optional
        State stiffness (N/m^2). Solids carry their stress through the constitutive update and ignore it. Default is
        0.0.
    exponent : float, optional
        State exponent. Solids carry their stress through the constitutive update and ignore it. Default is 1.0.
    mu : float, optional
        The viscosity, applied to every particle. A solid material keeps it at the default 0.0, otherwise the
        viscosity force adds to the momentum equation on top of the constitutive stress divergence. Default is 0.0.
    gamma : float, optional
        The surface tension, applied to every particle. A solid material keeps it at the default 0.0, otherwise the
        surface-tension force adds to the momentum equation on top of the constitutive stress divergence. Default is
        0.0.
    """

    sampler: SamplerType = "pbs"
    rho: PositiveFloat = 1000.0
    stiffness: NonNegativeFloat = 0.0
    exponent: PositiveFloat = 1.0
    mu: NonNegativeFloat = 0.0
    gamma: NonNegativeFloat = 0.0

    # Lateral stress coefficient of the stress-diffusion gravity correction. Derived by each material from its
    # elastic or frictional parameters, 1.0 meaning an isotropic correction.
    K0: ValidFloat = Field(default=1.0, exclude=True)

    # P-wave speed, which sets the impedance of the wall friction penalty and stays zero where the material carries
    # no elastic wave. Derived by each solid material from its elastic parameters.
    c_p: ValidFloat = Field(default=0.0, exclude=True)

    # Dispatch fields - set by subclass model_post_init, not user-specified. needs_stress turns the velocity
    # gradient and stress update kernels on for the materials that carry a constitutive stress.
    update_stress: Any = Field(default=None, exclude=True, repr=False)
    needs_stress: bool = Field(default=False, exclude=True, repr=False)

    # Auto-generated fields.
    idx: StrictInt | None = Field(default=None, exclude=True)

    @property
    def is_solid(self) -> bool:
        """
        Whether the material is a solid.

        Solid particles are excluded from the pressure solver, which lets each of them carry its own constitutive
        stress instead of the equation of state.
        """
        return False

    def model_post_init(self, context: Any) -> None:
        if self.update_stress is None:
            self.update_stress = self._update_stress_default

    @qd.func
    def _update_stress_default(
        self,
        dt: float,
        support_radius: float,
        sigma: qd.types.matrix(3, 3),
        L: qd.types.matrix(3, 3),
        stress_diffusion_sum: qd.types.matrix(3, 3),
        eps_p: float,
    ):
        return sigma, eps_p
