from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary, generate_dummy_particles
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils.misc import qd_to_numpy

from .particle_solver import ParticleSolver

if TYPE_CHECKING:
    from genesis.engine.entities import SPHEntity


@qd.data_oriented
class SPHSolver(ParticleSolver):
    material_cls = SPH.Base
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver

        # DFSPH parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error_percent
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations

        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # Boundary particles are appended to the particle arrays and reordered along with the simulated particles, so
        # every kernel that walks reordered particles needs to know whether the scene holds any. The friction
        # coefficient travels to the momentum kernel as an argument, a value of zero leaving the walls frictionless.
        self._boundary_friction = 0.0 if options.boundary_friction is None else options.boundary_friction
        self._has_boundary_particles = options.boundary_friction is not None

        # materials
        self._materials = list()
        self._materials_idx = list()
        self._materials_update_stress = list()

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.setup_boundary()

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

    def setup_boundary(self):
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
            # restitution=0.5,
        )

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            acc=gs.qd_vec3,  # acceleration
            rho=gs.qd_float,  # density
            p=gs.qd_float,  # pressure
            dfsph_factor=gs.qd_float,  # DFSPH use: Factor for Divergence and density solver
            drho=gs.qd_float,  # density deritivate
            L=gs.qd_mat3,  # velocity gradient
            sigma=gs.qd_mat3,  # Cauchy stress, tension positive
            eps_p=gs.qd_float,  # accumulated equivalent plastic strain
            vel_vir=gs.qd_vec3,  # virtual velocity of a boundary particle
            sigma_vir=gs.qd_mat3,  # virtual Cauchy stress of a boundary particle, tension positive
            vol_vir=gs.qd_float,  # virtual volume of a boundary particle, gauged to restore the kernel support
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
        )

        # static particle info
        struct_particle_info = qd.types.struct(
            rho=gs.qd_float,  # rest density
            mass=gs.qd_float,  # mass
            stiffness=gs.qd_float,
            exponent=gs.qd_float,
            mu=gs.qd_float,  # viscosity
            gamma=gs.qd_float,  # surface tension
            K0=gs.qd_float,  # lateral stress coefficient of the stress-diffusion gravity correction
            c_p=gs.qd_float,  # P-wave speed, the impedance the wall friction penalty scales by
            material_idx=gs.qd_int,
            is_solid=gs.qd_bool,
            is_boundary=gs.qd_bool,  # boundary particle, holding no material state
            boundary_normal=gs.qd_vec3,  # outward normal of the face the boundary particle belongs to
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
        )

        # construct fields
        # Boundary particles occupy the tail rows [n_particles, n_particles_total) of every particle array, and the
        # same reordered index space as the simulated particles.
        self.particles = struct_particle_state.field(
            shape=(self._n_particles_total, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._n_particles_total, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info = struct_particle_info.field(
            shape=(self._n_particles_total,), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles_total, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles_total, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info_reordered = struct_particle_info.field(
            shape=(self._n_particles_total, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles_total, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        # Stress diffusion term summed over the neighbours, read by the stress update within the same substep. It is
        # buffered because the velocity gradient sweep reads the stress field while the stress update writes it.
        if self._needs_stress:
            self.stress_diffusion_sum = qd.field(
                gs.qd_mat3, shape=(self._n_particles_total, self._B), layout=qd.Layout.SOA
            )

        if self._pressure_solver == "DFSPH":
            # Stiffness that the next Jacobi iteration applies, read by the neighbors of each particle
            self.particles_k_reordered = qd.field(gs.qd_float, shape=(self._n_particles, self._B), needs_grad=False)

            struct_env_df = qd.types.struct(
                err=gs.qd_float,  # density error summed over the active particles
                n_active=gs.qd_int,  # number of active particles, over which the density error is averaged
                is_solving=gs.qd_bool,
            )
            self.envs_df = struct_env_df.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

            struct_df_info = qd.types.struct(
                substep_dt=gs.qd_float,
                density_err_tol=gs.qd_float,  # average density error that ends the density solve, kg/m^3
                divergence_err_tol=gs.qd_float,  # average density rate that ends the divergence solve, kg/m^3/s
                stiffness_eps=gs.qd_float,
                jacobi_relaxation=gs.qd_float,
                factor_eps=gs.qd_float,
                n_neighbors_min=gs.qd_int,  # fewer neighbors leave the density rate of a particle out of the solve
            )
            self.df_info = struct_df_info.field(shape=(), needs_grad=False)

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles
        self._n_particles_total = self._n_particles

        self._coupler = self.sim._coupler

        self._needs_stress = any(material.needs_stress for material in self._materials)
        solid_materials = [material for material in self._materials if material.is_solid]

        if self._pressure_solver == "DFSPH" and solid_materials:
            gs.raise_exception("Solid SPH materials are only supported by the 'WCSPH' pressure solver.")

        if self.is_active:
            if self._has_boundary_particles:
                if not solid_materials:
                    gs.raise_exception(
                        "'boundary_friction' is set but the scene holds no solid SPH material to support."
                    )
                boundary_positions, boundary_normals = generate_dummy_particles(
                    self._lower_bound, self._upper_bound, self._particle_size, self._support_radius
                )
                self._n_particles_total = self._n_particles + boundary_positions.shape[0]
                gs.logger.info(
                    f"Generated {boundary_positions.shape[0]} boundary particles for the {self._n_particles} "
                    "simulated particles."
                )

            self.sh.build(self._n_particles_total, self._B)
            self.init_particle_fields()

            for entity in self.entities:
                entity._add_to_solver()

            if self._has_boundary_particles:
                self._kernel_add_boundary_particles(
                    self._sim.cur_substep_local,
                    self._n_particles,
                    boundary_positions.shape[0],
                    solid_materials[0].rho,
                    boundary_positions,
                    boundary_normals,
                )

            # Compute stable timestep for the given particle properties (mu) and support radius and print warning
            # if the current timestep is larger than the stable timestep. This formula is derived from
            # [_task_compute_non_pressure_forces] function, which computes the acceleration due to non-pressure forces.
            stable_dt = (
                self._stable_dt_coef
                * (0.01 * self._support_radius**2)
                / max(qd_to_numpy(self.particles_info.mu).max(), gs.EPS)
            )
            if self.substep_dt > stable_dt:
                gs.logger.warning(
                    f"Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep "
                    f"({stable_dt:.4g}) for SPH solver. "
                    "Please consider reducing the timestep, liquid viscosity (mu), or increasing the particle size."
                )

            # Compute the elastic-wave stable timestep for every solid material and warn if the current timestep
            # exceeds it. The timestep is never adjusted automatically because the substep count is resolved once at
            # build time from the scene options.
            if solid_materials:
                v_max = np.linalg.norm(qd_to_numpy(self.particles.vel), axis=-1).max()
                elastic_dt = min(0.3 * self._support_radius / (material.c_p + v_max) for material in solid_materials)
                if self.substep_dt > elastic_dt:
                    gs.logger.warning(
                        f"Current timestep ({self.substep_dt:.4g}) is larger than the elastic-wave stable timestep "
                        f"({elastic_dt:.4g}) for SPH solid materials, computed from the P-wave speed and the "
                        f"maximum particle velocity at build time ({v_max:.4g}). "
                        "Please consider reducing the timestep, the Young's modulus, or increasing the particle size."
                    )

            # TODO: Support per-particle density
            self._density0 = self.particles_info[0].rho

            if self._pressure_solver == "DFSPH":
                self.df_info.substep_dt[None] = self.substep_dt
                self.df_info.density_err_tol[None] = self._df_max_error_den * 0.01 * self._density0
                self.df_info.divergence_err_tol[None] = (
                    1.0 / self.substep_dt * self._df_max_error_div * 0.01 * self._density0
                )
                self.df_info.stiffness_eps[None] = 1e-5
                # The DFSPH factor inverts the diagonal of the pressure system alone, and the stiffness of the neighbors
                # of a particle adds up to as much again, so the diagonally scaled system has eigenvalues up to 2. A
                # full Jacobi update leaves the mode alternating from one particle to the next undamped, and overshoots
                # once the liquid compresses abruptly. Half the update cancels that mode, the others still converging.
                self.df_info.jacobi_relaxation[None] = 0.5
                self.df_info.factor_eps[None] = 1e-6
                self.df_info.n_neighbors_min[None] = 20

        # Kernels of this solver take the solver itself, so gravity has to be a field for them.
        self._build_gravity()

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_material(self, material):
        # Register material update methods if and only if the provided material is not already registered
        for material_i in self._materials:
            if material == material_i:
                material.idx = material_i.idx
                break
        else:
            material.idx = len(self._materials_idx)
            self._materials_idx.append(material.idx)
            self._materials_update_stress.append(material.update_stress)
        self._materials.append(material)

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "SPHEntity":
        self.add_material(material)

        entity = SPHEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )

        self.entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles_total, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # Copy to reordered, the active particles taking the first sorted indices (see compute_reordered_idx).
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            self.particles_ng_reordered[i_p, i_b].active = i_p < self.sh.cur_cnt[i_b]
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]

        if qd.static(self._coupler._rigid_sph):
            for i_p, i_g, i_b in qd.ndrange(self._n_particles_total, self._coupler.rigid_solver.n_geoms, self._B):
                if self.particles_ng[i_p, i_b].active:
                    self._coupler.sph_rigid_normal_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_g, i_b] = (
                        self._coupler.sph_rigid_normal[i_p, i_g, i_b]
                    )

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            if self.particles_ng[i_p, i_b].active:
                # only need to copy back dynamic state, i.e. self.particles
                self.particles[i_p, i_b] = self.particles_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_b]

        if qd.static(self._coupler._rigid_sph):
            for i_p, i_g, i_b in qd.ndrange(self._n_particles_total, self._coupler.rigid_solver.n_geoms, self._B):
                if self.particles_ng[i_p, i_b].active:
                    self._coupler.sph_rigid_normal[i_p, i_g, i_b] = self._coupler.sph_rigid_normal_reordered[
                        self.particles_ng[i_p, i_b].reordered_idx, i_g, i_b
                    ]

    @qd.func
    def _task_compute_rho(self, i, j, ret: qd.template(), i_b):
        # A boundary neighbour weighs its virtual volume, restoring the kernel weight the wall truncates for the solid
        # particle it acts on. Fluid particles take no support from the walls.
        volume_j = self._particle_volume
        if qd.static(self._has_boundary_particles):
            if self.particles_info_reordered[j, i_b].is_boundary:
                if self.particles_info_reordered[i, i_b].is_solid:
                    volume_j = self.particles_reordered[j, i_b].vol_vir
                else:
                    volume_j = gs.qd_float(0.0)
        ret += volume_j * self.cubic_kernel(
            (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos).norm()
        )

    @qd.kernel
    def _kernel_compute_rho(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_simulated = self.particles_ng_reordered[i_p, i_b].active
            if qd.static(self._has_boundary_particles):
                is_simulated = is_simulated and not self.particles_info_reordered[i_p, i_b].is_boundary
            if is_simulated:
                # Base density using the kernel at distance 0
                self.particles_reordered[i_p, i_b].rho = self._particle_volume * self.cubic_kernel(0.0)

                den = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, den, self._task_compute_rho, i_b
                )
                self.particles_reordered[i_p, i_b].rho += den

                self.particles_reordered[i_p, i_b].rho *= self.particles_info_reordered[i_p, i_b].rho

    @qd.func
    def _task_compute_non_pressure_forces(self, i, j, ret: qd.template(), i_b: qd.i32):
        d_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        dist = d_ij.norm()

        gamma_i = self.particles_info_reordered[i, i_b].gamma
        mass_i = self.particles_info_reordered[i, i_b].mass
        mu_i = self.particles_info_reordered[i, i_b].mu

        mass_j = self.particles_info_reordered[j, i_b].mass

        # -----------------------------
        # Surface Tension term
        # -----------------------------
        # If distance is bigger than _particle_size, use d_ij.norm() directly; otherwise clamp.
        effective_dist = dist if dist > self._particle_size else self._particle_size

        ret -= gamma_i / mass_i * mass_j * d_ij * self.cubic_kernel(effective_dist)

        # -----------------------------
        # Viscosity Force
        # -----------------------------
        v_ij = (self.particles_reordered[i, i_b].vel - self.particles_reordered[j, i_b].vel).dot(d_ij)

        # Some constant factor used in the viscosity formula
        d = 2 * (3 + 2)

        # The density of the neighbor j in batch b
        rho_j = self.particles_reordered[j, i_b].rho

        f_v = (
            d
            * mu_i
            * (mass_j / rho_j)
            * v_ij
            / (dist**2 + 0.01 * self._support_radius**2)
            * self.cubic_kernel_derivative(d_ij)
        )
        ret += f_v

    @qd.kernel
    def _kernel_compute_non_pressure_forces(self, f: qd.i32, t: float):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_simulated = self.particles_ng_reordered[i_p, i_b].active
            if qd.static(self._has_boundary_particles):
                is_simulated = is_simulated and not self.particles_info_reordered[i_p, i_b].is_boundary
            if is_simulated:
                acc = self._gravity[i_b]
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    acc,
                    self._task_compute_non_pressure_forces,
                    i_b,
                )

                # external force fields
                for i_ff in qd.static(range(len(self._ffs))):
                    acc += self._ffs[i_ff].get_acc(
                        self.particles_reordered[i_p, i_b].pos, self.particles_reordered[i_p, i_b].vel, t, i_p
                    )
                self.particles_reordered[i_p, i_b].acc = acc

    @qd.func
    def _task_compute_stress_forces(self, i, j, ret: qd.template(), i_b: qd.i32):
        rho_i = self.particles_reordered[i, i_b].rho
        rho_j = self.particles_reordered[j, i_b].rho
        mass_j = self.particles_info_reordered[j, i_b].mass
        sigma_j = self.particles_reordered[j, i_b].sigma
        grad_w = self.cubic_kernel_derivative(
            self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        )

        # A boundary neighbour acts through the solid particle it faces: it borrows that particle's rest density for
        # its mass and brings its own virtual volume and stress. It stands in for the mirror image of the centre
        # across the wall plane, whose computed density equals the centre's by lattice symmetry, so it carries
        # rho_j = rho_i and its sigma_vir / rho_j^2 weighs like a real neighbour of the same state. Fluid particles
        # take no support from the walls and keep a zero-weight pair, since a boundary particle carries no mass.
        if qd.static(self._has_boundary_particles):
            if self.particles_info_reordered[j, i_b].is_boundary and self.particles_info_reordered[i, i_b].is_solid:
                rho0_i = self.particles_info_reordered[i, i_b].rho
                rho_j = rho_i
                mass_j = self.particles_reordered[j, i_b].vol_vir * rho0_i
                sigma_j = self.particles_reordered[j, i_b].sigma_vir

                # Coulomb friction between the wall and the particle: the wall carries a tangential traction that
                # opposes the slip, bounded by mu times the normal pressure the pair presses the wall with. The bound
                # alone decides between holding and sliding. Below it the traction is proportional to the slip
                # velocity, which vanishes with that velocity, so a particle at rest feels nothing and a zero mu
                # leaves exactly the free-slip wall. The normal pressure is the sum of two wall-normal tractions of
                # the same sign and comparable magnitude: the material-side arm carries the normal traction of the
                # particle's stress, and the wall-side arm the normal traction of the particle's free-slip mirror
                # image.
                normal = self.particles_info_reordered[j, i_b].boundary_normal
                vel_i = self.particles_reordered[i, i_b].vel
                vel_t = vel_i - vel_i.dot(normal) * normal
                vel_t_norm = vel_t.norm()
                normal_pressure = qd.max(-normal.dot((self.particles_reordered[i, i_b].sigma + sigma_j) @ normal), 0.0)

                # The velocity-proportional branch carries the P-wave impedance of the material, rho0 * c_p,
                # computed from the initial density rho0 by construction. A settled particle runs at a density around
                # 0.8 rho0. An impedance times a velocity is the traction an acoustic wave of that particle velocity
                # carries, so the penalty stays within the elastic response of the material, and its arrest time, one
                # wave transit of a particle spacing, is the shortest the elastic timestep resolves. A larger
                # impedance would relax the slip faster than the scheme resolves and demand a shorter stable
                # timestep.
                friction_impedance = rho0_i * self.particles_info_reordered[i, i_b].c_p
                tau_fric = qd.min(ret.wall_friction * normal_pressure, friction_impedance * vel_t_norm)

                # The traction enters the momentum equation as the material-side arm of the stress divergence does,
                # m_b (T / rho_i^2) @ grad W, dimensional as [kg] [Pa] [m^6 kg^-2] [m^-4] = [m s^-2]. Grad W runs
                # along the outward normal of the wall lattice, so this tensor opposes the slip of the particle.
                t_hat = vel_t / qd.max(vel_t_norm, gs.EPS)
                traction_fric = -tau_fric * (
                    qd.Matrix.outer_product(t_hat, normal) + qd.Matrix.outer_product(normal, t_hat)
                )
                ret.acc += mass_j * (traction_fric / rho_i**2) @ grad_w

        # Stress divergence, symmetric formula
        ret.acc += mass_j * ((self.particles_reordered[i, i_b].sigma / rho_i**2 + sigma_j / rho_j**2) @ grad_w)

    @qd.kernel
    def _kernel_compute_stress_forces(self, f: qd.i32, wall_friction: qd.f32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_simulated = self.particles_ng_reordered[i_p, i_b].active
            if qd.static(self._has_boundary_particles):
                is_simulated = is_simulated and not self.particles_info_reordered[i_p, i_b].is_boundary
            if is_simulated:
                ret = qd.Struct(acc=qd.Vector.zero(gs.qd_float, 3), wall_friction=wall_friction)

                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_stress_forces,
                    i_b,
                )
                self.particles_reordered[i_p, i_b].acc += ret.acc

    @qd.kernel
    def _kernel_compute_eos_pressure(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_fluid = (
                self.particles_ng_reordered[i_p, i_b].active and not self.particles_info_reordered[i_p, i_b].is_solid
            )
            if qd.static(self._has_boundary_particles):
                is_fluid = is_fluid and not self.particles_info_reordered[i_p, i_b].is_boundary
            if is_fluid:
                rho0 = self.particles_info_reordered[i_p, i_b].rho
                stiff = self.particles_info_reordered[i_p, i_b].stiffness
                expnt = self.particles_info_reordered[i_p, i_b].exponent

                self.particles_reordered[i_p, i_b].rho = qd.max(self.particles_reordered[i_p, i_b].rho, rho0)

                self.particles_reordered[i_p, i_b].p = stiff * (
                    qd.pow(self.particles_reordered[i_p, i_b].rho / rho0, expnt) - 1.0
                )

                # Fluids enter the stress divergence as an isotropic pressure stress
                self.particles_reordered[i_p, i_b].sigma = -self.particles_reordered[i_p, i_b].p * qd.Matrix.identity(
                    gs.qd_float, 3
                )

    @qd.kernel
    def _kernel_advect_velocity(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_simulated = self.particles_ng_reordered[i_p, i_b].active
            if qd.static(self._has_boundary_particles):
                is_simulated = is_simulated and not self.particles_info_reordered[i_p, i_b].is_boundary
            if is_simulated:
                self.particles_reordered[i_p, i_b].vel += self._substep_dt * self.particles_reordered[i_p, i_b].acc

    @qd.kernel
    def _kernel_advect_position(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_simulated = self.particles_ng_reordered[i_p, i_b].active
            if qd.static(self._has_boundary_particles):
                is_simulated = is_simulated and not self.particles_info_reordered[i_p, i_b].is_boundary
            if is_simulated:
                pos = self.particles_reordered[i_p, i_b].pos
                new_vel = self.particles_reordered[i_p, i_b].vel

                # advect
                new_pos = pos + self._substep_dt * new_vel

                # impose boundary (pass b if needed)
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # update
                self.particles_reordered[i_p, i_b].vel = corrected_vel
                self.particles_reordered[i_p, i_b].pos = corrected_pos

    # ------------------------------------------------------------------------------------
    # --------------------------------- solid stress -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _task_compute_velocity_gradient(self, i, j, ret: qd.template(), i_b: qd.i32):
        x_ij = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        grad_w = self.cubic_kernel_derivative(x_ij)
        vol_j = self.particles_info_reordered[j, i_b].mass / self.particles_reordered[j, i_b].rho
        v_j = self.particles_reordered[j, i_b].vel

        is_boundary_j = False
        if qd.static(self._has_boundary_particles):
            is_boundary_j = self.particles_info_reordered[j, i_b].is_boundary
            if is_boundary_j:
                # The wall stands in for the centre's mirror image across the wall plane, so it weighs what that
                # neighbour would: vol_vir over the density ratio rho_i / rho0_i it carries, like every real
                # neighbour's mass / rho.
                vol_j = (
                    self.particles_reordered[j, i_b].vol_vir
                    * self.particles_info_reordered[i, i_b].rho
                    / self.particles_reordered[i, i_b].rho
                )
                v_j = self.particles_reordered[j, i_b].vel_vir

        ret.L += vol_j * qd.Matrix.outer_product(v_j - self.particles_reordered[i, i_b].vel, grad_w)

        # Stress diffusion term (Feng), accumulated as a stress rate. The gravity correction cancels the geostatic
        # stress difference, so the diffusion does not relax the weight of a settled column. The smoothing length is
        # the support radius, which is twice the particle size in the genesis SPH convention.
        if not is_boundary_j:
            psi = self.particles_reordered[i, i_b].sigma - self.particles_reordered[j, i_b].sigma
            psi += self.particles_info_reordered[i, i_b].rho * self._gravity[i_b].dot(x_ij) * ret.gravity_correction
            eta_sq = (0.1 * self._support_radius) ** 2
            ret.stress_diffusion_sum += psi * (x_ij.dot(grad_w) / (x_ij.norm_sqr() + eta_sq)) * vol_j

    @qd.kernel
    def _kernel_compute_velocity_gradient(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            if self.particles_ng_reordered[i_p, i_b].active and self.particles_info_reordered[i_p, i_b].is_solid:
                # Gravity correction of the stress diffusion, hoisted out of the neighbour sweep
                gravity = self._gravity[i_b]
                gravity_norm_sq = gravity.norm_sqr()
                gravity_correction = qd.Matrix.zero(gs.qd_float, 3, 3)
                if gravity_norm_sq > 1e-12:
                    K0 = self.particles_info_reordered[i_p, i_b].K0
                    gravity_correction = K0 * qd.Matrix.identity(gs.qd_float, 3) + (1.0 - K0) * (
                        qd.Matrix.outer_product(gravity, gravity) / gravity_norm_sq
                    )

                ret = qd.Struct(
                    L=qd.Matrix.zero(gs.qd_float, 3, 3),
                    stress_diffusion_sum=qd.Matrix.zero(gs.qd_float, 3, 3),
                    gravity_correction=gravity_correction,
                )

                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_velocity_gradient,
                    i_b,
                )

                self.particles_reordered[i_p, i_b].L = ret.L
                self.stress_diffusion_sum[i_p, i_b] = ret.stress_diffusion_sum

    @qd.kernel
    def _kernel_update_stress(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            if self.particles_ng_reordered[i_p, i_b].active and self.particles_info_reordered[i_p, i_b].is_solid:
                sigma = qd.Matrix.zero(gs.qd_float, 3, 3)
                eps_p = gs.qd_float(0.0)
                for material_idx in qd.static(self._materials_idx):
                    if self.particles_info_reordered[i_p, i_b].material_idx == material_idx:
                        sigma, eps_p = self._materials_update_stress[material_idx](
                            dt=self._substep_dt,
                            support_radius=self._support_radius,
                            sigma=self.particles_reordered[i_p, i_b].sigma,
                            L=self.particles_reordered[i_p, i_b].L,
                            stress_diffusion_sum=self.stress_diffusion_sum[i_p, i_b],
                            eps_p=self.particles_reordered[i_p, i_b].eps_p,
                        )
                self.particles_reordered[i_p, i_b].sigma = sigma
                self.particles_reordered[i_p, i_b].eps_p = eps_p

    # ------------------------------------------------------------------------------------
    # ------------------------------- boundary particles ---------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _task_extrapolate_boundary_state(self, i, j, ret: qd.template(), i_b: qd.i32):
        if self.particles_info_reordered[j, i_b].is_solid and not self.particles_info_reordered[j, i_b].is_boundary:
            x_ji = self.particles_reordered[j, i_b].pos - self.particles_reordered[i, i_b].pos
            vol_j = self.particles_info_reordered[j, i_b].mass / self.particles_reordered[j, i_b].rho
            weight = self.cubic_kernel(x_ji.norm()) * vol_j

            # K0 gravity correction of the neighbour, taken at its own material and in the direction from the wall to
            # the neighbour, so the extrapolated stress carries the geostatic increment between the two.
            gravity = self._gravity[i_b]
            gravity_norm_sq = gravity.norm_sqr()
            gravity_correction = qd.Matrix.zero(gs.qd_float, 3, 3)
            if gravity_norm_sq > 1e-12:
                K0 = self.particles_info_reordered[j, i_b].K0
                gravity_correction = K0 * qd.Matrix.identity(gs.qd_float, 3) + (1.0 - K0) * (
                    qd.Matrix.outer_product(gravity, gravity) / gravity_norm_sq
                )

            ret.den += weight
            ret.v_sum += self.particles_reordered[j, i_b].vel * weight
            ret.sigma_sum += (
                self.particles_reordered[j, i_b].sigma
                + self.particles_info_reordered[j, i_b].rho * gravity.dot(x_ji) * gravity_correction
            ) * weight

    @qd.kernel
    def _kernel_update_boundary_state(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            if self.particles_info_reordered[i_p, i_b].is_boundary:
                ret = qd.Struct(
                    den=gs.qd_float(0.0),
                    v_sum=qd.Vector.zero(gs.qd_float, 3),
                    sigma_sum=qd.Matrix.zero(gs.qd_float, 3, 3),
                )
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_extrapolate_boundary_state,
                    i_b,
                )

                # A boundary particle whose solid neighbourhood is negligible keeps an extrapolated state of zero.
                shepard = gs.qd_float(1.0)
                if ret.den > 1e-3:
                    shepard = 1.0 / ret.den

                v_ext = ret.v_sum * shepard
                sigma_ext = ret.sigma_sum * shepard

                # The wall contributes the nominal particle volume: at the lattice position of the mirror image of a
                # solid particle, that volume restores exactly the kernel weight the wall truncates. Scaling it by the
                # Shepard denominator instead would inflate it wherever the wall is partially covered by solids, and
                # over-restore the density there.
                self.particles_reordered[i_p, i_b].vol_vir = self._particle_volume

                normal = self.particles_info_reordered[i_p, i_b].boundary_normal

                # The stress channel of the wall is free-slip: it carries the mirror image of the extrapolated stress
                # across the wall plane, the reflection R sigma R with R = I - 2 n n^T, which keeps the normal
                # traction and reverses the tangential one. The tangential traction of a pair then cancels between
                # this image arm and the material-side arm, which carries the wall friction alone (see the Coulomb
                # friction in _task_compute_stress_forces).
                traction = sigma_ext @ normal
                traction_normal = normal.dot(traction)
                traction_shear = traction - traction_normal * normal
                sigma_vir = sigma_ext - 2.0 * (
                    qd.Matrix.outer_product(traction_shear, normal) + qd.Matrix.outer_product(normal, traction_shear)
                )

                # The wall applies no return mapping, so a tensile trace would pull the adjacent material: clamp it.
                if sigma_vir.trace() > 0.0:
                    sigma_vir = qd.Matrix.zero(gs.qd_float, 3, 3)

                # The mirror reverses the normal component of the relative velocity, which keeps the material from
                # crossing the wall, and carries the tangential one through as the extrapolation gives it, so the
                # velocity path injects no tangential drag of its own.
                v_wall = self.particles_reordered[i_p, i_b].vel
                v_vir = v_ext - 2.0 * (v_ext - v_wall).dot(normal) * normal

                self.particles_reordered[i_p, i_b].vel_vir = v_vir
                self.particles_reordered[i_p, i_b].sigma_vir = sigma_vir

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------
    # The DFSPH sweeps run over the simulated particles alone: boundary particles require a solid material and build
    # refuses the pressure solver together with one, so the sweeps need no boundary gate (see the refusal in build).

    @qd.func
    def _task_compute_DFSPH_factor(self, i, j, ret: qd.template(), i_b):
        # Fluid neighbors
        grad_j = -self._particle_volume * self.cubic_kernel_derivative(
            self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        )
        ret[3] += grad_j.norm_sqr()  # sum_grad_p_k
        for ii in qd.static(range(3)):  # grad_p_i
            ret[ii] -= grad_j[ii]

    @qd.kernel
    def _kernel_compute_DFSPH_factor(self, f: qd.i32):
        # Every environment takes part in the divergence solve that follows (see _kernel_update_convergence).
        for i_b in range(self._B):
            self.envs_df[i_b].is_solving = True
            self.envs_df[i_b].err = 0.0
            self.envs_df[i_b].n_active = 0

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                qd.atomic_add(self.envs_df[i_b].n_active, 1)

                sum_grad_p_k = gs.qd_float(0.0)
                grad_p_i = qd.Vector.zero(gs.qd_float, 3)

                # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
                ret = qd.Vector.zero(gs.qd_float, 4)

                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_compute_DFSPH_factor, i_b
                )

                sum_grad_p_k = ret[3]
                for ii in qd.static(range(3)):
                    grad_p_i[ii] = ret[ii]
                sum_grad_p_k += grad_p_i.norm_sqr()

                # Compute pressure stiffness denominator
                factor = gs.qd_float(0.0)
                if sum_grad_p_k > self.df_info[None].factor_eps:
                    factor = -1.0 / sum_grad_p_k
                else:
                    # A particle without neighbors carries no pressure into the next density solve.
                    factor = 0.0
                    self.particles_reordered[i_p, i_b].p = 0.0
                self.particles_reordered[i_p, i_b].dfsph_factor = factor

    @qd.func
    def _task_compute_density_time_derivative(self, i, j, ret: qd.template(), i_b):
        v_i = self.particles_reordered[i, i_b].vel
        v_j = self.particles_reordered[j, i_b].vel

        x_i = self.particles_reordered[i, i_b].pos
        x_j = self.particles_reordered[j, i_b].pos

        # Fluid neighbors
        ret.drho += self._particle_volume * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))
        ret.num_neighbors += 1

    @qd.kernel
    def _kernel_compute_density_time_derivative(self):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_df[i_b].is_solving and self.particles_ng_reordered[i_p, i_b].active:
                ret = qd.Struct(drho=0.0, num_neighbors=0)
                self.sh.for_all_neighbors(
                    i_p,
                    self.particles_reordered.pos,
                    self._support_radius,
                    ret,
                    self._task_compute_density_time_derivative,
                    i_b,
                )

                # only correct positive divergence
                drho = qd.max(ret.drho, 0.0)
                num_neighbors = ret.num_neighbors

                # Do not perform divergence solve when paritlce deficiency happens
                if num_neighbors < self.df_info[None].n_neighbors_min:
                    drho = 0.0

                self.particles_reordered[i_p, i_b].drho = drho

                # Stiffness that the next Jacobi iteration applies to cancel the density rate of the particle.
                self.particles_k_reordered[i_p, i_b] = drho * self.particles_reordered[i_p, i_b].dfsph_factor

                # The divergence error of the environment, summed for _kernel_update_convergence.
                qd.atomic_add(self.envs_df[i_b].err, self.particles_info_reordered[i_p, i_b].rho * drho)

    @qd.kernel
    def _kernel_predict_velocity(self, f: qd.i32):
        # Every environment takes part in the density solve that follows (see _kernel_update_convergence).
        for i_b in range(self._B):
            self.envs_df[i_b].is_solving = True
            self.envs_df[i_b].err = 0.0

        dt = self.df_info[None].substep_dt
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                # compute new velocities only considering non-pressure forces
                self.particles_reordered[i_p, i_b].vel = (
                    self.particles_reordered[i_p, i_b].vel + dt * self.particles_reordered[i_p, i_b].acc
                )

                # The density solve corrects a density error over the substep, which turns the DFSPH factor of the
                # divergence solve, a velocity correction per density rate, into an acceleration per density error.
                self.particles_reordered[i_p, i_b].dfsph_factor = self.particles_reordered[i_p, i_b].dfsph_factor * (
                    1.0 / dt**2
                )

    @qd.func
    def _task_compute_density_star(self, i, j, ret: qd.template(), i_b):
        v_i = self.particles_reordered[i, i_b].vel
        v_j = self.particles_reordered[j, i_b].vel
        x_i = self.particles_reordered[i, i_b].pos
        x_j = self.particles_reordered[j, i_b].pos
        ret += self._particle_volume * (v_i - v_j).dot(self.cubic_kernel_derivative(x_i - x_j))

    @qd.kernel
    def _kernel_compute_density_star(self):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_df[i_b].is_solving and self.particles_ng_reordered[i_p, i_b].active:
                delta = gs.qd_float(0.0)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, delta, self._task_compute_density_star, i_b
                )
                rho0 = self.particles_info_reordered[i_p, i_b].rho
                drho = self.particles_reordered[i_p, i_b].rho / rho0 + self.df_info[None].substep_dt * delta
                self.particles_reordered[i_p, i_b].drho = drho

                # Stiffness that the next Jacobi iteration applies to cancel the density error of the particle. The
                # stiffness accumulated by the density solve is carried over from one substep to the next, so the
                # iterations update it in both directions: they take back the part the liquid no longer needs, and
                # clamp it so that the pressure never pulls the liquid together. The update is relaxed (see
                # jacobi_relaxation in build).
                kappa = -self.particles_reordered[i_p, i_b].p / rho0
                k_i = (
                    self.df_info[None].jacobi_relaxation
                    * (drho - 1.0)
                    * self.particles_reordered[i_p, i_b].dfsph_factor
                )
                self.particles_k_reordered[i_p, i_b] = qd.min(kappa + k_i, 0.0) - kappa

                # The density error of the environment, summed for _kernel_update_convergence. A particle under
                # pressure counts its expansion as well, which the iterations correct by taking pressure back, while
                # a particle free of pressure only counts its compression, the liquid being free to expand.
                err_rel = drho - 1.0
                if kappa >= 0.0:
                    err_rel = qd.max(err_rel, 0.0)
                qd.atomic_add(self.envs_df[i_b].err, rho0 * qd.abs(err_rel))

    @qd.func
    def _func_stiffness(self, i_p: int, i_b: int, warm_start: bool):
        # The warm start applies the stiffness carried over from the previous substep in particles.p (see
        # _kernel_solver_iteration) to the particles compressed at the start of the substep.
        k_i = self.particles_k_reordered[i_p, i_b]
        if warm_start:
            k_i = 0.0
            rho0 = self.particles_info_reordered[i_p, i_b].rho
            if self.particles_reordered[i_p, i_b].rho > rho0:
                k_i = -self.particles_reordered[i_p, i_b].p / rho0
        return k_i

    @qd.func
    def _task_solver_iteration(self, i_p: int, j_p: int, ret: qd.template(), i_b: int):
        # The stiffness already holds the inverse of the density, which is the same for every particle of a liquid.
        k_sum = ret.k_i + self._func_stiffness(j_p, i_b, ret.is_warm_start)
        if qd.abs(k_sum) > self.df_info[None].stiffness_eps:
            grad_p_j = -self._particle_volume * self.cubic_kernel_derivative(
                self.particles_reordered[i_p, i_b].pos - self.particles_reordered[j_p, i_b].pos
            )
            ret.dv = ret.dv - ret.dv_scale * k_sum * grad_p_j

    @qd.kernel
    def _kernel_solver_iteration(self, solve_density: qd.template(), warm_start: bool):
        # Jacobi iteration: the velocity of every particle changes by the stiffness of its own and of its neighbors.
        # The divergence solve corrects velocities directly, while the density solve accelerates over the substep.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.envs_df[i_b].is_solving and self.particles_ng_reordered[i_p, i_b].active:
                dv_scale = gs.qd_float(1.0)
                if qd.static(solve_density):
                    dv_scale = self.df_info[None].substep_dt
                k_i = self._func_stiffness(i_p, i_b, warm_start)
                ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i, dv_scale=dv_scale, is_warm_start=warm_start)
                self.sh.for_all_neighbors(
                    i_p, self.particles_reordered.pos, self._support_radius, ret, self._task_solver_iteration, i_b
                )
                self.particles_reordered[i_p, i_b].vel = self.particles_reordered[i_p, i_b].vel + ret.dv

                if qd.static(solve_density):
                    # The stiffness k_i is the pressure this pass applies to cancel the density error, up to the
                    # -1/rho0 scaling carried by dfsph_factor, so summing -k_i * rho0 over the density solve yields
                    # the implicit pressure field that enforces incompressibility. This exposes in particles.p the
                    # same physical quantity that the equation of state provides under WCSPH. The warm start applies
                    # the pressure as it stands and drops the pressure of the particles it leaves out, which the
                    # neighbors reading it in the meantime leave out the same way.
                    rho0 = self.particles_info_reordered[i_p, i_b].rho
                    if warm_start:
                        if self.particles_reordered[i_p, i_b].rho <= rho0:
                            self.particles_reordered[i_p, i_b].p = 0.0
                    else:
                        self.particles_reordered[i_p, i_b].p = self.particles_reordered[i_p, i_b].p - k_i * rho0

    @qd.kernel
    def _kernel_update_convergence(self, solve_density: qd.template()) -> int:
        # Each environment stops iterating once the average density error of its own particles meets the tolerance,
        # so that its result is the same whatever the other environments simulate alongside it. The density rate
        # kernels sum the error of each environment, which is reset once read for the next call.
        n_envs_solving = 0
        for i_b in range(self._B):
            if self.envs_df[i_b].is_solving:
                err_tol = self.df_info[None].divergence_err_tol
                if qd.static(solve_density):
                    err_tol = self.df_info[None].density_err_tol
                if self.envs_df[i_b].err <= err_tol * self.envs_df[i_b].n_active:
                    self.envs_df[i_b].is_solving = False
                else:
                    qd.atomic_add(n_envs_solving, 1)
                self.envs_df[i_b].err = 0.0
        return n_envs_solving

    def _solve_incompressibility(self, solve_density: bool) -> None:
        if solve_density:
            kernel_compute_density_rate = self._kernel_compute_density_star
            n_iters_max = self._df_max_den_iters
        else:
            kernel_compute_density_rate = self._kernel_compute_density_time_derivative
            n_iters_max = self._df_max_div_iters

        # The convergence is checked before the first iteration: an environment that the warm start of the density
        # solve, or a divergence-free velocity field, already brings within tolerance does not iterate at all.
        if solve_density:
            self._kernel_solver_iteration(solve_density, warm_start=True)
        kernel_compute_density_rate()
        n_envs_solving = self._kernel_update_convergence(solve_density)
        n_iters = 0
        while n_envs_solving > 0 and n_iters < n_iters_max:
            self._kernel_solver_iteration(solve_density, warm_start=False)
            kernel_compute_density_rate()
            n_envs_solving = self._kernel_update_convergence(solve_density)
            n_iters += 1

        gs.logger.debug(f"DFSPH - {'density' if solve_density else 'divergence'} solve iterations: {n_iters}")

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def cubic_kernel(self, r_norm):
        """
        Cubic spline smoothing kernel.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q**2
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                res = 2 * k * (1.0 - q) ** 3
        return res

    @qd.func
    def cubic_kernel_derivative(self, r):
        """
        Derivative of cubic spline smoothing kernel.
        """
        res = qd.Vector.zero(gs.qd_float, 3)

        r_norm = r.norm()
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r_norm / h
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
            else:
                res = -6.0 * k * (1.0 - q) ** 2 * grad_q
        return res

    @qd.func
    def cubic_kernel_plane_integral(self, dist):
        """
        Integral of the cubic spline smoothing kernel over the plane at distance dist from its center.

        Closed form of 2 * pi * int_{|dist|}^{h} W(r) r dr. Its own integral along the plane normal equals
        half the kernel normalization: int_{0}^{h} of this quantity over dist is exactly 1/2, which makes
        forces weighted by it integrate to a per-unit-area magnitude over a covering layer of particles.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        q = qd.abs(dist) / h
        if q <= 1.0:
            q2 = q**2
            q3 = q2 * q
            if q <= 0.5:
                res = 0.0875 - (1.2 * q3 * q2 - 1.5 * q2**2 + 0.5 * q2)
            else:
                res = 0.1 - (q2 - 2.0 * q3 + 1.5 * q2**2 - 0.4 * q3 * q2)
            res = res * 16.0 / h
        return res

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self.is_active:
            self._kernel_reorder_particles(f)
            if self._pressure_solver == "WCSPH":
                self._kernel_compute_rho(f)
                # The boundary sweep weights its neighbours by their density, so it follows the density kernel.
                if self._has_boundary_particles:
                    self._kernel_update_boundary_state(f)
                if self._needs_stress:
                    self._kernel_compute_velocity_gradient(f)
                    self._kernel_update_stress(f)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_compute_eos_pressure(f)
                self._kernel_compute_stress_forces(f, self._boundary_friction)
                self._kernel_advect_velocity(f)
            elif self._pressure_solver == "DFSPH":
                self._kernel_compute_rho(f)
                self._kernel_compute_DFSPH_factor(f)
                self._solve_incompressibility(solve_density=False)
                self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
                self._kernel_predict_velocity(f)
                self._solve_incompressibility(solve_density=True)

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_advect_position(f)
            self._kernel_copy_from_reordered(f)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.L, state.sigma, state.eps_p, state.active)
            if self._pressure_solver == "DFSPH":
                self._kernel_set_pressure(state.pressure)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        L: qd.types.ndarray(),
        sigma: qd.types.ndarray(),
        eps_p: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
                for k in qd.static(range(3)):
                    self.particles[i_p, i_b].L[j, k] = L[i_b, i_p, j, k]
                    self.particles[i_p, i_b].sigma[j, k] = sigma[i_b, i_p, j, k]
            self.particles[i_p, i_b].eps_p = eps_p[i_b, i_p]
            self.particles_ng[i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.L, state.sigma, state.eps_p, state.active)
            if self._pressure_solver == "DFSPH":
                self._kernel_get_pressure(state.pressure)
        else:
            state = None
        return state

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        L: qd.types.ndarray(),
        sigma: qd.types.ndarray(),
        eps_p: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
                for k in qd.static(range(3)):
                    L[i_b, i_p, j, k] = self.particles[i_p, i_b].L[j, k]
                    sigma[i_b, i_p, j, k] = self.particles[i_p, i_b].sigma[j, k]
            eps_p[i_b, i_p] = self.particles[i_p, i_b].eps_p
            active[i_b, i_p] = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_set_pressure(self, pressure: qd.types.ndarray()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles[i_p, i_b].p = pressure[i_b, i_p]

    @qd.kernel
    def _kernel_get_pressure(self, pressure: qd.types.ndarray()):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            pressure[i_b, i_p] = self.particles[i_p, i_b].p

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles_total, self._B):
            is_rendered = self.particles_ng[i_p, i_b].active
            if qd.static(self._has_boundary_particles):
                is_rendered = is_rendered and not self.particles_info[i_p].is_boundary
            if is_rendered:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = is_rendered

    @qd.kernel
    def add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: float,
        mat_stiffness: float,
        mat_exponent: float,
        mat_mu: float,
        mat_gamma: float,
        mat_K0: float,
        mat_c_p: float,
        mat_idx: qd.i32,
        is_solid: qd.i32,
        pos: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].p = 0
            self.particles[i_p, i_b].L = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles[i_p, i_b].sigma = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles[i_p, i_b].eps_p = 0.0
            self.particles[i_p, i_b].vel_vir = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].sigma_vir = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles[i_p, i_b].vol_vir = self._particle_volume

        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            self.particles_info[i_p].rho = mat_rho
            self.particles_info[i_p].stiffness = mat_stiffness
            self.particles_info[i_p].exponent = mat_exponent
            self.particles_info[i_p].mu = mat_mu
            self.particles_info[i_p].gamma = mat_gamma
            self.particles_info[i_p].K0 = mat_K0
            self.particles_info[i_p].c_p = mat_c_p
            self.particles_info[i_p].material_idx = mat_idx
            self.particles_info[i_p].is_solid = qd.cast(is_solid, gs.qd_bool)
            self.particles_info[i_p].mass = self._particle_volume * mat_rho
            self.particles_info[i_p].is_boundary = False
            self.particles_info[i_p].boundary_normal = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_add_boundary_particles(
        self,
        f: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: qd.f32,
        pos: qd.types.ndarray(),
        normal: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = True
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].rho = mat_rho
            self.particles[i_p, i_b].vel_vir = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i_p, i_b].sigma_vir = qd.Matrix.zero(gs.qd_float, 3, 3)
            self.particles[i_p, i_b].vol_vir = self._particle_volume

        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            # Boundary particles hold no material state: they carry no mass and are not solid, which keeps them out of
            # the material kernels and out of the couplers. Their density is the rest density of a solid material, so
            # that a neighbour term dividing by it stays finite.
            self.particles_info[i_p].rho = mat_rho
            self.particles_info[i_p].mass = 0.0
            self.particles_info[i_p].is_solid = False
            self.particles_info[i_p].is_boundary = True
            # The material index names no registered material, so a consumer keying on it cannot take a wall row for a
            # simulated particle.
            self.particles_info[i_p].material_idx = -1
            for i in qd.static(range(3)):
                self.particles_info[i_p].boundary_normal[i] = normal[i_p_, i]

    # ----------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index')
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
                self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)
                # The pressure that DFSPH carries over belongs to the neighborhood the particle leaves.
                self.particles[i_p, i_b].p = 0.0

    @qd.kernel
    def _kernel_get_particles_pos(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                for i in qd.static(range(3)):
                    self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
                self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_get_particles_vel(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            # See '_kernel_set_particles_pos'
            if i_p >= 0:
                self.particles_ng[i_p, i_b].active = actives[i_b_, i_p_]
                self.particles[i_p, i_b].p = 0.0

    @qd.kernel
    def _kernel_get_particles_active(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_get_particles_mass(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), masses: qd.types.ndarray()
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            masses[i_b_, i_p_] = self.particles_info[i_p].mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def n_particles_total(self):
        """Number of particle rows in the solver arrays, the boundary particles included."""
        if self.is_built:
            return self._n_particles_total
        else:
            return sum([entity.n_particles for entity in self._entities])

    @property
    def pressure_solver(self) -> str:
        return self._pressure_solver

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
