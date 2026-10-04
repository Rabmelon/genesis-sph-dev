# Tests for SPH simulation - initial density and pressure validation
#
# Background:
# SPH is sensitive to the initial particle distribution, as it directly determines the initial density and pressure fields.
# To ensure numerical stability, particles must be initialized using a regular sampler that enforces near-uniform spacing.
# Irregular samplers (e.g. pbs, random) introduce local density fluctuations at initialization, which lead to large
# spurious pressure forces and can cause the simulation to become unstable or diverge.

import pytest
import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils.assertions import assert_allclose


def create_sph_scene(show_viewer, particle_size=0.01, pressure_solver="WCSPH"):
    """Create a basic SPH scene with regular sampler."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            pressure_solver=pressure_solver,
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    scene.build()

    return scene, liquid


@pytest.fixture(scope="function")
def sph_scene(show_viewer):
    """Fixture to create a basic SPH scene with regular sampler."""
    return create_sph_scene(show_viewer)


def get_sph_density_pressure(scene, n_particles):
    """
    Helper function to compute and retrieve density and pressure from SPH solver.

    This triggers the density computation kernel and returns the computed values.

    Parameters
    ----------
    scene : gs.Scene
        The built scene containing SPH solver.
    n_particles : int
        Number of particles to retrieve.

    Returns
    -------
    densities : np.ndarray
        Array of density values for all particles.
    pressures : np.ndarray
        Array of pressure values for all particles.
    """
    sph_solver = scene.sph_solver

    # Reorder particles into spatial grid cells for efficient neighbor queries,
    # then compute density. This mirrors what happens in substep_pre_coupling.
    # We must reorder first because _kernel_compute_rho operates on particles_reordered.
    sph_solver._kernel_reorder_particles(0)
    sph_solver._kernel_compute_rho(0)

    # Get density values from reordered particles (where _kernel_compute_rho stores results)
    densities = sph_solver.particles_reordered.rho.to_numpy()[:n_particles, 0]

    # Get material properties for pressure computation
    rho0 = sph_solver.particles_info[0].rho
    stiffness = sph_solver.particles_info[0].stiffness
    exponent = sph_solver.particles_info[0].exponent

    # Compute pressure using the same formula as the solver:
    # p = stiffness * ((max(rho, rho0) / rho0)^exponent - 1)
    densities_clamped = np.maximum(densities, rho0)
    pressures = stiffness * (np.power(densities_clamped / rho0, exponent) - 1.0)

    return densities, pressures


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_initial_density_regular_sampler(show_viewer, pressure_solver, tol):
    # A regular sampler places particles at uniform spacing: interior particles see a consistent near-rest
    # density while boundary particles sit lower (fewer neighbors inside the kernel support), so the mean lands
    # below the rest density with a bounded spread.
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho

    # Get initial density values
    densities, _ = get_sph_density_pressure(scene, n_particles)

    # Test 1: Mean density should be within reasonable bounds
    # For regular sampler, mean is typically 70-90% of rho0 due to boundary effects
    density_mean = densities.mean()
    assert 0.5 * rho0 < density_mean < rho0, (
        f"Mean initial density {density_mean:.2f} is outside expected range "
        f"[{0.5 * rho0:.2f}, {rho0:.2f}] for rest density {rho0}"
    )

    # Test 2: Density standard deviation should be bounded
    # Regular sampler produces relatively uniform density distribution
    density_std = densities.std()
    assert density_std < 0.1 * rho0, (
        f"Density standard deviation {density_std:.2f} is too high "
        f"(expected < {0.1 * rho0:.2f} for rest density {rho0})"
    )

    # Test 3: No extreme density values
    assert densities.min() > 0.3 * rho0, (
        f"Minimum density {densities.min():.2f} is too low (expected > {0.3 * rho0:.2f})"
    )
    assert densities.max() < 1.5 * rho0, (
        f"Maximum density {densities.max():.2f} is too high (expected < {1.5 * rho0:.2f})"
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_initial_pressure_regular_sampler(show_viewer, pressure_solver, tol):
    # Regular sampling yields densities at or below the rest density, and the equation of state clamps density
    # to the rest density, so the initial pressure must be exactly zero. A nonzero initial pressure would inject
    # spurious forces that destabilize the simulation.
    scene, _ = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho
    stiffness = sph_solver.particles_info[0].stiffness

    # Get initial pressure values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Test: For regular sampler, initial density should be <= rho0
    # which means pressure should be 0 (clamped)
    # All particles should have near-zero pressure initially
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero. Mean: {pressures.mean():.6f}, "
        f"Max: {pressures.max():.6f}. "
        f"This indicates density > rho0 for some particles, which shouldn't happen "
        f"with regular sampler."
    )

    # Also verify that density is indeed <= rho0 for most particles
    fraction_above_rho0 = (densities > rho0).sum() / n_particles
    assert fraction_above_rho0 < 0.01, (
        f"{fraction_above_rho0 * 100:.1f}% of particles have density > rho0. "
        f"Regular sampler should produce density <= rho0 for almost all particles."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_simulation_stability_regular_sampler(show_viewer, pressure_solver, tol):
    scene, liquid = create_sph_scene(show_viewer, pressure_solver=pressure_solver)
    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles

    # Get initial positions for comparison
    initial_pos = liquid.get_particles_pos().clone()

    # Run simulation for a few steps
    num_steps = 5
    max_velocities = []
    for step in range(num_steps):
        scene.step()
        pos = liquid.get_particles_pos()
        vel = liquid.get_particles_vel()
        assert not pos.isnan().any(), f"NaN values detected in positions at step {step}"
        assert not pos.isinf().any(), f"Inf values detected in positions at step {step}"
        assert not vel.isnan().any(), f"NaN values detected in velocities at step {step}"
        assert not vel.isinf().any(), f"Inf values detected in velocities at step {step}"

        # Track maximum velocity
        max_vel = vel.norm(dim=-1).max()
        max_velocities.append(max_vel)

    # Maximum velocity should be bounded (not exploding)
    # For stable simulation with gravity, velocities should be reasonable
    max_velocity_overall = max(max_velocities)
    assert max_velocity_overall < 10.0, (
        f"Maximum velocity {max_velocity_overall:.2f} exceeds reasonable bounds. This may indicate instability."
    )

    # Get final positions
    final_pos = liquid.get_particles_pos()

    # Check particles haven't moved too far (not exploding)
    displacement = (final_pos - initial_pos).norm(dim=-1).max()
    assert displacement < 1.0, (
        f"Maximum displacement {displacement:.4f} is too large after {num_steps} steps. This may indicate instability."
    )


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
@pytest.mark.parametrize("particle_size", [0.01, 0.02])
def test_density_consistency_different_particle_sizes(show_viewer, pressure_solver, particle_size, tol):
    # The SPH kernel support scales with the particle size, so the density distribution normalized by the rest
    # density must be independent of the particle size.
    scene, _ = create_sph_scene(
        show_viewer,
        particle_size=particle_size,
        pressure_solver=pressure_solver,
    )

    sph_solver = scene.sph_solver
    n_particles = sph_solver.n_particles
    rho0 = sph_solver.particles_info[0].rho

    # Get density values
    densities, pressures = get_sph_density_pressure(scene, n_particles)

    # Normalized metrics should be similar regardless of particle size
    normalized_mean = densities.mean() / rho0
    normalized_std = densities.std() / rho0

    # Mean density should be in similar range (50-100% of rho0)
    assert 0.5 < normalized_mean < 1.0, (
        f"Normalized mean density {normalized_mean:.3f} is outside expected range [0.5, 1.0] "
        f"for particle_size={particle_size}"
    )

    # Standard deviation should be bounded
    assert normalized_std < 0.1, (
        f"Normalized density std {normalized_std:.3f} is too high for particle_size={particle_size}"
    )

    # Pressure should still be zero
    assert np.allclose(pressures, 0.0, atol=1e-6), (
        f"Initial pressure is not zero for particle_size={particle_size}. Mean: {pressures.mean():.6f}"
    )


# =============================================================================
# DFSPH (Divergence-Free SPH) Pressure Solver Tests
# =============================================================================


# TODO: Add more tests for validating the underlying physical behavior of DFSPH.
@pytest.mark.required
def test_dfsph_simulation_builds_and_runs(show_viewer):
    scene, liquid = create_sph_scene(show_viewer, pressure_solver="DFSPH")

    assert scene.sph_solver is not None
    assert scene.sph_solver.n_particles > 0

    # Run a few simulation steps
    for _ in range(3):
        scene.step()

    pos = liquid.get_particles_pos()
    assert pos.shape[0] == scene.sph_solver.n_particles


@pytest.mark.required
@pytest.mark.parametrize("pressure_solver", ["WCSPH", "DFSPH"])
def test_pressure_carries_column_weight(show_viewer, pressure_solver):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            pressure_solver=pressure_solver,
            lower_bound=(-0.15, -0.15, 0.0),
            upper_bound=(0.15, 0.15, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, 0.0, 0.5),
            camera_lookat=(0.0, 0.0, 0.25),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.3, 0.3, 0.5),
        ),
        material=gs.materials.SPH.Liquid(
            sampler="regular",
        ),
    )
    scene.build()

    for _ in range(100):
        scene.step()

    sph_solver = scene.sph_solver
    pressures = qd_to_numpy(sph_solver.particles_reordered.p, transpose=True)[0]
    heights = qd_to_numpy(sph_solver.particles_reordered.pos, transpose=True)[0, :, 2]

    # The pressure at the bottom of a settled column carries the weight of the whole column, which is exactly
    # the particle count times the particle weight spread over the column footprint. The tolerance absorbs the
    # residual sloshing of the column and two boundary effects that pull in opposite directions: the wall
    # pressure force from the floor relieves part of the compression that the WCSPH equation of state reads,
    # while the positive-part clamp of the DFSPH density solve rectifies velocity noise into extra pressure.
    weight_per_area = sph_solver.n_particles * sph_solver.particles_info[0].mass * 9.81 / (0.3 * 0.3)
    p_bottom = pressures[heights < 0.05].mean()
    assert 0.4 * weight_per_area < p_bottom < 1.5 * weight_per_area

    # The pressure must decrease with height, following the hydrostatic gradient. The factors are lower bounds
    # on the ratio of the hydrostatic depths at the band centers, with slack for pressure fluctuations.
    p_middle = pressures[(heights >= 0.2) & (heights < 0.25)].mean()
    p_top = pressures[heights >= 0.3].mean()
    assert p_bottom > 1.5 * p_middle
    assert p_middle > 2.0 * p_top


# =============================================================================
# Solid (constitutive stress) SPH Tests
# =============================================================================


@pytest.mark.required
def test_rigid_rotation_keeps_stress_objective(show_viewer):
    # A rigid rotation carries no strain rate, so the elastic trial adds nothing and the Jaumann term alone turns the
    # stress the body already carries. A wrong sign there leaves the stress fixed in the laboratory frame and a
    # missing one lets the stress turn with the body only through the trial, so the two are told apart by which of the
    # two rotations of the initial stress the final stress matches, and by the invariants that neither may change.
    #
    # The constitutive sweep runs alone on a frozen rigid velocity field, which isolates the velocity gradient and the
    # stress update from the momentum equation. The zero-stress phase reads the kernel-complete particles alone: the
    # free-surface truncation of the kernel gradient gives a surface particle a spurious strain rate, whose
    # accumulation then hides the floating-point noise floor the check is after.
    box_center = np.array([0.0, 0.0, 0.5])
    box_size = 0.2
    pressure_0 = 1e6
    omega = 10.0
    n_substeps = 400

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.SPH.Elastic(
        rho=1600.0,
        E=1e6,
        nu=0.3,
        stress_diffusion=0.0,
        sampler="regular",
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=box_center,
            size=(box_size,) * 3,
        ),
        material=material,
    )
    scene.build()
    solver = scene.sph_solver

    # Rigid rotation about x at omega, v = omega x (pos - center)
    pos = cube.get_particles_pos()
    center = pos.mean(dim=0)
    vel = torch.zeros_like(pos)
    vel[:, 1] = -omega * (pos[:, 2] - center[2])
    vel[:, 2] = omega * (pos[:, 1] - center[1])
    cube.set_particles_vel(vel)

    for _ in range(n_substeps):
        solver._kernel_reorder_particles(0)
        solver._kernel_compute_rho(0)
        solver._kernel_compute_velocity_gradient(0)
        solver._kernel_update_stress(0)
        solver._kernel_copy_from_reordered(0)

    # The rigid rotation must leave the velocity field alone
    expected_vel = torch.zeros_like(pos)
    expected_vel[:, 1] = -omega * (pos[:, 2] - center[2])
    expected_vel[:, 2] = omega * (pos[:, 1] - center[1])
    assert_allclose(cube.get_particles_vel(), expected_vel, atol=1e-6)

    poss_0 = tensor_to_array(pos)
    half_box = np.full(3, 0.5 * box_size)
    kernel_complete = np.minimum(poss_0 - (box_center - half_box), (box_center + half_box) - poss_0)
    kernel_complete = kernel_complete.min(axis=1) >= solver.support_radius

    sigma_before = qd_to_numpy(solver.particles.sigma, transpose=True)[0]
    assert np.abs(sigma_before[kernel_complete]).max() < 1.0

    # K0 geostatic pre-stress, the stress state a settled soil column carries, then the same rotation
    sigma_0 = np.diag([-material.K0 * pressure_0, -material.K0 * pressure_0, -pressure_0]).astype(gs.np_float)
    solver.particles.sigma.from_numpy(np.broadcast_to(sigma_0, qd_to_numpy(solver.particles.sigma).shape).copy())

    for _ in range(n_substeps):
        solver._kernel_reorder_particles(0)
        solver._kernel_compute_rho(0)
        solver._kernel_compute_velocity_gradient(0)
        solver._kernel_update_stress(0)
        solver._kernel_copy_from_reordered(0)

    theta = n_substeps * solver.substep_dt * omega
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_plus = np.array([[1.0, 0.0, 0.0], [0.0, cos_theta, -sin_theta], [0.0, sin_theta, cos_theta]])
    rot_minus = np.array([[1.0, 0.0, 0.0], [0.0, cos_theta, sin_theta], [0.0, -sin_theta, cos_theta]])
    sigma_n = qd_to_numpy(solver.particles.sigma, transpose=True)[0]
    sigma_rotated = sigma_n.mean(axis=0)

    i1_0 = np.trace(sigma_0)
    sqrt_j2_0 = np.linalg.norm(sigma_0 - i1_0 / 3.0 * np.eye(3)) / np.sqrt(2.0)
    i1_rotated = np.trace(sigma_rotated)
    deviator_rotated = sigma_n - np.trace(sigma_n, axis1=-2, axis2=-1)[:, None, None] / 3.0 * np.eye(3)
    sqrt_j2_rotated = np.linalg.norm(deviator_rotated[kernel_complete], axis=(-2, -1)).mean() / np.sqrt(2.0)
    assert abs(i1_rotated - i1_0) < 1e-4 * abs(i1_0)
    assert abs(sqrt_j2_rotated - sqrt_j2_0) < 0.01 * sqrt_j2_0

    # The stress must follow the rotation of the body, both in magnitude and in the sign of its shear component
    sigma_norm = np.linalg.norm(sigma_0)
    residual = np.linalg.norm(sigma_rotated - rot_plus @ sigma_0 @ rot_plus.T) / sigma_norm
    residual_reversed = np.linalg.norm(sigma_rotated - rot_minus @ sigma_0 @ rot_minus.T) / sigma_norm
    assert residual < 0.15
    assert residual < 0.25 * residual_reversed
    assert sigma_rotated[1, 2] > 0.5 * (rot_plus @ sigma_0 @ rot_plus.T)[1, 2]


@pytest.mark.required
def test_uniaxial_strain_matches_elastic_modulus_and_poisson_ratio(show_viewer):
    # A pure uniaxial strain rate is the one loading whose elastic response is exactly known: the axial stress is the
    # constrained modulus K + 4G/3 times the axial strain rate, and the lateral-to-axial stress ratio is nu / (1 - nu).
    # Both are read from the stress the constitutive update produced for the strain rate the solver measured, so the
    # kernel-complete particles alone are trusted: a truncated stencil misses part of the strain rate and would make
    # the material look softer than it is.
    box_center = np.array([0.0, 0.0, 0.2])
    box_size = 0.16
    strain_rate = -1e-3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.SPH.Elastic(
        rho=1600.0,
        E=1e6,
        nu=0.3,
        stress_diffusion=0.0,
        sampler="regular",
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=box_center,
            size=(box_size,) * 3,
        ),
        material=material,
    )
    scene.build()
    solver = scene.sph_solver

    # v_z = strain_rate * (z - z_center), no lateral motion
    pos = cube.get_particles_pos()
    vel = torch.zeros_like(pos)
    vel[:, 2] = strain_rate * (pos[:, 2] - pos[:, 2].mean())
    cube.set_particles_vel(vel)

    solver._kernel_reorder_particles(0)
    solver._kernel_compute_rho(0)
    solver._kernel_compute_velocity_gradient(0)
    solver._kernel_update_stress(0)
    solver._kernel_copy_from_reordered(0)

    poss_0 = tensor_to_array(pos)
    half_box = np.full(3, 0.5 * box_size)
    kernel_complete = np.minimum(poss_0 - (box_center - half_box), (box_center + half_box) - poss_0)
    kernel_complete = kernel_complete.min(axis=1) >= solver.support_radius
    assert kernel_complete.any()

    strain_rate_zz = qd_to_numpy(solver.particles.L, transpose=True)[0][kernel_complete, 2, 2].mean()
    sigma = qd_to_numpy(solver.particles.sigma, transpose=True)[0][kernel_complete]
    constrained_modulus = sigma[:, 2, 2].mean() / (solver.substep_dt * strain_rate_zz)
    lateral_ratio = 0.5 * (sigma[:, 0, 0].mean() + sigma[:, 1, 1].mean()) / sigma[:, 2, 2].mean()
    poisson_ratio = lateral_ratio / (1.0 + lateral_ratio)
    young_modulus = constrained_modulus * (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio) / (1.0 - poisson_ratio)

    assert abs(poisson_ratio - material.nu) < 0.1 * material.nu
    assert abs(young_modulus - material.E) < 0.1 * material.E


@pytest.mark.required
def test_elastic_wave_front_speed_matches_p_wave_speed(show_viewer):
    # A piston holds the bottom of a bar at a constant axial velocity, which launches a step wave whose front travels
    # at the P-wave speed sqrt((K + 4G/3) / rho). The front is timed where the axial velocity of a slab reaches half
    # the piston velocity, and only the slabs between the piston and the free end are fitted, so neither the source
    # nor the reflected wave shapes the arrival times.
    particle_size = 0.02
    bar_size = np.array([0.16, 0.16, 0.4])
    bar_center = np.array([0.0, 0.0, 0.2])
    piston_velocity = 0.1
    n_steps = 140
    n_sample = 2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.SPH.Elastic(
        rho=1600.0,
        E=1e6,
        nu=0.3,
        stress_diffusion=0.0,
        sampler="regular",
    )
    bar = scene.add_entity(
        morph=gs.morphs.Box(
            pos=bar_center,
            size=bar_size,
        ),
        material=material,
    )
    scene.build()
    solver = scene.sph_solver

    poss_0 = tensor_to_array(bar.get_particles_pos())
    piston = np.flatnonzero(poss_0[:, 2] < 2 * particle_size)
    piston_vel = torch.zeros((piston.size, 3), dtype=gs.tc_float, device=gs.device)
    piston_vel[:, 2] = piston_velocity

    times, profiles = [], []
    for i_step in range(n_steps):
        bar.set_particles_vel(piston_vel, particles_idx_local=piston)
        scene.step()
        if (i_step + 1) % n_sample == 0:
            times.append((i_step + 1) * solver.substep_dt)
            profiles.append(tensor_to_array(bar.get_particles_vel()))
    times = np.array(times)
    profiles = np.stack(profiles)

    band_edges = np.arange(0.0, bar_size[2] + 1e-9, particle_size)
    band_velocity = np.full((len(times), band_edges.size - 1), np.nan)
    for i_band in range(band_edges.size - 1):
        in_band = (poss_0[:, 2] >= band_edges[i_band]) & (poss_0[:, 2] < band_edges[i_band + 1])
        band_velocity[:, i_band] = profiles[:, in_band, 2].mean(axis=1)

    band_centers = 0.5 * (band_edges[:-1] + band_edges[1:])
    target = 0.5 * piston_velocity
    arrivals = np.full(band_centers.size, np.nan)
    for i_band in range(band_centers.size):
        above = np.flatnonzero(band_velocity[:, i_band] >= target)
        if above.size == 0:
            continue
        i_first = above[0]
        if i_first > 0:
            t_prev, v_prev = times[i_first - 1], band_velocity[i_first - 1, i_band]
            arrivals[i_band] = t_prev + (target - v_prev) / (band_velocity[i_first, i_band] - v_prev) * (
                times[i_first] - t_prev
            )
        else:
            arrivals[i_band] = times[0]

    timed = np.isfinite(arrivals) & (band_centers >= 0.09) & (band_centers <= 0.35)
    assert timed.sum() >= 5
    front_speed = np.polyfit(arrivals[timed], band_centers[timed], 1)[0]

    bulk_modulus = material.E / (3.0 * (1.0 - 2.0 * material.nu))
    shear_modulus = material.E / (2.0 * (1.0 + material.nu))
    p_wave_speed = np.sqrt((bulk_modulus + 4.0 * shear_modulus / 3.0) / material.rho)
    assert 0.85 * p_wave_speed < front_speed < 1.15 * p_wave_speed


@pytest.mark.required
def test_stress_diffusion_relaxes_stress_discontinuity(show_viewer):
    # The stress diffusion is a Laplacian-like smoothing of the stress field, so an imposed normal-stress
    # discontinuity relaxes under it: the mean jump across the box midplane shrinks monotonically, a sign flip
    # reverses that, and a removed term freezes the jump. The reference entity keeps the diffusion coefficient at
    # zero, where the zero velocity field leaves the elastic trial with nothing to add, so any change there would
    # come from outside the diffusion term.
    box_size = 0.08
    amplitude = 1e4
    n_substeps = 20

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    diffused = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.15, 0.0, 0.2),
            size=(box_size,) * 3,
        ),
        material=gs.materials.SPH.Elastic(
            rho=1600.0,
            E=1e6,
            nu=0.3,
            stress_diffusion=0.1,
            sampler="regular",
        ),
    )
    frozen = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.15, 0.0, 0.2),
            size=(box_size,) * 3,
        ),
        material=gs.materials.SPH.Elastic(
            rho=1600.0,
            E=1e6,
            nu=0.3,
            stress_diffusion=0.0,
            sampler="regular",
        ),
    )
    scene.build()
    solver = scene.sph_solver

    # Tensile sigma_xx on the left half of each box, compressive on the right half
    n_diffused = diffused.n_particles
    pos_diffused = tensor_to_array(diffused.get_particles_pos())
    pos_frozen = tensor_to_array(frozen.get_particles_pos())
    is_left_diffused = pos_diffused[:, 0] < pos_diffused[:, 0].mean()
    is_left_frozen = pos_frozen[:, 0] < pos_frozen[:, 0].mean()
    sigma = np.zeros(qd_to_numpy(solver.particles.sigma).shape, dtype=gs.np_float)
    sigma[:n_diffused, :, 0, 0] = np.where(is_left_diffused, amplitude, -amplitude)[:, None]
    sigma[n_diffused:, :, 0, 0] = np.where(is_left_frozen, amplitude, -amplitude)[:, None]
    solver.particles.sigma.from_numpy(sigma)

    diffused_jumps, frozen_jumps = [], []
    for i_substep in range(n_substeps + 1):
        if i_substep > 0:
            solver._kernel_reorder_particles(0)
            solver._kernel_compute_rho(0)
            solver._kernel_compute_velocity_gradient(0)
            solver._kernel_update_stress(0)
            solver._kernel_copy_from_reordered(0)
        sigma_xx = qd_to_numpy(solver.particles.sigma, transpose=True)[0, :, 0, 0]
        diffused_xx = sigma_xx[:n_diffused]
        frozen_xx = sigma_xx[n_diffused:]
        diffused_jumps.append(diffused_xx[is_left_diffused].mean() - diffused_xx[~is_left_diffused].mean())
        frozen_jumps.append(frozen_xx[is_left_frozen].mean() - frozen_xx[~is_left_frozen].mean())

    assert np.all(np.diff(diffused_jumps) <= 0.0)
    assert diffused_jumps[-1] < 0.9 * diffused_jumps[0]
    assert_allclose(frozen_jumps[-1], frozen_jumps[0], atol=1e-6)


@pytest.mark.required
def test_drucker_prager_projects_stress_onto_yield_surface(show_viewer):
    # The return mapping has three outcomes for a stress state outside the cone: an elastic one when the state is
    # inside, a radial scaling of the deviator onto the cone, and the apex projection that caps the mean stress when
    # the cohesion has nothing left to hold. A cohesionless material at the apex has no tensile strength at all, and a
    # cohesive one is left with the uniaxial tensile strength cohesion / tan(phi) over the whole state.
    box_center = np.array([0.0, 0.0, 0.2])
    box_size = 0.08
    friction_angle = 30.0
    cohesion = 2e4
    confined_state = np.array([[-5e4, 3e4, 0.0], [3e4, -5e4, 0.0], [0.0, 0.0, -5e4]], dtype=gs.np_float)
    tensile_state = np.eye(3, dtype=gs.np_float) * 1e5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    granular = scene.add_entity(
        morph=gs.morphs.Box(
            pos=box_center,
            size=(box_size,) * 3,
        ),
        material=gs.materials.SPH.DruckerPrager(
            rho=1600.0,
            E=1e6,
            nu=0.3,
            friction_angle=friction_angle,
            cohesion=0.0,
            stress_diffusion=0.0,
        ),
    )
    cemented = scene.add_entity(
        morph=gs.morphs.Box(
            pos=box_center + np.array([0.15, 0.0, 0.0]),
            size=(box_size,) * 3,
        ),
        material=gs.materials.SPH.DruckerPrager(
            rho=1600.0,
            E=1e6,
            nu=0.3,
            friction_angle=friction_angle,
            cohesion=cohesion,
            stress_diffusion=0.0,
        ),
    )
    scene.build()
    solver = scene.sph_solver

    n_granular = granular.n_particles
    granular_slice = slice(0, n_granular)
    cemented_slice = slice(n_granular, n_granular + cemented.n_particles)

    # A confined state past the cone of the cohesionless material must be scaled onto the surface, which the plane
    # strain cone of the friction angle defines
    solver.particles.sigma.from_numpy(np.broadcast_to(confined_state, qd_to_numpy(solver.particles.sigma).shape).copy())
    solver._kernel_reorder_particles(0)
    solver._kernel_compute_rho(0)
    solver._kernel_compute_velocity_gradient(0)
    solver._kernel_update_stress(0)
    solver._kernel_copy_from_reordered(0)
    sigma = qd_to_numpy(solver.particles.sigma, transpose=True)[0]

    tan_phi = np.tan(np.radians(friction_angle))
    cone_denominator = np.sqrt(9.0 + 12.0 * tan_phi**2)
    mean_stress = np.trace(sigma[granular_slice], axis1=-2, axis2=-1) / 3.0
    deviator = sigma[granular_slice] - mean_stress[:, None, None] * np.eye(3)
    sqrt_j2 = np.linalg.norm(deviator, axis=(-2, -1)) / np.sqrt(2.0)
    yield_function = sqrt_j2 + 3.0 * tan_phi / cone_denominator * mean_stress
    assert np.abs(yield_function).max() < 0.1

    # The cemented material is still inside its cone, so its stress is left untouched
    assert_allclose(sigma[cemented_slice], confined_state, atol=1.0)

    # A tensile state is projected onto the apex of each material
    solver.particles.sigma.from_numpy(np.broadcast_to(tensile_state, qd_to_numpy(solver.particles.sigma).shape).copy())
    solver._kernel_reorder_particles(0)
    solver._kernel_compute_rho(0)
    solver._kernel_compute_velocity_gradient(0)
    solver._kernel_update_stress(0)
    solver._kernel_copy_from_reordered(0)
    sigma = qd_to_numpy(solver.particles.sigma, transpose=True)[0]
    assert_allclose(sigma[granular_slice], 0.0, atol=1e-6)
    tensile_strength = cohesion / tan_phi
    assert_allclose(sigma[cemented_slice], np.eye(3) * tensile_strength, atol=1e-3 * tensile_strength)


@pytest.mark.required
def test_drucker_prager_dilation_angle_drives_volumetric_plastic_strain(show_viewer):
    # A stress state past the cone is projected back with the non-associated flow rule, whose consistency parameter
    # carries the dilation angle. For a trial state with mean stress p, deviator s, sqrt(J2) = |s| / sqrt(2) and
    # yield function f, the return mapping is
    #   d_lambda = f / (G + 9 K alpha_phi alpha_psi),   s_new = (1 - d_lambda G / sqrt(J2)) s,
    #   p_new = p - 3 K alpha_psi d_lambda,
    # which lands exactly on the surface and gives the plastic strain increments
    #   d_eps_p_vol = tr(sigma_trial - sigma_new) / (3 K) = 3 alpha_psi d_lambda,
    #   d_eps_p_eq = sqrt(2/3) |dev(sigma_trial - sigma_new)| / (2 G) = d_lambda / sqrt(3),
    # so their ratio is 3 sqrt(3) alpha_psi. A shear-driven plastic step therefore dilates, and the equivalent
    # plastic strain is what the solver accumulates into eps_p.
    friction_angle = 30.0
    dilation_angle = 20.0
    initial_state = np.array([[-1e5, 1e5, 0.0], [1e5, -1e5, 0.0], [0.0, 0.0, -1e5]], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    material = gs.materials.SPH.DruckerPrager(
        rho=1600.0,
        E=1e6,
        nu=0.3,
        friction_angle=friction_angle,
        cohesion=0.0,
        dilation_angle=dilation_angle,
        stress_diffusion=0.0,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.08,) * 3,
        ),
        material=material,
    )
    scene.build()
    solver = scene.sph_solver

    solver.particles.sigma.from_numpy(np.broadcast_to(initial_state, qd_to_numpy(solver.particles.sigma).shape).copy())
    solver._kernel_reorder_particles(0)
    solver._kernel_compute_rho(0)
    solver._kernel_compute_velocity_gradient(0)
    solver._kernel_update_stress(0)
    solver._kernel_copy_from_reordered(0)

    # Every particle carries the same uniform trial state, so the response is read as a mean
    sigma = qd_to_numpy(solver.particles.sigma, transpose=True)[0].mean(axis=0)
    eps_p = qd_to_numpy(solver.particles.eps_p, transpose=True)[0].mean()

    tan_phi = np.tan(np.radians(friction_angle))
    cone_denominator = np.sqrt(9.0 + 12.0 * tan_phi**2)
    alpha_phi = tan_phi / cone_denominator
    alpha_psi = np.tan(np.radians(dilation_angle)) / cone_denominator
    bulk_modulus = material.E / (3.0 * (1.0 - 2.0 * material.nu))
    shear_modulus = material.E / (2.0 * (1.0 + material.nu))

    mean_trial = np.trace(initial_state) / 3.0
    deviator_trial = initial_state - mean_trial * np.eye(3)
    sqrt_j2_trial = np.linalg.norm(deviator_trial) / np.sqrt(2.0)
    yield_trial = sqrt_j2_trial + 3.0 * alpha_phi * mean_trial
    consistency = yield_trial / (shear_modulus + 9.0 * bulk_modulus * alpha_phi * alpha_psi)
    assert yield_trial > 0.0

    # The projection lands on the yield surface
    mean_after = np.trace(sigma) / 3.0
    deviator_after = sigma - mean_after * np.eye(3)
    yield_after = np.linalg.norm(deviator_after) / np.sqrt(2.0) + 3.0 * alpha_phi * mean_after
    assert abs(yield_after) < 0.1

    # The plastic volumetric strain is positive, and it stands to the accumulated equivalent plastic strain in the
    # 3 sqrt(3) alpha_psi ratio the flow rule prescribes
    eps_p_vol = np.trace(initial_state - sigma) / (3.0 * bulk_modulus)
    assert eps_p > 0.0
    assert_allclose(eps_p, consistency / np.sqrt(3.0), tol=1e-4)
    assert eps_p_vol > 0.0
    assert_allclose(eps_p_vol, 3.0 * alpha_psi * consistency, tol=1e-4)
    assert_allclose(eps_p_vol / eps_p, 3.0 * np.sqrt(3.0) * alpha_psi, tol=1e-4)


@pytest.mark.required
def test_soil_and_water_share_one_particle_set(show_viewer):
    # One particle set holding two materials: the fluid particles keep the equation of state and the solid ones carry
    # their own constitutive stress, and neither arm may take over the other's particles. The run is long enough for
    # the water to build a pressure field, which is what makes the fluid stress check meaningful.
    soil_start, soil_size = (-0.15, 0.0, 0.05), (0.1, 0.1, 0.1)
    water_start, water_size = (0.15, 0.0, 0.05), (0.1, 0.1, 0.1)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=10,
            gravity=(0.0, 0.0, -9.81),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.5, 0.4),
            camera_lookat=(0.0, 0.0, 0.05),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    soil = scene.add_entity(
        morph=gs.morphs.Box(
            pos=soil_start,
            size=soil_size,
        ),
        material=gs.materials.SPH.DruckerPrager(
            rho=1600.0,
            E=1e6,
            nu=0.3,
            friction_angle=30.0,
            cohesion=0.0,
        ),
    )
    water = scene.add_entity(
        morph=gs.morphs.Box(
            pos=water_start,
            size=water_size,
        ),
        material=gs.materials.SPH.Liquid(
            rho=1000.0,
            sampler="regular",
        ),
    )
    scene.build()

    for _ in range(250):
        scene.step()

    solver = scene.sph_solver
    soil_slice = slice(0, soil.n_particles)
    water_slice = slice(soil.n_particles, soil.n_particles + water.n_particles)
    is_solid = qd_to_numpy(solver.particles_info.is_solid)
    density = qd_to_numpy(solver.particles_info.rho)
    pressure = qd_to_numpy(solver.particles.p, transpose=True)[0]
    stress = qd_to_numpy(solver.particles.sigma, transpose=True)[0]
    velocity = qd_to_numpy(solver.particles.vel, transpose=True)[0]
    plastic_strain = qd_to_numpy(solver.particles.eps_p, transpose=True)[0]
    position = qd_to_numpy(solver.particles.pos, transpose=True)[0]

    assert is_solid[soil_slice].all()
    assert not is_solid[water_slice].any()
    assert_allclose(density[soil_slice], 1600.0, atol=1e-4)
    assert_allclose(density[water_slice], 1000.0, atol=1e-4)
    assert np.isfinite(stress).all() and np.isfinite(velocity).all()
    assert np.isfinite(pressure).all() and np.isfinite(position).all() and np.isfinite(plastic_strain).all()

    # Only the fluid particles read the equation of state, and only they carry it as their stress. Solid particles keep
    # the hard zero the solver writes into the pressure field.
    assert np.abs(pressure[soil_slice]).max() <= 0.0
    assert np.abs(pressure[water_slice]).max() > 0.0
    assert_allclose(stress[water_slice], -pressure[water_slice][:, None, None] * np.eye(3), atol=1e-12)

    # Each material keeps its own mass, which the entity reports from its own particles
    assert_allclose(soil.get_mass(), solver.particle_volume * 1600.0 * soil.n_particles, tol=1e-6)
    assert_allclose(water.get_mass(), solver.particle_volume * 1000.0 * water.n_particles, tol=1e-6)
    mass_first = soil.get_mass()
    mass_second = soil.get_mass()
    assert_allclose(mass_first, mass_second, atol=gs.EPS)
