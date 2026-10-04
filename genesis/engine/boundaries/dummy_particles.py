import numpy as np

import genesis as gs


def generate_dummy_particles(lower, upper, spacing, support_radius):
    """Generate the boundary particle lattice that covers the six faces of the simulation domain.

    The lattice holds one layer every `spacing`, the first half a spacing outside the face and the last one reaching
    past the support radius. Face priority follows the axes, x then y then z: a face spans the extended range along
    every axis whose own faces come later, and the domain range along the axes whose faces come earlier, so edges and
    corners carry exactly one particle each, named after the face that owns them.

    Parameters
    ----------
    lower : array_like, shape (3,)
        Lower corner of the simulation domain, in meters.
    upper : array_like, shape (3,)
        Upper corner of the simulation domain, in meters.
    spacing : float
        Lattice spacing of the simulated particles, in meters.
    support_radius : float
        Radius of the support of the smoothing kernel, in meters.

    Returns
    -------
    positions : np.ndarray, shape (n_particles, 3)
        Positions of the boundary particles.
    normals : np.ndarray, shape (n_particles, 3)
        Outward normal of the face the particle belongs to.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    n_layers = int(np.ceil(support_radius / spacing)) + 1
    offsets = 0.5 * spacing + spacing * np.arange(n_layers)
    extent = offsets[-1]

    coords_interior = list()
    coords_extended = list()
    for i_axis in range(3):
        coords_interior.append(np.arange(lower[i_axis], upper[i_axis] + 0.5 * spacing, spacing))
        coords_extended.append(
            np.arange(lower[i_axis] - extent, upper[i_axis] + extent + 0.5 * spacing, spacing)
        )

    positions = list()
    normals = list()
    for i_axis, i_axis_u, i_axis_v in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        coords_u = (coords_extended if i_axis_u > i_axis else coords_interior)[i_axis_u]
        coords_v = (coords_extended if i_axis_v > i_axis else coords_interior)[i_axis_v]
        uu, vv = np.meshgrid(coords_u, coords_v, indexing="ij")
        for side in (-1.0, 1.0):
            face_coord = lower[i_axis] if side < 0 else upper[i_axis]
            normal = np.zeros(3, dtype=gs.np_float)
            normal[i_axis] = side
            for offset in offsets:
                plane = np.empty((uu.size, 3), dtype=gs.np_float)
                plane[:, i_axis] = face_coord + side * offset
                plane[:, i_axis_u] = uu.ravel()
                plane[:, i_axis_v] = vv.ravel()
                positions.append(plane)
                normals.append(np.broadcast_to(normal, plane.shape))

    return np.concatenate(positions), np.concatenate(normals)
