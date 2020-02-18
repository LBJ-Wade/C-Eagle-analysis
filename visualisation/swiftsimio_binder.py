

def generate_map(*args, parallel = False):
    """
    SWIFTSIMIO WRAPPER
    ------------------
    The key here is that only particles in the domain [0, 1] in x, and [0, 1] in y will be visible in the image.
    You may have particles outside of this range; they will not crash the code, and may even contribute to the image
    if their  smoothing lengths overlap with [0, 1]. You will need to re-scale your data such that it lives within
    this range.
    out will be a 2D numpy grid of shape [res, res]. You will need to re-scale this back to your original dimensions
    to get it in the correct units, and do not forget that it now represents the smoothed quantity per surface area.

    ================================

    @jit(nopython=True, fastmath=True)
    def scatter(x: float64, y: float64, m: float32, h: float32, res: int) -> ndarray

        Creates a scatter plot of:
        + x: the x-positions of the particles. Must be bounded by [0, 1].
        + y: the y-positions of the particles. Must be bounded by [0, 1].
        + m: the masses (or otherwise weights) of the particles
        + h: the smoothing lengths of the particles
        + res: the number of pixels.

    ================================

    This ignores boundary effects.
    Note that explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.

    :param parallel: (boolean) default = False
        Triggers the use of Numba decorators for parallel rendering.

    :param kwargs: parse the kwargs used by swiftsimio.visualisation.projection.scatter_parallel

    :return: nd array
    """
    from swiftsimio.visualisation.projection import scatter, scatter_parallel

    if parallel:
        return scatter_parallel(*args)
    else:
        return scatter(*args)


def generate_volume(*args, parallel = False):
    """
    SWIFTSIMIO WRAPPER
    ------------------
    The key here is that only particles in the domain [0, 1] in x, [0, 1] in y and [0, 1] in z will be visible in the image.
    You may have particles outside of this range; they will not crash the code, and may even contribute to the image
    if their  smoothing lengths overlap with [0, 1]. You will need to re-scale your data such that it lives within
    this range.
    out will be a 3D numpy grid of shape [res, res, res]. You will need to re-scale this back to your original
    dimensions to get it in the correct units, and do not forget that it now represents the smoothed quantity per
    surface volume.

    ================================

    @jit(nopython=True, fastmath=True)
    def scatter(
        x: float64, y: float64, z: float64, m: float32, h: float32, res: int
    ) -> ndarray:

        Creates a voxel grid of:
        + x: the x-positions of the particles. Must be bounded by [0, 1].
        + y: the y-positions of the particles. Must be bounded by [0, 1].
        + z: the z-positions of the particles. Must be bounded by [0, 1].
        + m: the masses (or otherwise weights) of the particles
        + h: the smoothing lengths of the particles
        + res: the number of voxels along one axis, i.e. this returns a cube
               of res * res * res..

    ================================

    This ignores boundary effects.
    Note that explicitly defining the types in this function allows
    for a 25-50% performance improvement. In our testing, using numpy
    floats and integers is also an improvement over using the numba ones.

    :param parallel: (boolean) default = False
        Triggers the use of Numba decorators for parallel rendering.

    :param kwargs: parse the kwargs used by swiftsimio.visualisation.volume_render.scatter_parallel

    :return: nd array
    """
    from swiftsimio.visualisation.volume_render import scatter, scatter_parallel

    if parallel:
        return scatter_parallel(*args)
    else:
        return scatter(*args)


if __name__ == '__main__':
    import sys
    import os.path

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    import matplotlib.colors as colors

    from cluster import Cluster
    from testing import angular_momentum
    import numpy as np


    def rescale(X, x_min, x_max):
        nom = (X - X.min(axis=0)) * (x_max - x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        # denom[np.where(denom == 0.)[0]] = 1.
        return x_min + nom / denom

    cluster = Cluster(simulation_name='celr_e', clusterID=0, redshift='z000p000')
    r500 = cluster.group_r500()
    r500 = cluster.comoving_length(r500)
    mass = cluster.particle_masses('gas')
    mass = cluster.comoving_mass(mass)
    SPH_kernel = cluster.particle_SPH_smoothinglength()
    SPH_kernel = cluster.comoving_length(SPH_kernel)

    coords, vel = angular_momentum.derotate(cluster, align='gas', aperture_radius=r500, cluster_rest_frame=True)

    spatial_filter = np.where(
        (np.abs(coords[:,0]) < 3*r500) &
        (np.abs(coords[:,1]) < 3*r500) &
        (np.abs(coords[:,2]) < 3*r500))[0]
    coords = coords[spatial_filter, :]
    vel = vel[spatial_filter, :]
    mass = mass[spatial_filter]
    SPH_kernel = SPH_kernel[spatial_filter]

    res = np.int(50)

    bins_x = np.linspace(-np.min(coords[:, 0]), np.max(coords[:, 0]), res)
    pixel_area = (bins_x[1] - bins_x[0]) **2

    from unyt import hydrogen_mass, speed_of_light, thompson_cross_section
    kSZ = np.multiply((vel.T * mass).T, (-1) * thompson_cross_section / (speed_of_light * hydrogen_mass *
                                                                         1.16))

    x = np.asarray(rescale(coords[:,0], 0, 1), dtype = np.float64)
    y = np.asarray(rescale(coords[:,1], 0, 1), dtype = np.float64)
    z = np.asarray(rescale(coords[:,2], 0, 1), dtype = np.float64)

    m = np.asarray(kSZ[:, 2], dtype = np.float32)
    h = np.asarray(SPH_kernel, dtype = np.float32)

    temp_map = generate_map(x, y, m, h, res, parallel=True)
    print(temp_map)
    norm = colors.SymLogNorm(linthresh=np.percentile(np.abs(m), 5), linscale=0.5, vmin=-np.abs(m).max(), vmax=np.abs(
        m).max())

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    cs = ax.imshow(temp_map, cmap = 'seismic', extent = (np.min(coords[:,0]), np.max(coords[:,0]),
                                                         np.min(coords[:,1]), np.max(coords[:,1])))
    cbar = fig.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.show()


