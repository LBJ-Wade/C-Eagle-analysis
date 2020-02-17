

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

    cluster = Cluster(simulation_name='celr_e', clusterID=0, redshift='z000p000')
    r500 = cluster.group_r500()
    r500 = cluster.comoving_length(r500)
    mass = cluster.particle_masses('gas')
    mass = cluster.comoving_mass(mass)
    SPH_kernel = cluster.particle_SPH_smoothinglength()
    SPH_kernel = cluster.comoving_length(SPH_kernel)

    coords, vel = angular_momentum.derotate(cluster, align='gas', aperture_radius=r500, cluster_rest_frame=True)

    from unyt import hydrogen_mass, speed_of_light, thompson_cross_section
    kSZ = np.multiply((vel.T * mass).T, (-1) * thompson_cross_section / (speed_of_light * hydrogen_mass * 1.16))

    x = np.asarray((coords[:,0] - np.min(coords[:,0]))/(np.max(coords[:,0]) - np.min(coords[:,0])), dtype = np.float64)
    y = np.asarray((coords[:,1] - np.min(coords[:,1]))/(np.max(coords[:,1]) - np.min(coords[:,1])), dtype = np.float64)
    z = np.asarray((coords[:,2] - np.min(coords[:,2]))/(np.max(coords[:,2]) - np.min(coords[:,2])), dtype = np.float64)

    res = np.int(200)

    bins_x = np.linspace(-np.min(coords[:, 0]), np.max(coords[:, 0]), res)
    bins_y = np.linspace(-np.min(coords[:, 1]), np.max(coords[:, 1]), res)
    bins_z = np.linspace(-np.min(coords[:, 2]), np.max(coords[:, 2]), res)
    pixel_area = (bins_x[1] - bins_x[0]) * (bins_y[1] - bins_y[0])


    m = np.asarray(mass * vel[:,2] * thompson_cross_section / (pixel_area * speed_of_light * hydrogen_mass * 1.16),
                   dtype = np.float32)
    h = np.asarray(SPH_kernel, dtype = np.float32)




    temp_map = generate_map(x, y, np.log10(m**2), h, res, parallel=True)
    norm = colors.SymLogNorm(linthresh=1e-5, linscale=0.5, vmin=-np.abs(m).max(), vmax=np.abs(m).max())

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(temp_map)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


