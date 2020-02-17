
import sys
sys.path.append("..") # Adds higher directory to python modules path.

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
    from cluster import Cluster
    from testing import angular_momentum
    import numpy as np

    cluster = Cluster(simulation_name='celr_e', clusterID=0, redshift='z000p000')
    r500 = cluster.group_r500()
    r500 = cluster.comoving_length(r500)
    mass = cluster.particle_masses('gas')
    mass = cluster.comoving_mass(mass)
    SPH_kernel = cluster.particle_SPH_smoothinglength()

    coords, vel = angular_momentum.derotate(cluster, align='gas', aperture_radius=r500, cluster_rest_frame=True)

    from unyt import hydrogen_mass, speed_of_light, thompson_cross_section

    plot_limit = 3 * r500
    nbins = 100
    bins = np.linspace(-plot_limit, plot_limit, nbins)
    pixel_area = (bins[1] - bins[0]) ** 2
    kSZ = np.multiply((vel.T * mass).T, (-1) * thompson_cross_section / (speed_of_light * hydrogen_mass * 1.16))

    temp_map = generate_map(coords[:,0]/np.max(coords[:,0]), coords[:,0]/np.max(coords[:,1]), kSZ, SPH_kernel, 200,
                          parallel=True)

    from matplotlib.pyplot import imsave
    from matplotlib.colors import LogNorm

    # Normalize and save
    imsave("~/temp_map.png", LogNorm()(temp_map.value), cmap="twilight")

