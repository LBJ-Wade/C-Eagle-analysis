

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

    print('[ SWIFTSIMIO ]\t ==> Invoking `projection` front-end binding.')
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

    print('[ SWIFTSIMIO ]\t ==> Invoking `volume_render` front-end binding.')
    if parallel:
        return scatter_parallel(*args)
    else:
        return scatter(*args)





