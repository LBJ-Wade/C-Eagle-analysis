from swiftsimio.visualisation.volume_render import scatter

def generate_map(parallel = False, **kwargs):
    """
    SWIFTSIMIO WRAPPER
    ------------------
    The key here is that only particles in the domain [0, 1] in x, and [0, 1] in y will be visible in the image.
    You may have particles outside of this range; they will not crash the code, and may even contribute to the image
    if their  smoothing lengths overlap with [0, 1]. You will need to re-scale your data such that it lives within
    this range.
    out will be a 2D numpy grid of shape [res, res]. You will need to re-scale this back to your original dimensions
    to get it in the correct units, and do not forget that it now represents the smoothed quantity per surface area.

    :param parallel: (boolean) default = False
        Triggers the use of Numba decorators for parallel rendering.

    :param kwargs: parse the kwargs used by swiftsimio.visualisation.projection.scatter_parallel

    :return: nd array
    """
    from swiftsimio.visualisation.projection import scatter, scatter_parallel

    if parallel:
        return scatter_parallel(**kwargs)
    else:
        return scatter(**kwargs)


def generate_volume(parallel = False, **kwargs):
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

    :param parallel: (boolean) default = False
        Triggers the use of Numba decorators for parallel rendering.

    :param kwargs: parse the kwargs used by swiftsimio.visualisation.volume_render.scatter_parallel

    :return: nd array
    """
    from swiftsimio.visualisation.volume_render import scatter, scatter_parallel

    if parallel:
        return scatter_parallel(**kwargs)
    else:
        return scatter(**kwargs)

