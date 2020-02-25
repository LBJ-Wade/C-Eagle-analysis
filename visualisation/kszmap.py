import sys
import os.path
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt

import swiftsimio_binder as swift
from unyt import hydrogen_mass, speed_of_light, thompson_cross_section

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cluster import Cluster
from testing import angular_momentum


def rescale(X, x_min, x_max):
    """
    Rescaled the array of input to the range [x_min, x_max] linearly.
    This method is often used in the context of making maps with matplotlib.pyplot.inshow.
    The matrix to be accepted must contain arrays in the [0,1] range.

    :param X: numpy.ndarray
        This is the input array to be rescaled.
    :param x_min: float or int
        The lower boundary for the array to have.
    :param x_max: float or int
        The upper boundary for the new array to have.

    :return: numpy.ndarray
        The array, linearly rescaled to the range [x_min, x_max].
    """
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    return x_min + nom / denom

class KSZMAP:

    REQUIRES = {'partType0': ['coordinates', 'velocities', 'temperature', 'sphkernel', 'mass']}

    def __init__(self,
                 cluster: Cluster,
                 resolution: int = 200,
                 aperture: float = None,
                 plotlimits: float = None):
        """

        :param cluster:
        :param resolution:
        :param aperture:
        :param plotlimits:
        """

        # Impose cluster requirements
        cluster.set_requires(self.REQUIRES)
        cluster.import_requires()

        # Initialise the KSZ map fields
        self.cluster = cluster
        self.resolution = resolution
        self.aperture = cluster.r500 if aperture == None else aperture
        self.plotlimits = 3*cluster.r500 if plotlimits == None else plotlimits


    def make_panel(self, axes: plt.Axes.axes, projection: str) -> plt.imshow:
        """
        Returns the
        :param projection:
        :return:
        """
        # Derotate cluster
        coords, vel = angular_momentum.derotate(self.cluster,
                                                align='gas',
                                                aperture_radius=self.aperture,
                                                cluster_rest_frame=True)

        # Filter particles
        spatial_filter = np.where((np.abs(coords[:, 0]) < self.plotlimits) &
                                  (np.abs(coords[:, 1]) < self.plotlimits) &
                                  (self.cluster.partType0_temperature > 1e5))[0]

        coords     = coords[spatial_filter, :]
        vel        = vel[spatial_filter, :]
        mass       = self.cluster.partType0_mass[spatial_filter]
        SPH_kernel = self.cluster.partType0_sphkernel[spatial_filter]

        constant_factor = (-1) * thompson_cross_section / (speed_of_light * hydrogen_mass * 1.16)
        kSZ = np.multiply((vel.T * mass).T, constant_factor)

        x = np.asarray(rescale(coords[:, 0], 0, 1), dtype=np.float64)
        y = np.asarray(rescale(coords[:, 1], 0, 1), dtype=np.float64)
        z = np.asarray(rescale(coords[:, 2], 0, 1), dtype=np.float64)
        m = np.asarray(kSZ[:, 2], dtype=np.float32)
        h = np.asarray(SPH_kernel, dtype=np.float32)

        # Generate the sph-smoothed map
        temp_map = swift.generate_map(x, y, m, h, self.resolution, parallel=True)
        norm = colors.SymLogNorm(linthresh=1e-5, linscale=0.5,
                                 vmin=-np.abs(temp_map).max(),
                                 vmax=np.abs(temp_map).max())

        # Attach the image to the Axes class
        image = axes.imshow(temp_map, cmap='seismic', norm=norm,
                            extent=(-self.plotlims, self.plotlims,
                                    -self.plotlims, self.plotlims))

        return image


    def test(self):

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        panel = self.make_panel(ax, 'xy')
        cbar = fig.colorbar(panel)
        cbar.ax.minorticks_off()
        ax.set_xlabel(r'$x\ /\mathrm{Mpc}$')
        ax.set_ylabel(r'$y\ /\mathrm{Mpc}$')
        plt.show()


if __name__ == '__main__':
    # Create a cluster object
    cluster = Cluster(simulation_name='celr_e', clusterID=0, redshift='z000p000')
    # cluster.info()
    # Create a KSZMAP object and link it to the cluster object
    test_map = KSZMAP(cluster)
    # Test the map output
    test_map.test()