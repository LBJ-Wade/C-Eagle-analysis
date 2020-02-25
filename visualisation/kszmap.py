import sys
import os.path
import numpy as np
import matplotlib.colors as colors

import swiftsimio_binder as swift

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

    def __init__(self, cluster: Cluster):
        self.cluster = cluster
        self.cluster.requires = {'partType0': ['coordinates', 'velocities', 'temperature', 'sphkernel', 'mass']}

    def test(self):

        r500 = self.cluster.group_r500()
        mass = self.cluster.particle_masses('gas')
        SPH_kernel = self.cluster.particle_SPH_smoothinglength()
        temperature = self.cluster.particle_temperature()

        plotlims = 3*r500
        coords, vel = angular_momentum.derotate(self.cluster, align='gas', aperture_radius=plotlims,
                                                cluster_rest_frame=True)

        spatial_filter = np.where(
            (np.abs(coords[:,0]) < plotlims) &
            (np.abs(coords[:,1]) < plotlims) &
            (temperature > 1e5))[0]
        coords = coords[spatial_filter, :]
        vel = vel[spatial_filter, :]
        mass = mass[spatial_filter]
        SPH_kernel = SPH_kernel[spatial_filter]

        res = np.int(1000)

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

        temp_map = swift.generate_map(x, y, m, h, res, parallel=True)
        norm = colors.SymLogNorm(linthresh=1e-5, linscale=0.5, vmin=-np.abs(temp_map).max(), vmax=np.abs(temp_map).max())

        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        cs = ax.imshow(temp_map, cmap = 'seismic', norm = norm,
                       extent = (-plotlims, plotlims,
                                 -plotlims, plotlims)
                       )
        cbar = fig.colorbar(cs)
        cbar.ax.minorticks_off()
        ax.set_xlabel(r'$x\ \mathrm{Mpc}$')
        ax.set_ylabel(r'$y\ \mathrm{Mpc}$')
        plt.show()


if __name__ == '__main__':
    # Create a cluster object
    cluster = Cluster(simulation_name='celr_e', clusterID=0, redshift='z000p000')
    # Create a KSZMAP object and link it to the cluster object
    test_map = KSZMAP(cluster)
    # Test the map output
    test_map.test()