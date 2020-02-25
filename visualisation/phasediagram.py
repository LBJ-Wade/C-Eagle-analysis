import sys
import os.path
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cluster import Cluster, Simulation
from visualisation import rendering


class PhaseDiagram(Simulation, rendering.Map):

    REQUIRES = {'partType0': ['coordinates', 'temperature', 'sphdensity', 'mass']}

    # Inherit only some methods
    info = Simulation.__dict__["info"]
    get_centers_from_bins = rendering.Map.__dict__["get_centers_from_bins"]
    bins_meshify = rendering.Map.__dict__["bins_meshify"]


    def __init__(self,
                 cluster: Cluster,
                 resolution: int = 200,
                 aperture: float = None,
                 density_bounds: list = None,
                 temperature_bounds: list = None):
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
        self.density_bounds = [1e-8, 1e5] if density_bounds == None else density_bounds
        self.temperature_bounds = [1e3, 1e8] if temperature_bounds == None else temperature_bounds


    def make_panel(self, axes: plt.Axes.axes) -> plt.pcolormesh:
        """
        Returns the
        :param projection:
        :return:
        """

        # Filter particles
        spatial_filter = np.where((self.cluster.partType0_temperature > 1e4))[0]


        mass = self.cluster.mass_units(self.cluster.partType0_mass[spatial_filter], unit_system='astro')
        density = self.cluster.density_units(self.cluster.partType0_sphdensity[spatial_filter], unit_system='nHcgs')
        temperature = self.cluster.partType0_temperature[spatial_filter]

        x_bins = np.logspace(np.log10(self.density_bounds[0]), np.log10(self.density_bounds[1]), self.resolution)
        y_bins = np.logspace(np.log10(self.temperature_bounds[0]), np.log10(self.temperature_bounds[1]), self.resolution)
        A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
        Cx, Cy = self.bins_meshify(density, temperature, x_bins, y_bins)
        count = self.bins_evaluate(density, temperature, x_bins, y_bins, weights=mass) #/ A_pix

        # Logarithmic normalization
        norm = colors.LogNorm()  # (vmin=10 ** -2, vmax=10 ** 1)

        count2 = np.ma.masked_where(count == 0, count)
        cmap = plt.get_cmap('CMRmap')
        cmap.set_bad(color='grey', alpha=1)

        image = axes.pcolormesh(Cx, Cy, count2, cmap=cmap, norm=norm)

        return image


    def test(self):

        fig = plt.figure(figsize=(6, 6))
        axes = fig.add_subplot(111)

        image = self.make_panel(axes)

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel("Density [$n_H$ cm$^{-3}$]")
        axes.set_ylabel("Temperature [K]")

        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes)
        cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
        cbar = plt.colorbar(image, cax=cax2, orientation='vertical')
        cbar.set_label("$Mass$ [M_\odot]", labelpad=17)
        # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
        cax2.xaxis.set_ticks_position("top")

        plt.show()


if __name__ == '__main__':
    # Create a cluster object
    cluster = Cluster(simulation_name='celr_e', clusterID=0, redshift='z000p000')
    # Create a PhaseDiagram object and link it to the cluster object
    t_rho_diagram = PhaseDiagram(cluster)
    t_rho_diagram.info()
    # Test the map output
    t_rho_diagram.test()