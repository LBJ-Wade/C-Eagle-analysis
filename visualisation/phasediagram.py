import matplotlib
matplotlib.use('Agg')
import sys
import os.path
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation
from visualisation import rendering


class PhaseDiagram(Simulation, rendering.Map):

    REQUIRES = {'partType0': ['coordinates', 'temperature', 'sphdensity', 'mass']}
    pathSave = '/cosma6/data/dp004/dc-alta2/C-Eagle-analysis-work'

    # Inherit only some methods
    info = Simulation.__dict__["info"]
    get_centers_from_bins = rendering.Map.__dict__["get_centers_from_bins"]
    bins_meshify = rendering.Map.__dict__["bins_meshify"]


    def __init__(self,
                 cluster: Cluster,
                 resolution: int = 300,
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
        self.aperture = 5*cluster.r200 if aperture == None else aperture
        self.density_bounds = [1e-8, 1e5] if density_bounds == None else density_bounds
        self.temperature_bounds = [1e3, 1e11] if temperature_bounds == None else temperature_bounds


    def make_panel(self, axes: plt.Axes.axes) -> plt.pcolormesh:
        """
        Returns the
        :param projection:
        :return:
        """
        radial_dist = self.cluster.radial_distance_CoP(self.cluster.partType0_coordinates)
        spatial_filter = np.where(radial_dist < self.aperture)[0]

        mass        = self.cluster.mass_units(self.cluster.partType0_mass[spatial_filter], unit_system='astro')
        density     = self.cluster.density_units(self.cluster.partType0_sphdensity[spatial_filter], unit_system='nHcgs')
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

    def make_cluster_label(self, axes: plt.Axes.axes):
        items_labels = r"""THERMODYNAMIC PHASE SPACE
                        Cluster {:s}\ {:d}
                        $z$ = {:.3f}
                        $R_{{500\ true}}$ = {:.2f} Mpc
                        Aperture radius = {:.2f} Mpc""".format(self.cluster.simulation,
                                                              self.cluster.clusterID,
                                                              self.cluster.z,
                                                              self.cluster.r500,
                                                              self.aperture)

        print(items_labels)
        axes.text(0.03, 0.97, items_labels,
                  horizontalalignment='left',
                  verticalalignment='top',
                  transform=axes.transAxes,
                  size = 15)


    def setup_plot(self):
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(7, 6))
        axes = fig.add_subplot(111)

        image = self.make_panel(axes)
        self.make_cluster_label(axes)

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlabel(r"$\rho\ \mathrm{[n_H\ cm^{-3}]}$")
        axes.set_ylabel(r"$T\ \mathrm{[K]}$")

        # Plot selection boundary
        axes.plot([1e-8, 1e-1, 1e2], [1e4, 1e4, 1e5], marker = None, linestyle = '-', color = 'orange', lw = 2)

        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes)
        cax2 = ax2_divider.append_axes("right", size="3%", pad="1%")
        cbar = plt.colorbar(image, cax=cax2, orientation='vertical')
        cbar.set_label(r"$M\ \mathrm{[M_\odot]}$", labelpad=17)
        # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
        cax2.xaxis.set_ticks_position("top")


        filename_out = self.pathSave + '/phasediagrams/' + self.cluster.simulation_name + '/' + \
                       self.cluster.cluster_prefix + str(self.cluster.clusterID) + '_' + self.cluster.redshift

        if not os.path.exists(self.pathSave + '/phasediagrams/' + self.cluster.simulation_name):
            os.makedirs(self.pathSave + '/phasediagrams/' + self.cluster.simulation_name)

        aperture = ("%05.2f" % self.aperture).replace('.', 'p')
        plt.tight_layout()

        plt.savefig(filename_out + "_aperture" + aperture + "_bins" + str(self.resolution) + ".png", dpi=300)



def test_simple():

    # Create a cluster object
    cluster = Cluster(simulation_name='macsis', clusterID=0, redshift='z000p000')
    # cluster.info()

    # Create a PhaseDiagram object and link it to the cluster object
    t_rho_diagram = PhaseDiagram(cluster)
    # t_rho_diagram.info()

    # Test the map output
    t_rho_diagram.setup_plot()

def test_loop_apertures(i):
    # Create a cluster object
    cluster = Cluster(simulation_name='ceagle', clusterID=i, redshift='z000p000')
    apertures = cluster.generate_apertures()

    for aperture in apertures:
        # Create a PhaseDiagram object and link it to the cluster object
        t_rho_diagram = PhaseDiagram(cluster, aperture=aperture)
        t_rho_diagram.info()

        # Test the map output
        t_rho_diagram.setup_plot()

def test_loop_redshifts(i):

    simulation = Simulation(simulation_name='celr_b')
    redshifts = simulation.redshiftAllowed

    for z in redshifts:
        # Create a cluster object
        cluster = Cluster(simulation_name='celr_b', clusterID=i, redshift=z)
        cluster.info()

        # Create a PhaseDiagram object and link it to the cluster object
        t_rho_diagram = PhaseDiagram(cluster)
        # t_rho_diagram.info()

        # Test the map output
        t_rho_diagram.setup_plot()




if __name__ == '__main__':
    # test_simple()
    exec(open('visualisation/light_mode.py').read())
    for sim in ['celr_e', 'celr_b', 'macsis', 'ceagle']:

        cluster = Cluster(simulation_name=sim, clusterID=0, redshift='z000p000')
        t_rho_diagram = PhaseDiagram(cluster)
        t_rho_diagram.setup_plot()

    # from mpi4py import MPI
    #
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    # test_loop_apertures(rank)
    # test_loop_redshifts(rank)