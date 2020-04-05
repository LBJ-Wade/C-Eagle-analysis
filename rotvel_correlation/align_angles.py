"""
------------------------------------------------------------------
FILE:   rotvel_correlation/align_angles.py
AUTHOR: Edo Altamura
DATE:   15-03-2020
------------------------------------------------------------------
The save python package generates partial raw data from the simulation
data. The pull.py file accesses and reads these data and returns
then in a format that can be used for plotting, further processing
etc.
-------------------------------------------------------------------
"""
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import sys
import os.path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils import resample
from typing import List, Dict, Tuple, Union
import itertools

exec(open(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation
from import_toolkit._cluster_retriever import redshift_str2num
from import_toolkit.progressbar import ProgressBar
from read import pull

class CorrelationMatrix(pull.FOFRead):

    def __init__(self, cluster: Cluster):

        super().__init__(cluster)

    def get_data(self):
        return self.pull_rot_vel_alignment_angles()

    def get_apertures(self):
        return self.pull_apertures()

    @staticmethod
    def get_percentiles(data: np.ndarray, percentiles: list = [15.9, 50, 84.1]) -> np.ndarray:
        data = np.asarray(data)
        return np.array([np.percentile(data, percent, axis=0) for percent in percentiles])

    @ProgressBar()
    def bootstrap(self, data: np.ndarray, n_iterations: Union[int, float] = 1e3) -> Dict[str, Tuple[float, float]]:
        """
        Class method to compute the median/percentile statistics of a 1D dataset using the
        bootstrap resampling. The bootstrap allows to compute the dispersion of values in the
        computed median/percentiles.
        The method checks for the input dimensionality, then creates many realisations of the
        initial dataset while resampling & replacing elements. The random seed is controlled
        by a for loop of n_iterations (default 1000) iterations. It then computes the percentiles
        for each realisation of the dataset and pushes the percentile results into a master stats
        array.
        From this master array of shape (n_iterations, 3), where all percentiles are gathered,
        the mean and standard deviations are computed for each type of percentile and returned
        in the form of a dictionary: the 3 percentiles with quoted mean and std.

        :param data: expect 1D numpy array
            Lists are also handled, provided they make it though the static typing condition.

        :param n_iterations: expect int (default det to 1000)
            The number of realisations for the bootstrap resampling. Recommended to be > 1e3.

        :return:  Dict[str, Tuple[float, float]]
            The mean and standard deviation (from the bootstrap sample), quoted for each
            percentile and median.
        """
        n_iterations = int(n_iterations)
        data = np.asarray(data)
        assert data.ndim is 1, f"Expected `data` to have dimensionality 1, got {data.ndim}."
        stats_resampled = np.zeros((0, 3), dtype=np.float)

        counter = 0
        for seed in range(n_iterations):
            data_resampled = resample(data, replace=True, n_samples=len(data), random_state=seed)
            stats = self.get_percentiles(data_resampled, percentiles=[15.9, 50, 84.1])
            stats_resampled = np.concatenate((stats_resampled, [stats]), axis=0)
            yield ((counter + 1) / n_iterations)
            counter += 1

        stats_resampled_MEAN = np.mean(stats_resampled, axis=0)
        stats_resampled_STD  = np.std(stats_resampled, axis=0)
        bootstrap_stats = {
            'percent16': (stats_resampled_MEAN[0], stats_resampled_STD[0]),
            'median50' : (stats_resampled_MEAN[1], stats_resampled_STD[1]),
            'percent84': (stats_resampled_MEAN[2], stats_resampled_STD[2])
        }

        return bootstrap_stats

    def plot_matrix(self, angle_matrix, apertures):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        im = ax.imshow(angle_matrix, cmap='Set3_r', vmin=0, vmax=180, origin='lower')
        # Manipulate the colorbar on the side
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, ticks=np.linspace(0, 180, 13))
        cbar.ax.minorticks_off()
        cbar.ax.set_ylabel(r'$\Delta \theta$ \quad [degrees]', rotation=270, size=25, labelpad=30)

        x_labels = [
            r'$\mathbf{\widehat{v_{pec}}}$',
            r'$\mathbf{\widehat{L}}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(gas)}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(DM)}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(stars)}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(BH)}$',
            r'$\mathbf{\widehat{L}}^{(gas)}$',
            r'$\mathbf{\widehat{L}}^{(DM)}$',
            r'$\mathbf{\widehat{L}}^{(stars)}$',
            r'$\mathbf{\widehat{L}}^{(BH)}$'
        ]

        ticks_major = np.arange(0, len(x_labels), 1)
        ticks_minor = np.arange(-0.5, len(x_labels), 1)
        ax.set_xticks(ticks_major, minor=False)
        ax.set_yticks(ticks_major, minor=False)
        ax.set_xticks(ticks_minor, minor=True)
        ax.set_yticks(ticks_minor, minor=True)
        ax.set_xticklabels(x_labels, rotation = 90)
        ax.set_yticklabels(x_labels)

        # Compute percentiles
        percentiles = self.get_percentiles(angle_matrix)
        percent16 = percentiles[0]
        median50 = percentiles[1]
        percent84 = percentiles[2]

        # Loop over data dimensions and create text annotations.
        iterator = itertools.product(np.arange(len(x_labels)), repeat=2)
        for i, j in iterator:
            if i is not j:
                 ax.text(j, i, r"${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                            median50[i, j],
                            percent84[i, j] - median50[i, j],
                            median50[i, j] - percent16[i, j]),
                            ha="center", va="center", color="k", size = 12)

        ax.set_title(r"Aperture = {:.2f}".format(apertures) + r"\ $R_{500\ true}$", size = 25)
        ax.grid(b = True, which='minor',color='w', linestyle='-', linewidth=10, alpha = 1)

        ax.tick_params(which='minor', length=0, color='w')
        ax.tick_params(which='major', length=0, color='w')
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor('w')

        fig.tight_layout()

class TrendZ:

    # Inherit methods from the CorrelationMatrix class
    get_percentiles = staticmethod(CorrelationMatrix.get_percentiles)
    bootstrap = CorrelationMatrix.bootstrap

    def __init__(self):
        pass

    def get_apertures(self, cluster: Cluster):
        read = pull.FOFRead(cluster)
        return read.pull_apertures()

    @staticmethod
    def get_centers_from_bins(bins):
        """ return centers from bin sequence """
        return (bins[:-1] + bins[1:]) / 2

    @staticmethod
    def get_centers_from_log_bins(bins):
        """ return centers from bin sequence """
        return np.sqrt(bins[:-1] * bins[1:])

    def plot_z_trends(self,
                      simulation_name: str = None,
                      aperture_id: int = 10):

        warnings.filterwarnings("ignore")
        sim = Simulation(simulation_name=simulation_name)
        path = os.path.join(sim.pathSave, sim.simulation_name, 'rotvel_correlation')
        if not os.path.exists(path):
            os.makedirs(path)
        aperture_id_str = f'Aperture {aperture_id}'
        cluster = Cluster(simulation_name=simulation_name, clusterID=0, redshift='z000p000')
        aperture_float = self.get_apertures(cluster)[aperture_id] / cluster.r200
        z_master = np.array([redshift_str2num(z) for z in sim.redshiftAllowed])
        z_master = z_master[z_master < 1.8]
        print(f"{sim.simulation:=^100s}")
        print(f"{aperture_id_str:^100s}\n")

        if  os.path.isfile(os.path.join(path, f'redshift_rotTvelT_simstats_aperture_{aperture_id}.npy')):
            print("Retrieving npy files...")
            angle_master = np.load(os.path.join(path, f'redshift_rotTvelT_simstats_aperture_{aperture_id}.npy'))
            angle_master = np.asarray(angle_master)

        else:
            print(f"{'':<30s} {' process ID ':^25s} | {' halo ID ':^15s} | {' halo redshift ':^20s}\n")
            angle_master = np.zeros((sim.totalClusters, len(z_master)), dtype=np.float)
            iterator = itertools.product(sim.clusterIDAllowed, sim.redshiftAllowed)
            for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
                print(f"{'Processing...':<30s} {process_n:^25d} | {halo_id:^15d} | {halo_z:^20s}")
                if sim.sample_completeness[halo_id, sim.redshiftAllowed.index(halo_z)]:
                    cluster = Cluster(simulation_name=simulation_name,
                                      clusterID=halo_id,
                                      redshift=halo_z,
                                      fastbrowsing=True)
                    read = pull.FOFRead(cluster)
                    angle = read.pull_rot_vel_angle_between('Total_angmom', 'Total_ZMF')[aperture_id]
                    angle_master[halo_id, sim.redshiftAllowed.index(halo_z)] = angle

            np.save(os.path.join(path, f'redshift_rotTvelT_simstats_aperture_{aperture_id}.npy'), angle_master)

        percent16_mean = np.zeros_like(z_master)
        median50_mean  = np.zeros_like(z_master)
        percent84_mean = np.zeros_like(z_master)
        percent16_std  = np.zeros_like(z_master)
        median50_std   = np.zeros_like(z_master)
        percent84_std  = np.zeros_like(z_master)

        # BOOTSTRAP SAMPLING
        for idx, redshift in np.ndenumerate(z_master):
            boot_stats = self.bootstrap(angle_master.T[idx], n_iterations=1e2)
            percent16_mean [idx] = boot_stats['percent16'][0]
            median50_mean [idx]  = boot_stats['median50'][0]
            percent84_mean [idx] = boot_stats['percent84'][0]
            percent16_std[idx]   = boot_stats['percent16'][1]
            median50_std[idx]    = boot_stats['median50'][1]
            percent84_std[idx]   = boot_stats['percent84'][1]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        sim_colors = {
            'ceagle' : 'pink',
            'celr_e' : 'lime',
            'celr_b' : 'orange',
            'macsis' : 'aqua',
        }
        ax.axhline(90, linestyle='--', color='k', alpha=0.5, linewidth=2)

        ax.plot(z_master, percent84_mean, color=sim_colors[sim.simulation_name],
                alpha=1, linestyle='none', marker='^', markersize=10)
        ax.plot(z_master, median50_mean, color=sim_colors[sim.simulation_name],
                alpha=1, linestyle='none', marker='o', markersize=10)
        ax.plot(z_master, percent16_mean, color=sim_colors[sim.simulation_name],
                alpha=1, linestyle='none', marker='v', markersize=10)
        ax.plot(z_master, percent84_mean, color = sim_colors[sim.simulation_name],
                alpha = 0.8, drawstyle='steps-mid', linestyle='--', lw=1.5)
        ax.plot(z_master, median50_mean, color = sim_colors[sim.simulation_name],
                alpha = 0.8,  drawstyle='steps-mid', linestyle='-', lw=1.5)
        ax.plot(z_master, percent16_mean, color = sim_colors[sim.simulation_name],
                alpha = 0.8, drawstyle='steps-mid', linestyle='-.', lw=1.5)
        ax.fill_between(z_master, percent84_mean - percent84_std, percent84_mean + percent84_std,
                        color = sim_colors[sim.simulation_name], alpha = 0.2, step='mid', edgecolor='none')
        ax.fill_between(z_master, median50_mean - median50_std,  median50_mean + median50_std,
                        color = sim_colors[sim.simulation_name], alpha = 0.2, step='mid', edgecolor='none')
        ax.fill_between(z_master, percent16_mean - percent16_std, percent16_mean + percent16_std,
                        color = sim_colors[sim.simulation_name], alpha = 0.2, step='mid', edgecolor='none')

        perc84 = Line2D([], [], color='k', marker='^', linestyle='--', markersize=10, label=r'$84^{th}$ percentile')
        perc50 = Line2D([], [], color='k', marker='o', linestyle='-', markersize=10, label=r'median')
        perc16 = Line2D([], [], color='k', marker='v', linestyle='-.', markersize=10, label=r'$16^{th}$ percentile')
        patch_ceagle = Patch(facecolor=sim_colors['ceagle'], label='C-EAGLE', edgecolor='k', linewidth=1)
        patch_celre = Patch(facecolor=sim_colors['celr_e'], label='CELR-E', edgecolor='k', linewidth=1)
        patch_celrb = Patch(facecolor=sim_colors['celr_b'], label='CELR-B', edgecolor='k', linewidth=1)
        patch_macsis = Patch(facecolor=sim_colors['macsis'], label='MACSIS', edgecolor='k', linewidth=1)

        leg1 = ax.legend(handles=[perc84, perc50, perc16], loc='lower right', handlelength=3, fontsize=20)
        leg2 = ax.legend(handles=[patch_ceagle, patch_celre, patch_celrb, patch_macsis], loc='lower left',
                         handlelength=1,
                         fontsize=20)
        ax.add_artist(leg1)
        ax.add_artist(leg2)

        items_labels = r""" REDSHIFT TRENDS
                            $\Delta \theta \equiv (\mathbf{{L,\widehat{{CoP}},v_{{pec}}}})$
                            Number of clusters: {:d}
                            $z$ = {:.2f} - {:.2f}
                            Aperture radius = {:.2f} $R_{{200\ true}}$""".format(sim.totalClusters,
                                                                   z_master[0],
                                                                   z_master[-1],
                                                                   aperture_float)

        print(items_labels)
        ax.text(0.03, 0.97, items_labels,
                  horizontalalignment='left',
                  verticalalignment='top',
                  transform=ax.transAxes,
                  size=15)

        ax.set_xlabel(r"$z$", size=25)
        ax.set_ylabel(r"$\Delta \theta$ \quad [degrees]", size=25)
        ax.set_ylim(0, 180)
        plt.savefig(os.path.join(path, f'redshift_rotTvelT_aperture_{aperture_id}.png'), dpi=300)


if __name__ == '__main__':


    def test():
        cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
        matrix = CorrelationMatrix(cluster)
        data = matrix.get_data()
        aperture = matrix.get_apertures()

        matrix.plot_matrix(data[15], aperture[15])

    def all_clusters(apertureidx):
        simulation = Simulation(simulation_name='celr_e')
        matrix_list = []
        aperture_list = []
        aperture_self_similar = []
        for id in simulation.clusterIDAllowed:
            cluster = Cluster(simulation_name='celr_e', clusterID=id, redshift='z000p101')
            print(f'Analysing cluster {cluster.clusterID}')
            matrix = CorrelationMatrix(cluster)
            data = matrix.get_data()[apertureidx]
            aperture = matrix.get_apertures()[apertureidx]
            matrix_list.append(data)
            aperture_list.append(aperture)
            aperture_self_similar.append(aperture/cluster.r500)

        if not os.path.exists(os.path.join(simulation.pathSave, simulation.simulation_name, 'rotvel_correlation')):
            os.makedirs(os.path.join(simulation.pathSave, simulation.simulation_name, 'rotvel_correlation'))

        average_aperture = np.mean(aperture_self_similar, axis=0)

        matrix.plot_matrix(matrix_list, average_aperture)
        print(f"Saving matrix | aperture {apertureidx} | redshift {cluster.redshift}")
        plt.savefig(os.path.join(simulation.pathSave, simulation.simulation_name, 'rotvel_correlation',
                                 f'meanPMstd_{cluster.redshift}_aperture_{apertureidx}.png'))



    trendz = TrendZ()
    # for aperture in range(20):
    #     if aperture % size is rank:
    trendz.plot_z_trends(simulation_name='celr_e', aperture_id=10)




