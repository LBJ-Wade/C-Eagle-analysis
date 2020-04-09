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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
[[            Lists are also handled, provided they make it though the static typing condition.

        :param n_iterations: expect int (default det to 1000)
            The number of realisations for the bootstrap resampling. Recommended to be > 1e3.

        :return:  Dict[str, Tuple[float, float]]
            The mean and standard deviation (from the bootstrap sample), quoted for each
            percentile and median.
        """
        N_iters = int(n_iterations)
        data = np.asarray(data)
        assert data.ndim is 1, f"Expected 'data' to have dimensionality 1, got {data.ndim}."
        stats_resampled = np.zeros((0, 3), dtype=np.float)

        counter = 0
        for seed in range(N_iters):
            data_resampled = resample(data, replace=True, n_samples=len(data), random_state=seed)
            stats = self.get_percentiles(data_resampled, percentiles=[15.9, 50, 84.1])
            stats_resampled = np.concatenate((stats_resampled, [stats]), axis=0)
            yield ((counter + 1) /N_iters)
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
        warnings.filterwarnings("ignore")
        self.aperture_id = 10
        self.simulation = None
        self.figure = None
        self.path = None
        self.bootstrap_niters = 10**4

    def set_aperture(self, aperture_id: int):
        self.aperture_id = aperture_id
        aperture_id_str = f'Aperture {self.aperture_id}'
        print(f"{aperture_id_str:^100s}\n")

    def set_simulation(self, simulation_name: str = None) -> None:
        self.simulation = Simulation(simulation_name=simulation_name)
        self.path = os.path.join(self.simulation.pathSave, self.simulation.simulation_name, 'rotvel_correlation')
        print(f"{self.simulation.simulation:=^100s}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def set_figure(self, new_figure: Figure) -> None:
        """
        Set a new `figure` attribute to the class.

        :param new_axes:  expect a matplotlib.figure.Figure object
            The new matplotlib.figure.Figure environment to build the diagram in.

        :return: None
        """
        self.figure = new_figure

    def set_bootstrap_niters(self, niters: Union[int, float]) -> None:
        self.bootstrap_niters = int(niters)

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

    def make_simstats(self):
        assert self.simulation is not None
        z_master = np.array([redshift_str2num(z) for z in self.simulation.redshiftAllowed])
        z_master = z_master[z_master < 1.8]

        print(f"{'':<30s} {' process ID ':^25s} | {' halo ID ':^15s} | {' halo redshift ':^20s}\n")
        angle_master = np.zeros((self.simulation.totalClusters, len(z_master), 2), dtype=np.float)
        iterator = itertools.product(self.simulation.clusterIDAllowed, self.simulation.redshiftAllowed)

        for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
            print(f"{'Processing...':<30s} {process_n:^25d} | {halo_id:^15d} | {halo_z:^20s}")
            if self.simulation.sample_completeness[halo_id, self.simulation.redshiftAllowed.index(halo_z)]:
                cluster = Cluster(simulation_name=self.simulation.simulation_name,
                                  clusterID=halo_id,
                                  redshift=halo_z,
                                  fastbrowsing=True)
                read = pull.FOFRead(cluster)
                angle = read.pull_rot_vel_angle_between('Total_angmom', 'Total_ZMF')[self.aperture_id]
                angle_master[halo_id, self.simulation.redshiftAllowed.index(halo_z)][0] = redshift_str2num(halo_z)
                angle_master[halo_id, self.simulation.redshiftAllowed.index(halo_z)][1] = angle

        print(f"Saving npy files: redshift_rotTvelT_simstats_aperture_{self.aperture_id}.npy")
        np.save(os.path.join(self.path, f'redshift_rotTvelT_simstats_aperture_{self.aperture_id}.npy'), angle_master)

    def make_simbootstrap(self):
        # z_master = np.array([redshift_str2num(z) for z in self.simulation.redshiftAllowed])
        # z_master = z_master[z_master < 1.8]

        if not os.path.isfile(os.path.join(self.path, f'redshift_rotTvelT_simstats_aperture_{self.aperture_id}.npy')):
            warnings.warn(f"File redshift_rotTvelT_simstats_aperture_{self.aperture_id}.npy not found.")
            print("self.make_simstats() activated.")
            self.make_simstats()

        print(f"Retrieving npy files: redshift_rotTvelT_simstats_aperture_{self.aperture_id}.npy")
        angle_master = np.load(os.path.join(self.path, f'redshift_rotTvelT_simstats_aperture_{self.aperture_id}.npy'))
        angle_master = np.asarray(angle_master)

        # Bin data from angle_master
        redshift_data = angle_master[:,:,0].flatten()
        angle_data = angle_master[:,:,1].flatten()
        redshift_data_bin_edges = np.histogram_bin_edges(redshift_data, bins=15)
        redshift_data_bin_centres = self.get_centers_from_bins(redshift_data_bin_edges)
        redshift_data_bin_idx = np.digitize(redshift_data, redshift_data_bin_edges)

        percent16_mean = np.zeros_like(redshift_data_bin_idx, dtype=float)
        median50_mean = np.zeros_like(redshift_data_bin_idx, dtype=float)
        percent84_mean = np.zeros_like(redshift_data_bin_idx, dtype=float)
        percent16_std = np.zeros_like(redshift_data_bin_idx, dtype=float)
        median50_std = np.zeros_like(redshift_data_bin_idx, dtype=float)
        percent84_std = np.zeros_like(redshift_data_bin_idx, dtype=float)

        for idx in range(len(redshift_data_bin_edges)):
            _angle_data_binned = angle_data[redshift_data_bin_idx == idx+1]
            if len(_angle_data_binned) is not 0:
                boot_stats = self.bootstrap(_angle_data_binned, n_iterations=self.bootstrap_niters)
            percent16_mean[idx] = boot_stats['percent16'][0]
            median50_mean[idx] = boot_stats['median50'][0]
            percent84_mean[idx] = boot_stats['percent84'][0]
            percent16_std[idx] = boot_stats['percent16'][1]
            median50_std[idx] = boot_stats['median50'][1]
            percent84_std[idx] = boot_stats['percent84'][1]

        sim_bootstrap = np.asarray([
            [redshift_data_bin_centres.tolist(), redshift_data_bin_edges.tolist()],
            [percent16_mean.tolist(), percent16_std.tolist()],
            [median50_mean.tolist(), median50_std.tolist()],
            [percent84_mean.tolist(), percent84_std.tolist()]
        ])

        print(f"Saving npy files: redshift_rotTvelT_bootstrap_aperture_{self.aperture_id}.npy")
        np.save(os.path.join(self.path, f'redshift_rotTvelT_bootstrap_aperture_{self.aperture_id}.npy'), sim_bootstrap)

    def plot_z_trends(self, axis: Axes = None) -> None:

        if axis is None:
            axis = self.figure.add_subplot(111)

        # z_master = np.array([redshift_str2num(z) for z in self.simulation.redshiftAllowed])
        # z_master = z_master[z_master < 1.8]
        cluster = Cluster(simulation_name=self.simulation.simulation_name, clusterID=0, redshift='z000p000')
        aperture_float = self.get_apertures(cluster)[self.aperture_id] / cluster.r200

        if not os.path.isfile(os.path.join(self.path, f'redshift_rotTvelT_bootstrap_aperture_{self.aperture_id}.npy')):
            warnings.warn(f"File redshift_rotTvelT_bootstrap_aperture_{self.aperture_id}.npy not found.")
            print("self.make_simbootstrap() activated.")
            self.make_simbootstrap()

        print(f"Retrieving npy files: redshift_rotTvelT_bootstrap_aperture_{self.aperture_id}.npy")
        sim_bootstrap = np.load(os.path.join(self.path, f'redshift_rotTvelT_bootstrap_aperture_{self.aperture_id}.npy'))
        sim_bootstrap = np.asarray(sim_bootstrap)

        items_labels = f""" REDSHIFT TRENDS
                            Number of clusters: {self.simulation.totalClusters:d}
                            $z$ = {sim_bootstrap[0,0,0]:.2f} - {sim_bootstrap[0,0,-1]:.2f}
                            Aperture radius = {aperture_float:.2f} $R_{{200\ true}}$"""
        print(items_labels)

        sim_colors = {
            'ceagle' : 'pink',
            'celr_e' : 'lime',
            'celr_b' : 'orange',
            'macsis' : 'aqua',
        }

        axis.axhline(90, linestyle='--', color='k', alpha=0.5, linewidth=2)

        axis.plot(sim_bootstrap[0,0], sim_bootstrap[1,0], color=sim_colors[self.simulation.simulation_name],
                alpha=1, linestyle='none', marker='^', markersize=10)
        axis.plot(sim_bootstrap[0,0], sim_bootstrap[2,0], color=sim_colors[self.simulation.simulation_name],
                alpha=1, linestyle='none', marker='o', markersize=10)
        axis.plot(sim_bootstrap[0,0], sim_bootstrap[1,0], color=sim_colors[self.simulation.simulation_name],
                alpha=1, linestyle='none', marker='v', markersize=10)
        axis.plot(sim_bootstrap[0,0], sim_bootstrap[3,0], color = sim_colors[self.simulation.simulation_name],
                alpha = 0.8, drawstyle='steps-mid', linestyle='--', lw=1.5)
        axis.plot(sim_bootstrap[0,0], sim_bootstrap[2,0], color = sim_colors[self.simulation.simulation_name],
                alpha = 0.8,  drawstyle='steps-mid', linestyle='-', lw=1.5)
        axis.plot(sim_bootstrap[0,0], sim_bootstrap[1,0], color = sim_colors[self.simulation.simulation_name],
                alpha = 0.8, drawstyle='steps-mid', linestyle='-.', lw=1.5)
        axis.fill_between(sim_bootstrap[0,0],
                               sim_bootstrap[3,0] - sim_bootstrap[3,1],
                               sim_bootstrap[3,0] + sim_bootstrap[3,1],
                               color = sim_colors[self.simulation.simulation_name],
                               alpha = 0.2, step='mid', edgecolor='none')
        axis.fill_between(sim_bootstrap[0,0],
                               sim_bootstrap[2,0] - sim_bootstrap[2,1],
                               sim_bootstrap[2,0] + sim_bootstrap[2,1],
                               color = sim_colors[self.simulation.simulation_name],
                               alpha = 0.2, step='mid', edgecolor='none')
        axis.fill_between(sim_bootstrap[0,0],
                               sim_bootstrap[1, 0] - sim_bootstrap[1, 1],
                               sim_bootstrap[1, 0] + sim_bootstrap[1, 1],
                               color = sim_colors[self.simulation.simulation_name],
                               alpha = 0.2, step='mid', edgecolor='none')

        perc84 = Line2D([], [], color='k', marker='^', linestyle='--', markersize=10, label=r'$84^{th}$ percentile')
        perc50 = Line2D([], [], color='k', marker='o', linestyle='-', markersize=10, label=r'median')
        perc16 = Line2D([], [], color='k', marker='v', linestyle='-.', markersize=10, label=r'$16^{th}$ percentile')
        patch_ceagle = Patch(facecolor=sim_colors['ceagle'], label='C-EAGLE', edgecolor='k', linewidth=1)
        patch_celre  = Patch(facecolor=sim_colors['celr_e'], label='CELR-E', edgecolor='k', linewidth=1)
        patch_celrb  = Patch(facecolor=sim_colors['celr_b'], label='CELR-B', edgecolor='k', linewidth=1)
        patch_macsis = Patch(facecolor=sim_colors['macsis'], label='MACSIS', edgecolor='k', linewidth=1)

        leg1 = axis.legend(handles=[perc84, perc50, perc16], loc='lower right', handlelength=3, fontsize=20)
        leg2 = axis.legend(handles=[patch_ceagle, patch_celre, patch_celrb, patch_macsis],
                                loc='lower left', handlelength=1, fontsize=20)
        axis.add_artist(leg1)
        axis.add_artist(leg2)
        axis.text(0.03, 0.97, items_labels,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=axis.transAxes,
                        size=15)

        axis.set_xlabel(r"$z$", size=25)
        axis.set_ylabel(r"$\Delta \theta \equiv (\mathbf{L},\widehat{CoP},\mathbf{v_{pec}})$ \quad [degrees]", size=25)
        axis.set_ylim(0, 180)

    def plot_z_trend_histogram(self, axis: Axes = None) -> None:

        if axis is None:
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111)


    def save_z_trend(self):
        plt.savefig(os.path.join(self.path, f'redshift_rotTvelT_aperture_{self.aperture_id}.png'), dpi=300)

    @classmethod
    def run_from_dict(cls, setup: Dict[str, Union[int, str, List[str], Figure]]):

        if setup['run_mode'] is 'single_plot':
            self = cls()
            self.set_aperture(setup['aperture_id'])
            self.set_simulation(setup['simulation_name'])
            self.set_bootstrap_niters(setup['bootstrap_niters'])
            self.set_figure(setup['figure'])
            ax = self.figure.add_subplot(111)
            self.plot_z_trends(axis=ax)
            self.save_z_trend()

        elif setup['run_mode'] is 'multi_sim':
            self = cls()
            self.set_figure(setup['figure'])
            self.set_bootstrap_niters(setup['bootstrap_niters'])
            if type(setup['aperture_id']) is int:
                setup['aperture_id'] = [setup['aperture_id']]
            for aperture in setup['aperture_id']:
                if int(aperture) % size is rank:
                    self.set_aperture(aperture)
                    for sim in setup['simulation_name']:
                        self.set_simulation(sim)
                        self.plot_z_trends(axis=None)
                    self.save_z_trend()
            
        elif setup['run_mode'] is 'multi_bootstrap':
            self = cls()
            self.set_figure(setup['figure'])
            self.set_bootstrap_niters(setup['bootstrap_niters'])
            if type(setup['aperture_id']) is int:
                setup['aperture_id'] = [setup['aperture_id']]
            for aperture in setup['aperture_id']:
                if int(aperture) % size is rank:
                    self.set_aperture(aperture)
                    for sim in setup['simulation_name']:
                        self.set_simulation(sim)
                        self.make_simbootstrap()

        return cls

if __name__ == '__main__':


    def test():
        cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
        matrix = CorrelationMatrix(cluster)
        data = matrix.get_data()
        aperture = matrix.get_apertures()

        matrix.plot_matrix(data[15], aperture[15])

    def all_clusters(apertureidx):
        simulation = Simulation(simulation_name='celr_b')
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
                                 f'{simulation.simulation_name}_meanPMstd_{cluster.redshift}_aperture'
                                 f'_{apertureidx}.png'))


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    setup = {
        'run_mode'        : 'multi_sim',
        'aperture_id'     : 10,
        'simulation_name' : ['celr_b'],
        'bootstrap_niters': 1e3,
        'figure'          : fig,
    }
    trend_z = TrendZ.run_from_dict(setup)




