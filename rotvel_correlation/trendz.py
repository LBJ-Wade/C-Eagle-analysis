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
from rotvel_correlation.align_angles import CorrelationMatrix


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
	def get_widths_from_bins(bins):
		""" return centers from bin sequence """
		# return bins[1:] - bins[:-1]
		return np.array([np.abs(bins[i+1] - bins[i]) for i in range(len(bins)-1)])

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
				angle = read.pull_rot_vel_angle_between('ParType0_angmom', 'ParType4_angmom')[self.aperture_id]
				angle_master[halo_id, self.simulation.redshiftAllowed.index(halo_z), 0] = redshift_str2num(halo_z)
				angle_master[halo_id, self.simulation.redshiftAllowed.index(halo_z), 1] = angle

		print(f"Saving npy files: redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy")
		np.save(os.path.join(self.path, f'redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy'), angle_master)

	def make_simbootstrap(self):

		if not os.path.isfile(os.path.join(self.path, f'redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy')):
			warnings.warn(f"File redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy not found.")
			print("self.make_simstats() activated.")
			self.make_simstats()

		print(f"Retrieving npy files: redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy")
		angle_master = np.load(os.path.join(self.path, f'redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy'))
		angle_master = np.asarray(angle_master)

		# Bin data from angle_master
		redshift_data = angle_master[:,:,0].flatten()
		angle_data = angle_master[:,:,1].flatten()
		redshift_data_bin_centres = np.linspace(0., 1.8, 15)
		redshift_data_bin_edges = self.get_centers_from_bins(redshift_data_bin_centres)
		pre_redshift_extension = redshift_data_bin_centres[0] - abs(redshift_data_bin_centres[0] - redshift_data_bin_edges[0])
		post_redshift_extension = redshift_data_bin_centres[-1] + abs(redshift_data_bin_centres[-1] - redshift_data_bin_edges[-1])
		redshift_data_bin_edges = np.insert(redshift_data_bin_edges, 0, pre_redshift_extension)
		redshift_data_bin_edges = np.insert(redshift_data_bin_edges, len(redshift_data_bin_edges), post_redshift_extension)
		redshift_data_bin_idx = np.digitize(redshift_data, redshift_data_bin_edges)
		redshift_data_bin_widths = self.get_widths_from_bins(redshift_data_bin_edges)
		redshift_data_bin_widths[0] = redshift_data_bin_widths[0]/2

		# Compute percentiles for the z-trends
		percent16_mean = np.zeros_like(redshift_data_bin_centres, dtype=float)
		median50_mean = np.zeros_like(redshift_data_bin_centres, dtype=float)
		percent84_mean = np.zeros_like(redshift_data_bin_centres, dtype=float)
		percent16_std = np.zeros_like(redshift_data_bin_centres, dtype=float)
		median50_std = np.zeros_like(redshift_data_bin_centres, dtype=float)
		percent84_std = np.zeros_like(redshift_data_bin_centres, dtype=float)

		for idx in range(len(redshift_data_bin_centres)):
			_angle_data_binned = angle_data[redshift_data_bin_idx == idx+1]
			if len(_angle_data_binned) is not 0:
				boot_stats = self.bootstrap(_angle_data_binned, n_iterations=self.bootstrap_niters)
				percent16_mean[idx] = boot_stats['percent16'][0]
				median50_mean[idx] = boot_stats['median50'][0]
				percent84_mean[idx] = boot_stats['percent84'][0]
				percent16_std[idx] = boot_stats['percent16'][1]
				median50_std[idx] = boot_stats['median50'][1]
				percent84_std[idx] = boot_stats['percent84'][1]
			else:
				percent16_mean[idx] = np.nan
				median50_mean[idx] = np.nan
				percent84_mean[idx] = np.nan
				percent16_std[idx] = np.nan
				median50_std[idx] = np.nan
				percent84_std[idx] = np.nan

		# Compute the collapsed histograms


		sim_bootstrap = np.array([
			[redshift_data_bin_centres, redshift_data_bin_widths],
			[percent16_mean, percent16_std],
			[median50_mean, median50_std],
			[percent84_mean, percent84_std]
		])

		print(f"Saving npy files: redshift_rot0rot4_bootstrap_aperture_{self.aperture_id}.npy")
		np.save(os.path.join(self.path, f'redshift_rot0rot4_bootstrap_aperture_{self.aperture_id}.npy'), sim_bootstrap)

	@ProgressBar()
	def make_simhist(self):

		if not os.path.isfile(os.path.join(self.path, f'redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy')):
			warnings.warn(f"File redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy not found.")
			print("self.make_simstats() activated.")
			self.make_simstats()

		print(f"Retrieving npy files: redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy")
		angle_master = np.load(os.path.join(self.path, f'redshift_rot0vel4_simstats_aperture_{self.aperture_id}.npy'))
		angle_master = np.asarray(angle_master)

		# Bin data from angle_master
		angle_data = angle_master[:,:,1].flatten()
		angle_data_bin_edges = np.linspace(0, 180, 21)
		angle_data_bin_count = np.histogram(angle_data, bins=angle_data_bin_edges)[0]
		angle_data_bin_centres = self.get_centers_from_bins(angle_data_bin_edges)
		angle_data_bin_widths = self.get_widths_from_bins(angle_data_bin_edges)

		# Bootstrap the histograms and compute stats
		N_iters = int(self.bootstrap_niters)
		stats_resampled = np.zeros((N_iters, len(angle_data_bin_count)), dtype=np.float)
		counter = 0
		for seed in range(N_iters):
			data_resampled = resample(angle_data, replace=True, n_samples=len(angle_data), random_state=seed)
			stats_resampled[seed] = np.histogram(data_resampled, bins=angle_data_bin_edges)[0]
			yield ((counter + 1) / N_iters)
			counter += 1

		stats_resampled_MEAN = np.mean(stats_resampled, axis=0)
		stats_resampled_STD = np.std(stats_resampled, axis=0)

		sim_bootstrap = np.array([
			angle_data_bin_centres, angle_data_bin_widths,
			stats_resampled_MEAN, stats_resampled_STD,
		])

		print(f"Saving npy files: redshift_rot0rot4_histogram_aperture_{self.aperture_id}.npy")
		np.save(os.path.join(self.path, f'redshift_rot0rot4_histogram_aperture_{self.aperture_id}.npy'), sim_bootstrap)

	def plot_z_trends(self, axis: Axes = None) -> None:

		if axis is None:
			axis = self.figure.add_subplot(111)

		cluster = Cluster(simulation_name=self.simulation.simulation_name, clusterID=0, redshift='z000p000')
		aperture_float = self.get_apertures(cluster)[self.aperture_id] / cluster.r200

		if not os.path.isfile(os.path.join(self.path, f'redshift_rot0rot4_bootstrap_aperture_{self.aperture_id}.npy')):
			warnings.warn(f"File redshift_rot0rot4_bootstrap_aperture_{self.aperture_id}.npy not found.")
			print("self.make_simbootstrap() activated.")
			self.make_simbootstrap()

		print(f"Retrieving npy files: redshift_rot0rot4_bootstrap_aperture_{self.aperture_id}.npy")
		sim_bootstrap = np.load(os.path.join(self.path, f'redshift_rot0rot4_bootstrap_aperture_'
		f'{self.aperture_id}.npy'), allow_pickle=True)
		sim_bootstrap = np.asarray(sim_bootstrap)

		items_labels = f""" REDSHIFT TRENDS
							Number of clusters: {self.simulation.totalClusters:d}
							$z$ = 0.0 - 1.8
							Aperture radius = {aperture_float:.2f} $R_{{200\ true}}$"""
		print(items_labels)

		sim_colors = {
			'ceagle' : 'pink',
			'celr_e' : 'lime',
			'celr_b' : 'orange',
			'macsis' : 'aqua',
		}

		axis.axhline(90, linestyle='--', color='k', alpha=0.5, linewidth=2)

		axis.plot(sim_bootstrap[0,0], sim_bootstrap[3,0], color=sim_colors[self.simulation.simulation_name],
				alpha=1, linestyle='none', marker='^', markersize=10)
		axis.plot(sim_bootstrap[0,0], sim_bootstrap[2,0], color=sim_colors[self.simulation.simulation_name],
				alpha=1, linestyle='none', marker='o', markersize=10)
		axis.plot(sim_bootstrap[0,0], sim_bootstrap[1,0], color=sim_colors[self.simulation.simulation_name],
				alpha=1, linestyle='none', marker='v', markersize=10)

		for marker_index in range(len(sim_bootstrap[0, 0])):

			if marker_index is 0:
				align_toggle = 'edge'
				x_edge_left = sim_bootstrap[0,0][marker_index]
				x_edge_right = sim_bootstrap[0,0][marker_index]+sim_bootstrap[0,1][marker_index]
			else:
				align_toggle = 'center'
				x_edge_left = sim_bootstrap[0,0][marker_index]-sim_bootstrap[0,1][marker_index]/2
				x_edge_right = sim_bootstrap[0,0][marker_index]+sim_bootstrap[0,1][marker_index]/2

			axis.plot([x_edge_left, x_edge_right],
					  [sim_bootstrap[3,0][marker_index], sim_bootstrap[3,0][marker_index]],
					  color = sim_colors[self.simulation.simulation_name],
					  alpha = 0.8, linestyle='--', lw=1.5)

			axis.plot([x_edge_left, x_edge_right],
					  [sim_bootstrap[2,0][marker_index], sim_bootstrap[2,0][marker_index]],
					  color = sim_colors[self.simulation.simulation_name],
					  alpha = 0.8, linestyle='-', lw=1.5)

			axis.plot([x_edge_left, x_edge_right],
					  [sim_bootstrap[1,0][marker_index], sim_bootstrap[1,0][marker_index]],
					  color = sim_colors[self.simulation.simulation_name],
					  alpha = 0.8, linestyle='-.', lw=1.5)

			axis.bar(sim_bootstrap[0,0][marker_index], 2*sim_bootstrap[3,1][marker_index],
					 bottom=sim_bootstrap[3,0][marker_index]-sim_bootstrap[3,1][marker_index],
					 width=sim_bootstrap[0,1][marker_index],
					 align=align_toggle,
					 color = sim_colors[self.simulation.simulation_name],
					 alpha = 0.2,
					 edgecolor='none', linewidth=0)
			axis.bar(sim_bootstrap[0,0][marker_index], 2*sim_bootstrap[2,1][marker_index],
					 bottom=sim_bootstrap[2,0][marker_index]-sim_bootstrap[2,1][marker_index],
					 width=sim_bootstrap[0,1][marker_index],
					 align=align_toggle,
					 color = sim_colors[self.simulation.simulation_name],
					 alpha = 0.2,
					 edgecolor='none', linewidth=0)
			axis.bar(sim_bootstrap[0,0][marker_index], 2*sim_bootstrap[1,1][marker_index],
					 bottom=sim_bootstrap[1,0][marker_index]-sim_bootstrap[1,1][marker_index],
					 width=sim_bootstrap[0,1][marker_index],
					 align=align_toggle,
					 color = sim_colors[self.simulation.simulation_name],
					 alpha = 0.2,
					 edgecolor='none', linewidth=0)

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
		axis.set_ylabel(r"$\Delta \theta \equiv (\mathbf{L},\mathrm{\widehat{CoP}},\mathbf{v_{pec}})$\quad[degrees]", size=25)
		axis.set_ylim(0, 180)

	def plot_z_trend_histogram(self, axis: Axes = None, polar: bool = True, normed: bool = True) -> None:

		if axis is None:
			axis = self.figure.add_subplot(111)

		cluster = Cluster(simulation_name=self.simulation.simulation_name, clusterID=0, redshift='z000p000')
		aperture_float = self.get_apertures(cluster)[self.aperture_id] / cluster.r200

		if not os.path.isfile(os.path.join(self.path, f'redshift_rot0rot4_histogram_aperture_{self.aperture_id}.npy')):
			warnings.warn(f"File redshift_rot0rot4_histogram_aperture_{self.aperture_id}.npy not found.")
			print("self.make_simhist() activated.")
			self.make_simhist()

		print(f"Retrieving npy files: redshift_rot0rot4_histogram_aperture_{self.aperture_id}.npy")
		sim_hist = np.load(os.path.join(self.path, f'redshift_rot0rot4_histogram_aperture_{self.aperture_id}.npy'), allow_pickle=True)
		sim_hist = np.asarray(sim_hist)

		if normed:
			norm_factor = np.sum(self.simulation.sample_completeness)
			sim_hist[2] /= norm_factor
			sim_hist[3] /= norm_factor
			y_label = r"Sample fraction"
		else:
			y_label = r"Number of samples"


		items_labels = f""" REDSHIFT TRENDS - HISTOGRAM
							Number of clusters: {self.simulation.totalClusters:d}
							$z$ = 0.0 - 1.8
							Total samples: {np.sum(self.simulation.sample_completeness):d} $\equiv N_\mathrm{{clusters}} \cdot N_\mathrm{{redshifts}}$
							Aperture radius = {aperture_float:.2f} $R_{{200\ true}}$"""
		print(items_labels)

		sim_colors = {
				'ceagle': 'pink',
				'celr_e': 'lime',
				'celr_b': 'orange',
				'macsis': 'aqua',
		}

		axis.axvline(90, linestyle='--', color='k', alpha=0.5, linewidth=2)
		axis.step(sim_hist[0], sim_hist[2], color=sim_colors[self.simulation.simulation_name], where='mid')
		axis.fill_between(sim_hist[0], sim_hist[2]+sim_hist[3], sim_hist[2]-sim_hist[3],
		                  step='mid',
		                  color=sim_colors[self.simulation.simulation_name],
		                  alpha=0.2,
		                  edgecolor='none',
		                  linewidth=0
		                  )

		axis.set_ylabel(y_label, size=25)
		axis.set_xlabel(r"$\Delta \theta \equiv (\mathbf{L}_\mathrm{gas},\mathrm{\widehat{CoP}},\mathbf{L}_\mathrm{stars})$\quad[degrees]", size=25)
		axis.set_xlim(0, 180)
		axis.set_ylim(0, 0.1)
		axis.text(0.03, 0.97, items_labels,
		          horizontalalignment='left',
		          verticalalignment='top',
		          transform=axis.transAxes,
		          size=15)

		if polar:
			inset_axis = self.figure.add_axes([0.75, 0.65, 0.25, 0.25], projection='polar')
			inset_axis.patch.set_alpha(0)  # Transparent background
			inset_axis.set_theta_zero_location('N')
			inset_axis.set_thetamin(0)
			inset_axis.set_thetamax(180)
			inset_axis.set_xticks(np.pi/180. * np.linspace(0,  180, 5, endpoint=True))
			inset_axis.set_yticks([])
			inset_axis.step(sim_hist[0]/180*np.pi, sim_hist[2], color=sim_colors[self.simulation.simulation_name], where='mid')
			inset_axis.fill_between(sim_hist[0]/180*np.pi, sim_hist[2] + sim_hist[3], sim_hist[2] - sim_hist[3],
			                  step='mid',
			                  color=sim_colors[self.simulation.simulation_name],
			                  alpha=0.2,
			                  edgecolor='none',
			                  linewidth=0
			                  )

		patch_ceagle = Patch(facecolor=sim_colors['ceagle'], label='C-EAGLE', edgecolor='k', linewidth=1)
		patch_celre = Patch(facecolor=sim_colors['celr_e'], label='CELR-E', edgecolor='k', linewidth=1)
		patch_celrb = Patch(facecolor=sim_colors['celr_b'], label='CELR-B', edgecolor='k', linewidth=1)
		patch_macsis = Patch(facecolor=sim_colors['macsis'], label='MACSIS', edgecolor='k', linewidth=1)

		leg2 = axis.legend(handles=[patch_ceagle, patch_celre, patch_celrb, patch_macsis],
		                   loc='lower center', handlelength=1, fontsize=20)
		axis.add_artist(leg2)

		# arrow_radius = np.max(sim_hist[2])/1.5
		# vpec_direction = np.average(sim_hist[0], weights=sim_hist[2]/np.max(sim_hist[2]))/180*np.pi
		# kw = dict(arrowstyle="<|-", color='k', lw=1.5)
		# inset_axis.annotate(r"$\mathbf{L}$", xytext=(np.pi/2, arrow_radius), xy=(0, 0), xycoords='polar', arrowprops=kw, ha="center", va="center")
		# inset_axis.annotate(r"$\mathbf{v_{pec}}$", xytext=(vpec_direction, arrow_radius), xy=(vpec_direction, 0), xycoords='polar', arrowprops=kw,
		#                     ha="center", va="center")

		# inset_axis.arrow(0, 0, 0, arrow_radius, alpha=0.5, width=0.15, edgecolor='black', facecolor='black', lw=20, zorder=5)
		# inset_axis.arrow(0, 0, np.pi/2, arrow_radius, alpha=0.5, width=0.15, edgecolor='black', facecolor='black', lw=20, zorder=5)


	def save_z_trend(self, common_folder: bool = False) -> None:
		extension = "png"

		if common_folder:
			common_path = os.path.join(self.simulation.pathSave, 'rotvel_correlation')
			if not os.path.exists(common_path):
				os.makedirs(common_path)
			print(f"Saving {extension} figure: redshift_rot0rot4_aperture_{self.aperture_id}.{extension}")
			plt.savefig(os.path.join(common_path, f'redshift_rot0rot4_aperture_{self.aperture_id}.{extension}'))

		else:
			print(f"Saving {extension} figure: {self.simulation.simulation_name}_redshift_rot0rot4_aperture_{self.aperture_id}.{extension}")
			plt.savefig(os.path.join(self.path, f'{self.simulation.simulation_name}_redshift_rot0rot4_aperture_{self.aperture_id}.{extension}'), dpi=300)


	def save_z_trend_hist(self, common_folder: bool = False) -> None:
		extension = "png"

		if common_folder:
			common_path = os.path.join(self.simulation.pathSave, 'rotvel_correlation')
			if not os.path.exists(common_path):
				os.makedirs(common_path)
			print(f"Saving {extension} figure: redshift_rot0rot4_histogram_aperture_{self.aperture_id}.{extension}")
			plt.savefig(os.path.join(common_path, f'redshift_rot0rot4_histogram_aperture_{self.aperture_id}.{extension}'))

		else:
			print(f"Saving {extension} figure: {self.simulation.simulation_name}_redshift_rot0rot4_histogram_aperture_{self.aperture_id}.{extension}")
			plt.savefig(os.path.join(self.path, f'{self.simulation.simulation_name}_redshift_rot0rot4_histogram_aperture_{self.aperture_id}.{extension}'), dpi=300)



	@classmethod
	def run_from_dict(cls, setup: Dict[str, Union[int, str, List[str], Figure]]):
		self = cls()
		if setup['run_mode'] is 'single_plot':
			self.set_aperture(setup['aperture_id'])
			self.set_simulation(setup['simulation_name'])
			self.set_bootstrap_niters(setup['bootstrap_niters'])
			self.set_figure(setup['figure'])
			ax = self.figure.add_subplot(111)
			self.plot_z_trends(axis=ax)
			self.save_z_trend()
			plt.cla()
			self.plot_z_trend_histogram(axis=ax)
			self.save_z_trend_hist()


		elif setup['run_mode'] is 'multi_sim':
			self.set_bootstrap_niters(setup['bootstrap_niters'])
			if type(setup['aperture_id']) is int:
				setup['aperture_id'] = [setup['aperture_id']]
			for aperture in setup['aperture_id']:
				if int(aperture) % size is rank:
					self.set_aperture(aperture)
					fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
					self.set_figure(fig)
					for sim in setup['simulation_name']:
						self.set_simulation(sim)
						self.plot_z_trends(axis=axes[setup['simulation_name'].index(sim)])
					self.save_z_trend(common_folder=True)
					plt.clf()
					fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
					self.set_figure(fig)
					for sim in setup['simulation_name']:
						self.set_simulation(sim)
						self.plot_z_trend_histogram(axis=axes[setup['simulation_name'].index(sim)])
					self.save_z_trend_hist(common_folder=True)

		elif setup['run_mode'] is 'multi_bootstrap':
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

	fig = plt.figure(figsize=(10, 10))
	# ax = fig.add_subplot(111)
	#
	# grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
	#
	# # Define the axes
	# ax_main = fig.add_subplot(grid[:-1, :-1])
	# ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
	# ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

	setup = {
		'run_mode'        : 'multi_sim',
		'aperture_id'     : [1, 7, 10, 15],
		'simulation_name' : ['celr_b', 'celr_e', 'macsis'],
		'bootstrap_niters': 5e4,
		'figure'          : fig,
	}
	trend_z = TrendZ.run_from_dict(setup)

