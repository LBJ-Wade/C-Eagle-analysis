import os, sys
import itertools
import warnings
from typing import List
import numpy as np
import slack
import socket

# Graphics packages
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
plt.style.use("mnras.mplstyle")

# Internal packages
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import utils

def save_plot(filepath: str, to_slack: bool = False, **kwargs) -> None:
	"""
	Function to parse the plt.savefig method and send the file to a slack channel.
	:param filepath: The path+filename+extension where to save the figure
	:param to_slack: Gives the option to send the the plot to slack
	:param kwargs: Other kwargs to parse into the plt.savefig
	:return: None
	"""
	plt.savefig(filepath, **kwargs)
	if to_slack:
		print(
				"[+] Forwarding to the `#personal` Slack channel",
				f"\tDir: {os.path.dirname(filepath)}",
				f"\tFile: {os.path.basename(filepath)}",
				sep='\n'
		)
		slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
		slack_msg = f"Host: {socket.gethostname()}\nDir: {os.path.dirname(filepath)}\nFile: {os.path.basename(filepath)}"
		try:
			# Send files to Slack: init slack client with access token
			client = slack.WebClient(token=slack_token)
			client.files_upload(file=filepath, initial_comment=slack_msg, channels='#personal')
		except:
			warnings.warn("[-] Failed to broadcast plot to Slack channel.")

def median_plot(axes: plt.Axes, x: np.ndarray, y: np.ndarray,  **kwargs):

	perc84 = Line2D([], [], color='k', marker='^', linewidth=1, linestyle='-', markersize=3, label=r'$84^{th}$ percentile')
	perc50 = Line2D([], [], color='k', marker='o', linewidth=1, linestyle='-', markersize=3, label=r'median')
	perc16 = Line2D([], [], color='k', marker='v', linewidth=1, linestyle='-', markersize=3, label=r'$16^{th}$ percentile')
	legend = axes.legend(handles=[perc84, perc50, perc16], loc='lower right', handlelength=2)
	axes.add_artist(legend)
	data_plot = utils.medians_2d(x, y, **kwargs)
	axes.errorbar(data_plot['median_x'], data_plot['median_y'], yerr=data_plot['err_y'],
	              marker='o', ms=2, alpha=1, linestyle='-', capsize=0, linewidth=0.5)
	axes.errorbar(data_plot['median_x'], data_plot['percent16_y'], yerr=data_plot['err_y'],
	              marker='v', ms=2, alpha=1, linestyle='-', capsize=0, linewidth=0.5)
	axes.errorbar(data_plot['median_x'], data_plot['percent84_y'], yerr=data_plot['err_y'],
	              marker='^', ms=2, alpha=1, linestyle='-', capsize=0, linewidth=0.5)

def kde_plot(axes: plt.Axes, x: np.ndarray, y: np.ndarray, **kwargs):

	kde_results = utils.kde_2d(x, y, **kwargs)
	clevels = axes.contourf(*kde_results, 10, cmap='YlGn_r')
	# axes.contour(clevels, levels=cset.levels[::2], cmap='YlGn_r')

	# Delete outer levels
	for level in clevels.collections:
		for kp, path in reversed(list(enumerate(level.get_paths()))):
			verts = path.vertices  # (N,2)-shape array of contour line coordinates
			diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
			dataset_diameter = max([(x.max()-x.min()), (y.max()-y.min())])
			if diameter > 0.7*dataset_diameter:
				del (level.get_paths()[kp])

	# Identify points within contours
	inside = np.full_like(x, False, dtype=bool)
	for level in clevels.collections:
		for kp, path in reversed(list(enumerate(level.get_paths()))):
			inside |= path.contains_points(tuple(zip(*(x,y))))
	ax.scatter(x[~inside], y[~inside], marker='.', color='g', s=2, alpha=0.1)


def snap_label(axes: plt.Axes, redshift: str, aperture: int) -> None:
	label = f"BAHAMAS\n$z$={utils.redshift_str2num(redshift):2.2f}\n$R_\\mathrm{{aperture}}$={utils.aperture_labels[aperture]}"
	axes.text(0.95, 0.95, label, transform=axes.transAxes, horizontalalignment='right', verticalalignment='top',)


redshift = 'z000p000'
aperture = 7
#-----------------------------------------------------------------

m500 = utils.read_snap_output(redshift, apertureID=aperture, dataset='m500')
c_l  = utils.read_snap_output(redshift, apertureID=aperture, dataset='c_l')
figname = f'bahamas_hyd_alignment_{redshift}.png'


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale("log")
ax.set_yscale("linear")
ax.set_xlabel(utils.datasets_names['m500'])
ax.set_ylabel(utils.datasets_names['c_l'])
kde_plot(ax, m500, c_l[:,1,1], axscales = ['log', 'linear'], gridbins=300)
median_plot(ax, m500, c_l[:,1,1], axscales = ['log', 'linear'], binning_method = 'equalnumber')
snap_label(ax, redshift, aperture)
save_plot(os.path.join(utils.basepath, figname), to_slack=True, dpi=300)