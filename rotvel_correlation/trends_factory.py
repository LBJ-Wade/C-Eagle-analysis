import sys
import os
import warnings
import itertools
from typing import Union, List
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

exec(open(os.path.abspath(os.path.join(
		os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation
from rotvel_correlation.simstats import Simstats

warnings.filterwarnings("ignore")
pathSave = '/cosma6/data/dp004/dc-alta2/C-Eagle-analysis-work/rotvel_correlation'

def bayesian_blocks(t):
    """Bayesian Blocks Implementation
    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed
    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges
    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size
    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges
    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)
    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]
        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]
        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]
    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]
    return edges[change_points]

# Print some overall stats about the datasets
read_apertures = [Simstats(simulation_name='macsis', aperture_id=i).read_simstats() for i in range(20)]
for apid, stat in enumerate(read_apertures):
    print(f"Aperture radius {apid} \t --> \t {stat['R_aperture'][0]/stat['R_200_crit'][0]:1.2f} R_200_crit")
del read_apertures
read_redshifts = [Simstats(simulation_name=i, aperture_id=0).read_simstats() for i in ['macsis', 'celr_e']]
for sim_name, stat in zip(['macsis', 'celr_e'], read_redshifts):
    print('\n')
    for zid, redshift in enumerate(stat.query('cluster_id == 0')['redshift_float']):
        print(f"Simulation: {sim_name:<10s} Redshift {zid:2d} --> {redshift:1.2f}")
del read_redshifts

# Start with one single aperture
aperture_id = 9
simstats = list()
simstats.append(Simstats(simulation_name='macsis', aperture_id=aperture_id))
simstats.append(Simstats(simulation_name='celr_e', aperture_id=aperture_id))
simstats.append(Simstats(simulation_name='celr_b', aperture_id=aperture_id))
stats_out = [sim.read_simstats() for sim in simstats]
attrs = [sim.read_metadata() for sim in simstats]
print(stats_out[0].info())

# Create SQL query
query_COLLECTIVE = list()
query_COLLECTIVE.append('redshift_float < 0.02')
query_COLLECTIVE.append('M_200_crit > 10**9')
query_COLLECTIVE.append('thermodynamic_merging_index_T < 1')
stats_filtered = [stat.query(' and '.join(query_COLLECTIVE)) for stat in stats_out]

# Generate plots catalog
x_labels = ['redshift_float', 'R_500_crit', 'R_aperture', 'M_2500_crit', 'M_aperture_T',
            'peculiar_velocity_T_magnitude', 'angular_momentum_T_magnitude',
            'dynamical_merging_index_T', 'thermodynamic_merging_index_T',
            'substructure_fraction_T']
y_labels = ['M_200_crit','rotTvelT','rot0rot4','rot1rot4','dynamical_merging_index_T',
            'thermodynamic_merging_index_T','substructure_fraction_T']
data_entries = list(itertools.product(x_labels, y_labels))
x_labels = []
y_labels = []
for entry in data_entries:
    if entry[0] is not entry[1]:
        x_labels.append(entry[0])
        y_labels.append(entry[1])
xscale = []
yscale = []
for x in x_labels:
    scale = 'log' if 'M' in x or 'R' in x or 'velocity' in x else 'linear'
    xscale.append(scale)
for y in y_labels:
    scale = 'log' if 'M' in y or 'R' in y or 'velocity' in y else 'linear'
    yscale.append(scale)
data_summary = {
    'x' : x_labels,
    'y' : y_labels,
    'xscale' : xscale,
    'yscale' : yscale,
}
summary = pd.DataFrame(data=data_summary, columns=data_summary.keys())
summary = summary[summary['y'].str.contains('rot')]
summary = summary[~summary['x'].str.contains('redshift')]
print(summary)

# Activate the plot factory
data_entries = summary.to_dict('r')
for entry_index, data_entry in enumerate(data_entries):
    filename = f"_{data_entry['x'].replace('_', '')}_{data_entry['y'].replace('_', '')}_aperture"f"{aperture_id}.pdf"
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    gs.update(wspace=0., hspace=0.)
    info_ax0 = fig.add_subplot(gs[0]); info_ax0.axis('off')
    ax1 = fig.add_subplot(gs[1])
    info_ax1 = fig.add_subplot(gs[2]); info_ax1.axis('off')
    ax2 = fig.add_subplot(gs[3], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[4], sharex=ax2, sharey=ax2)
    ax4 = fig.add_subplot(gs[5], sharex=ax3, sharey=ax3)
    ax = [ax1, ax2, ax3, ax4]
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[2].get_yticklabels(), visible=False)
    plt.setp(ax[3].get_yticklabels(), visible=False)
    xlims = [np.min(pd.concat(stats_filtered)[data_entry['x']]), np.max(pd.concat(stats_filtered)[data_entry['x']])]
    ylims = [np.min(pd.concat(stats_filtered)[data_entry['y']]), np.max(pd.concat(stats_filtered)[data_entry['y']])]
    label_x = data_entry['x']
    label_y = data_entry['y']
    # label_x = attrs[0]['Columns/labels'][data_entry['x']].replace('{{', '{').replace('}', '}}')
    # label_y = attrs[0]['Columns/labels'][data_entry['y']].replace('{{', '{').replace('}', '}}')
    ax[0].set_ylabel(label_y)
    ax[1].set_ylabel(label_y)
    ax[1].set_xlabel(label_x)
    ax[2].set_xlabel(label_x)
    ax[3].set_xlabel(label_x)
    simstats_palette = ['#1B9E77','#D95F02','#7570B3','#E7298A']
    items_info = (
            label_x.split('[')[0].strip('quad'),
            label_y.split('[')[0],
            np.sum([attr['Number of clusters'] for attr in attrs]),
            attrs[0]['Redshift bounds'],
            stats_filtered[0]['R_aperture'][0] / stats_filtered[0]['R_200_crit'][0],
    )
    items_labels = r"""
            %s \textemdash\ %s
            Number of clusters: %d
            $z$ = %s
            Aperture radius = %2.2f $R_{{200\ true}}$""".format(*items_info)

    info_ax0.text(0.03, 0.97, items_labels, horizontalalignment='left', verticalalignment='top', size=15, transform=info_ax0.transAxes)

    axisinfo_kwargs = dict(
            horizontalalignment='right',
            verticalalignment='top',
            size=15
    )
    handles = [Patch(facecolor=simstats_palette[i], label=attrs[i]['Simulation'], edgecolor='k', linewidth=1) for i in range(len(attrs))]
    leg = info_ax1.legend(handles=handles, loc='lower right', handlelength=1, fontsize=20)
    info_ax1.add_artist(leg)
    ##################################################################################################
    # SCATTER PLOTS #
    ##################################################################################################
    for ax_idx, axes in enumerate(ax):
        axes.set_xscale(data_entry['xscale'])
        axes.set_yscale(data_entry['yscale'])
        axes.tick_params(direction='in', length=5, top=True, right=True)
        if ax_idx == 0:
            axes.scatter(
                    pd.concat(stats_filtered)[data_entry['x']],
                    pd.concat(stats_filtered)[data_entry['y']],
                    s=5,
                    c=simstats_palette[ax_idx-1]
            )
            axes.text(0.95, 0.95, r'\textsc{Total}', transform=axes.transAxes, **axisinfo_kwargs)
        else:
            axes.scatter(
                    stats_filtered[ax_idx-1][data_entry['x']],
                    stats_filtered[ax_idx-1][data_entry['y']],
                    s=5,
                    c=simstats_palette[ax_idx-1]
            )
            axes.text(0.95, 0.95, f"\textsc{{{attrs[ax_idx-1]['Simulation']}}}", transform=axes.transAxes, **axisinfo_kwargs)

    plt.savefig(os.path.join(pathSave, 'scatterplot'+filename))
    print(f"[+] Plot {entry_index:3d}/{len(data_entries)} Figure saved: {'scatterplot'+filename}")


    ##################################################################################################
    # kde PLOTS #
    ##################################################################################################
    fig_kde = fig
    ax_kde = [fig_kde.axes[i] for i in [1, 3, 4, 5]]
    for axes in ax_kde:
        for artist in axes.lines + axes.collections:
            artist.remove()

    x_space = np.linspace(xlims[0], xlims[1], 101)
    y_space = np.linspace(ylims[0], ylims[1], 101)
    if data_entry['xscale'] is 'log':
        x_space = np.linspace(np.log10(xlims[0]), np.log10(xlims[1]), 101)
    if data_entry['yscale'] is 'log':
        y_space = np.linspace(np.log10(ylims[0]), np.log10(ylims[1]), 101)
    xx, yy = np.meshgrid(x_space, y_space)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    for ax_idx, axes in enumerate(ax_kde):
        if ax_idx == 0:
            x = pd.concat(stats_filtered)[data_entry['x']]
            y = pd.concat(stats_filtered)[data_entry['y']]
            values = np.vstack([x if data_entry['xscale'] is 'linear' else np.log10(x), y])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            #cfset = axes.contourf(xx, yy, f, cmap='Blues')
            cset = axes.contour(xx if data_entry['xscale'] is 'linear' else 10**xx, yy, f, colors=simstats_palette[ax_idx-1])
            axes.scatter(x, y, s=3, c=simstats_palette[ax_idx-1], alpha=0.2)
            axes.text(0.95, 0.95, r'\textsc{Total}', transform=axes.transAxes, **axisinfo_kwargs)
        else:
            x = stats_filtered[ax_idx-1][data_entry['x']]
            y = stats_filtered[ax_idx-1][data_entry['y']]
            values = np.vstack([x if data_entry['xscale'] is 'linear' else np.log10(x), y])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            #cfset = axes.contourf(xx, yy, f, cmap='Blues')
            cset = axes.contour(xx if data_entry['xscale'] is 'linear' else 10**xx, yy, f, colors=simstats_palette[ax_idx-1])
            axes.scatter(x, y, s=3, c=simstats_palette[ax_idx-1], alpha=0.2)
            axes.text(0.95, 0.95, f"\textsc{{{attrs[ax_idx-1]['Simulation']}}}", transform=axes.transAxes, **axisinfo_kwargs)

    plt.savefig(os.path.join(pathSave, 'kdeplot_'+filename))
    print(f"[+] Plot {entry_index:3d}/{len(data_entries)} Figure saved: {'kdeplot_'+filename}")
    ##################################################################################################
    # MEDIAN PLOTS #
    ##################################################################################################
