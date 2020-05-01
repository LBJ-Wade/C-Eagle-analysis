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
        fit_vec -= 3  # 4 comes from the prior on the number of changepoints
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
sys.stdout = open(os.devnull, 'w')
read_apertures = [Simstats(simulation_name='macsis', aperture_id=i).read_simstats() for i in range(20)]
sys.stdout = sys.__stdout__
for apid, stat in enumerate(read_apertures):
    print(f"Aperture radius {apid} \t --> \t {stat['R_aperture'][0]/stat['R_200_crit'][0]:1.2f} R_200_crit")
del read_apertures
sys.stdout = open(os.devnull, 'w')
read_redshifts = [Simstats(simulation_name=i, aperture_id=0).read_simstats() for i in ['macsis', 'celr_e']]
sys.stdout = sys.__stdout__
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
print(f"\n{' stats_out DATASET INFO ':-^40s}")
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
    scale = 'log' if 'M' in x or 'velocity' in x else 'linear'
    xscale.append(scale)
for y in y_labels:
    scale = 'log' if 'M' in y or 'velocity' in y else 'linear'
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
print(f"\n{' summary DATASET PLOTS INFO ':-^40s}\n", summary)



# Activate the plot factory
print(f"\n{' RUNNING PLOT FACTORY... ':-^40s}")
data_entries = summary.to_dict('r')
for entry_index, data_entry in enumerate(data_entries):
    filename = f"_{data_entry['x'].replace('_', '')}_{data_entry['y'].replace('_', '')}_aperture"f"{aperture_id}.png"
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

    # Unresolved issue with the Latex labels
    # Some contain an extra `$` at the end of the string, which should not be there.
    label_x = attrs[0]['Columns/labels'][data_entry['x']]
    label_y = attrs[0]['Columns/labels'][data_entry['y']]
    if label_x.endswith('$'): label_x = label_x.rstrip('$')
    if label_y.endswith('$'): label_y = label_y.rstrip('$')
    ax[0].set_ylabel(label_y)
    ax[1].set_ylabel(label_y)
    ax[1].set_xlabel(label_x)
    ax[2].set_xlabel(label_x)
    ax[3].set_xlabel(label_x)

    simstats_palette = ['#1B9E77','#D95F02','#7570B3','#E7298A']

    z_range = [np.min(pd.concat(stats_filtered)['redshift_float']),
               np.max(pd.concat(stats_filtered)['redshift_float'])]
    z_range_str = f'{z_range[0]:1.2f} - {z_range[1]:1.2f}' if round(z_range[0]) < round(z_range[1]) else f'{z_range[0]:1.2f}'
    items_labels = [
        f"{label_x.split(r'quad')[0]} -\\ {label_y.split(r'quad')[0]}",
        f"Number of clusters: {np.sum([attr['Number of clusters'] for attr in attrs]):d}",
        f"$z$ = {z_range_str:s}",
        f"Aperture radius = {stats_filtered[0]['R_aperture'][0] / stats_filtered[0]['R_200_crit'][0]:2.2f} $R_{{200\\ true}}$"
    ]
    info_ax0.text(0.03, 0.97, '\n'.join(items_labels), horizontalalignment='left', verticalalignment='top', size=15, transform=info_ax0.transAxes)

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
            axes.text(0.95, 0.95, f'\\textsc{{Total}}', transform=axes.transAxes, **axisinfo_kwargs)
        else:
            axes.scatter(
                    stats_filtered[ax_idx-1][data_entry['x']],
                    stats_filtered[ax_idx-1][data_entry['y']],
                    s=5,
                    c=simstats_palette[ax_idx-1]
            )
            axes.text(0.95, 0.95, f"\\textsc{{{attrs[ax_idx-1]['Simulation']}}}", transform=axes.transAxes, **axisinfo_kwargs)

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
            axes.text(0.95, 0.95, f'\\textsc{{Total}}', transform=axes.transAxes, **axisinfo_kwargs)
        else:
            x = stats_filtered[ax_idx-1][data_entry['x']]
            y = stats_filtered[ax_idx-1][data_entry['y']]
            values = np.vstack([x if data_entry['xscale'] is 'linear' else np.log10(x), y])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            #cfset = axes.contourf(xx, yy, f, cmap='Blues')
            cset = axes.contour(xx if data_entry['xscale'] is 'linear' else 10**xx, yy, f, colors=simstats_palette[ax_idx-1])
            axes.scatter(x, y, s=3, c=simstats_palette[ax_idx-1], alpha=0.2)
            axes.text(0.95, 0.95, f"\\textsc{{{attrs[ax_idx-1]['Simulation']}}}", transform=axes.transAxes, **axisinfo_kwargs)

    plt.savefig(os.path.join(pathSave, 'kdeplot'+filename))
    print(f"[+] Plot {entry_index:3d}/{len(data_entries)} Figure saved: {'kdeplot'+filename}")


    ##################################################################################################
    # MEDIAN PLOTS #
    ##################################################################################################
    fig_median = fig
    ax_median = [fig_median.axes[i] for i in [1, 3, 4, 5]]
    for axes in ax_median:
        for artist in axes.lines + axes.collections:
            artist.remove()

    perc84 = Line2D([], [], color='k', marker='^', linestyle='-.', markersize=12, label=r'$84^{th}$ percentile')
    perc50 = Line2D([], [], color='k', marker='o', linestyle='-', markersize=12, label=r'median')
    perc16 = Line2D([], [], color='k', marker='v', linestyle='--', markersize=12, label=r'$16^{th}$ percentile')
    leg1 = fig_median.axes[2].legend(handles=[perc84, perc50, perc16], loc='center right', handlelength=2, fontsize=20)
    fig_median.axes[2].add_artist(leg1)
    xlims = [np.min(pd.concat(stats_filtered)[data_entry['x']]), np.max(pd.concat(stats_filtered)[data_entry['x']])]
    ylims = [np.min(pd.concat(stats_filtered)[data_entry['y']]), np.max(pd.concat(stats_filtered)[data_entry['y']])]
    x_space = np.linspace(np.log10(xlims[0]), np.log10(xlims[1]), 101)
    y_space = np.linspace(ylims[0], ylims[1], 101)

    for ax_idx, axes in enumerate(ax_median):
        axes.set_xlim([xlims[0] - 0.1 * np.diff(xlims), xlims[1] + 0.1 * np.diff(xlims)])
        axes.set_ylim([ylims[0] - 0.1 * np.diff(ylims), ylims[1] + 0.1 * np.diff(ylims)])
        axes_to_data = axes.transAxes + axes.transData.inverted()
        ax_frame = axes_to_data.transform
        if ax_idx == 0:
            x = pd.concat(stats_filtered)[data_entry['x']]
            y = pd.concat(stats_filtered)[data_entry['y']]

            # Compute the candlestick widths
            ax_xlims = axes.get_xlim()
            ax_ylims = axes.get_ylim()
            width = ax_xlims[1] - ax_xlims[0] if data_entry['xscale'] is 'linear' else np.log10(ax_xlims[1]) - np.log10(ax_xlims[0])
            height = ax_ylims[1] - ax_ylims[0] if data_entry['yscale'] is 'linear' else np.log10(ax_ylims[1]) - np.log10(ax_ylims[0])
            candlestick_h_kwargs = dict(align='edge',
                                        left=np.median(x),
                                        height=0.05 * height,
                                        xerr=np.std(x) / np.sqrt(len(x)),
                                        ecolor='k',
                                        edgecolor='k',
                                        facecolor=simstats_palette[ax_idx - 1],
                                        alpha=1
                                        )
            candlestick_v_kwargs = dict(align='edge',
                                        bottom=np.median(y),
                                        width=0.05 * width,
                                        yerr=np.std(y) / np.sqrt(len(y)),
                                        ecolor='k',
                                        edgecolor='k',
                                        facecolor=simstats_palette[ax_idx - 1],
                                        alpha=1
                                        )

            x_bin_stats = bayesian_blocks(x) if data_entry['xscale'] is 'linear' else 10 ** bayesian_blocks(np.log10(x))
            median_y, edges, _ = st.binned_statistic(x, y, statistic='median', bins=x_bin_stats)
            percent84_y, _, _  = st.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 84), bins=x_bin_stats)
            percent16_y, _, _  = st.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 16), bins=x_bin_stats)
            count_y, _, _      = st.binned_statistic(x, y, statistic='count', bins=x_bin_stats)
            std_y, _, _        = st.binned_statistic(x, y, statistic='std', bins=x_bin_stats)
            median_x = edges[:-1] + np.diff(edges) / 2
            axes.scatter(x, y, s=3, c=simstats_palette[ax_idx - 1], alpha=0.2)
            axes.errorbar(median_x, median_y, yerr=std_y / np.sqrt(count_y),
                          marker='o', ms=8, c=simstats_palette[ax_idx - 1], alpha=1,
                          linestyle='-', capsize=0)
            axes.errorbar(median_x, percent16_y, yerr=std_y / np.sqrt(count_y),
                          marker='v', ms=8, c=simstats_palette[ax_idx - 1], alpha=1,
                          linestyle='--', capsize=0)
            axes.errorbar(median_x, percent84_y, yerr=std_y / np.sqrt(count_y),
                          marker='^', ms=8, c=simstats_palette[ax_idx - 1], alpha=1,
                          linestyle='-.', capsize=0)

            axes.barh(ax_frame((0, 0))[1], np.percentile(x, 84) - np.median(x), **candlestick_h_kwargs)
            axes.barh(ax_frame((0, 0))[1], np.percentile(x, 16) - np.median(x), **candlestick_h_kwargs)
            axes.barh(ax_frame((0, 0))[1], 0, **candlestick_h_kwargs)
            axes.bar(ax_frame((0, 0))[0], np.percentile(y, 84) - np.median(y), **candlestick_v_kwargs)
            axes.bar(ax_frame((0, 0))[0], np.percentile(y, 16) - np.median(y), **candlestick_v_kwargs)
            axes.bar(ax_frame((0, 0))[0], 0, **candlestick_v_kwargs)
            axes.text(0.95, 0.95, '\\textsc{Total}', transform=axes.transAxes, **axisinfo_kwargs)
        else:
            x = stats_filtered[ax_idx - 1][data_entry['x']]
            y = stats_filtered[ax_idx - 1][data_entry['y']]

            # Compute the candlestick widths
            ax_xlims = axes.get_xlim()
            ax_ylims = axes.get_ylim()
            width  = ax_xlims[1]-ax_xlims[0] if data_entry['xscale'] is 'linear' else np.log10(ax_xlims[1])-np.log10(ax_xlims[0])
            height = ax_ylims[1]-ax_ylims[0] if data_entry['yscale'] is 'linear' else np.log10(ax_ylims[1])-np.log10(ax_ylims[0])
            candlestick_h_kwargs = dict(align='edge',
                                        left=np.median(x),
                                        height=0.05*height,
                                        xerr=np.std(x) / np.sqrt(len(x)),
                                        ecolor='k',
                                        edgecolor='k',
                                        facecolor=simstats_palette[ax_idx - 1],
                                        alpha=1
                                        )
            candlestick_v_kwargs = dict(align='edge',
                                        bottom=np.median(y),
                                        width=0.05*width,
                                        yerr=np.std(y) / np.sqrt(len(y)),
                                        ecolor='k',
                                        edgecolor='k',
                                        facecolor=simstats_palette[ax_idx - 1],
                                        alpha=1
                                        )

            x_bin_stats = bayesian_blocks(x) if data_entry['xscale'] is 'linear' else 10 ** bayesian_blocks(np.log10(x))
            median_y, edges, _ = st.binned_statistic(x, y, statistic='median', bins=x_bin_stats)
            percent84_y, _, _  = st.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 84), bins=x_bin_stats)
            percent16_y, _, _  = st.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 16), bins=x_bin_stats)
            count_y, _, _      = st.binned_statistic(x, y, statistic='count', bins=x_bin_stats)
            std_y, _, _        = st.binned_statistic(x, y, statistic='std', bins=x_bin_stats)
            median_x = edges[: -1] + np.diff(edges) / 2
            axes.scatter(x, y, s=3, c=simstats_palette[ax_idx - 1], alpha=0.2)
            axes.errorbar(median_x, median_y, yerr=std_y / np.sqrt(count_y),
                          marker='o', ms=8, c=simstats_palette[ax_idx - 1], alpha=1,
                          linestyle='-', capsize=0)
            axes.errorbar(median_x, percent16_y, yerr=std_y / np.sqrt(count_y),
                          marker='v', ms=8, c=simstats_palette[ax_idx - 1], alpha=1,
                          linestyle='--', capsize=0)
            axes.errorbar(median_x, percent84_y, yerr=std_y / np.sqrt(count_y),
                          marker='^', ms=8, c=simstats_palette[ax_idx - 1], alpha=1,
                          linestyle='-.', capsize=0)
            axes.barh(ax_frame((0, 0))[1], np.percentile(x, 84) - np.median(x), **candlestick_h_kwargs)
            axes.barh(ax_frame((0, 0))[1], np.percentile(x, 16) - np.median(x), **candlestick_h_kwargs)
            axes.barh(ax_frame((0, 0))[1], 0, **candlestick_h_kwargs)
            axes.bar(ax_frame((0, 0))[0], np.percentile(y, 84) - np.median(y), **candlestick_v_kwargs)
            axes.bar(ax_frame((0, 0))[0], np.percentile(y, 16) - np.median(y), **candlestick_v_kwargs)
            axes.bar(ax_frame((0, 0))[0], 0, **candlestick_v_kwargs)
            axes.text(0.95, 0.95, f"\\textsc{{{attrs[ax_idx - 1]['Simulation']}}}", transform=axes.transAxes, **axisinfo_kwargs)

    plt.savefig(os.path.join(pathSave, 'median' + filename))
    print(f"[+] Plot {entry_index:3d}/{len(data_entries)} Figure saved: {'median' + filename}")
