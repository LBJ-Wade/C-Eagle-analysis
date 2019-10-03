import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as patches

import map_plot_parameters as plotpar
import map_synthetizer as mapgen
import gas_fractions_gridSpec_calculator as data
import map_kSZ_substructure_match as submatch

from os.path import exists
from os import makedirs, chdir
from itertools import count
from matplotlib import cm
from matplotlib.ticker import NullFormatter
from matplotlib.colorbar import Colorbar

def source_data(switchargs):
    redshift, projection, norm, max_HaloNum = switchargs
    # Setup variables
    dir_name = 'Substructure_Match_Output_NoSelection'
    nbins = 600
    rfov = 5
    master_selection = {'H': [], 'I': [], 'R': [], 'M': [], 'Fg': [], 'Vr': [], 'MV': []}

    # Loop over halos
    for num_halo in range(max_HaloNum):
        print('loading halo ' + str(num_halo))

        # define file names
        save_name_selected = 'selected_kSZ' + '_halo' + str(num_halo) + '_z' + str(
            redshift).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '_proj' + str(projection)

        # load data from files
        sg_selection = np.load(dir_name + '//' + save_name_selected + '.npy').item()

        # append data to master files
        for key in master_selection.keys():
            if len(sg_selection[key]) > 0:
                master_selection[key] = np.concatenate((master_selection[key], sg_selection[key])) if len(master_selection[key])>0 else sg_selection[key]

    return master_selection

def main_figure_normed_new(redshift = 0.57, projection = 0, nbins=50, output='show', caller = '', max_HaloNum = 10):
    plotpar.set_defaults_plot()
    nullfmt = NullFormatter()
    fig = plt.figure(1, figsize=(10, 10))

    # Now, create the gridspec structure, as required
    gs = gridspec.GridSpec(ncols=3, nrows=4,
                           height_ratios=[0.065, 1, 1, 0.5],
                           width_ratios=[1, 1, 0.5])

    # 3 rows, 4 columns, each with the required size ratios.
    # Also make sure the margins and spacing are appropriate

    gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.08, hspace=0.12)

    # Note: I set the margins to make it look good on my screen ...
    # BUT: this is irrelevant for the saved image, if using bbox_inches='tight'in savefig !

    # Note: Here, I use a little trick. I only have three vertical layers of plots :
    # a scatter plot, a histogram, and a line plot. So, in principle, I could use a 3x3 structure.
    # However, I want to have the histogram 'closer' from the scatter plot than the line plot.
    # So, I insert a 4th layer between the histogram and line plot,
    # keep it empty, and use its thickness (the 0.2 above) to adjust the space as required.

    colormap = cm.get_cmap('viridis_r')
    hist_color = 'blue'
    histedge_color = 'blue'

    if caller.lower() == 'martin':
        Martin_Dream_CMap = mapgen.GreyMap(256)
        Martin_Hist_Color = Martin_Dream_CMap(0.1)
        Martin_HistEdge_Color = Martin_Dream_CMap(0.99)
        colormap = Martin_Dream_CMap
        hist_color = Martin_Hist_Color
        #histedge_color = Martin_HistEdge_Color

    if caller.lower() == 'edo':
        Martin_Dream_CMap = mapgen.Martin_Dream_CMap(256)
        Martin_Hist_Color = Martin_Dream_CMap(0.99)
        colormap = Martin_Dream_CMap
        hist_color = Martin_Hist_Color

    if caller.lower() == 'manchester':
        Manchester_CMap = mapgen.Manchester_CMap(256)
        Manchester_Hist_Color = Manchester_CMap(0.99)
        colormap = Manchester_CMap
        hist_color = Manchester_Hist_Color

    # LABELS
    label_n = r'$\log_{10} n_{sub}$'
    label_M = r'$M/M_\odot$'
    label_R = r'$R/R_{200}$'
    label_f = r'$f_{g}$'
    label_v = r'$v_{z}/\mathrm{(km\ s^{-1})}$'

    # GRIDS & NBINS
    grid_on = False

    # DATA
    switchargs = (redshift, projection, True, max_HaloNum)
    master_selection = source_data(switchargs)

    # Search option:
    #master_data = search_halos(max_HaloNum=max_HaloNum)

    # loop over plots
    for j in [0, 1]:
        for i in [0, 1]:

            print('Block started')

            if i == 0 and j == 0:
                u = master_selection['R']
                v = master_selection['Fg']

            if i == 1 and j == 0:
                u = master_selection['M']
                v = master_selection['Fg']

            if i == 0 and j == 1:
                u = master_selection['R']
                v = master_selection['Vr']

            if i == 1 and j == 1:
                u = master_selection['M']
                v = master_selection['Vr']

            x_min_LIN, x_max_LIN = np.min(u), np.max(u)
            x_min_LOG, x_max_LOG = np.log10(x_min_LIN), np.log10(x_max_LIN)
            y_min_LIN, y_max_LIN = np.min(v), np.max(v)
            if j == 0: y_min_LIN, y_max_LIN = 0, 0.3

            # First, the scatter plot
            ax1 = fig.add_subplot(gs[j + 1, i])
            ax1.tick_params(labelbottom='off')
            print('\tComputing 2dhist \t\t (%1i, %1i)' % (j + 1, i))

            # # Get the optimal number of bins based on knuth_bin_width
            # N_xbins = int((np.max(x)-np.min(x))/knuth_bin_width(x)) + 1
            # N_ybins = int((np.max(y)-np.min(y))/knuth_bin_width(y)) + 1

            # Compute bins
            N_xbins = nbins
            N_ybins = N_xbins
            bins_LOG = np.logspace(x_min_LOG, x_max_LOG, num=N_xbins)
            bins_LIN = np.linspace(y_min_LIN, y_max_LIN, num=N_ybins)
            bin_centers_LIN = mapgen.get_centers_from_bins(bins_LIN)
            bin_centers_LOG = mapgen.get_centers_from_log_bins(bins_LOG)

            # 2D DISTRIBUTIONS
            Cx, Cy = mapgen.bins_meshify(u, v, bins_LOG, bins_LIN)
            selection_count = mapgen.bins_evaluate(u, v, bins_LOG, bins_LIN, weights=None)

            selection_count[selection_count==0] = np.nan
            count = np.log10(selection_count)

            cmap = plt.get_cmap(colormap)
            cmap.set_bad(color='w',alpha=1)

            # MARGINAL DISTRIBUTIONS

            #from scipy.stats.contingency import margins
            #x_marginal, y_marginal = margins(count)
            #x_marginal, y_marginal = x_marginal.T[0], y_marginal[0]

            horz_marginals, _ = np.histogram(u, bins_LOG)
            vert_marginals, _ = np.histogram(v, bins_LIN)

            horz_marginals = np.log10(horz_marginals)
            vert_marginals = np.log10(vert_marginals)


            # norm = colors.NoNorm()
            plt1 = ax1.pcolormesh(Cx, Cy, count, cmap=colormap)#, norm=norm)
            ax1.grid(grid_on)
            ax1.set_xlim([x_min_LIN, x_max_LIN])
            ax1.set_ylim([y_min_LIN, y_max_LIN])
            ax1.set_xscale('log')
            ax1.set_yscale('linear')
            ax1.set_xlabel(r' ')  # Force this empty !
            ax1.set_ylabel(label_f)

            if j == 0:
                ax1.set_ylabel(label_f)
            elif j == 1:
                ax1.set_ylabel(label_v)

            # Colorbar
            if j==0 and i ==0:
                cbax = fig.add_subplot(gs[j, :-1])
                print('\tComputing colorbar \t\t (%1i, %1i)' % (j, i))
                cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
                cb.set_label(label_n, labelpad=10)
                #cb.set_ticks([0., 0.25, 0.5, 0.75, 1.])
                #cb.set_ticklabels([r'$0$', r'$1/4$', r'$1/2$', r'$3/4$', r'$1$'])
                trig_vertical_hist = 0

            # VERTICAL HISTOGRAM
            if i != 0:
                ax1v = fig.add_subplot(gs[j + 1, i + 1])
                print('\tComputing vert hist \t (%1i, %1i)' % (j + 1, i + 1))
                ax1v.barh(bin_centers_LIN, vert_marginals, height=np.diff(bins_LIN), edgecolor=histedge_color, linewidth=2, color=hist_color)
                ax1v.barh(bin_centers_LIN, vert_marginals, height=np.diff(bins_LIN), edgecolor=hist_color,
                          color=hist_color)
                #ax1v.bar(bins_LIN, x_marginal, orientation='horizontal', edgecolor='red', histtype='step', cumulative=-1)
                #ax1v.hist(x_marginal, bins=bins_LIN, orientation='horizontal', color='k', histtype='step')
                #ax1v.hist(x_marginal, bins=bins_LIN, orientation='horizontal', color='red', histtype='step', cumulative=-1)
                ax1v.set_yticks(ax1.get_yticks())  # Ensures we have the same ticks as the scatter plot !
                ax1v.set_xscale('linear')
                ax1v.set_yscale('linear')
                #ax1v.set_xticks([0., 0.5, 1.])
                ax1v.tick_params(labelleft=False)
                ax1v.set_ylim(ax1.get_ylim())
                ax1.yaxis.set_major_formatter(nullfmt)
                ax1.set_ylabel('')
                ax1v.grid(grid_on)
                if j==0:
                    ax1v.tick_params(labelbottom=False)
                    ax1v.xaxis.set_major_formatter(ticker.NullFormatter())
                    ax1v.xaxis.set_minor_formatter(ticker.NullFormatter())
                    #ax1v.set_xlabel(r' ')
                    #ax1v.set_xticklabels([r'', r'', r''])
                if j==1:
                    ax1v.set_xticks(ax1v.get_xticks())
                    ax1v.set_xlim(ax1v.get_xlim())
                    ax1v.set_xlabel(label_n)
                    #ax1v.set_xticklabels([r'$0$', r'$1/2$', r'$1$'])
                trig_vertical_hist = False

            # Percentiles
            percents = [15.9, 50, 84.1]
            percent_str = [r'$16\%$', r'$50\%$', r'$84\%$']
            clr = ['orange', 'blue', 'green']
            percent_ticks = np.percentile(v, percents)
            if trig_vertical_hist:
                percent_str = np.flipud(percent_str)
                clr = np.flipud(clr)
                ax1v_TWIN = ax1v.twinx()
                ax1v_TWIN.set_ylim(ax1.get_ylim())
                ax1v_TWIN.tick_params(axis='y', which='both', labelleft=False, labelright=True)
                ax1v_TWIN.set_yticks(percent_ticks)
                ax1v_TWIN.set_yticklabels(percent_str)
                for percent_tick, c, tick in zip(percent_ticks, clr, ax1v_TWIN.yaxis.get_major_ticks()):
                    tick.label1.set_color(c)
                    ax1v_TWIN.axhline(y=percent_tick, color=c, linestyle='--')
                percent_str = np.flipud(percent_str)
                clr = np.flipud(clr)

            # HORIZONTAL HISTOGRAM
            if j != 0:
                ax1h = fig.add_subplot(gs[j + 2, i])
                print('\tComputing horiz hist \t (%1i, %1i)' % (j + 2, i))
                ax1h.bar(bin_centers_LOG, horz_marginals, width=np.diff(bins_LOG), log=True, edgecolor=histedge_color, linewidth=2, color=hist_color)
                ax1h.bar(bin_centers_LOG, horz_marginals, width=np.diff(bins_LOG), log=True, edgecolor=hist_color,
                         color=hist_color)
                #ax1h.hist(y_marginal, bins=bins_LOG, orientation='vertical', color='k', histtype='step')
                #ax1h.hist(y_marginal, bins=bins_LOG, orientation='vertical', color='red', histtype='step', cumulative=True)
                ax1h.set_xticks(ax1.get_xticks())  # Ensures we have the same ticks as the scatter plot!
                ax1h.set_xlim(ax1.get_xlim())
                ax1h.set_xscale('log')
                ax1h.set_yscale('linear')
                if i == 0:
                    ax1h.set_xlabel(label_R)
                    ax1h.set_ylabel(label_n)
                    ax1.xaxis.set_major_formatter(ticker.NullFormatter())
                    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
                    ax1h.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: (r'${{:.{:1d}f}}$'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
                    ax1h.xaxis.set_minor_formatter(ticker.NullFormatter())

                elif i == 1:
                    ax1h.set_xlabel(label_M)
                    ax1h.tick_params(labelleft=False)
                    ax1h.set_ylabel('')
                    ax1h.set_yticks(ax1h.get_yticks())  # Ensures we have the same ticks as the scatter plot!
                    ax1h.set_ylim(ax1h.get_ylim())

                ax1h.grid(grid_on)
                #ax1h.set_yticks([0., 0.5, 1.])
                #ax1h.set_yticklabels([r'$0$', r'$1/2$', r'$1$'])

            #for i in range(len(percents)): ax1h.axvline(x=percent_ticks[i], color=clr[i], linestyle='--')
            print('Block completed\n')

    if output.lower() == 'show':
        fig.show()

    elif output.lower() == 'save':
        dir_name = 'MatchedSG_Characteristics'
        if not exists(dir_name): makedirs(dir_name)
        save_name = 'all_char_norm_new' + '_z' + str(redshift).replace(".", "") + '_proj' + str(
                projection) + '_nbins' + str(nbins) + '_NoSelection_maxhalo' + str(max_HaloNum) + '_' + caller + '.pdf'
        fig.savefig(dir_name + '//' + save_name, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

    elif output.lower() == 'none':
        pass

    else:
        print("Error: Invalid request")

if __name__ == "__main__":
    main_figure_normed_new(projection = 0, nbins=60, output='save', caller='martin', max_HaloNum = 390)