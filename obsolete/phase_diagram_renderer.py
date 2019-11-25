import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
from obsolete import map_synthetizer as mapgen

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from os import makedirs
from os.path import exists


def phase_diagram_master(axes, redshift, nbins = 400, max_halo=10, selection = 'all', bg='w'):
    master_density = []
    master_temperature = []
    for num_halo in np.arange(max_halo):
        print('Ímporting halo ' + str(num_halo))

        # Import data
        path = extract.path_from_cluster_name(num_halo, simulation_type='gas')
        file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))
        r200 = extract.group_r200(path, file)
        group_CoP = extract.group_centre_of_potential(path, file)

        file = extract.file_name_hdf5(subject='particledata', redshift=extract.redshift_floatTostr(redshift))

        # Gas particles
        part_type = extract.particle_type('gas')
        density = extract.particle_SPH_density(path, file, part_type)/1.16 # Account for the fact that there are H/He atoms
        coordinates = extract.particle_coordinates(path, file, part_type)
        temperature = extract.particle_temperature(path, file, part_type)
        group_number = extract.group_number(path, file, part_type)
        subgroup_number = extract.subgroup_number(path, file, part_type)

        # Retrieve coordinates
        x = coordinates[:, 0] - group_CoP[0]
        y = coordinates[:, 1] - group_CoP[1]
        z = coordinates[:, 2] - group_CoP[2]

        # Rescale to comoving coordinates
        h = extract.file_hubble_param(path, file)
        x = profile.comoving_length(x, h, redshift)
        y = profile.comoving_length(y, h, redshift)
        z = profile.comoving_length(z, h, redshift)
        r200 = profile.comoving_length(r200, h, redshift)
        density = profile.comoving_density(density, h, redshift)
        density = profile.density_units(density, unit_system='nHcgs')

        # Compute radial distance
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        index = 0
        # Select particles within 5*r200
        if selection.lower() == 'all':
            index = np.where((r < 5 * r200) & (group_number == 1) & (subgroup_number > -1))[0]
        if selection.lower() == 'sub':
            index = np.where((r < 5 * r200) & (group_number == 1) & (subgroup_number > 0) & (subgroup_number < 10000))[0]
        if selection.lower() == 'icm':
            index = np.where((r < 5 * r200) & (group_number == 1) & (subgroup_number == 0))[0]

        density = density[index]
        temperature = temperature[index]

        master_density.append(density)
        master_temperature.append((temperature))


    # Bin data
    x_Data = np.concatenate(master_density)
    y_Data = np.concatenate(master_temperature)
    x_bins = np.logspace(np.min(np.log10(x_Data)), np.max(np.log10(x_Data)), nbins)
    y_bins = np.logspace(np.min(np.log10(y_Data)), np.max(np.log10(y_Data)), nbins)
    #A_pix = (x_bins[1:] - x_bins[:-1]) * (y_bins[1] - y_bins[0])
    #Ex, Ey = np.meshgrid(x_bins, y_bins)
    #A_pix = np.asarray([np.multiply((Ex[i][1:]-Ex[i][:-1]),(Ey[i+1][0]-Ey[i][0])) for i in range(np.shape(Ex)[0]-1)])
    Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
    #count = np.divide(mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights=None), A_pix)
    count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights=None)

    # Logarithmic normalization
    norm = mpl.colors.LogNorm()  # (vmin=10 ** -2, vmax=10 ** 1)

    count2 = np.ma.masked_where(count == 0, count)
    cmap = plt.get_cmap('CMRmap')
    cmap.set_bad(color=bg, alpha=1)

    img = axes.pcolormesh(Cx, Cy, count2, cmap=cmap, norm=norm)
    axes.set_facecolor('k')
    axes.set_xscale('log')
    axes.set_yscale('log')
    #axes.axvline(x=0.1, linewidth=1, color='w', linestyle='dotted')
    axes.axhline(y=1e5, linewidth=1, color='w', linestyle='dashed')
    #__M__ axes.axhline(y=1e5, linewidth=1, color='darkgrey', linestyle='dashed')
    axes.set_xlabel(r'$n_{\mathrm{H}}/\mathrm{cm}^{3}$')
    axes.set_xlim(2e-6, 1e3)
    #axes.set_xlabel(r'$\rho/(M_\odot\ kpc^{-3})$')
    if selection=='all':
        axes.set_ylabel(r'$T/\mathrm{K}$')
        t = axes.text(.15, 10 ** 9, r'$\mathrm{ICM\ +\ SUBHALOS}$', color='w', fontsize = 15)
        #__M__t = axes.text(1, 10 ** 9, r'$\mathrm{ALL\ GAS}$', color='w', fontsize=15)

    else:
        axes.set_ylabel(r' ')
        t = axes.text(.2, 10 ** 9, r'$\mathrm{SUBHALOS\ ONLY}$', color='w', fontsize = 15)
        #__M__t = axes.text(.07, 10 ** 9, r'$\mathrm{SUBHALO\ GAS\ ONLY}$', color='w', fontsize=15)
    t.set_bbox(dict(facecolor='k', alpha=0.9, edgecolor='grey'))
    #__M__t.set_bbox(dict(facecolor='w', alpha=0.15, edgecolor='darkgrey'))
    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("top", size="3%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
    #cbar.set_label(r'$N_{\mathrm{particles}} / (\mathrm{K}\ \mathrm{cm}^{-3})$', labelpad=-60)
    cbar.set_label(r'$N_{\mathrm{particles}}$', labelpad=-60)
    # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
    cax2.xaxis.set_ticks_position("top")



def phasediag_1x2(redshift, output='show', nbins=600, max_halo=10):
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 7), sharey=True)
    phase_diagram_master(axes[0], redshift, nbins=nbins, max_halo=max_halo, selection='all', bg='k')
    phase_diagram_master(axes[1], redshift, nbins=nbins, max_halo=max_halo, selection='sub', bg='k')
    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        save_name = 'phase-diagram_nHcgs_2x1_halo' + str(max_halo) + '_z' + str(redshift).replace(".", "") + '_nbins' + str(
            nbins) + '.png'
        dir_name = 'phase_diagrams_T-Rho_master390'

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' + save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)

if __name__ == "__main__":
    redshift = 0.57
    phasediag_1x2(redshift, output='save', nbins=500, max_halo=390)

