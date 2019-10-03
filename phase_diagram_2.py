import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from os import makedirs, chdir
from os.path import exists


def phase_diagram_Martin(num_halo, redshift, nbins = 400, bg='k', output = 'show', selection = 'all'):
    # Import data
    path = 		extract.path_from_cluster_name(num_halo, simulation_type = 'gas')
    file = 		extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    r200 = 		extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)

    file = 		extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))

    # Gas particles
    part_type = extract.particle_type('gas')
    density = extract.particle_SPH_density(path, file, part_type)
    coordinates = extract.particle_coordinates(path, file, part_type)
    temperature = extract.particle_temperature(path, file, part_type)
    group_number = extract.group_number(path, file, part_type)
    subgroup_number = extract.subgroup_number(path, file, part_type)


    # Retrieve coordinates
    x = coordinates[:,0] - group_CoP[0]
    y = coordinates[:,1] - group_CoP[1]
    z = coordinates[:,2] - group_CoP[2]

    # Rescale to comoving coordinates
    h = extract.file_hubble_param(path, file)
    x = profile.comoving_length(x, h, redshift)
    y = profile.comoving_length(y, h, redshift)
    z = profile.comoving_length(z, h, redshift)
    r200 = profile.comoving_length(r200, h, redshift)
    density = profile.comoving_density(density, h, redshift)
    density = profile.density_units(density, unit_system = 'astro')

    # Compute radial distance
    r = np.sqrt(x**2+y**2+z**2)

    index = 0
    # Select particles within 5*r200
    if selection.lower() == 'all':
        index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number > -1))[0]
    if selection.lower() == 'sub':
        index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number > 0))[0]
    if selection.lower() == 'icm':
        index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number == 0))[0]

    density = density[index]
    temperature = temperature[index]

    # Bin data
    x_Data = density*10**9
    y_Data = temperature
    x_bins = np.logspace(np.min(np.log10(x_Data)), np.max(np.log10(x_Data)), nbins)
    y_bins = np.logspace(np.min(np.log10(y_Data)), np.max(np.log10(y_Data)), nbins)
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
    Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
    count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None) / A_pix

    # Generate plot
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))

    # Logarithmic normalization
    norm = mpl.colors.LogNorm()#(vmin=10 ** -2, vmax=10 ** 1)

    count2 = np.ma.masked_where(count == 0, count)
    cmap = plt.get_cmap('CMRmap')
    cmap.set_bad(color=bg, alpha=1)

    img = axes.pcolormesh(Cx, Cy, count2, cmap=cmap, norm = norm)
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel(r'$\rho/(M_\odot\ kpc^{-3})$')
    axes.set_ylabel(r'$T/K$')


    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
    cbar.set_label(r'$N_{particles} / (K\ M_\odot\ kpc^{-3})$', labelpad=17)
    #cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
    cax2.xaxis.set_ticks_position("top")

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        save_name = 'phase-diagram_halo' + str(num_halo) + '_z'+ str(redshift).replace(".", "") + '_nbins' + str(nbins) + '_' + selection + '.pdf'
        dir_name = 'phase_diagrams_T-Rho_halos'

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' +save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)

def phase_diagram_Edo(num_halo, redshift, nbins = 400, bg='k', output = 'show', selection = 'all', sgnum = None):
    # Import data
    path = 		extract.path_from_cluster_name(num_halo, simulation_type = 'gas')
    file = 		extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    r200 = 		extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)

    file = 		extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))

    # Gas particles
    part_type = extract.particle_type('gas')
    density = extract.particle_SPH_density(path, file, part_type)
    coordinates = extract.particle_coordinates(path, file, part_type)
    temperature = extract.particle_temperature(path, file, part_type)
    group_number = extract.group_number(path, file, part_type)
    subgroup_number = extract.subgroup_number(path, file, part_type)


    # Retrieve coordinates
    x = coordinates[:,0] - group_CoP[0]
    y = coordinates[:,1] - group_CoP[1]
    z = coordinates[:,2] - group_CoP[2]

    # Rescale to comoving coordinates
    h = extract.file_hubble_param(path, file)
    x = profile.comoving_length(x, h, redshift)
    y = profile.comoving_length(y, h, redshift)
    z = profile.comoving_length(z, h, redshift)
    r200 = profile.comoving_length(r200, h, redshift)
    density = profile.comoving_density(density, h, redshift)
    density = profile.density_units(density, unit_system = 'astro')

    # Compute radial distance
    r = np.sqrt(x**2+y**2+z**2)

    index = 0
    # Select particles within 5*r200
    if selection.lower() == 'all':
        index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number > -1))[0]
    if selection.lower() == 'sub':
        if sgnum == None:
            index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number > 0))[0]
        else:
            index = np.where((r < 5 * r200) & (group_number == 1) & (subgroup_number == int(sgnum)))[0]
    if selection.lower() == 'icm':
        index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number == 0))[0]


    density = density[index]
    temperature = temperature[index]

    iii = np.where(temperature > 10**5)[0]
    print(len(iii)/ len(temperature))
    # Bin data
    x_Data = density*10**9
    y_Data = temperature
    x_bins = np.logspace(np.min(np.log10(x_Data)), np.max(np.log10(x_Data)), nbins)
    y_bins = np.logspace(np.min(np.log10(y_Data)), np.max(np.log10(y_Data)), nbins)
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
    Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
    count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None) / A_pix

    # Generate plot
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))

    # Logarithmic normalization
    norm = mpl.colors.LogNorm()#(vmin=10 ** -2, vmax=10 ** 1)

    count2 = np.ma.masked_where(count == 0, count)
    cmap = plt.get_cmap('CMRmap')
    cmap.set_bad(color=bg, alpha=1)

    img = axes.pcolormesh(Cx, Cy, count2, cmap=cmap, norm = norm)
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel(r'$\rho/(M_\odot\ kpc^{-3})$')
    axes.set_ylabel(r'$T/K$')


    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("top", size="3%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
    cbar.set_label(r'$N_{particles} / (K\ M_\odot\ kpc^{-3})$', labelpad=17)
    #cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
    cax2.xaxis.set_ticks_position("top")

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        save_name = 'phase-diagram_EDO_halo' + str(num_halo) + '_z'+ str(redshift).replace(".", "") + '_nbins' + str(nbins) + '_' + selection+ str(sgnum) + '.jpg'
        dir_name = 'phase_diagrams_T-Rho_halos//sub_test'

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' +save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)

def phase_diagram_master(redshift, nbins = 500, output = 'save', selection = 'all', bg='w'):
    master_density = []
    master_temperature = []
    for num_halo in np.arange(390):
        print('Ímporting halo ' + str(num_halo))

        # Import data
        path = extract.path_from_cluster_name(num_halo, simulation_type='gas')
        file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))
        r200 = extract.group_r200(path, file)
        group_CoP = extract.group_centre_of_potential(path, file)

        file = extract.file_name_hdf5(subject='particledata', redshift=extract.redshift_floatTostr(redshift))

        # Gas particles
        part_type = extract.particle_type('gas')
        density = extract.particle_SPH_density(path, file, part_type)
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
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
    Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
    count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights=None) / A_pix

    # Generate plot
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    # Logarithmic normalization
    norm = mpl.colors.LogNorm()  # (vmin=10 ** -2, vmax=10 ** 1)

    count2 = np.ma.masked_where(count == 0, count)
    cmap = plt.get_cmap('CMRmap')
    cmap.set_bad(color=bg, alpha=1)

    img = axes.pcolormesh(Cx, Cy, count2, cmap=cmap, norm=norm)
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel(r'$n_{\mathrm{H}}/\mathrm{cm}^{3}$')
    #axes.set_xlabel(r'$\rho/(M_\odot\ kpc^{-3})$')
    axes.set_ylabel(r'$T/\mathrm{K}$')

    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("top", size="3%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
    cbar.set_label(r'$N_{\mathrm{particles}} / (\mathrm{K}\ \mathrm{cm}^{-3})$', labelpad=-70)
    # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
    cax2.xaxis.set_ticks_position("top")

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        save_name = 'phase-diagram_nHcgs_halo' + str(10) + '_z' + str(redshift).replace(".", "") + '_nbins' + str(
            nbins) + '_' + selection + '.png'
        dir_name = 'phase_diagrams_T-Rho_master390'

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' + save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)

def call_maps(rank):
    num_halo = rank
    redshift = 0.57

    # Call function:
    # selection options: 'all'/'icm'/'sub'
    for selection in ['all','sub','icm']:
        for nbins in [600]:
            phase_diagram_Edo(redshift, nbins=nbins, output='save', selection=selection, bg='k')

            #if selection == 'sub':
            #    for sg in [1073741824]:
            #        phase_diagram_Edo(num_halo, redshift, nbins = nbins, selection=selection, bg='k', output = 'show', sgnum = int(sg))
            #else:
            #    phase_diagram_Edo(num_halo, redshift, nbins=nbins, selection=selection, bg='k', output='show', sgnum=None)



# **************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python phase_diagram_2.py

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process:', rank)
    plt.ion()
    call_maps(rank)

# **************************************************************************************************