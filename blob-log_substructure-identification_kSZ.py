import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen
import kernel_convolver as kernconv
import distance_cosmology as cosmo
import subhalo_marker as sgmark

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.constants import c, sigma_T
from scipy.stats import gamma
from os import makedirs, chdir
from os.path import exists

from skimage.feature import blob_dog, blob_log, blob_doh

# from mpi4py.futures import MPIPoolExecutor

# Turn off FutureWarnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def substructure_identification(map, rfov, r200):
    if map.shape[0] != map.shape[1]:
        print("Map is not square!")
        exit(0)

    nbins = len(map)
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    dx, dy = x_bins[1] - x_bins[0], y_bins[1] - y_bins[0]

    # Grayscale image
    map_array = np.reshape(np.log10(np.abs(map)), (1, (nbins - 1) ** 2))
    img_par_scale = 0.75 * np.max(map_array)
    img_par_offset = np.percentile(map_array, 75) / img_par_scale

    aux_map = img_par_offset - np.log10(np.abs(map)) / img_par_scale

    aux_map[aux_map == -np.inf] = 0
    aux_map[aux_map < 0] = 0
    aux_map[aux_map > 1] = 1

    _, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type='gauss')

    # Blob identification
    test = fwhm / (60 * dx)

    min_sigma = test * 1
    max_sigma = test * 10
    num_sigma = 80
    threshold = 0.05

    blobs_log = blob_log(aux_map, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

    # Take raw output from the blob log algorithm
    bly = blobs_log[:, 0]
    blx = blobs_log[:, 1]
    blr = blobs_log[:, 2]

    # Create data storage for the blobl markers and convert to plot axis coordinate system
    blob_catalog_dict = {'I': 0,
                    'X': 0,
                    'Y': 0,
                    'R': 0}

    blob_catalog_dict['I'] = np.arange(len(blx))
    blob_catalog_dict['X'] = (blx - nbins / 2 + 1) / nbins * rfov * r200 * 2
    blob_catalog_dict['Y'] = (bly - nbins / 2 + 1) / nbins * rfov * r200 * 2
    blob_catalog_dict['R'] = blr / nbins * rfov * r200 * 2

    return blob_catalog_dict


def map_kSZ_intensity(num_halo, redshift, simulation_type, nbins=100, rfov=2, output='show', title=True,
                      plot_groups='FoF'):
    # Import data
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))
    r200 = extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)

    # Gas particles
    file = extract.file_name_hdf5(subject='particledata', redshift=extract.redshift_floatTostr(redshift))
    redshift_short = redshift

    part_type = extract.particle_type('gas')
    mass = extract.particle_masses(path, file, part_type)
    coordinates = extract.particle_coordinates(path, file, part_type)
    velocities = extract.particle_velocity(path, file, part_type)
    temperatures = extract.particle_temperature(path, file, part_type)
    group_number = extract.group_number(path, file, part_type)
    subgroup_number = extract.subgroup_number(path, file, part_type)
    tot_rest_frame, _ = profile.total_mass_rest_frame(path, file)
    # gas_rest_frame, _ = profile.cluster_average_momentum(path, file, part_type)

    # Retrieve coordinates & velocities
    x = coordinates[:, 0] - group_CoP[0]
    y = coordinates[:, 1] - group_CoP[1]
    z = coordinates[:, 2] - group_CoP[2]
    vx = velocities[:, 0] - tot_rest_frame[0]
    vy = velocities[:, 1] - tot_rest_frame[1]
    vz = velocities[:, 2] - tot_rest_frame[2]

    # Rescale to comoving coordinates
    h = extract.file_hubble_param(path, file)
    redshift = extract.file_redshift(path, file)
    x = profile.comoving_length(x, h, redshift)
    y = profile.comoving_length(y, h, redshift)
    z = profile.comoving_length(z, h, redshift)
    r200 = profile.comoving_length(r200, h, redshift)
    vx = profile.comoving_velocity(vx, h, redshift)
    vy = profile.comoving_velocity(vy, h, redshift)
    vz = profile.comoving_velocity(vz, h, redshift)
    vx = profile.velocity_units(vx, unit_system='SI')
    vy = profile.velocity_units(vy, unit_system='SI')
    vz = profile.velocity_units(vz, unit_system='SI')
    mass = profile.comoving_mass(mass, h, redshift)
    mass = profile.mass_units(mass, unit_system='SI')
    T = temperatures

    # Compute radial distance
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Particle selection
    min_gn = 0
    min_T = 10 ** 5
    max_r = 5

    if plot_groups == 'FoF':
        min_sgn = 0
    elif plot_groups == 'subgroups':
        min_sgn = 1
    else:
        print("[ERROR] The (sub)groups you are trying to plot are not defined.")
        exit(1)


    # TODO: index = np.where((r < max_r*r200) & (group_number in gn_catalogue) & (subgroup_number in sg_catalogue) & (T > min_T))[0]
    index = np.where((r < max_r * r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn) & (T > min_T))[0]
    mass, T = mass[index], T[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    x = x * Mpc_to_arcmin
    y = y * Mpc_to_arcmin
    z = z * Mpc_to_arcmin
    r200 = r200 * Mpc_to_arcmin

    # Bin data
    cmap = ['gnuplot', 'gnuplot', 'gnuplot']
    # cmap = [mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = False)]
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$y_{kSZ}$',
                 r'$y_{kSZ}$',
                 r'$y_{kSZ}$']
    weight_function = r'$y_{kSZ} = - \frac{\sigma_T}{A_{pix} \mu_e m_H c} \sum_{i=0}^{N_{l.o.s.} m^{g}_i v^{r}_i}$'

    # Compute angular bins
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    dx, dy = x_bins[1] - x_bins[0], y_bins[1] - y_bins[0]

    m_H = 1.6737236 * 10 ** (-27)  # Hydrogen atom mass in kg
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0]) * (3.0856776 * 10 ** 22 / Mpc_to_arcmin) ** 2

    # Set up index permutations
    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Set up vectors
    pp_xyz = np.asarray([x, y, z])
    pp_v_ = np.asarray([vx, vy, vz])

    for i in [0]:
        # line of sight momentum weights - conversion to required variable type
        mass = mass.astype(np.longdouble)
        weight = pp_v_[ijk[i][2]].astype(np.longdouble)

        # Histogram calculation
        Cx, Cy = mapgen.bins_meshify(pp_xyz[ijk[i][0]], pp_xyz[ijk[i][1]], x_bins, y_bins)
        count_mv = mapgen.bins_evaluate(pp_xyz[ijk[i][0]], pp_xyz[ijk[i][1]], x_bins, y_bins, weights=mass * weight)

        # Compute kSZ
        kSZ = -count_mv * sigma_T.value / (A_pix * c.value * m_H * 1.16)

        # Convolution
        kernel_Type = 'gauss'
        kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type=kernel_Type)
        kernel = np.array(kernel)
        kSZmap = convolve(kSZ, kernel)
        result_map = np.abs(kSZmap)

        # Grayscale image
        kSZarray = np.reshape(np.log10(np.abs(kSZmap)), (1, (nbins-1)**2))
        img_par_scale = 0.75*np.max(kSZarray)
        img_par_offset = np.percentile(kSZarray, 75)/img_par_scale


        aux_map = img_par_offset-np.log10(np.abs(kSZmap))/img_par_scale

        aux_map[aux_map == -np.inf] = 0
        aux_map[aux_map < 0] = 0
        aux_map[aux_map > 1] = 1

        # Blob identification
        test = fwhm / (60 * dx)

        min_sigma = test*1
        max_sigma = test*10
        num_sigma = 80
        threshold = 0.05

        blobs_log = blob_log(aux_map, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

        # Logarithmic normalization
        norm = colors.SymLogNorm(linthresh=10 ** -7, linscale=0.6, vmin=0, vmax=np.abs(result_map).max())

        # Plot image
        img = axes[i].pcolor(Cx, Cy, result_map, cmap=cmap[i], norm=norm)

        # Render elements in plots
        axes[i].set_aspect('equal')
        axes[i].add_artist(Circle((0, 0), radius=r200, color='white', fill=False, linestyle='--', label=r'$R_{200}$'))
        axes[i].add_artist(Circle((0, 0), radius=5 * r200, color='white', fill=False, linewidth=0.5, linestyle='-',
                                  label=r'$R_{200}$'))
        axes[i].set_xlim(-rfov * r200, rfov * r200)
        axes[i].set_ylim(-rfov * r200, rfov * r200)
        axes[i].set_xlabel(xlabel[i])
        axes[i].set_ylabel(ylabel[i])
        axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size=15)

        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes[i])
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
        cbar.set_label(cbarlabel[i], labelpad=-70)
        cax2.xaxis.set_ticks_position("top")
        print("Plot run completed:\t", i)

    for i in [1]:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        img = axes[i].pcolor(Cx, Cy, aux_map, cmap='gray', norm=norm)

        # Render elements in plots
        axes[i].set_aspect('equal')
        axes[i].add_artist(Circle((0, 0), radius=r200, color='white', fill=False, linestyle='--', label=r'$R_{200}$'))
        axes[i].add_artist(Circle((0, 0), radius=5 * r200, color='white', fill=False, linewidth=0.5, linestyle='-',
                                  label=r'$R_{200}$'))
        axes[i].set_xlim(-rfov * r200, rfov * r200)
        axes[i].set_ylim(-rfov * r200, rfov * r200)
        axes[i].set_xlabel(xlabel[i-1])
        axes[i].set_ylabel(ylabel[i-1])
        axes[i].annotate(thirdAX[i-1], (0.03, 0.03), textcoords='axes fraction', size=15)

        # Take raw output from the blob log algorithm
        bly = blobs_log[:,0]
        blx = blobs_log[:,1]
        blr = blobs_log[:,2]

        # Create data storage for the blobl markers and convert to plot axis coordinate system
        catalog_dict = {'I': 0,
                        'X': 0,
                        'Y': 0,
                        'R': 0}

        catalog_dict['I'] = np.arange(len(blx))
        catalog_dict['X'] = (blx - nbins / 2 + 1) / nbins * rfov * r200 * 2
        catalog_dict['Y'] = (bly - nbins / 2 + 1) / nbins * rfov * r200 * 2
        catalog_dict['R'] = blr/ nbins * rfov * r200 * 2

        #"""
        for k in range(len(blx)):
            circ = plt.Circle((catalog_dict['X'][k], catalog_dict['Y'][k]), catalog_dict['R'][k], color='green', linewidth=1, fill=False)
            axes[i].add_patch(circ)
        #"""

        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes[i])
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
        cbar.set_label(cbarlabel[i], labelpad=-70)
        cax2.xaxis.set_ticks_position("top")


    for i in [2]:
        axes[i].set_frame_on(False)
        axes[i].set_xticklabels('')
        axes[i].set_yticklabels('')
        axes[i].set_aspect('equal')
        font14 = {'size': 14}
        font12 = {'size': 12}
        axes[i].text(0.1, 1, r'$\mathrm{Calibration\ of\ the\ LoG\ algorithm}$', fontdict = font14)
        if title and plot_groups == 'FoF':
            axes[i].text(0.1,0.9,
                r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f \qquad \mathrm{ICM + subhalos}$' % (num_halo, redshift), fontdict = font14)
        if title and plot_groups == 'subgroups':
            axes[i].text(0.1,0.9,
                r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f \qquad \mathrm{subhalos}$' % (num_halo, redshift), fontdict = font14)

        axes[i].text(0.1, 0.8, r'$\mathrm{Grayscale\ image\ parameters:}$', fontdict = font14)
        axes[i].text(0.2, 0.75, r'$\mathrm{aux\_map = (img\_par\_offset-log_{10}(abs(kSZmap)))/(img\_par\_scale)}$', fontdict = font12)
        axes[i].text(0.2, 0.70, r'$\mathrm{img\_par\_offset = %4.2f}$' % img_par_offset, fontdict = font12)
        axes[i].text(0.2, 0.65, r'$\mathrm{img\_par\_scale = %4.2f}$' % img_par_scale, fontdict=font12)

        axes[i].text(0.1, 0.55, r'$\mathrm{LoG\ parameters:}$', fontdict = font12)
        axes[i].text(0.2, 0.50, r'$\mathrm{min/max\ sigma:}$', fontdict = font12)
        axes[i].text(0.7, 0.50, r'$\mathrm{%4.2f\ /\ %4.2f}$'% (min_sigma, max_sigma), fontdict = font12)
        axes[i].text(0.2, 0.45, r'$\mathrm{number\ of\ sigmas:}$', fontdict = font12)
        axes[i].text(0.7, 0.45, r'$\mathrm{%d}$' % num_sigma, fontdict = font12)
        axes[i].text(0.2, 0.40, r'$\mathrm{threshold:}$', fontdict = font12)
        axes[i].text(0.7, 0.40, r'$\mathrm{%4.2f}$' % threshold, fontdict = font12)

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save':
        # WARNING: Changing these folder and file names will break map_kSZ_intensity_marker-match.py dependence
        dir_name = 'Substructure-Identification-kSZ'
        file_name = 'blob_catalog_kSZ_halo' + str(num_halo) + '_z' + str(redshift_short).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins) + '.npy'
        save_name = 'substruct-id-kSZ' + '_halo' + str(num_halo) + '_z' + str(redshift_short).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins)

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' + save_name + '.pdf')

        if os.path.isfile(dir_name + '//' + file_name):
            os.remove(dir_name + '//' + file_name)

        np.save(dir_name + '//' + file_name, catalog_dict)

        # Generate metadata.txt
        """
        import plot_metadata as meta
        args = (
        num_halo, simulation_type, redshift, angular_distance, min_gn, min_sgn, min_T, max_r, weight_function, nbins,
        rfov, kernel_Type, fwhm, r200, r200 / Mpc_to_arcmin)
        meta.metadata_file(args, dir_name + '//' + save_name)
        """

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)


def call_kSZ_map(rank):
    num_halo = rank
    simulation_type = 'gas'
    redshift = 0.57
    plot_groups = 'FoF'  # options: "FoF" or "subgroups"
    subhalo_threshold = 90  # percentile threshold for subhalo markers

    # Call function:
    map_kSZ_intensity(num_halo, redshift, simulation_type,
                      nbins=300,
                      rfov=5,
                      output='show',
                      title=True,
                      plot_groups=plot_groups)


# **************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python blob-log_substructure-identification_kSZ.py

if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process:', rank)
    call_kSZ_map(rank)

# **************************************************************************************************
