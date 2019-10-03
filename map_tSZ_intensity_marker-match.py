import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen
import kernel_convolver as kernconv
import distance_cosmology as cosmo
import subhalo_marker as sgmark
import subhalo_selection as subsel
from bloblog_sub_id import substructure_identification

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.constants import c, sigma_T, k_B, m_e
from os import makedirs, chdir
from os.path import exists

#from mpi4py.futures import MPIPoolExecutor

# Turn off FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def map_tSZ_intensity(num_halo, redshift, simulation_type, nbins = 100, rfov = 2, output = 'show', title = True, plot_groups = 'FoF'):
    # Import data
    path =         extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
    file =         extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    r200 =         extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)
    fsgtab = [path, file]

    # Gas particles
    file = extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
    redshift_short = redshift

    part_type = extract.particle_type('gas')
    mass = extract.particle_masses(path, file, part_type)
    coordinates = extract.particle_coordinates(path, file, part_type)
    velocities = extract.particle_velocity(path, file, part_type)
    temperatures = extract.particle_temperature(path, file, part_type)
    group_number = extract.group_number(path, file, part_type)
    subgroup_number = extract.subgroup_number(path, file, part_type)
    tot_rest_frame, _ = profile.total_mass_rest_frame(path, file)
    #gas_rest_frame, _ = profile.cluster_average_momentum(path, file, part_type)

    # Retrieve coordinates & velocities
    x = coordinates[:,0] - group_CoP[0]
    y = coordinates[:,1] - group_CoP[1]
    z = coordinates[:,2] - group_CoP[2]
    vx = velocities[:,0] - tot_rest_frame[0]
    vy = velocities[:,1] - tot_rest_frame[1]
    vz = velocities[:,2] - tot_rest_frame[2]
        
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
    vx = profile.velocity_units(vx, unit_system = 'SI')
    vy = profile.velocity_units(vy, unit_system = 'SI')
    vz = profile.velocity_units(vz, unit_system = 'SI')
    mass = profile.comoving_mass(mass, h, redshift)
    mass = profile.mass_units(mass, unit_system = 'SI')
    T = temperatures

    # Compute radial distance
    r = np.sqrt(x**2+y**2+z**2)

    # Particle selection
    min_gn = 0
    min_T = 10**5
    max_r = 5

    if plot_groups == 'FoF': min_sgn = 0
    elif plot_groups == 'subgroups': min_sgn = 1
    else:
        print("[ERROR] The (sub)groups you are trying to plot are not defined.")
        exit(1)


    index = np.where((r < max_r * r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn ) & (T > min_T))[0]
    mass, T = mass[index], T[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Generate plot frame
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1)*180*60/angular_distance
    x = x*Mpc_to_arcmin
    y = y*Mpc_to_arcmin
    z = z*Mpc_to_arcmin
    r200 = r200*Mpc_to_arcmin

    # Bin data
    cmap = ['inferno', 'inferno', 'inferno']
    #cmap = [mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = False)]
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$y_{tSZ}$', r'$y_{tSZ}$', r'$y_{tSZ}$']
    weight_function = r'$y_{tSZ} = - \frac{\sigma_T}{A_{pix} \mu_e m_H c} \sum_{i=0}^{N_{l.o.s.} m^{g}_i v^{r}_i}$'

    # Compute angular bins
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)

    m_H = 1.6737236 * 10 ** (-27)  # Hydrogen atom mass in kg
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0]) * (3.0856776 * 10 ** 22 / Mpc_to_arcmin) ** 2
    const = sigma_T.value * k_B.value / (m_e.value * c.value ** 2 * m_H * 1.16)

    # Set up index permutations
    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Set up vectors
    pp_xyz = np.asarray([x, y, z])
    pp_v_ = np.asarray([vx, vy, vz])

    for i in [0, 1, 2]:
        # line of sight momentum weights - conversion to required variable type
        mass = mass.astype(np.longdouble)
        weight = T.astype(np.longdouble)

        # Histogram calculation
        Cx, Cy = mapgen.bins_meshify(pp_xyz[ijk[i][0]], pp_xyz[ijk[i][1]], x_bins, y_bins)
        count_mT = mapgen.bins_evaluate(pp_xyz[ijk[i][0]], pp_xyz[ijk[i][1]], x_bins, y_bins, weights = mass*weight)

        # Compute tSZ
        tSZ = count_mT * const / (A_pix)

        # Convolution
        kernel_Type = 'gauss'
        kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type = kernel_Type)
        kernel = np.array(kernel)
        tSZmap = convolve(tSZ, kernel)

        blob_catalogue = substructure_identification(np.asarray(tSZmap), rfov, r200)
        tSZmap[tSZmap == 0] = 10**-10

        subhalo_catalog = subsel.compute_catalog(num_halo, redshift_short, i, output="Return")
        sg_xyz = sgmark.subhalo_marks_filtered(fsgtab[0], fsgtab[1], subhalo_catalog['I'])

        # Linear normalization
        #norm = mapgen.MidpointNormalize(vmin=tSZmap.min(), vmax=tSZmap.max(), midpoint=0)

        # Logarithmic normalization
        norm = colors.LogNorm(vmin=10 ** -10, vmax=10 ** -3)
        #norm = colors.SymLogNorm(linthresh=10**-7, linscale=0.6, vmin=-np.abs(tSZmap).max(), vmax=np.abs(tSZmap).max())

        # Plot image
        img = axes[i].pcolor(Cx, Cy, tSZmap, cmap=cmap[i], norm= norm)

        # Plot subhalos from SUBFIND selection
        axes[i].scatter(sg_xyz[ijk[i][0]], sg_xyz[ijk[i][1]], s = 15, marker='x', facecolors='w')

        # Plot markers as identified by bloblog
        blx = blob_catalogue['X']
        bly = blob_catalogue['Y']
        blr = blob_catalogue['R']

        # """
        for k in range(len(blx)):
            circ = plt.Circle((blx[k], bly[k]), blr[k], color='c', linewidth=1, fill=False)
            # print('adding circle: ', str((x - nbins / 2 + 1) / nbins * x_bins[-1]), str((y - nbins / 2 +1) / nbins * y_bins[-1]), str(r))
            axes[i].add_patch(circ)
        # """

        # Render elements in plots
        axes[i].set_aspect('equal')
        axes[i].add_artist(Circle((0,0), radius=r200, color = 'black', fill = False, linestyle = '--', label = r'$R_{200}$'))
        axes[i].add_artist(Circle((0,0), radius=5*r200, color = 'black', fill = False, linewidth = 0.5,linestyle = '-', label = r'$R_{200}$'))
        axes[i].set_xlim(-rfov*r200, rfov*r200)
        axes[i].set_ylim(-rfov*r200, rfov*r200)
        axes[i].set_xlabel(xlabel[i])
        axes[i].set_ylabel(ylabel[i])
        axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size = 15)
        if title and plot_groups == 'FoF': 
            axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f \qquad \mathrm{ICM + subhalos}$' % (num_halo, redshift),  pad=94)
        if title and plot_groups == 'subgroups': 
            axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f \qquad \mathrm{subhalos}$' % (num_halo, redshift),  pad=94)

        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes[i])
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
        cbar.set_label(cbarlabel[i], labelpad= -70)
        #cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])    
        cax2.xaxis.set_ticks_position("top")
        print("Plot run completed:\t", i)

    # Define output
    if output == 'show':
        plt.show()

    elif output == 'save': 
        dir_name = 'tSZ maps marker-match'
        save_name = 'tSZmap_' + plot_groups + '_mark-match_halo' + str(num_halo) + '_z' + str(redshift_short).replace(".","") + '_rfov' + str(rfov) + '_nbins' + str(nbins)

        if not exists(dir_name): makedirs(dir_name)

        plt.savefig(dir_name + '//' + save_name + '.pdf')

        # Generate metadata.txt
        import plot_metadata as meta 
        #args = (num_halo, simulation_type, redshift, angular_distance, min_gn, min_sgn, min_T, max_r, weight_function,  nbins, rfov, kernel_Type, fwhm, r200, r200/Mpc_to_arcmin)
        #meta.metadata_file(args, dir_name + '//' + save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)


def call_tSZ_map(num_halo):
    simulation_type = 'gas'
    redshift = 0.57
    plot_groups = 'FoF' # options: "FoF" or "subgroups"
    subhalo_threshold = 90  # percentile threshold for subhalo markers

    # Call function:
    # WARNING: Parameters must match blob catalog
    map_tSZ_intensity(num_halo, redshift, simulation_type,
                      nbins = 200,
                      rfov = 5,
                      output = 'save',
                      title = False,
                      plot_groups = plot_groups)


#**************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python map_tSZ_intensity_marker-match.py

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process:', rank)
    call_tSZ_map(rank)

#**************************************************************************************************
