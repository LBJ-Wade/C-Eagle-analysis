import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen
import kernel_convolver as kernconv
import distance_cosmology as cosmo
import subhalo_marker as sgmark

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.constants import c, sigma_T
from os import makedirs, chdir
from os.path import exists

#from mpi4py.futures import MPIPoolExecutor

# Turn off FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def map_kSZ_intensity(num_halo, redshift, simulation_type, nbins = 100, rfov = 2, output = 'show', title = True, plot_groups = 'FoF', plot_sg = False, sgth = 0):
    # Import data
    path =         extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
    file =         extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    r200 =         extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)

    # Obtain subhalo data and markers from subgroup_tab
    if (plot_sg): sg_x_y_pz_gn_ms, sg_y_z_px_gn_ms, sg_x_z_py_gn_ms = sgmark.subhalo_marks(path, file, sgth)


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

    index = np.where((r < max_r*r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn ) & (T > min_T))[0]    
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
    cmap = ['seismic', 'seismic', 'seismic_r']
    #cmap = [mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = False)]
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

    m_H = 1.6737236 * 10 ** (-27)  # Hydrogen atom mass in kg
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0]) * (3.0856776 * 10 ** 22 / Mpc_to_arcmin) ** 2

    for i in [0, 1, 2]:
        # Handle data
        if i == 0:
            x_Data = x
            y_Data = y
            weight = vz
            if (plot_sg): sg = sg_x_y_pz_gn_ms
        elif i == 1:
            x_Data = y
            y_Data = z
            weight = vx
            if (plot_sg): sg = sg_y_z_px_gn_ms
        elif i == 2:
            x_Data = x
            y_Data = z
            weight = vy
            if (plot_sg): sg = sg_x_z_py_gn_ms

        Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)

        # line of sight momentum weights
        mass = mass.astype(np.longdouble)
        weight = weight.astype(np.longdouble)

        count_mv = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = mass*weight)

        # Compute kSZ
        kSZ = -count_mv*sigma_T.value/(A_pix*c.value*m_H*1.16)

        # Convolution
        kernel_Type = 'gauss'
        kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type = kernel_Type)
        kernel = np.array(kernel)
        kSZmap = convolve(kSZ, kernel)

        # !!! applying non-continous filter
        kSZmap = sgmark.substruct_detect(kSZmap)

        # Convert kSZ y-parameter to symmetric log scale
        # log_pos = np.where(kSZmap>0)[0]
        # log_neg = np.where(kSZmap<0)[0]
        # kSZmap[log_pos] = np.log10(np.abs(kSZmap[log_pos]))
        # kSZmap[log_neg] = -np.log10(np.abs(kSZmap[log_neg]))

        # norm = mapgen.MidpointNormalize(vmin=kSZmap.min(), vmax=kSZmap.max(), midpoint=0)

        norm = colors.SymLogNorm(linthresh=10**-7, linscale=0.6, vmin=-np.abs(kSZmap).max(), vmax=np.abs(kSZmap).max())
        img = axes[i].pcolor(Cx, Cy, kSZmap, cmap=cmap[i], norm= norm)

        # Plot subhalos
        if (plot_sg): axes[i].scatter(sg[0, :], sg[1, :], s=sg[4, :], marker='o', edgecolors='black', facecolors='None')

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
        dir_name = 'kSZ maps w-marks'
        if not exists(dir_name): 
            makedirs(dir_name)
        save_name = 'kSZmap_' + plot_groups + '_w-marks_halo'+str(num_halo) + '_z' + str(redshift_short).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins)
        plt.savefig(dir_name + '//' + save_name + '.pdf')

        # Generate metadata.txt
        import plot_metadata as meta 
        args = (num_halo, simulation_type, redshift, angular_distance, min_gn, min_sgn, min_T, max_r, weight_function,  nbins, rfov, kernel_Type, fwhm, r200, r200/Mpc_to_arcmin, sgth)
        meta.metadata_file(args, dir_name + '//' + save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)


def call_kSZ_map(num_halo):
    simulation_type = 'gas'
    redshift = 0.57
    plot_groups = 'FoF' # options: "FoF" or "subgroups"
    subhalo_threshold = 90  # percentile threshold for subhalo markers

    # Call function:
    map_kSZ_intensity(num_halo, redshift, simulation_type,
                      nbins = 300,
                      rfov = 5,
                      output = 'show',
                      title = True,
                      plot_groups = plot_groups,
                      plot_sg = True,
                      sgth = subhalo_threshold)


#**************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python map_kSZ_intensity_w-marks_M.py

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process:', rank)
    call_kSZ_map(rank)

#**************************************************************************************************
