import clusters_retriever as extract
from visualisation import map_plot_parameters as plotpar
import cluster_profiler as profile
from obsolete import map_synthetizer as mapgen, plot_metadata as meta
from import_toolkit import distance_cosmology as cosmo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.convolution import convolve
from os import makedirs
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
    file =         extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
    redshift_short = redshift

    # Gas particles
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
    cmap = ['Blues', 'Blues', 'Blues']
    # cmap = [mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = False)]
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$y_{tSZ}$',
                 r'$y_{tSZ}$',
                 r'$y_{tSZ}$']
    weight_function = r'$y_{tSZ} = - \frac{\sigma_T}{A_{pix} \mu_e m_H c} \sum_{i=0}^{N_{l.o.s.} m^{g}_i T^{g}_i}$'
    for i in [0, 1, 2]:
        # Handle data
        if i == 0:
            x_Data = x
            y_Data = y
            weight = T
        elif i == 1:
            x_Data = y
            y_Data = z
            weight = T
        elif i == 2:
            x_Data = x
            y_Data = z
            weight = T

        # Compute angular bins
        x_bins = np.linspace(-rfov*r200, rfov*r200, nbins)
        y_bins = np.linspace(-rfov*r200, rfov*r200, nbins)
        Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)

        from astropy.constants import c, sigma_T, k_B, m_e
        m_H = 1.6737236*10**(-27) # Hydrogen atom mass in kg
        A_pix = (x_bins[1] - x_bins[0])*(y_bins[1] - y_bins[0])*(3.0856776*10**22/Mpc_to_arcmin)**2
        const = sigma_T.value*k_B.value/(m_e.value*c.value**2 * m_H *1.16)
        # line of sight momentum weights
        mass = mass.astype(np.longdouble)
        weight = weight.astype(np.longdouble)
        count_mT = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = mass*weight)
        # Compute tSZ
        tSZ = count_mT*const/(A_pix)

        # convolution
        kernel_Type = 'gauss'
        #kernel = Gaussian2DKernel(stddev=2)
        kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type = kernel_Type)
        kernel = np.array(kernel)
        tSZmap = convolve(tSZ, kernel)

        # norm = mapgen.MidpointNormalize(vmin=tSZmap.min(), vmax=tSZmap.max(), midpoint=0)
        c
        # norm = colors.PowerNorm(gamma=0.2)
        img = axes[i].pcolor(Cx, Cy, tSZmap, cmap=cmap[i], norm= norm)

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
        dir_name = 'tSZ maps'
        if not exists(dir_name): 
            makedirs(dir_name)
        save_name = 'tSZmap_' + plot_groups + '_halo'+str(num_halo) + '_z' + str(redshift_short).replace(".", "") + '_rfov' + str(rfov) + '_nbins' + str(nbins)
        plt.savefig(dir_name + '//' + save_name + '.pdf')

        # Generate metadata.txt
        args = (num_halo, simulation_type, redshift, angular_distance, min_gn, min_sgn, min_T, max_r, weight_function,  nbins, rfov, kernel_Type, fwhm, r200, r200/Mpc_to_arcmin)
        meta.metadata_file(args, dir_name + '//' + save_name)

    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)


def map_tSZ_ICM(num_halo, redshift, simulation_type, nbins = 100, rfov = 2, output = 'show', title = True):
    map_tSZ_intensity(num_halo, redshift, simulation_type, nbins = nbins, rfov = rfov, output = output, title = True, plot_groups = 'FoF')

def map_tSZ_subgroups(num_halo, redshift, simulation_type, nbins = 100, rfov = 2, output = 'show', title = True):
    map_tSZ_intensity(num_halo, redshift, simulation_type, nbins = nbins, rfov = rfov, output = output, title = True, plot_groups = 'subgroups')

def call_tSZ_map(num_halo):
    simulation_type = 'gas'
    redshift = 0.57
    bins = 500
    map_tSZ_ICM(num_halo, redshift, simulation_type, nbins = bins, rfov = 5, output = 'save', title = True)
    map_tSZ_subgroups(num_halo, redshift, simulation_type, nbins = bins, rfov = 5, output = 'save', title = True)


# Test just one map
# call_tSZ_map(0)


#**************************************************************************************************
# MPI implementation

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python map_kSZ_intensity.py

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('********************\nMPI process:', rank)
    call_tSZ_map(rank)


    #with MPIPoolExecutor() as executor:
    #executor.map(call_kSZ_map, rank)

