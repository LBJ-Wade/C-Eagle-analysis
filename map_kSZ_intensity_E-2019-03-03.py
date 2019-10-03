import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen
import kernel_convolver as kernconv
# import cosmolopy !!! USELESS 50 YEARS OLD PACKAGE

import numpy as np
import astropy
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from os import makedirs, chdir
from os.path import exists
#from mpi4py.futures import MPIPoolExecutor


def map_kSZ_intensity(num_halo, redshift, simulation_type, nbins, rfov, output = 'show', title = True):
    # Import data
    path =         extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
    file =         extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    r200 =         extract.group_r200(path, file)
    print("R200 comoving: %8.2f"% (r200))
    group_CoP = extract.group_centre_of_potential(path, file)
    file =         extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
    
    # Gas particles
    part_type = extract.particle_type('gas')
    mass = extract.particle_masses(path, file, part_type)
    coordinates = extract.particle_coordinates(path, file, part_type)
    velocities = extract.particle_velocity(path, file, part_type)
    temperatures = extract.particle_temperature(path, file, part_type)
    density = extract.particle_SPH_density(path, file, part_type = part_type)
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

    h = extract.file_hubble_param(path, file)

    
    # Rescale to comoving coordinates
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
    density = profile.comoving_density(density, h, redshift)
    density = profile.density_units(density, unit_system = 'SI')
    T = temperatures

    # Compute radial distance
    r = np.sqrt(x**2+y**2+z**2)

    # Particle selection
    print("*** *** Subhalos only *** ***")
    min_gn = 0
    min_sgn = 0
    min_T = 10**5
    max_r = 5
    index = np.where((r < max_r*r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn ) & (T > min_T))[0]
    mass, T = mass[index], T[index]
    density = density[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Generate plot
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

    # Convert to angular distances
    redshift = extract.file_redshift(path, file)
    cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307)
    angular_distance = cosmo.luminosity_distance(redshift)
    print("angular_diameter_distance: ", angular_distance)
    Mpc_to_arcmin = np.power(np.pi, -1)*180*60/angular_distance.value

    x = x*Mpc_to_arcmin
    y = y*Mpc_to_arcmin
    z = z*Mpc_to_arcmin
    r200 = r200*Mpc_to_arcmin


    from astropy.constants import c, m_e, sigma_T
    kSZ_const = - sigma_T.value/c.value
    # electon number density
    n_e = density/m_e.value
    #print(n_e)

    # Assign plot elements
    cmap = ["viridis", "viridis", "viridis"]
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$\log_{10} |Y_{kSZ}|$',
                 r'$\log_{10} |Y_{kSZ}|$',
                 r'$\log_{10} |Y_{kSZ}|$']

    weight_function = 'Y_{kSZ} = - sigma_T/c* int_{l.o.s.} v_r*n_e * dl'
    # Loop over 3 projections
    for i in [0, 1, 2]:
        # Handle data
        if i == 0:
            x_Data = x
            y_Data = y
            weight = vz*n_e*kSZ_const
        elif i == 1:
            x_Data = y
            y_Data = z
            weight = vx*n_e*kSZ_const
        elif i == 2:
            y_Data = z
            x_Data = x
            weight = vy*n_e*kSZ_const

        # Define bins
        x_bins = np.linspace(-rfov*r200, rfov*r200, nbins)
        y_bins = np.linspace(-rfov*r200, rfov*r200, nbins)
        Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)

        # Mass-weighted velocity
        count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = weight)
        #count[count == 0] = 1



        # convolution
        kernel_Type = 'gauss'
        kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type = kernel_Type)
        kernel = np.array(kernel)
        kSZmap = convolve(count, kernel)

        #norm = mapgen.MidpointNormalize(vmin=np.log10(kSZmap).min(), vmax=np.log10(kSZmap).max(), midpoint=0)
        img = axes[i].pcolor(Cx, Cy, np.log10(np.abs(kSZmap)), cmap=cmap[i])#, norm= norm, alpha = 0.2)

        # Render elements in plots
        axes[i].set_aspect('equal')
        axes[i].add_artist(Circle((0,0), radius=r200, color = 'black', fill = False, linestyle = '--', label = r'$R_{200}$'))
        axes[i].add_artist(Circle((0,0), radius=5*r200, color = 'black', fill = False, linewidth = 0.5,linestyle = '-', label = r'$R_{200}$'))
        axes[i].set_xlim(-rfov*r200, rfov*r200)
        axes[i].set_ylim(-rfov*r200, rfov*r200)
        axes[i].set_xlabel(xlabel[i])
        axes[i].set_ylabel(ylabel[i])
        axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size = 15)
        if title: 
            axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift),  pad=-14)
        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes[i])
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
        cbar.set_label(cbarlabel[i], labelpad= -70)
        #cbar.set_clim(-8, 0)
        #cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])    
        cax2.xaxis.set_ticks_position("top")
        print("Plot run completed:", i)

    output = 'save'
    # Define output
    if output == 'show': 
        plt.show()
    elif output == 'save': 
        dir_name = 'parallel-out//test'
        if not exists(dir_name): 
            makedirs(dir_name)
        save_name = 'subhalos_partType'+part_type+'_halo'+str(num_halo)+'z'+str(redshift).replace(".", "")
        plt.savefig(dir_name + '//'+save_name+'.pdf')
    else:
        print("[ERROR] The output type you are trying to select is not defined.")
        exit(1)

    # Generate metadata.txt
    import plot_metadata as meta 
    args = (num_halo, simulation_type, redshift, angular_distance.value, min_gn, min_sgn, min_T, max_r, weight_function,  nbins, rfov, kernel_Type, fwhm, r200, r200/Mpc_to_arcmin)
    meta.metadata_file(args, dir_name + '//'+save_name)






#############################################################################################


def call_kSZ_map(halo):
    simulation_type = 'gas'
    map_kSZ_intensity(halo, 0.57, simulation_type = simulation_type, nbins = 500, rfov=2)

call_kSZ_map(0)
