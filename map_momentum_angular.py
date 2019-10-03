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
#from mpi4py.futures import MPIPoolExecutor


def map_kSZ_intensity(num_halo, redshift, simulation_type, nbins, rfov):
    # Import data
    path =         extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
    file =         extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
    r200 =         extract.group_r200(path, file)
    print(r200)
    group_CoP = extract.group_centre_of_potential(path, file)
    file =         extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
    
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

    h = extract.file_hubble_param(path, file)

    
    # Rescale to comoving coordinates
    x = profile.comoving_length(x, h, redshift)
    y = profile.comoving_length(y, h, redshift)
    z = profile.comoving_length(z, h, redshift)
    r200 = profile.comoving_length(r200, h, redshift)
    vx = profile.comoving_velocity(vx, h, redshift)
    vy = profile.comoving_velocity(vy, h, redshift)
    vz = profile.comoving_velocity(vz, h, redshift)
    
    vx = profile.velocity_units(vx, unit_system = 'astro')
    vy = profile.velocity_units(vy, unit_system = 'astro')
    vz = profile.velocity_units(vz, unit_system = 'astro')
    mass = profile.comoving_mass(mass, h, redshift)
    mass = profile.mass_units(mass, unit_system = 'astro')
    T = temperatures

    # Compute radial distance
    r = np.sqrt(x**2+y**2+z**2)

    # Particle selection
    index = np.where((r < 5*r200) & (group_number > -1) & (subgroup_number > -1) & (T > 10**5))[0]

    mass, T = mass[index], T[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Generate plot
    plotpar.set_defaults_plot()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

    # Convert to angular distances
    #cosmo = {'omega_M_0' : 0.307, 'omega_lambda_0' : 0.693, 'h' : 0.6777}
    #cosmo = cosmolopy.set_omega_k_0(cosmo)
    redshift = extract.file_redshift(path, file)
    #angular_distance = cosmolopy.angular_diameter_distance(redshift, z0 = 0, **cosmo)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    angular_distance = cosmo.luminosity_distance(redshift)
    print("angular_diameter_distance: ", angular_distance)
    Mpc_to_arcmin = np.power(np.pi, -1)*180*60/angular_distance*astropy.units.Mpc

    x = x*Mpc_to_arcmin
    y = y*Mpc_to_arcmin
    z = z*Mpc_to_arcmin
    r200 = r200*Mpc_to_arcmin

    # Bin data
    cmap = [plt.get_cmap('seismic'), plt.get_cmap('seismic'), plt.get_cmap('seismic_r')]
    #cmap = [mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = True), mapgen.modified_spectral_cmap(Reversed = False)]
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$\sum_{i} m_i v_{z, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$',
                 r'$\sum_{i} m_i v_{x, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$',
                 r'$\sum_{i} m_i v_{y, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$']
    for i in [0, 1, 2]:
        # Handle data
        if i == 0:
            x_Data = x
            y_Data = y
            weight = vz
        elif i == 1:
            x_Data = y
            y_Data = z
            weight = vx
        elif i == 2:
            x_Data = x
            y_Data = z
            weight = vy

        x_bins = np.linspace(-rfov*r200, rfov*r200, nbins)
        y_bins = np.linspace(-rfov*r200, rfov*r200, nbins)
        Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
        # line of sight momentum weights
        count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = mass*weight)

        # convolution
        kernel, _ = kernconv.nika2_kernel(x_bins, y_bins)
        kernel = np.array(kernel)
        kSZmap = convolve(count, kernel)

        norm = mapgen.MidpointNormalize(vmin=kSZmap.min(), vmax=kSZmap.max(), midpoint=0)
        img = axes[i].pcolor(Cx, Cy, kSZmap, cmap=cmap[i], norm= norm)


        # Render elements in plots
        axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift),  pad=94)
        axes[i].set_aspect('equal')
        axes[i].add_artist(Circle((0,0), radius=r200, color = 'black', fill = False, linestyle = '--', label = r'$R_{200}$'))
        axes[i].add_artist(Circle((0,0), radius=5*r200, color = 'black', fill = False, linewidth = 0.5,linestyle = '-', label = r'$R_{200}$'))
        axes[i].set_xlim(-rfov*r200, rfov*r200)
        axes[i].set_ylim(-rfov*r200, rfov*r200)
        axes[i].set_xlabel(xlabel[i])
        axes[i].set_ylabel(ylabel[i])
        axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size = 15)
        #if title: 
        #    axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))
        # Colorbar adjustments
        ax2_divider = make_axes_locatable(axes[i])
        cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
        cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
        cbar.set_label(cbarlabel[i], labelpad= -70)
        #cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])    
        cax2.xaxis.set_ticks_position("top")
        print("run completed:", i)

    #outfilename = 'parallel-out//I_kSZ_halo' + str(num_halo) +'_z016_' + str(nbins) + 'bins_' + str(rfov) + 'rfov.pdf'
    #plt.savefig(outfilename)
    plt.show()
    #plt.savefig('test.pdf')

def call_kSZ_map(halo):
    simulation_type = 'gas'
    map_kSZ_intensity(halo, 0.57, simulation_type = simulation_type, nbins = 200, rfov=2)

# $$$ CMD: >> mpiexec -n <number-of-threads> python <file>
# $$$ CMD: >> mpiexec -n 10 python map_kSZ_intensity_M-2019-03-05_14-00-00.py
call_kSZ_map(0)


"""
if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process:', rank)
    call_kSZ_map(rank)
    #with MPIPoolExecutor() as executor:
    #executor.map(call_kSZ_map, rank)
"""
