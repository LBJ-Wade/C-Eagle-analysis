import clusters_retriever as extract
import cluster_profiler as profile
from obsolete import map_synthetizer as mapgen
import kernel_convolver as kernconv
from import_toolkit import distance_cosmology as cosmo

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.convolution import convolve
from astropy.constants import c, sigma_T, k_B, m_e

def render_M(axes, num_halo, redshift, simulation_type, projection=0, nbins=100, rfov=5):
    # Import data
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))
    r200 = extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)
    fsgtab = [path, file]

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
    min_sgn = 0
    min_T = 10 ** 5
    max_r = 5

    index = np.where((r < max_r * r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn) & (T > min_T))[0]
    mass, T = mass[index], T[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    x = x * Mpc_to_arcmin
    y = y * Mpc_to_arcmin
    z = z * Mpc_to_arcmin
    r200 = r200 * Mpc_to_arcmin

    # Colors and Labels
    cmap = 'binary'
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$M_{\mathrm{gas}} / (M_\odot\ \mathrm{arcmin}^2)$', r'$M_{\mathrm{gas}} / (M_\odot\ \mathrm{arcmin}^2)$', r'$M_{\mathrm{gas}} / (M_\odot\ \mathrm{arcmin}^2)$']

    # Compute angular bins
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)

    m_H = 1.6737236 * 10 ** (-27)  # Hydrogen atom mass in kg
    S_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0])
    Mconst = 1.998 * 10 ** 30

    # Set up index permutations
    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Set up vectors
    pp_xyz = np.asarray([x, y, z])
    pp_v_ = np.asarray([vx, vy, vz])
    mass = mass.astype(np.longdouble)

    # Create mesh
    Cx, Cy = mapgen.bins_meshify(pp_xyz[ijk[projection][0]], pp_xyz[ijk[projection][1]], x_bins, y_bins)

    # Prepare Kernel
    kernel_Type = 'gauss'
    kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type=kernel_Type)
    kernel = np.array(kernel)

    """ ---------- Mass map ------------- """
    weight = mass

    # Histogram calculation
    count_M = mapgen.bins_evaluate(pp_xyz[ijk[projection][0]], pp_xyz[ijk[projection][1]], x_bins, y_bins,
                                   weights=weight)

    M = count_M / (Mconst * S_pix)

    Mmap = convolve(M, kernel)
    Mmap[Mmap == 0] = np.min(Mmap[Mmap > 0])

    if axes is None:
        return Mmap

    # Logarithmic normalization
    Mnorm = colors.LogNorm(vmin=10 ** 8)

    # Plot image
    img = axes.pcolor(Cx, Cy, Mmap, cmap=cmap, norm=Mnorm)

    # Render elements in plots
    axes.set_aspect('equal')
    axes.add_artist(Circle((0, 0), radius=r200, color='white', fill=False, linestyle='--', label=r'$R_{200}$'))
    axes.add_artist(
        Circle((0, 0), radius=5 * r200, color='black', fill=False, linewidth=0.5, linestyle='-', label=r'$R_{200}$'))
    axes.set_xlim(-rfov * r200, rfov * r200)
    axes.set_ylim(-rfov * r200, rfov * r200)
    axes.set_xlabel(xlabel[projection])
    axes.set_ylabel(ylabel[projection])
    axes.annotate(thirdAX[projection], (0.03, 0.03), textcoords='axes fraction', size=15)

    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
    cbar.set_label(cbarlabel[projection], labelpad=-70)
    cax2.xaxis.set_ticks_position("top")

    return Mmap


def render_kSZ(axes, num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, cbar_color='seismic'):
    # Import data
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))
    r200 = extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)

    # Gas particles
    file = extract.file_name_hdf5(subject='particledata', redshift=extract.redshift_floatTostr(redshift))

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
    min_sgn = 0
    min_T = 10 ** 5
    max_r = 5

    index = np.where((r < max_r * r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn) & (T > min_T))[0]
    mass, T = mass[index], T[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    x = x * Mpc_to_arcmin
    y = y * Mpc_to_arcmin
    z = z * Mpc_to_arcmin
    r200 = r200 * Mpc_to_arcmin

    cmap = [cbar_color, cbar_color, cbar_color+'_r']
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$y_{kSZ}$', r'$y_{kSZ}$', r'$y_{kSZ}$']

    # Compute angular bins
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)

    m_H = 1.6737236 * 10 ** (-27)  # Hydrogen atom mass in kg
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0]) * (3.0856776 * 10 ** 22 / Mpc_to_arcmin) ** 2

    # Set up index permutations
    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Set up vectors
    pp_xyz = np.asarray([x, y, z])
    pp_v_ = np.asarray([vx, vy, vz])

    # line of sight momentum weights - conversion to required variable type
    mass = mass.astype(np.longdouble)
    weight = pp_v_[ijk[projection][2]].astype(np.longdouble)

    # Histogram calculation
    Cx, Cy = mapgen.bins_meshify(pp_xyz[ijk[projection][0]], pp_xyz[ijk[projection][1]], x_bins, y_bins)
    count_mv = mapgen.bins_evaluate(pp_xyz[ijk[projection][0]], pp_xyz[ijk[projection][1]], x_bins, y_bins, weights=mass * weight)

    # Compute kSZ
    kSZ = -count_mv * sigma_T.value / (A_pix * c.value * m_H * 1.16)

    # Convolution
    kernel_Type = 'gauss'
    kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type=kernel_Type)
    kernel = np.array(kernel)
    kSZmap = convolve(kSZ, kernel)

    if axes is None:
        return kSZmap

    # Logarithmic normalization
    norm = colors.SymLogNorm(linthresh=10 ** -7, linscale=0.6, vmin=-np.abs(kSZmap).max(), vmax=np.abs(kSZmap).max())

    # Plot image
    img = axes.pcolor(Cx, Cy, kSZmap, cmap=cmap[projection], norm=norm)

    # Render elements in plots
    axes.set_aspect('equal')
    axes.add_artist(Circle((0, 0), radius=r200, color='black', fill=False, linestyle='--', label=r'$R_{200}$'))
    axes.add_artist(
        Circle((0, 0), radius=5 * r200, color='black', fill=False, linewidth=0.5, linestyle='-', label=r'$R_{200}$'))
    axes.set_xlim(-rfov * r200, rfov * r200)
    axes.set_ylim(-rfov * r200, rfov * r200)
    axes.set_xlabel(xlabel[projection])
    axes.set_ylabel(ylabel[projection])
    axes.annotate(thirdAX[projection], (0.03, 0.03), textcoords='axes fraction', size=15)

    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
    cbar.set_label(cbarlabel[projection], labelpad=-70)
    cax2.xaxis.set_ticks_position("top")

    return kSZmap

def render_tSZ(axes, num_halo, redshift, simulation_type, projection = 0, nbins = 100, rfov = 5, cbar_color='inferno'):
    # Import data
    path = extract.path_from_cluster_name(num_halo, simulation_type=simulation_type)
    file = extract.file_name_hdf5(subject='groups', redshift=extract.redshift_floatTostr(redshift))
    r200 = extract.group_r200(path, file)
    group_CoP = extract.group_centre_of_potential(path, file)

    # Gas particles
    file = extract.file_name_hdf5(subject='particledata', redshift=extract.redshift_floatTostr(redshift))

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
    min_sgn = 0
    min_T = 10 ** 5
    max_r = 5

    index = np.where((r < max_r * r200) & (group_number >= min_gn) & (subgroup_number >= min_sgn) & (T > min_T))[0]
    mass, T = mass[index], T[index]
    x, y, z = x[index], y[index], z[index]
    vx, vy, vz = vx[index], vy[index], vz[index]

    # Convert to angular distances
    angular_distance = cosmo.angular_diameter_D(redshift)
    Mpc_to_arcmin = np.power(np.pi, -1) * 180 * 60 / angular_distance
    x = x * Mpc_to_arcmin
    y = y * Mpc_to_arcmin
    z = z * Mpc_to_arcmin
    r200 = r200 * Mpc_to_arcmin

    cmap = [cbar_color, cbar_color, cbar_color]
    xlabel = [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
    ylabel = [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
    thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
    cbarlabel = [r'$y_{tSZ}$', r'$y_{tSZ}$', r'$y_{tSZ}$']

    # Compute angular bins
    x_bins = np.linspace(-rfov * r200, rfov * r200, nbins)
    y_bins = np.linspace(-rfov * r200, rfov * r200, nbins)

    m_H = 1.6737236 * 10 ** (-27)  # Hydrogen atom mass in kg
    A_pix = (x_bins[1] - x_bins[0]) * (y_bins[1] - y_bins[0]) * (3.0856776 * 10 ** 22 / Mpc_to_arcmin) ** 2
    tSZconst = sigma_T.value * k_B.value / (m_e.value * c.value ** 2 * m_H * 1.16)

    # Set up index permutations
    ijk = np.asarray([[0, 1, 2], [1, 2, 0], [0, 2, 1]])

    # Set up vectors
    pp_xyz = np.asarray([x, y, z])

    # Prepare Kernel
    kernel_Type = 'gauss'
    kernel, fwhm = kernconv.nika2_kernel(x_bins, y_bins, kernel_Type=kernel_Type)
    kernel = np.array(kernel)

    # mass
    mass = mass.astype(np.longdouble)

    # Histogram calculation
    Cx, Cy = mapgen.bins_meshify(pp_xyz[ijk[projection][0]], pp_xyz[ijk[projection][1]], x_bins, y_bins)

    weight = T.astype(np.longdouble)

    # Histogram calculation
    count_mT = mapgen.bins_evaluate(pp_xyz[ijk[projection][0]], pp_xyz[ijk[projection][1]], x_bins, y_bins,
                                    weights=mass * weight)

    # Compute tSZ
    tSZ = count_mT * tSZconst / A_pix

    # Convolution
    tSZmap = convolve(tSZ, kernel)
    tSZmap[tSZmap == 0] = 10 ** -10

    if axes is None:
        return tSZmap

    # Logarithmic normalization
    tSZnorm = colors.LogNorm(vmin=10 ** -10, vmax=10 ** -3)

    # Plot image
    img = axes.pcolor(Cx, Cy, tSZmap, cmap=cmap[projection], norm=tSZnorm)

    # Render elements in plots
    axes.set_aspect('equal')
    axes.add_artist(Circle((0, 0), radius=r200, color='black', fill=False, linestyle='--', label=r'$R_{200}$'))
    axes.add_artist(
        Circle((0, 0), radius=5 * r200, color='white', fill=False, linewidth=0.5, linestyle='-', label=r'$R_{200}$'))
    axes.set_xlim(-rfov * r200, rfov * r200)
    axes.set_ylim(-rfov * r200, rfov * r200)
    axes.set_xlabel(xlabel[projection])
    axes.set_ylabel(ylabel[projection])
    axes.annotate(thirdAX[projection], (0.03, 0.03), textcoords='axes fraction', size=15, color='w')

    # Colorbar adjustments
    ax2_divider = make_axes_locatable(axes)
    cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
    cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
    cbar.set_label(cbarlabel[projection], labelpad=-70)
    cax2.xaxis.set_ticks_position("top")

    return tSZmap