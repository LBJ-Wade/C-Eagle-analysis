import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen
import cosmolopy

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def map_kSZ_intensity(num_halo, redshift, simulation_type, nbins):
	# Import data
	path = 		extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
	file = 		extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
	r200 = 		extract.group_r200(path, file)
	group_CoP = extract.group_centre_of_potential(path, file)
	file = 		extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
	
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

	h = 0.67777

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
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

	# Convert to angular distances
	cosmo = {'omega_M_0' : 0.307, 'omega_lambda_0' : 0.693, 'h' : 0.6777}
	cosmo = cosmolopy.set_omega_k_0(cosmo)
	zz = 0.576777
	angular_distance = cosmolopy.angular_diameter_distance(zz, **cosmo)
	Mpc_to_rad = 1/angular_distance
	print("conversion: ", Mpc_to_rad)

	x, y, z = x*Mpc_to_rad, y*Mpc_to_rad, z*Mpc_to_rad

	# Bin data
	cmap = ['seismic', 'seismic', 'seismic_r']
	xlabel = [r'$x\mathrm{/Mpc}$', r'$y\mathrm{/Mpc}$', r'$x\mathrm{/Mpc}$']
	ylabel = [r'$y\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$']
	thirdAX = [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
	cbarlabel = [r'$\sum_{i} m_i v_{z, i} / \sum_{i} m_i \ [\mathrm{km\ s^{-1}}]$',
				 r'$\sum_{i} m_i v_{x, i} / \sum_{i} m_i \ [\mathrm{km\ s^{-1}}]$',
				 r'$\sum_{i} m_i v_{y, i} / \sum_{i} m_i \ [\mathrm{km\ s^{-1}}]$']

	for i in [0,1,2]:
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

		x_bins = np.linspace(np.min(x_Data), np.max(x_Data), nbins)
		y_bins = np.linspace(np.min(y_Data), np.max(y_Data), nbins)
		print(np.shape(x_Data), np.shape(y_Data), np.shape(x_bins), np.shape(y_bins))
		Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
		# line of sight momentum weights
		count_mv = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = mass*weight)
		# mass weights
		count_m = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = mass)
		# average mass weighted velocity
		count_m[count_m == 0] = 1
		count = np.divide(count_mv, count_m)
		norm = mapgen.MidpointNormalize(vmin=count.min(), vmax=count.max(), midpoint=0)
		img = axes[i].pcolormesh(Cx, Cy, count, cmap=cmap[i], norm= norm)

		# Render elements in plots
		axes[i].set_aspect('equal')
		#axes[i].add_artist(Circle((0,0), radius=r200, color = 'black', fill = False, linestyle = '--', label = r'$R_{200}$'))
		#axes[i].add_artist(Circle((0,0), radius=5*r200, color = 'black', fill = False, linewidth = 0.5,linestyle = '-', label = r'$R_{200}$'))
		#axes[i].set_xlim(-5.1*r200, 5.1*r200)
		#axes[i].set_ylim(-5.1*r200, 5.1*r200)
		axes[i].set_xlabel(xlabel[i])
		axes[i].set_ylabel(ylabel[i])
		axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size = 15)
		#if title: 
		#	axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))
		# Colorbar adjustments
		ax2_divider = make_axes_locatable(axes[i])
		cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
		cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
		cbar.set_label(cbarlabel[i], labelpad=17)
		#cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
		cax2.xaxis.set_ticks_position("top")

	plt.show()

simulation_type = 'gas'
map_kSZ_intensity(0, 0.57, simulation_type = simulation_type, nbins = 800)