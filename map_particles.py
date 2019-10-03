import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Circle
from os import makedirs, chdir
from os.path import exists



def map_particles(num_halo, redshift, simulation_type = 'gas', output = 'show', title = True, save_name = 'Map_particles_gas', nbins = 400):
	# Import data
	path = 		extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
	file = 		extract.file_name_hdf5(subject = 'groups', redshift = extract.redshift_floatTostr(redshift))
	r200 = 		extract.group_r200(path, file)
	group_CoP = extract.group_centre_of_potential(path, file)
	file = 		extract.file_name_hdf5(subject = 'particledata', redshift = extract.redshift_floatTostr(redshift))
	#print(r200)
	h = extract.file_hubble_param(path, file)


	# Gas particles
	part_type = extract.particle_type('gas')
	mass = extract.particle_masses(path, file, part_type)
	coordinates = extract.particle_coordinates(path, file, part_type)
	velocities = extract.particle_velocity(path, file, part_type)
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

	# Compute radial distance
	r = np.sqrt(x**2+y**2+z**2)

	# Select particles within 5*r200
	index = np.where((r < 5*r200) & (group_number > -1) & (subgroup_number > -1))[0]
	mass = mass[index]
	x, y, z = x[index], y[index], z[index]
	vx, vy, vz = vx[index], vy[index], vz[index]

	# Generate plot
	plotpar.set_defaults_plot()
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))

	# Bin data
	#nbins = 250
	cmap = 'terrain_r'
	################################################################################################
	x_Data = x
	y_Data = y
	x_bins = np.linspace(np.min(x_Data), np.max(x_Data), nbins)
	y_bins = np.linspace(np.min(y_Data), np.max(y_Data), nbins)
	Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
	count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None)
	img = axes[0].pcolor(Cx, Cy, np.log10(count), cmap=cmap)
	axes[0].set_xlabel(r'$x\mathrm{/Mpc}$')
	axes[0].set_ylabel(r'$y\mathrm{/Mpc}$')
	axes[0].annotate(r'$\bigotimes z$', (0.03, 0.03), textcoords='axes fraction', size = 15)
	# Colorbar adjustments
	ax2_divider = make_axes_locatable(axes[0])
	cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
	cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
	#cbar.set_label(r'$\log_{10}(N_{particles})$', labelpad=17)
	#cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
	cax2.xaxis.set_ticks_position("top")

	
	x_Data = y
	y_Data = z
	x_bins = np.linspace(np.min(x_Data), np.max(x_Data), nbins)
	y_bins = np.linspace(np.min(y_Data), np.max(y_Data), nbins)
	Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
	count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None)
	img = axes[1].pcolor(Cx, Cy, np.log10(count), cmap=cmap)
	axes[1].set_xlabel(r'$y\mathrm{/Mpc}$')
	axes[1].set_ylabel(r'$z\mathrm{/Mpc}$')
	axes[1].annotate(r'$\bigotimes x$', (0.03, 0.03), textcoords='axes fraction', size = 15)
	# Colorbar adjustments
	ax2_divider = make_axes_locatable(axes[1])
	cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
	cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
	#cbar.set_label(r'$\log_{10}(N_{particles})$', labelpad=17)
	#cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
	cax2.xaxis.set_ticks_position("top")


	x_Data = x
	y_Data = z
	x_bins = np.linspace(np.min(x_Data), np.max(x_Data), nbins)
	y_bins = np.linspace(np.min(y_Data), np.max(y_Data), nbins)
	Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
	count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None)
	img = axes[2].pcolor(Cx, Cy, np.log10(count), cmap=cmap)
	axes[2].set_xlabel(r'$x\mathrm{/Mpc}$')
	axes[2].set_ylabel(r'$z\mathrm{/Mpc}$')
	axes[2].annotate(r'$\bigodot y$', (0.03, 0.03), textcoords='axes fraction', size = 15)
	# Colorbar adjustments
	ax2_divider = make_axes_locatable(axes[2])
	cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
	cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
	cbar.set_label(r'$\log_{10}(N_{particles})$', labelpad=17)
	#cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
	cax2.xaxis.set_ticks_position("top")
	################################################################################################

	# Plot r200 circle
	for i in [0,1,2]:
		axes[i].set_aspect('equal')
		axes[i].add_artist(Circle((0,0), radius=r200, color = 'red', fill = False, linestyle = '--', label = r'$R_{200}$'))
		axes[i].add_artist(Circle((0,0), radius=5*r200, color = 'black', fill = False, linewidth = 0.5,linestyle = '-', label = r'$R_{200}$'))
		axes[i].set_xlim(-5.1*r200, 5.1*r200)
		axes[i].set_ylim(-5.1*r200, 5.1*r200)
		if title: 
			axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))

	# Define output
	if output == 'show': 
		plt.show()
	elif output == 'save': 
		dir_name = 'Map particles'
		if not exists(dir_name): 
			makedirs(dir_name)
		plt.savefig(dir_name + '//'+save_name+'_partType'+part_type+'_halo'+str(num_halo)+'z'+str(redshift).replace(".", "")+'.pdf')
	else:
		print("[ERROR] The output type you are trying to select is not defined.")
		exit(1)


# ***********************************
#      Example of implementation
# ***********************************
'''
# Specify object
simulation_type = 'gas'
num_halo = 0
redshift = 0.57
h = 0.67777

for num_halo in range(0,10):
	print('\nExamining halo\t', num_halo)
	#map_particles(num_halo, redshift, simulation_type = 'gas', output = 'save', title = True, save_name = 'Map_particles_gas', nbins = 400)

from pdf_graph_merger import merge_pdf
chdir('Map particles')
merge_pdf('Map_particles_gas', out_filename = 'Map_particles_gas')

print(' - - - - - - - - - \nEnd of file.')
'''