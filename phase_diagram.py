import clusters_retriever as extract
import map_plot_parameters as plotpar
import cluster_profiler as profile
import map_synthetizer as mapgen

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from os import makedirs, chdir
from os.path import exists


def phase_diagram(num_halo, redshift, output = 'show', title = True, save_name = 'Central_group_all_part_halo_'):
	# Import data
	path = 		extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
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
	x = profile.comoving_length(x, h, redshift)
	y = profile.comoving_length(y, h, redshift)
	z = profile.comoving_length(z, h, redshift)
	r200 = profile.comoving_length(r200, h, redshift)
	density = profile.comoving_density(density, h, redshift)
	density = profile.density_units(density, unit_system = 'astro')

	# Compute radial distance
	r = np.sqrt(x**2+y**2+z**2)

	# Select particles within 5*r200
	index = np.where((r < 5*r200) & (group_number > -1) & (subgroup_number > 0))[0]
	density = density[index]
	temperature = temperature[index]

	# Bin data
	nbins = 600
	x_Data = density
	y_Data = temperature
	x_bins = np.logspace(np.min(np.log10(x_Data)), np.max(np.log10(x_Data)), nbins)
	y_bins = np.logspace(np.min(np.log10(y_Data)), np.max(np.log10(y_Data)), nbins)
	Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
	count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None)

	# Generate plot
	plotpar.set_defaults_plot()
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))

	img = axes.pcolor(Cx, Cy, np.log10(count+1), cmap='viridis')
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlabel(r'$\rho/(M_\odot\ pc^{-3})$')
	axes.set_ylabel(r'$T/K$')
	if title: 
		axes.set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))

	# Colorbar adjustments
	ax2_divider = make_axes_locatable(axes)
	cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
	cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
	cbar.set_label(r'$\log_{10}(N_{particles})$', labelpad=17)
	#cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
	cax2.xaxis.set_ticks_position("top")

	# Define output
	if output == 'show': 
		plt.show()
	elif output == 'save': 
		dir_name = 'Temperature-Density phase diagrams'
		if not exists(dir_name): 
			makedirs(dir_name)
		plt.savefig(dir_name + '//'+save_name+str(num_halo)+'z'+str(redshift).replace(".", "")+'.pdf')
	else:
		print("[ERROR] The output type you are trying to select is not defined.")
		exit(1)


def phase_diagram_Martin(num_halo, redshift, output = 'show', title = True, save_name = 'Central_group_all_part_halo_'):
	# Import data
	path = 		extract.path_from_cluster_name(num_halo, simulation_type = simulation_type)
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
	x = profile.comoving_length(x, h, redshift)
	y = profile.comoving_length(y, h, redshift)
	z = profile.comoving_length(z, h, redshift)
	r200 = profile.comoving_length(r200, h, redshift)
	density = profile.comoving_density(density, h, redshift)
	density = profile.density_units(density, unit_system = 'astro')

	# Compute radial distance
	r = np.sqrt(x**2+y**2+z**2)

	# Select particles within 5*r200
	index = np.where((r < 5*r200) & (group_number == 1) & (subgroup_number == 0))[0]
	density = density[index]
	temperature = temperature[index]

	# Bin data
	nbins = 600
	x_Data = density
	y_Data = temperature
	x_bins = np.logspace(np.min(np.log10(x_Data)), np.max(np.log10(x_Data)), nbins)
	y_bins = np.logspace(np.min(np.log10(y_Data)), np.max(np.log10(y_Data)), nbins)
	Cx, Cy = mapgen.bins_meshify(x_Data, y_Data, x_bins, y_bins)
	count = mapgen.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights = None)

	count = np.ma.masked_where(count == 0, count)
	cmap = plt.get_cmap('viridis')
	cmap.set_bad(color = 'w', alpha = 1)

	# Generate plot
	plotpar.set_defaults_plot()
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))

	img = axes.pcolor(Cx, Cy, np.log10(count), cmap=cmap)
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlabel(r'$\rho/(M_\odot\ pc^{-3})$')
	axes.set_ylabel(r'$T/K$')
	if title: 
		axes.set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))

	# Colorbar adjustments
	ax2_divider = make_axes_locatable(axes)
	cax2 = ax2_divider.append_axes("right", size="3%", pad="2%")
	cbar = plt.colorbar(img, cax=cax2, orientation='vertical')
	cbar.set_label(r'$\log_{10}(N_{particles})$', labelpad=17)
	#cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])	
	cax2.xaxis.set_ticks_position("top")

	# Define output
	if output == 'show': 
		plt.show()
	elif output == 'save': 
		dir_name = 'Temperature-Density phase diagrams'
		if not exists(dir_name): 
			makedirs(dir_name)
		plt.savefig(dir_name + '//'+save_name+str(num_halo)+'_z'+str(redshift).replace(".", "")+'.pdf')
	else:
		print("[ERROR] The output type you are trying to select is not defined.")
		exit(1)

# ***********************************
#      Example of implementation
# ***********************************

# Specify object
simulation_type = 'gas'
#num_halo = 9
redshift = 0.57
h = 0.67777

for num_halo in range(4, 5):
	print('\nExamining halo\t', num_halo)
	phase_diagram(num_halo, redshift, output = 'show', title = True, save_name = 'Phase-diag_central-only_halo_000')

#from pdf_graph_merger import merge_pdf
#chdir('Temperature-Density phase diagrams')
#merge_pdf('Central_group_all_particles_halo_', out_filename = 'Central_group_all_particles')

print(' - - - - - - - - - \nEnd of file.')
