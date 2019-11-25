"""
This file contains methods for generating a txt file to be attached to each map.
In the txt file there are the metadata relative to the plot, the halo, selection method and weighting functions.
The methods here return strings.
"""

def metadata_file(args, fileName):

	'''
	args = (
			num_halo, 
			simulation_type, 
			redshift, 
			angular_distance.value, 
			min_gn, 
			min_sgn, 
			min_T, 
			max_r,
			weight_function,  
			nbins, 
			rfov, 
			kernel_Type, 
			fwhm, 
			r200, 
			r200/Mpc_to_arcmin			
			)
	'''

	fileName = fileName + '.txt'
	meta_data = ('0\tnum_halo:\t', 
				 '1\tsimulation_type:\t', 
				 '2\tredshift:\t', 
				 '3\tangular_distance_Mpc:\t', 
				 '4\tmin_gn:\t', 
				 '4\tmin_sgn:\t', 
				 '5\tmin_Temperature:\t', 
				 '6\tmax_radius:\t', 
				 '7\tweighting_function:\t', 
				 '8\tnun_bins:\t', 
				 '9\tr_fov:\t', 
				 '10\tkernel_Type:\t', 
				 '11\tfwhm_PSF:\t', 
				 '12\tr200_angular_arcmin:\t', 
				 '13\tr200_physical_Mpc:\t',
				 '14\tsubhalo_markers_threshold:\t')

	f = open(fileName,'w')
	for var in range(len(args)):
		line = meta_data[var] + str(args[var]) + '\n'
		f.write(line)
	f.close()