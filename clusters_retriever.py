import numpy as np
import h5py as h5
import os
import sys

path_separator = '/'

def convertPath(path):
	"""
	# os.sep #### The character used by the operating system to separate pathname components. 
	# This is '/' for POSIX and '\\' for Windows. Note that knowing this is not sufficient to be able to parse or concatenate pathnames — 
	# use os.path.split() and os.path.join() — but it is occasionally useful.
	"""
	separator = os.path.sep
	if separator != path_separator:
		path = path.replace(os.path.sep, path_separator)
	return path

def halo_Num(n):
	"""
	Returns the halo number in format e.g. 0001, 0010, 0100, 1000
	OUTPUT_TYPE: str
	"""
	return '%04d' % (n,)

def path_from_cluster_name(cluster_number, simulation_type = 'gas'):
	"""
	ARGS:
		cluster_number: type = positive int 
			number of the cluster in the simulation
		simulation_type: HYDRO or DMO - default = HYDRO == 'gas'
		redshift: default = '22' i.e. z=0

	RETURNS:
		string type. Path of the hdf5 file to extract data from.
	"""
	# Set working directory as the directory of this file.
	os.chdir(sys.path[0])#
	# Remember to move up TWO directories!
	master_directory = os.path.sep.join(os.getcwd().split(os.path.sep)[:-2])
	# Construct halo number
	cluster_ID = 'halo_' + halo_Num(cluster_number)

	return os.path.join(master_directory, simulation_type, cluster_ID)


def file_name_hdf5(subject = 'particledata', redshift = '022'):
	"""
	ARGS:
		subject: particle data or group data
			select if want all particles or just data from groups.
		redshift: default = '22' i.e. z=0

	RETURNS:
		string type. Name of the hdf5 file to extract data from.
	"""
	sbj_string = subject + '_' + redshift
	if subject == 'particledata':
		file_string = os.path.join(sbj_string, 'eagle_subfind_particles_' + redshift + '.0.hdf5')
	elif subject == 'groups':
		file_string = os.path.join(sbj_string, 'eagle_subfind_tab_' + redshift + '.0.hdf5')
	else:
		print("[ERROR] subject file type not recognised. Must be 'particledata' or 'groups'.")
		exit(1)

	return file_string


def group_centre_of_potential(path, file):
	"""
	AIM: reads the FoF group central of potential from the path and file given
	RETURNS: type = np.array of 3 doubles
	ACCESS DATA: e.g. group_CoP[0] for getting the x value
	"""
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['/FOF/GroupCentreOfPotential']
	group_centre_of_potential = hd5set[0]
	h5file.close()
	return group_centre_of_potential


def group_r200(path, file):
	"""
	AIM: reads the FoF virial radius from the path and file given
	RETURNS: type = double
	"""
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset=h5file["/FOF/Group_R_Crit200"]
	temp=h5dset[...]
	r200c=temp[0]
	h5file.close()
	return r200c

def extract_header_attribute(path, file, element_number):
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset=h5file["/Header"]	
	attr_name = list(h5dset.attrs.keys())[element_number]
	attr_value = list(h5dset.attrs.values())[element_number]
	h5file.close()
	return attr_name, attr_value

def extract_header_attribute_name(path, file, element_name):
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset=h5file["/Header"]	
	attr_name = h5dset.attrs.get(element_name, default=None)
	attr_value = h5dset.attrs.get(element_name, default=None)
	h5file.close()
	return attr_name, attr_value


def file_hubble_param(path, file):
	"""
	AIM: retrieves the Hubble parameter of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'HubbleParam')
	return attr_value

def file_comic_time(path, file):
	"""
	AIM: retrieves the Hubble parameter of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'Time')
	return attr_value


def file_redshift(path, file):
	"""
	AIM: retrieves the redshift of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'Redshift')
	return attr_value


def file_Ngroups(path, file):
	"""
	AIM: retrieves the redshift of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'Ngroups')
	return attr_value


def file_Nsubgroups(path, file):
	"""
	AIM: retrieves the redshift of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'Nsubgroups')
	return attr_value


def file_OmegaBaryon(path, file):
	"""
	AIM: retrieves the redshift of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'OmegaBaryon')
	return attr_value


def file_Omega0(path, file):
	"""
	AIM: retrieves the redshift of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'Omega0')
	return attr_value


def file_OmegaLambda(path, file):
	"""
	AIM: retrieves the redshift of the file
	RETURNS: type = double
	"""
	_ , attr_value = extract_header_attribute_name(path, file, 'OmegaLambda')
	return attr_value

def subgroups_centre_of_potential(path, file):
	"""
	AIM: reads the subgroups central of potential from the path and file given
	RETURNS: type = 2D np.array
	FORMAT:	 sub#	  x.sub_CoP 	y.sub_CoP     z.sub_CoP
				0  [[			,				,			],
				1	[			,				,			],
				2	[			,				,			],
				3	[			,				,			],
				4	[			,				,			],
				5	[			,				,			],
				.						.
				.						.
				.						.					]]

	"""
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['/Subhalo/CentreOfPotential']
	sub_CoP = hd5set[...]
	h5file.close()
	return sub_CoP

def subgroups_centre_of_mass(path, file):
	"""
	AIM: reads the subgroups central of mass from the path and file given
	RETURNS: type = 2D np.array
	FORMAT:	 sub#	  x.sub_CoM 	y.sub_CoM     z.sub_CoM
				0  [[			,				,			],
				1	[			,				,			],
				2	[			,				,			],
				3	[			,				,			],
				4	[			,				,			],
				5	[			,				,			],
				.						.
				.						.
				.						.					]]

	"""
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['/Subhalo/CentreOfMass']
	sub_CoM = hd5set[...]
	h5file.close()
	return sub_CoM


def subgroups_velocity(path, file):
	"""
	AIM: reads the subgroups 3d velocities from the path and file given
	RETURNS: type = 2D np.array
	FORMAT:	 sub#	    vx.sub 	 	 vy.sub       	vz.sub
				0  [[			,				,			],
				1	[			,				,			],
				2	[			,				,			],
				3	[			,				,			],
				4	[			,				,			],
				5	[			,				,			],
				.						.
				.						.
				.						.					]]

	"""
	# Import data from hdf5 file
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['Subhalo/Velocity']
	sub_v = hd5set[...]
	h5file.close()
	return sub_v

def subgroups_mass(path, file):
	"""
	AIM: reads the subgroups masses from the path and file given
	RETURNS: type = 1D np.array
	"""	
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['Subhalo/Mass']
	sub_m = hd5set[...]
	h5file.close()
	return sub_m	

def subgroups_mass_type(path, file):
	"""
	AIM: reads the subgroups mass types from the path and file given
	RETURNS: type = 2D np.array
	"""		
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['Subhalo/MassType']
	sub_mt = hd5set[...]
	h5file.close()
	return sub_mt	

def subgroups_number_of(path, file):
	"""
	AIM: reads the number of subgroups in FoF group from the path and file given
	RETURNS: type = 1D np.array
	"""	
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['FOF/NumOfSubhalos']
	sub_N = hd5set[...]
	h5file.close()
	return sub_N

def subgroups_group_number(path, file):
	"""
	AIM: reads the group number of subgroups from the path and file given
	RETURNS: type = 1D np.array
	"""
	h5file=h5.File(os.path.join(path, file),'r')
	hd5set=h5file['Subhalo/GroupNumber']
	sub_gn = hd5set[...]
	h5file.close()
	return sub_gn

def particle_type(part_type = 'gas'):
	"""
	AIM: returns a string characteristic of the particle type selected
	RETURNS: string of number 0<= n <= 5
	"""
	if part_type == 'gas' or part_type == 0 or part_type == '0': return '0'
	elif part_type == 'highres_DM' or part_type == 1 or part_type == '1': return '1'
	elif part_type == 'lowres_DM' or part_type == 2 or part_type == '2': return '2'
	elif part_type == 'lowres_DM' or part_type == 3 or part_type == '3': return '3'
	elif part_type == 'stars' or part_type == 4 or part_type == '4': return '4'
	elif part_type == 'black_holes' or part_type == 5 or part_type == '5': return '5'
	else:
		print("[ERROR] You entered the wrong particle type!")
		exit(1)


def group_number(path, file, part_type):
	"""
	RETURNS: np.array
	"""
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType' + part_type + '/GroupNumber']
	group_number = h5dset[...]
	h5file.close()
	return group_number


def subgroup_number(path, file, part_type):
	"""
	RETURNS: np.array
	"""
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType' + part_type + '/SubGroupNumber']
	sub_group_number = h5dset[...]
	h5file.close()
	return sub_group_number

def particle_coordinates(path, file, part_type):
	"""
	RETURNS: 2D np.array
	"""
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType' + part_type + '/Coordinates']
	part_coord = h5dset[...]
	h5file.close()
	return part_coord

def particle_velocity(path, file, part_type):
	"""
	RETURNS: 2D np.array
	"""
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType' + part_type + '/Velocity']
	part_vel = h5dset[...]
	h5file.close()
	return part_vel

def particle_masses(path, file, part_type):
	"""
	RETURNS: 2D np.array
	"""
	h5file=h5.File(os.path.join(path, file),'r')
	if (part_type != '1'):
		h5dset = h5file['/PartType' + part_type + '/Mass']
		part_mass = h5dset[...]
	elif part_type == '1':
		part_mass = np.ones_like(group_number(path, file, part_type))*0.422664
	h5file.close()
	return part_mass


def particle_temperature(path, file, part_type = '0'):
	"""
	RETURNS: 1D np.array
	"""
	# Check that we are extracting the temperature of gas particles
	if part_type is not '0':
		print("[ERROR] Trying to extract the temperature of non-gaseous particles.")
		exit(1)
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType0/Temperature']
	part_temperature = h5dset[...]
	h5file.close()
	return part_temperature


def particle_SPH_density(path, file, part_type = '0'):
	"""
	RETURNS: 1D np.array
	"""
	# Check that we are extracting the temperature of gas SPH density
	if part_type is not '0':
		print("[ERROR] Trying to extract the SPH density of non-gaseous particles.")
		exit(1)
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType0/Density']
	part_density = h5dset[...]
	h5file.close()
	return part_density


def particle_metallicity(path, file, part_type = '0'):
	"""
	RETURNS: 1D np.array
	"""
	# Check that we are extracting the temperature of gas SPH density
	if part_type is not '0':
		print("[ERROR] Trying to extract the metallicity of non-gaseous particles.")
		exit(1)
	h5file=h5.File(os.path.join(path, file),'r')
	h5dset = h5file['/PartType0/Metallicity']
	part_metallicity = h5dset[...]
	h5file.close()
	return part_metallicity


def redshift_strTofloat(redshift):
	"""
	INPUT: string with redshift identifier as reported by Gadget
		e.g. '022' = {redshift} 0
	"""
	if redshift == '022': return 0
	if redshift == '016': return 0.57

def redshift_floatTostr(redshift):
	"""
	INPUT: double expressing the redshift
		e.g. 0 = '022'
	"""
	if redshift == 0: return '022'
	if redshift == 0.57: return '016'



###################################################################################
'''
### Example implementation
num_halo = 0
print("*****Test group_tab data")
path = path_from_cluster_name(num_halo, simulation_type = 'gas')

file = file_name_hdf5(subject = 'groups', redshift = '022')

print(os.path.join(path, file))

r200 = group_r200(path, file)
group_CoP = group_centre_of_potential(path, file)

print('CoP x value = ', group_CoP[0])
print('R200 = ', r200)

print("*****Test particle data")
file = file_name_hdf5(subject = 'groups', redshift = '016')
print(os.path.join(path, file))
part_type = particle_type('gas')
mass = particle_temperature(path, file, part_type)
print(mass)
print(' - - - - - - - - - \nEnd of file.')
'''


