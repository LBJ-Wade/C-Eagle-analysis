import numpy as np
import h5py as h5
import os
import sys

path_separator = '/'

#####################################################
#													#
#					F I L E  						#
# 				M A N A G E M E N T 				#
#													#
#####################################################

def halo_Num(n: int):
	"""
	Returns the halo number in format e.g. 00, 01, 02
	OUTPUT_TYPE: str
	"""
	if n > -1 and n < 30:
		return '%02d' % (n,)
	else:
		print("[ERROR] Cluster number out of bounds (00 ... 29). The C-EAGLE dataset has 30 clusters.")
		sys.exit(1)

def redshift_num2str(z: float) -> str:
	"""
	Converts the redshift of the snapshot from numerical to text, in a format compatible with the file names.
	E.g. float z = 2.16 ---> str z = 'z002p160'.
	"""
	integer_z, decimal_z = divmod(z, 1)
	integer_z = '%03d' % (int(integer_z),)
	decimal_z = '%03d' % (int(decimal_z),)
	return 'z' + integer_z + 'p' + decimal_z

def redshift_str2num(z: str) -> float:
	"""
	Converts the redshift of the snapshot from text to numerical, in a format compatible with the file names.
	E.g. float z = 2.16 <--- str z = 'z002p160'.
	"""
	z = z.strip('z').replace('p', '.')
	return float(z)

def path_from_cluster_name(cluster_number):
	"""
	ARGS:
		cluster_number: type = positive int 
			number of the cluster in the simulation

	RETURNS:
		string type. Path of the hdf5 file to extract data from.
	"""
	os.chdir(sys.path[0])	# Set working directory as the directory of this file.
	master_directory = 	'/cosma5/data/dp004/C-EAGLE/Complete_Sample'
	cluster_ID = 		'CE_' + halo_Num(cluster_number)
	data_dir = 			'data'

	return os.path.join(master_directory, cluster_ID, data_dir)

def get_redshift_catalogue():
	z_dict = {
		'z_type': # either 'snapshot' or 'snipshot'
			['snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 
			'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 
			'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 
			'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 'snapshot', 
			'snapshot', 'snapshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 
			'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot', 'snipshot'],
		'z_IDNumber': # The sequential number of the sna/ipshots
			['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', 
			'012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', 
			'024', '025', '026', '027', '028', '029', '000', '001', '002', '003', '004', '005', 
			'006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', 
			'018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', 
			'030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', 
			'042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', 
			'054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', 
			'066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', 
			'078', '079', '080', '081'],
		'z_value': # The value of the sna/ipshot redshifts
			['z014p003', 'z006p772', 'z004p614', 'z003p512', 'z002p825', 'z002p348', 'z001p993', 
			'z001p716', 'z001p493', 'z001p308', 'z001p151', 'z001p017', 'z000p899', 'z000p795', 
			'z000p703', 'z000p619', 'z000p543', 'z000p474', 'z000p411', 'z000p366', 'z000p352', 
			'z000p297', 'z000p247', 'z000p199', 'z000p155', 'z000p113', 'z000p101', 'z000p073', 
			'z000p036', 'z000p000', 'z010p873', 'z008p988', 'z007p708', 'z006p052', 'z005p478', 
			'z005p008', 'z004p279', 'z003p989', 'z003p736', 'z003p313', 'z003p134', 'z002p972', 
			'z002p691', 'z002p567', 'z002p453', 'z002p250', 'z002p158', 'z002p073', 'z001p917', 
			'z001p846', 'z001p779', 'z001p656', 'z001p599', 'z001p544', 'z001p443', 'z001p396', 
			'z001p351', 'z001p266', 'z001p226', 'z001p188', 'z001p116', 'z001p082', 'z001p049', 
			'z000p986', 'z000p956', 'z000p927', 'z000p872', 'z000p846', 'z000p820', 'z000p771', 
			'z000p748', 'z000p725', 'z000p681', 'z000p660', 'z000p639', 'z000p599', 'z000p580', 
			'z000p562', 'z000p525', 'z000p508', 'z000p491', 'z000p458', 'z000p442', 'z000p426', 
			'z000p395', 'z000p381', 'z000p366', 'z000p338', 'z000p324', 'z000p311', 'z000p284', 
			'z000p272', 'z000p259', 'z000p234', 'z000p223', 'z000p211', 'z000p188', 'z000p177', 
			'z000p166', 'z000p144', 'z000p133', 'z000p123', 'z000p103', 'z000p093', 'z000p083', 
			'z000p063', 'z000p054', 'z000p045', 'z000p026', 'z000p018', 'z000p009', 'z000p000']
	}
	return z_dict

def file_dir_hdf5(cluster_num: int = 0, subject: str = 'particledata', redshift = 0.0) -> str:
	"""
	ARGS:
		subject: particle data or group data
			select if want all particles or just data from groups.
		redshift: default = '22' i.e. z=0

	RETURNS:
		string type. Name of the hdf5 directory to extract data from.
	"""
	# Validate redshift
	if type(redshift) is float: 
		redshift = redshift_num2str(redshift)
	elif type(redshift) is str:
		pass
	else:
		print('[ERROR] Redshift variable type is neither float nor str.')
		exit(1)
	
	if redshift not in get_redshift_catalogue()['z_value']:
		print('[ERROR] Redshift entered does not correspond to simulated snapshot or snipshot')
		exit(1)
	
	redshift_i = get_redshift_catalogue()['z_value'].index(redshift)
	redshift_index = get_redshift_catalogue()['z_IDNumber'][redshift_i]

	sbj_string = subject + '_' + redshift_index + '_' + redshift
	file_dir = os.path.join(path_from_cluster_name(cluster_num), sbj_string)
	file_list = os.listdir(file_dir)
	
	if subject == 'particledata':	prefix = 'eagle_subfind_particles_'
	elif subject == 'groups':		prefix = 'eagle_subfind_tab_'
	elif subject == 'snapshot':
		print("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
		exit(1)
	elif subject == 'snipshot':
		print("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
		exit(1)
	elif subject == 'hsmldir':
		print("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
		exit(1)
	elif subject == 'groups_snip':
		print("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
		exit(1)
	else:
		print("[ERROR] subject file type not recognised. Must be 'particledata' or 'groups' or 'snapshot' or 'snipshot' or 'hsmldir' or 'groups_snip'.")
		exit(1)

	filter(lambda x: x.startswith(prefix), file_list)
	return file_dir, file_list

#####################################################
#													#
#					D A T A   						#
# 				M A N A G E M E N T 				#
#													#
#####################################################

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


print(file_dir_hdf5(subject= 'groups', redshift = 0.0))




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


