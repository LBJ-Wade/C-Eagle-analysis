from __future__ import print_function, division, absolute_import

import numpy as np
import h5py as h5
import os
from functools import wraps
# from numba import jit
# import threading

import redshift_catalogue_ceagle as zcat

#################################
#                               #
# 	G L O B    M E T H O D S    #
#							    #
#################################

def halo_Num(n: int):
	"""
	Returns the halo number in format e.g. 00, 01, 02
	"""
	return '%02d' % (n,)

def redshift_str2num(z: str):
	"""
	Converts the redshift of the snapshot from text to numerical,
	in a format compatible with the file names.
	E.g. float z = 2.16 <--- str z = 'z002p160'.
	"""
	z = z.strip('z').replace('p', '.')
	return round(float(z), 3)

def redshift_num2str(z: float):
	"""
	Converts the redshift of the snapshot from numerical to
	text, in a format compatible with the file names.
	E.g. float z = 2.16 ---> str z = 'z002p160'.
	"""
	integer_z, decimal_z = str(z).split('.')
	return 'z' + integer_z.ljust(3, '0') + 'p' + decimal_z.rjust(3, '0')



def free_memory(var_list, invert = False):
	"""
	Function for freeing memory dynamically.
	invert allows to delete all local variables that are NOT in var_list.
	"""
	if not invert:
		for name in var_list:
			if not name.startswith('_') and name in dir():
				del globals()[name]
	if invert:
		for name in dir():
			if name in var_list and not name.startswith('_'):
				del globals()[name]


#################################
#                               #
#	   S I M U L A T I O N      #
# 			C L A S S           #
#							    #
#################################

class Simulation:

	def __init__(self):
		self.simulation = 'C-EAGLE'
		self.computer = 'cosma.dur.ac.uk'
		self.pathData = '/cosma5/data/dp004/C-EAGLE/Complete_Sample'
		self.pathSave = '/cosma6/data/dp004/dc-alta2/C-Eagle-analysis-work'
		self.totalClusters = 30
		self.clusterIDAllowed = np.linspace(0, self.totalClusters-1, self.totalClusters, dtype=np.int)
		self.subjectsAllowed = ['particledata',	'groups', 'snapshot', 'snipshot', 'hsmldir', 'groups_snip']
		self.redshiftAllowed = zcat.group_data()['z_value']

	def set_pathData(self, newPath: str):
		self.pathData = newPath

	def set_totalClusters(self, newNumber: int):
		self.totalClusters = newNumber

	def get_redshiftAllowed(self, dtype = float):
		"""	Access the allowed redshifts in the simulation.	"""
		if dtype == str:
			return self.redshiftAllowed
		if dtype == float:
			return [redshift_str2num(z) for z in self.redshiftAllowed]

#################################
#                               #
#		  C L U S T E R  	    #
# 			C L A S S           #
#							    #
#################################

class Cluster (Simulation):

	def __init__(self, *args, clusterID:int = 0, redshift:float = 0.0, **kwargs):
		super().__init__()

		# Initialise and validate attributes
		self.set_clusterID(clusterID)
		self.set_redshift(redshift)

		# Pass attributed into kwargs
		kwargs['clusterID'] = self.clusterID
		kwargs['redshift'] = self.redshift

		# Set additional attributes from methods
		self.hubble_param = self.file_hubble_param()
		self.comic_time = self.file_comic_time()
		self.redshift = self.file_redshift()
		# self.Ngroups = self.file_Ngroups()
		# self.Nsubgroups = self.file_Nsubgroups()
		self.OmegaBaryon = self.file_OmegaBaryon()
		self.Omega0 = self.file_Omega0()
		self.OmegaLambda = self.file_OmegaLambda()

	# Change and validate Cluster attributes
	def set_clusterID(self, clusterID: int):
		try:
			assert (type(clusterID) is int), 'clusterID must be integer.'
			assert (clusterID in self.clusterIDAllowed), 'clusterID out of bounds (00 ... 29).'
		except AssertionError:
			raise
		else:
			self.clusterID = clusterID

	def set_redshift(self, redshift: float):
		try:
			assert (type(redshift) is float), 'redshift must be float.'
			assert (redshift >= 0.0), 'Negative redshift.'
			assert (redshift_num2str(redshift) in self.redshiftAllowed), 'Redshift not valid.'
		except AssertionError:
			raise
		else:
			self.redshift = round(redshift, 3)

	def path_from_cluster_name(self):
		"""
		RETURNS: string type. Path of the hdf5 file to extract data from.
		"""
		#os.chdir(sys.path[0])	# Set working directory as the directory of this file.
		master_directory = 	self.pathData
		cluster_ID = 'CE_' + halo_Num(self.clusterID)
		data_dir = 'data'
		return os.path.join(master_directory, cluster_ID, data_dir)


	#####################################################
	#													#
	#				D E C O R A T O R S  				#
	# 									 				#
	#####################################################

	def data_subject(**decorator_kwargs):
		"""
		This decorator adds functionality to the data import functions.
		It dynamically creates attributes and filepaths, to be then
		passed onto the actual import methods.
		:param decorator_kwargs: subject = (str)
		:return: decorated function with predefined **kwargs
		The **Kwargs are dynamically allocated to the external methods.
		"""
		# print("Reading ", decorator_kwargs['subject'], " files.")

		def wrapper(f):  # a wrapper for the function
			@wraps(f)
			def decorated_function(self, *args, **kwargs):  # the decorated function

				redshift_str = redshift_num2str(round(self.redshift, 3))
				redshift_i = self.redshiftAllowed.index(redshift_str)
				redshift_index = zcat.group_data()['z_IDNumber'][redshift_i]
				sbj_string = decorator_kwargs['subject'] + '_' + redshift_index + '_' + redshift_str
				file_dir = os.path.join(self.path_from_cluster_name(), sbj_string)
				file_list = os.listdir(file_dir)

				if decorator_kwargs['subject'] == 'particledata':
					prefix = 'eagle_subfind_particles_'
				elif decorator_kwargs['subject'] == 'groups':
					prefix = 'eagle_subfind_tab_'
				elif decorator_kwargs['subject'] == 'snapshot':
					raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
				elif decorator_kwargs['subject'] == 'snipshot':
					raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
				elif decorator_kwargs['subject'] == 'hsmldir':
					raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")
				elif decorator_kwargs['subject'] == 'groups_snip':
					raise ("[WARNING] This feature is not yet implemented in clusters_retriever.py.")

				# Transfer function state into the **kwargs
				# These **kwargs are accessible to the decorated class methods
				kwargs['subject'] = decorator_kwargs['subject']
				kwargs['file_dir'] = file_dir
				kwargs['file_list'] = [x for x in file_list if x.startswith(prefix)]
				kwargs['file_list_sorted'] = sorted([os.path.join(file_dir, file) for file in kwargs['file_list']])

				return f(self, *args, **kwargs)

			return decorated_function

		return wrapper


	#####################################################
	#													#
	#					D A T A   						#
	# 				M A N A G E M E N T 				#
	#													#
	#####################################################
	@data_subject(subject = "groups")
	def group_centre_of_potential(self, *args, **kwargs):
		"""
		AIM: reads the FoF group central of potential from the path and file given
		RETURNS: type = np.array of 3 doubles
		ACCESS DATA: e.g. group_CoP[0] for getting the x value
		"""

		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		hd5set=h5file['/FOF/GroupCentreOfPotential']
		sub_CoP = hd5set[...]
		h5file.close()
		pos = sub_CoP[0]
		free_memory(['pos'], invert = True)
		return pos

	@data_subject(subject = "particledata")
	def group_centre_of_mass(self, *args, **kwargs):
		"""
		AIM: reads the FoF group central of mass from the path and file given
		RETURNS: type = np.array of 3 doubles
		ACCESS DATA: e.g. group_CoM[0] for getting the x value
		"""
		#TODO: compute the centre of mass from particledata
		# h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
		# hd5set = h5file['/Subhalo/CentreOfPotential']
		# sub_CoP = hd5set[...]
		# h5file.close()
		# pos = sub_CoP[0]
		# free_memory(['pos'], invert=True)
		# return pos

	@data_subject(subject = "groups")
	def group_r200(self, *args, **kwargs):
		"""
		AIM: reads the FoF virial radius from the path and file given
		RETURNS: type = double
		"""
		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		h5dset=h5file["/FOF/Group_R_Crit200"]
		temp=h5dset[...]
		h5file.close()
		r200c=temp[0]
		free_memory(['r200c'], invert=True)
		return r200c

	@data_subject(subject = "groups")
	def group_r500(self, *args, **kwargs):
		"""
		AIM: reads the FoF virial radius from the path and file given
		RETURNS: type = double
		"""
		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		h5dset=h5file["/FOF/Group_R_Crit500"]
		temp=h5dset[...]
		h5file.close()
		r500c=temp[0]
		free_memory(['r500c'], invert=True)
		return r500c

	@data_subject(subject = "groups")
	def group_mass(self, *args, **kwargs):
		"""
		AIM: reads the FoF virial radius from the path and file given
		RETURNS: type = double
		"""
		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		h5dset=h5file["/FOF/GroupMass"]
		temp=h5dset[...]
		h5file.close()
		m_tot=temp[0]
		free_memory(['m_tot'], invert=True)
		return m_tot

	@data_subject(subject="groups")
	def group_M200(self, *args, **kwargs):
		"""
        AIM: reads the FoF virial radius from the path and file given
        RETURNS: type = double
        """
		h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
		h5dset = h5file["/FOF/Group_M_Crit200"]
		temp = h5dset[...]
		h5file.close()
		m200 = temp[0]
		free_memory(['m200'], invert=True)
		return m200

	@data_subject(subject = "groups")
	def group_M500(self, *args, **kwargs):
		"""
		AIM: reads the FoF virial radius from the path and file given
		RETURNS: type = double
		"""
		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		h5dset=h5file["/FOF/Group_M_Crit500"]
		temp=h5dset[...]
		h5file.close()
		m500=temp[0]
		free_memory(['m500'], invert=True)
		return m500

	@data_subject(subject = "groups")
	def extract_header_attribute(self, element_number, *args, **kwargs):
		# Import data from hdf5 file
		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		h5dset=h5file["/Header"]
		attr_name = list(h5dset.attrs.keys())[element_number]
		attr_value = list(h5dset.attrs.values())[element_number]
		h5file.close()
		return attr_name, attr_value

	@data_subject(subject = "groups")
	def extract_header_attribute_name(self, element_name, *args, **kwargs):
		# Import data from hdf5 file
		h5file=h5.File(kwargs['file_list_sorted'][0],'r')
		h5dset=h5file["/Header"]
		attr_name = h5dset.attrs.get(element_name, default=None)
		attr_value = h5dset.attrs.get(element_name, default=None)
		h5file.close()
		return attr_name, attr_value

	def file_hubble_param(self):
		"""
		AIM: retrieves the Hubble parameter of the file
		RETURNS: type = double
		"""
		_ , attr_value = self.extract_header_attribute_name('HubbleParam')
		return attr_value

	def file_comic_time(self):
		_ , attr_value = self.extract_header_attribute_name('Time')
		return attr_value

	def file_redshift(self):
		_ , attr_value = self.extract_header_attribute_name('Redshift')
		return attr_value

	def file_OmegaBaryon(self):
		_ , attr_value = self.extract_header_attribute_name('OmegaBaryon')
		return attr_value

	def file_Omega0(self):
		_ , attr_value = self.extract_header_attribute_name('Omega0')
		return attr_value

	def file_OmegaLambda(self):
		_ , attr_value = self.extract_header_attribute_name('OmegaLambda')
		return attr_value

	def file_Ngroups(self):
		_, attr_value = self.extract_header_attribute_name('TotNgroups')
		return attr_value

	def file_Nsubgroups(self):
		_, attr_value = self.extract_header_attribute_name('TotNsubgroups')
		return attr_value

	def file_NumPart_Total(self):
		"""
		[
			NumPart_Total(part_type0),
			NumPart_Total(part_type1),
			NumPart_Total(part_type2),
			NumPart_Total(part_type3),
			NumPart_Total(part_type4),
			NumPart_Total(part_type5)
		]

		:return: array of 6 elements
		"""
		_, attr_value = self.extract_header_attribute_name('NumPart_Total')
		return attr_value


	@data_subject(subject="groups")
	def NumOfSubhalos(self, *args, central_FOF = None,**kwargs):
		"""
		AIM: retrieves the redshift of the file
		RETURNS: type = int

		NOTES: there is no file for the FOF group number array. Instead,
				there is an array in /FOF for the number of subhalos in each
				FOF group. Used to gather each subgroup number
		"""
		if central_FOF:
			h5file = h5.File(kwargs['file_list_sorted'][0], 'r')
			h5dset = h5file["/FOF/NumOfSubhalos"]
			attr_value = h5dset[...]
			h5file.close()
			Ngroups = attr_value[0]
			free_memory(['Ngroups'], invert=True)
			return Ngroups

		elif central_FOF is None or not central_FOF:
			Ngroups = np.zeros(0 ,dtype=np.int)
			for path in kwargs['file_list_sorted']:
				h5file=h5.File(path,'r')
				h5dset=h5file["/FOF/NumOfSubhalos"]
				attr_value = h5dset[...]
				h5file.close()
				Ngroups = np.concatenate((Ngroups, attr_value), axis = 0)
			free_memory(['Ngroups'], invert=True)
			return Ngroups


	@data_subject(subject="groups")
	def SubGroupNumber(self, *args, central_FOF = None,**kwargs):
		"""
		AIM: reads the group number of subgroups from the path and file given
		RETURNS: type = 1/2D np.array

		if central_FOF:
			returns []

		"""
		if central_FOF:
			# Build the sgn list artificially: less overhead in opening files
			n_subhalos = self.NumOfSubhalos(*args, central_FOF=True, **kwargs)
			sgn_list = np.linspace(0, n_subhalos - 1, n_subhalos, dtype=np.int)
			free_memory(['sgn_list'], invert=True)
			return sgn_list

		elif central_FOF is None or not central_FOF:
			sgn_list = np.zeros(0 ,dtype=np.int)
			for path in kwargs['file_list_sorted']:
				h5file=h5.File(path,'r')
				h5dset=h5file["/Subhalo/SubGroupNumber"]
				sgn_sublist = h5dset[...]
				h5file.close()
				sgn_list = np.concatenate((sgn_list, sgn_sublist), axis = 0)
			# Check that the len of array is == total no of subhalos
			assert np.sum(self.NumOfSubhalos(*args, central_FOF=False, **kwargs)) == sgn_list.__len__()
			free_memory(['sgn_list'], invert=True)
			return sgn_list

	@data_subject(subject="groups")
	def subgroups_centre_of_potential(self, *args, **kwargs):
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
		pos = np.zeros( (0,3) ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/Subhalo/CentreOfPotential']
			sub_CoP = hd5set[...]
			h5file.close()
			pos = np.concatenate((pos, sub_CoP), axis = 0)
			free_memory(['pos'], invert = True)
		return pos

	@data_subject(subject="groups")
	def subgroups_centre_of_mass(self, *args, **kwargs):
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

		pos = np.zeros( (0,3) ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/Subhalo/CentreOfMass']
			sub_CoM = hd5set[...]
			h5file.close()
			pos = np.concatenate((pos, sub_CoM), axis = 0)
			free_memory(['pos'], invert = True)
		return pos

	@data_subject(subject="groups")
	def subgroups_velocity(self, *args, **kwargs):
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

		vel = np.zeros( (0,3) ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/Subhalo/Velocity']
			sub_v = hd5set[...]
			h5file.close()
			vel = np.concatenate((vel, sub_v), axis = 0)
			free_memory(['vel'], invert = True)
		return vel

	@data_subject(subject="groups")
	def subgroups_mass(self, *args, **kwargs):
		"""
		AIM: reads the subgroups masses from the path and file given
		RETURNS: type = 1D np.array
		"""

		mass = np.zeros(0 ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/Subhalo/Mass']
			sub_m = hd5set[...]
			h5file.close()
			mass = np.concatenate((mass, sub_m))
			free_memory(['mass'], invert = True)
		return mass

	@data_subject(subject="groups")
	def subgroups_kin_energy(self, *args, **kwargs):
		"""
		AIM: reads the subgroups kinetic energy from the path and file given
		RETURNS: type = 1D np.array
		"""
		kin_energy = np.zeros(0 ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/Subhalo/KineticEnergy']
			sub_ke = hd5set[...]
			h5file.close()
			kin_energy = np.concatenate((kin_energy, sub_ke), axis = 0)
			free_memory(['kin_energy'], invert = True)
		return kin_energy

	@data_subject(subject="groups")
	def subgroups_therm_energy(self, *args, **kwargs):
		"""
		AIM: reads the subgroups thermal energy from the path and file given
		RETURNS: type = 1D np.array
		"""
		therm_energy = np.zeros(0 ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/Subhalo/ThermalEnergy']
			sub_th = hd5set[...]
			h5file.close()
			therm_energy = np.concatenate((therm_energy, sub_th), axis = 0)
			free_memory(['therm_energy'], invert = True)
		return therm_energy

	# def subgroups_mass_type(self):
	# 	"""
	# 	AIM: reads the subgroups mass types from the path and file given
	# 	RETURNS: type = 2D np.array
	# 	"""
	# 	# Import data from hdf5 file
	# 	if self.subject != 'groups':
	# 		raise ValueError('subject of data must be groups.')
	# 	massType = np.zeros(0)
	# 	for path in self.filePaths:
	# 		h5file=h5.File(path,'r')
	# 		hd5set=h5file['/Subhalo/MassType']
	# 		sub_mType = hd5set[...]
	# 		h5file.close()
	# 		massType = np.concatenate((massType, sub_mType), axis = 0)
	# 		free_memory(['massType'], invert = True)
	# 	return massType
	#
	# def subgroups_number_of(self):
	# 	"""
	# 	AIM: reads the number of subgroups in FoF group from the path and file given
	# 	RETURNS: type = 1D np.array
	# 	"""
	# 	# Import data from hdf5 file
	# 	if self.subject != 'groups':
	# 		raise ValueError('subject of data must be groups.')
	# 	sub_N_tot = np.zeros(0 ,dtype=np.int)
	# 	for path in self.filePaths:
	# 		h5file=h5.File(path,'r')
	# 		hd5set=h5file['FOF/NumOfSubhalos']
	# 		sub_N = hd5set[...]
	# 		h5file.close()
	# 		sub_N_tot = np.concatenate((sub_N_tot, sub_N), axis = 0)
	# 	return sub_N_tot



	def particle_type(self, part_type = 'gas'):
		"""
		AIM: returns a string characteristic of the particle type selected
		RETURNS: string of number 0<= n <= 5
		"""
		if part_type == 'gas' 			or part_type == 0 or part_type == '0': return '0'
		elif part_type == 'highres_DM' 	or part_type == 1 or part_type == '1': return '1'
		elif part_type == 'lowres_DM' 	or part_type == 2 or part_type == '2': return '2'
		elif part_type == 'lowres_DM' 	or part_type == 3 or part_type == '3': return '3'
		elif part_type == 'stars' 		or part_type == 4 or part_type == '4': return '4'
		elif part_type == 'black_holes' or part_type == 5 or part_type == '5': return '5'
		else:
			print("[ERROR] You entered the wrong particle type!")
			exit(1)

	@data_subject(subject="particledata")
	def group_number_part(self, part_type, *args, **kwargs):
		"""
		RETURNS: np.array
		"""
		group_number = np.zeros(0 ,dtype=np.int)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType' + part_type + '/GroupNumber']
			sub_gn = hd5set[...]
			h5file.close()
			group_number = np.concatenate((group_number, sub_gn), axis = 0)
		return group_number

	@data_subject(subject="particledata")
	def subgroup_number_part(self, part_type, *args, **kwargs):
		"""
		RETURNS: np.array
		"""
		sub_group_number = np.zeros(0 ,dtype=np.int)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType' + part_type + '/SubGroupNumber']
			sub_gn = hd5set[...]
			h5file.close()
			sub_group_number = np.concatenate((sub_group_number, sub_gn), axis = 0)
		return sub_group_number

	@data_subject(subject="particledata")
	def particle_coordinates(self, part_type, *args, **kwargs):
		"""
		RETURNS: 2D np.array
		"""
		pos = np.zeros( (0,3) ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType' + part_type + '/Coordinates']
			sub_pos = hd5set[...]
			h5file.close()
			pos = np.concatenate((pos, sub_pos), axis = 0)
			free_memory(['pos'], invert = True)
		return pos

	@data_subject(subject="particledata")
	def particle_velocity(self, part_type, *args, **kwargs):
		"""
		RETURNS: 2D np.array
		"""
		part_vel = np.zeros( (0,3) ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType' + part_type + '/Velocity']
			sub_vel = hd5set[...]
			h5file.close()
			part_vel = np.concatenate((part_vel, sub_vel), axis = 0)
			free_memory(['part_vel'], invert = True)
		return part_vel

	@data_subject(subject="particledata")
	def particle_masses(self, part_type, *args, **kwargs):
		"""
		RETURNS: 2D np.array
		"""
		if (part_type != '1'):
			part_mass = np.zeros(0 ,dtype=np.float)
			for path in kwargs['file_list_sorted']:
				h5file=h5.File(path,'r')
				hd5set=h5file['/PartType' + part_type + '/Mass']
				sub_m = hd5set[...]
				h5file.close()
				part_mass = np.concatenate((part_mass, sub_m), axis = 0)
				free_memory(['part_mass'], invert = True)
		elif part_type == '1':
			part_mass = np.ones_like(self.group_number(part_type))*0.422664
		return part_mass

	@data_subject(subject="particledata")
	def particle_temperature(self, part_type = '0', *args, **kwargs):
		"""
		RETURNS: 1D np.array
		"""
		# Check that we are extracting the temperature of gas particles
		if part_type is not '0':
			print("[ERROR] Trying to extract the temperature of non-gaseous particles.")
			exit(1)
		temperature = np.zeros(0 ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType0/Temperature']
			sub_T = hd5set[...]
			h5file.close()
			temperature = np.concatenate((temperature, sub_T), axis = 0)
			free_memory(['temperature'], invert = True)
		return temperature

	@data_subject(subject="particledata")
	def particle_SPH_density(self, part_type = '0', *args, **kwargs):
		"""
		RETURNS: 1D np.array
		"""
		# Check that we are extracting the temperature of gas SPH density
		#TODO write as assert
		if part_type is not '0':
			print("[ERROR] Trying to extract the SPH density of non-gaseous particles.")
			exit(1)
		densitySPH = np.zeros(0 ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType0/Density']
			sub_den = hd5set[...]
			h5file.close()
			densitySPH = np.concatenate((densitySPH, sub_den), axis = 0)
			free_memory(['densitySPH'], invert = True)
		return densitySPH

	@data_subject(subject="particledata")
	def particle_metallicity(self, part_type = '0', *args, **kwargs):
		"""
		RETURNS: 1D np.array
		"""
		# Check that we are extracting the temperature of gas SPH density
		if part_type is not '0':
			print("[ERROR] Trying to extract the metallicity of non-gaseous particles.")
			exit(1)
		metallicity = np.zeros(0 ,dtype=np.float)
		for path in kwargs['file_list_sorted']:
			h5file=h5.File(path,'r')
			hd5set=h5file['/PartType0/Metallicity']
			sub_Z = hd5set[...]
			h5file.close()
			metallicity = np.concatenate((metallicity, sub_Z), axis = 0)
			free_memory(['metallicity'], invert = True)
		return metallicity
