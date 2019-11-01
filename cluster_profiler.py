"""
This file contains declaration and definition of functions for a basic
disgnostics of the cluster examined.
"""

import numpy as np
import numba as jit
import clusters_retriever as extract


def cluster_average_velocity(velocity3D):
	"""
	INPUT: np.array 2D exoressing the 3-component velocity of the particles.
	RETURNS: a np.array 1D with the 3-components of the bulk velocity of the particles.
	USES: work in the FoF group's rest frame by computing the mean velocity.
	NOTES: the rest-mass frame depends on the mass-weights of the particles,
		hence on part-type.
		THIS IS ONLY AVERAGE VELOCITY
	"""
	return  [np.mean(velocity3D[:,0]),
			 np.mean(velocity3D[:,1]),
			 np.mean(velocity3D[:,2])]


def cluster_average_momentum(self, part_type):
	"""
	Returns the mass-weighted average of the velocities, i.e. zero-momentum frame
		for a SINGLE particle type.
	"""
	part_type = extract.Cluster.particle_type(self, part_type)
	mass = extract.Cluster.particle_masses(self, part_type)
	velocity = extract.Cluster.particle_velocity(self, part_type)
	sum_of_masses = np.sum(mass)
	rest_frame = [np.sum(mass*velocity[:,0])/sum_of_masses,
				  np.sum(mass*velocity[:,1])/sum_of_masses,
				  np.sum(mass*velocity[:,2])/sum_of_masses]

	return rest_frame, sum_of_masses

def subhalo_average_momentum(path, file, part_type):
	part_type = extract.particle_type(part_type)
	sgmass = extract.subgroups_mass(path, file)
	sgmasstype = extract.subgroups_mass_type(path, file)
	sgvelocity = extract.subgroups_velocity(path, file)
	numofsg = extract.subgroups_number_of(path, file)
	#groupnumber = extract.group_number(path, file, part_type)
	#print("PT:", type(part_type))
	sub_mom = np.transpose([sgmass, sgmass, sgmass])*sgvelocity*np.transpose([sgmasstype[:,int(part_type)], sgmasstype[:,int(part_type)], sgmasstype[:,int(part_type)]])
	#momentum = np.multiply(np.transpose([mass, mass, mass]), velocity)

	#for i in range(len(momentum))
	#	sub_mom[fuctionIndex(groupnumber[i], subnumber[i])] += momentum[i]
	#sub_mom = [sum([mass[i]*velocity[i] for i in range(len(mass)) if (subnumber[i] == j & groupnumber[i] == 0)]) for j in range(max(subnumber))]
	return sub_mom

def total_mass_rest_frame(self):
	"""
	Returns the mass-weighted average of the velocities, i.e. zero-momentum frame
		for a ALL particle types.
	"""
	part_type_momentum = []
	part_type_totMass = []
	for part_type in ['gas', 'highres_DM', 'stars', 'black_holes']:
		rest_frame, sum_of_masses = cluster_average_momentum(self, part_type)
		part_type_momentum.append(rest_frame)
		part_type_totMass.append(sum_of_masses)
	sum_of_masses = np.sum(part_type_totMass)

	# Explicitly calculate the averaged rest frame
	rest_frame = [(part_type_momentum[0][0]*part_type_totMass[0] + part_type_momentum[1][0]*part_type_totMass[1] + part_type_momentum[2][0]*part_type_totMass[2] + part_type_momentum[3][0]*part_type_totMass[3])/sum_of_masses,
				  (part_type_momentum[0][1]*part_type_totMass[0] + part_type_momentum[1][1]*part_type_totMass[1] + part_type_momentum[2][1]*part_type_totMass[2] + part_type_momentum[3][1]*part_type_totMass[3])/sum_of_masses,
				  (part_type_momentum[0][2]*part_type_totMass[0] + part_type_momentum[1][2]*part_type_totMass[1] + part_type_momentum[2][2]*part_type_totMass[2] + part_type_momentum[3][2]*part_type_totMass[3])/sum_of_masses]

	return rest_frame, sum_of_masses


def filter_sphere(r, unfiltered_data, radius):
	"""
	CREATED: 12.02.2019
	LAST MODIFIED: 12.02.2019

	INPUTS:
		r: [np.array] or [list] (NB r and unfiltered_data must be of same length and same type.)
			r is the radial distance of the particles from the centre (of potential or of mass).

		unfiltered_data: [np.array] or [list] (NB r and unfiltered_data must be of same length and same type.)
			the data to filter with respect to the radial distance

		radius: [float] or [double] or [int]
			is the threshold radius. Select all particles within the spherical surface defined by the radius.

	RETURNS: the array of filtered data.

	USE: filter particles within r200 or 5*r200.
	"""
	# Select particles within radius
	index = np.where(r < radius)[0]
	filtered_data = unfiltered_data[index]
	return filtered_data

def density_units(density, unit_system = 'SI'):
	"""
	CREATED: 12.02.2019
	LAST MODIFIED: 12.02.2019

	INPUTS: density np.array

			metric system used: 'SI' or 'cgs' or astronomical 'astro'
	"""
	if unit_system == 'SI':
		# kg*m^-3
		return density*6.769911178294543*10**-28
	elif unit_system == 'cgs':
		# g*cm^-3
		return density*6.769911178294543*10**-31
	elif unit_system == 'astro':
		# solar masses / (parsec)^3
		return density*6.769911178294543*np.power(3.086, 3)/1.9891 * 10**-10
	elif unit_system == 'nHcgs':
		return density*6.769911178294543*10**-31/(1.674*10**-24)
	else:
		print("[ERROR] Trying to convert SPH density to an unknown metric system.")
		exit(1)


def velocity_units(velocity, unit_system = 'SI'):
	"""
	CREATED: 14.02.2019
	LAST MODIFIED: 14.02.2019

	INPUTS: velocity np.array

			metric system used: 'SI' or 'cgs' or astronomical 'astro'
	"""
	if unit_system == 'SI':
		# m/s
		return velocity*1000
	elif unit_system == 'cgs':
		# cm/s
		return velocity*100000
	elif unit_system == 'astro':
		# km/s
		return velocity*1
	else:
		print("[ERROR] Trying to convert velocity to an unknown metric system.")
		exit(1)

def mass_units(mass, unit_system = 'SI'):
	"""
	CREATED: 14.02.2019
	LAST MODIFIED: 14.02.2019

	INPUTS: mass np.array

			metric system used: 'SI' or 'cgs' or astronomical 'astro'
	"""
	if unit_system == 'SI':
		# m/s
		return mass*1.9891*10**40
	elif unit_system == 'cgs':
		# cm/s
		return mass*1.9891*10**43
	elif unit_system == 'astro':
		# km/s
		return mass*10**10
	else:
		print("[ERROR] Trying to convert mass to an unknown metric system.")
		exit(1)

def momentum_units(momentum, unit_system = 'SI'):
	"""
	CREATED: 07.03.2019
	LAST MODIFIED: 07.03.2019

	INPUTS: momentum np.array

			metric system used: 'SI' or 'cgs' or astronomical 'astro'
	"""
	if unit_system == 'SI':
		# m/s
		return momentum*1.9891*10**43
	elif unit_system == 'cgs':
		# cm/s
		return momentum*1.9891*10**48
	elif unit_system == 'astro':
		# km/s
		return momentum*10**10
	else:
		print("[ERROR] Trying to convert mass to an unknown metric system.")
		exit(1)

def comoving_density(density, hubble_par, redshift):
	"""
	Rescales the density from the comoving coordinates to the physical coordinates
	"""
	scale_factor = 1/(redshift + 1)
	return density*hubble_par**2*scale_factor**-3

def comoving_length(coord, hubble_par, redshift):
	"""
	Rescales the density from the comoving length to the physical length
	"""
	scale_factor = 1/(redshift + 1)
	return coord*scale_factor/hubble_par

def comoving_velocity(vel, hubble_par, redshift):
	"""
	Rescales the density from the comoving velocity to the physical velocity
	"""
	scale_factor = 1/(redshift + 1)
	return vel*np.sqrt(scale_factor)

def comoving_mass(mass, hubble_par, redshift):
	"""
	Rescales the density from the comoving mass to the physical mass
	"""
	return mass/hubble_par

def comoving_momentum(mom, hubble_par, redshift):
	"""
	Rescales the momentum from the comoving to the physical
	"""
	scale_factor = 1/(redshift + 1)
	return mom*np.sqrt(scale_factor)/hubble_par
