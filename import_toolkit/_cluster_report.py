"""
------------------------------------------------------------------
FILE:   _cluster_report.py
AUTHOR: Edo Altamura
DATE:   21-05-2020
------------------------------------------------------------------
This file is an extension of the cluster.Cluster class. It provides
class methods for performing basic analysis procedures on C-EAGLE
data from the /cosma5 data system.
This file contains a mixin class, affiliated to cluster.Cluster.
Mixins are classes that have no data of their own — only methods —
so although you inherit them, you never have to call super() on them.
They working principle is based on OOP class inheritance.
-------------------------------------------------------------------
"""

import numpy as np
from typing import Dict, Union
from unyt import hydrogen_mass, boltzmann_constant, gravitational_constant, parsec, solar_mass
import warnings

# Delete the units from Unyt constants
hydrogen_mass = float(hydrogen_mass.value)
boltzmann_constant = float(boltzmann_constant.value)
G_SI = float(gravitational_constant.value)
G_astro = float(gravitational_constant.in_units('(1e6*pc)*(km/s)**2/(1e10*solar_mass)').value)
parsec = float((1 * parsec).in_units('m').value)
solar_mass = float(solar_mass.value)


class Mixin:

	def group_fofinfo(self, aperture_radius: float = None) -> Dict[str, Union[np.ndarray, np.float]]:
		if aperture_radius is None: warnings.warn(f'Aperture radius not defined.')
		fof_dict = {
				'hubble_param'       : self.hubble_param,
				'comic_time'         : self.comic_time,
				'redshift'           : self.z,
				'OmegaBaryon'        : self.OmegaBaryon,
				'Omega0'             : self.Omega0,
				'OmegaLambda'        : self.OmegaLambda,
				'centre_of_potential': self.centre_of_potential,
				'r_aperture'         : aperture_radius,
				'r200'               : self.r200,
				'r500'               : self.r500,
				'r2500'              : self.r2500,
				'mfof'               : self.Mtot,
				'm200'               : self.M200,
				'm500'               : self.M500,
				'm2500'              : self.M2500,
				'NumOfSubhalos'      : self.NumOfSubhalos
		}
		return fof_dict

	def group_dynamics(self, aperture_radius: float = None) -> Dict[str, np.ndarray]:
		"""
		Method that computes the cluster's morphology information from particles within a
		specified aperture.
		If the aperture is not specified, it is set by default to the true R500 of the cluster
		radius from the centre of potential and computes the morphology analysis using the relative
		static methods in this Mixin class.
		The method also checks that the necessary datasets are loaded into the cluster object.
		The information is gathered in dictionary-form output and contains the following analysis
		datasets, relative to the given aperture:

			- Inertia tensor (3x3 matrix): np.ndarray
			- Eigenvalues of the inertia tensor, per unit mass 3x np.float
				lambda_i / mass_within_aperture = (semi-axis length)**2
				They represent the semi-axes of the ellipsoid.
			- Eigenvectors of the inertia tensor 3x 1D array
				They are normalised and express the orientation of the semi-axes of the ellipsoid.
			- Triaxiality 1x np.float:
				triaxiality = (a**2-b**2)/(a**2-c**2)
			-Circularity 1x np.float:
				circularity = c/a

		Each of these datasets is structured as follows:

			[
				Dynamics_dataset_allParticleTypes,
				Dynamics_dataset_ParticleType0,
				Dynamics_dataset_ParticleType1,
				Dynamics_dataset_ParticleType4,
			]

		:param out_allPartTypes: default = False
		:param aperture_radius: default = None (R500)
		:return: expected a numpy array of dimension 1 if all particletypes are combined, or
			dimension 2 if particle types are returned separately.
		"""
		if aperture_radius is None:
			aperture_radius = self.r500
			warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:2.2f} Mpc.')

		mass = np.zeros(0, dtype=np.float)
		coords = np.zeros((0, 3), dtype=np.float)
		velocity = np.zeros((0, 3), dtype=np.float)
		temperature = np.zeros(0, dtype=np.float)

		N_particles = np.zeros(0, dtype=np.int)
		aperture_mass = np.zeros(0, dtype=np.float)
		centre_of_mass = np.zeros((0, 3), dtype=np.float)
		zero_momentum_frame = np.zeros((0, 3), dtype=np.float)
		angular_momentum = np.zeros((0, 3), dtype=np.float)
		angular_velocity = np.zeros((0, 3), dtype=np.float)
		specific_angular_momentum = np.zeros(0, dtype=np.float)
		circular_velocity = np.zeros(0, dtype=np.float)
		spin_parameter = np.zeros(0, dtype=np.float)
		substructure_mass = np.zeros(0, dtype=np.float)
		substructure_fraction = np.zeros(0, dtype=np.float)
		thermal_energy = np.zeros(0, dtype=np.float)
		kinetic_energy = np.zeros(0, dtype=np.float)
		dynamical_merging_index = np.zeros(0, dtype=np.float)
		thermodynamic_merging_index = np.zeros(0, dtype=np.float)

		for part_type in ['4', '1', '0']:
			assert hasattr(self, f'partType{part_type}_coordinates')
			assert hasattr(self, f'partType{part_type}_velocity')
			assert hasattr(self, f'partType{part_type}_mass')
			if part_type is '0': assert hasattr(self, f'partType{part_type}_temperature')
			assert hasattr(self, f'partType{part_type}_subgroupnumber')
			radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
			aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
			del radial_dist
			_mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
			_velocity = getattr(self, f'partType{part_type}_velocity')[aperture_radius_index]
			_coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
			_temperature = getattr(self, f'partType{part_type}_temperature')[aperture_radius_index] if part_type is '0' else np.array([])
			_subgroupnumber = getattr(self, f'partType{part_type}_subgroupnumber')[aperture_radius_index]
			if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
			if _velocity.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
			if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
			if part_type is '0' and _temperature.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
			if _subgroupnumber.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")

			mass = np.concatenate((mass, _mass), axis=0)
			coords = np.concatenate((coords, _coords), axis=0)
			velocity = np.concatenate((velocity, _velocity), axis=0)
			temperature = np.concatenate((temperature, _temperature), axis=0)

			_N_particles = len(_mass)
			N_particles = np.append(N_particles, _N_particles)

			_aperture_mass = np.sum(_mass)
			aperture_mass = np.append(aperture_mass, _aperture_mass)

			_coords_norm = np.subtract(_coords, self.centre_of_potential)
			_centre_of_mass = self.centre_of_mass(_mass, _coords)
			centre_of_mass = np.concatenate((centre_of_mass, _centre_of_mass[None, :]), axis=0)

			_zero_momentum_frame = self.zero_momentum_frame(_mass, _velocity)
			zero_momentum_frame = np.concatenate((zero_momentum_frame, _zero_momentum_frame[None, :]), axis=0)

			_velocity_norm = np.subtract(_velocity, _zero_momentum_frame)
			_angular_momentum = self.angular_momentum(_mass, _coords_norm, _velocity_norm)
			angular_momentum = np.concatenate((angular_momentum, _angular_momentum[None, :]), axis=0)

			_angular_velocity = np.linalg.inv(self.inertia_tensor(_mass, _coords_norm)) @ _angular_momentum
			angular_velocity = np.concatenate((angular_velocity, _angular_velocity[None, :]), axis=0)

			_specific_angular_momentum = np.linalg.norm(_angular_momentum) / _aperture_mass
			specific_angular_momentum = np.append(specific_angular_momentum, _specific_angular_momentum)

			_circular_velocity = np.sqrt(G_astro * _aperture_mass / aperture_radius)
			circular_velocity = np.append(circular_velocity, _circular_velocity)

			_spin_parameter = np.linalg.norm(_angular_momentum) / (_aperture_mass * aperture_radius * _circular_velocity * np.sqrt(2))
			spin_parameter = np.append(spin_parameter, _spin_parameter)

			sgn_index = np.where(_subgroupnumber == 0)[0]
			_substructure_mass = _aperture_mass - np.sum(_mass[sgn_index])
			substructure_mass = np.append(substructure_mass, _substructure_mass)

			_substructure_fraction = _substructure_mass / _aperture_mass
			substructure_fraction = np.append(substructure_fraction, _substructure_fraction)

			_thermal_energy = self.thermal_energy(self.mass_units(_mass), _temperature) * np.power(10., -46) if part_type is '0' else 0.
			thermal_energy = np.append(thermal_energy, _thermal_energy)

			_kinetic_energy = self.kinetic_energy(self.mass_units(_mass), self.velocity_units(_velocity_norm)) * np.power(10., -46)
			kinetic_energy = np.append(kinetic_energy, _kinetic_energy)

			_dynamical_merging_index = np.linalg.norm(self.centre_of_potential - _centre_of_mass) / aperture_radius
			dynamical_merging_index = np.append(dynamical_merging_index, _dynamical_merging_index)

			_thermodynamic_merging_index = _kinetic_energy / _thermal_energy if part_type is '0' else 0.
			thermodynamic_merging_index = np.append(thermodynamic_merging_index, _thermodynamic_merging_index)

			del _mass
			del _N_particles
			del _velocity
			del _coords
			del _temperature
			del _subgroupnumber
			del _aperture_mass
			del _coords_norm
			del _centre_of_mass
			del _zero_momentum_frame
			del _velocity_norm
			del _angular_momentum
			del _angular_velocity
			del _specific_angular_momentum
			del _circular_velocity
			del _spin_parameter
			del sgn_index
			del _substructure_mass
			del _substructure_fraction
			del _thermal_energy
			del _kinetic_energy
			del _dynamical_merging_index
			del _thermodynamic_merging_index

		_N_particles = len(mass)
		N_particles = np.append(N_particles, _N_particles)

		_aperture_mass = np.sum(mass)
		aperture_mass = np.append(aperture_mass, _aperture_mass)

		coords_norm = np.subtract(coords, self.centre_of_potential)
		_centre_of_mass = self.centre_of_mass(mass, coords)
		centre_of_mass = np.concatenate((centre_of_mass, _centre_of_mass[None, :]), axis=0)

		_zero_momentum_frame = self.zero_momentum_frame(mass, velocity)
		zero_momentum_frame = np.concatenate((zero_momentum_frame, _zero_momentum_frame[None, :]), axis=0)

		velocity_norm = np.subtract(velocity, _zero_momentum_frame)
		_angular_momentum = self.angular_momentum(mass, coords_norm, velocity_norm)
		angular_momentum = np.concatenate((angular_momentum, _angular_momentum[None, :]), axis=0)

		_angular_velocity = np.linalg.inv(self.inertia_tensor(mass, coords_norm)) @ _angular_momentum
		angular_velocity = np.concatenate((angular_velocity, _angular_velocity[None, :]), axis=0)

		_specific_angular_momentum = np.linalg.norm(_angular_momentum) / _aperture_mass
		specific_angular_momentum = np.append(specific_angular_momentum, _specific_angular_momentum)

		_circular_velocity = np.sqrt(G_astro * _aperture_mass / aperture_radius)
		circular_velocity = np.append(circular_velocity, _circular_velocity)

		_spin_parameter = np.linalg.norm(_angular_momentum) / (_aperture_mass * aperture_radius * _circular_velocity * np.sqrt(2))
		spin_parameter = np.append(spin_parameter, _spin_parameter)

		_substructure_mass = np.sum(substructure_mass)
		substructure_mass = np.append(substructure_mass, _substructure_mass)

		_substructure_fraction = _substructure_mass / _aperture_mass
		substructure_fraction = np.append(substructure_fraction, _substructure_fraction)

		_thermal_energy = np.sum(thermal_energy)
		thermal_energy = np.append(thermal_energy, _thermal_energy)

		_kinetic_energy = self.kinetic_energy(self.mass_units(mass), self.velocity_units(velocity_norm)) * np.power(10., -46)
		kinetic_energy = np.append(kinetic_energy, _kinetic_energy)

		_dynamical_merging_index = np.linalg.norm(self.centre_of_potential - _centre_of_mass) / aperture_radius
		dynamical_merging_index = np.append(dynamical_merging_index, _dynamical_merging_index)

		_thermodynamic_merging_index = np.sum(thermodynamic_merging_index)
		thermodynamic_merging_index = np.append(thermodynamic_merging_index, _thermodynamic_merging_index)

		del _N_particles
		del mass
		del coords
		del velocity
		del temperature
		del _aperture_mass
		del coords_norm
		del _centre_of_mass
		del _zero_momentum_frame
		del velocity_norm
		del _angular_momentum
		del _angular_velocity
		del _specific_angular_momentum
		del _circular_velocity
		del _spin_parameter
		del _substructure_mass
		del _substructure_fraction
		del _thermal_energy
		del _kinetic_energy
		del _dynamical_merging_index
		del _thermodynamic_merging_index

		dynamic_dict = {
				'N_particles'                : N_particles[::-1],
				'aperture_mass'              : aperture_mass[::-1],
				'centre_of_mass'             : centre_of_mass[::-1],
				'zero_momentum_frame'        : zero_momentum_frame[::-1],
				'angular_momentum'           : angular_momentum[::-1],
				'angular_velocity'           : angular_velocity[::-1],
				'specific_angular_momentum'  : specific_angular_momentum[::-1],
				'circular_velocity'          : circular_velocity[::-1],
				'spin_parameter'             : spin_parameter[::-1],
				'substructure_mass'          : substructure_mass[::-1],
				'substructure_fraction'      : substructure_fraction[::-1],
				'thermal_energy'             : thermal_energy[::-1],
				'kinetic_energy'             : kinetic_energy[::-1],
				'dynamical_merging_index'    : dynamical_merging_index[::-1],
				'thermodynamic_merging_index': thermodynamic_merging_index[::-1]
		}
		return dynamic_dict

	def group_morphology(self, aperture_radius: float = None) -> Dict[str, np.ndarray]:
		"""
		Method that computes the cluster's morphology information from particles within a
		specified aperture.
		If the aperture is not specified, it is set by default to the true R500 of the cluster
		radius from the centre of potential and computes the morphology analysis using the relative
		static methods in this Mixin class.
		The method also checks that the necessary datasets are loaded into the cluster object.
		The information is gathered in dictionary-form output and contains the following analysis
		datasets, relative to the given aperture:

			- Inertia tensor (3x3 matrix): np.ndarray
			- Eigenvalues of the inertia tensor, per unit mass 3x np.float
				lambda_i / mass_within_aperture = (semi-axis length)**2
				They represent the semi-axes of the ellipsoid.
			- Eigenvectors of the inertia tensor 3x 1D array
				They are normalised and express the orientation of the semi-axes of the ellipsoid.
			- Triaxiality 1x np.float:
				triaxiality = (a**2-b**2)/(a**2-c**2)
			- Sphericity 1x np.float:
				sphericity = c/a
			- Elongation 1x np.float:
				elongation = c/a

		Each of these datasets is structured as follows:

			[
				Morphology_dataset_allParticleTypes,
				Morphology_dataset_ParticleType0,
				Morphology_dataset_ParticleType1,
				Morphology_dataset_ParticleType4,
			]

		:param out_allPartTypes: default = False
		:param aperture_radius: default = None (R500)
		:return: expected a numpy array of dimension 1 if all particletypes are combined, or
			dimension 2 if particle types are returned separately.
		"""
		if aperture_radius is None:
			aperture_radius = self.r500
			warnings.warn(f'Aperture radius set to default R_500,true. = {self.r500:.2f} Mpc.')

		mass = np.zeros(0, dtype=np.float)
		coords = np.zeros((0, 3), dtype=np.float)
		inertia_tensor = np.zeros((0, 9), dtype=np.float)
		eigenvalues = np.zeros((0, 3), dtype=np.float)
		eigenvectors = np.zeros((0, 9), dtype=np.float)
		triaxiality = np.zeros(0, dtype=np.float)
		sphericity = np.zeros(0, dtype=np.float)
		elongation = np.zeros(0, dtype=np.float)

		for part_type in ['4', '1', '0']:
			assert hasattr(self, f'partType{part_type}_coordinates')
			assert hasattr(self, f'partType{part_type}_mass')
			radial_dist = self.radial_distance_CoP(getattr(self, f'partType{part_type}_coordinates'))
			aperture_radius_index = np.where(radial_dist < aperture_radius)[0]
			del radial_dist
			_mass = getattr(self, f'partType{part_type}_mass')[aperture_radius_index]
			_coords = getattr(self, f'partType{part_type}_coordinates')[aperture_radius_index]
			if _mass.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
			if _coords.__len__() == 0: warnings.warn(f"Array PartType{part_type} is empty - check filtering.")
			_coords = np.subtract(_coords, self.centre_of_potential)
			mass = np.concatenate((mass, _mass), axis=0)
			coords = np.concatenate((coords, _coords), axis=0)

			_inertia_tensor = self.inertia_tensor(_mass, _coords)
			inertia_tensor = np.concatenate((inertia_tensor, _inertia_tensor.ravel()[None, :]), axis=0)

			_eigenvalues, _eigenvectors = self.principal_axes_ellipsoid(_inertia_tensor, eigenvalues=True)
			_eigenvalues /= np.sum(_mass)
			# Sort eigenvalues from largest to smallest
			_eigenvalues_sorted = np.sort(_eigenvalues)[::-1]
			_eigenvectors_sorted = np.zeros_like(_eigenvectors)
			for counter, val in np.ndenumerate(_eigenvalues_sorted):
				index = np.where(_eigenvalues == val)[0]
				_eigenvectors_sorted[counter] = _eigenvectors[index]

			eigenvalues = np.concatenate((eigenvalues, _eigenvalues_sorted[None, :]), axis=0)
			eigenvectors = np.concatenate((eigenvectors, _eigenvectors_sorted.ravel()[None, :]), axis=0)

			_triaxiality = (_eigenvalues_sorted[0] - _eigenvalues_sorted[1]) / (_eigenvalues_sorted[0] - _eigenvalues_sorted[2])
			triaxiality = np.append(triaxiality, _triaxiality)

			_sphericity = np.sqrt(_eigenvalues_sorted[2]) / np.sqrt(_eigenvalues_sorted[0])
			sphericity = np.append(sphericity, _sphericity)

			_elongation = np.sqrt(_eigenvalues_sorted[1]) / np.sqrt(_eigenvalues_sorted[0])
			elongation = np.append(elongation, _elongation)

			del _mass
			del _coords
			del _inertia_tensor
			del _eigenvalues
			del _eigenvalues_sorted
			del _eigenvectors_sorted
			del _triaxiality
			del _sphericity
			del _elongation

		_inertia_tensor = self.inertia_tensor(mass, coords)
		inertia_tensor = np.concatenate((inertia_tensor, _inertia_tensor.ravel()[None, :]), axis=0)

		_eigenvalues, _eigenvectors = self.principal_axes_ellipsoid(_inertia_tensor, eigenvalues=True)
		_eigenvalues /= np.sum(mass)
		# Sort eigenvalues from largest to smallest
		_eigenvalues_sorted = np.sort(_eigenvalues)[::-1]
		_eigenvectors_sorted = np.zeros_like(_eigenvectors)
		for counter, val in np.ndenumerate(_eigenvalues_sorted):
			index = np.where(_eigenvalues == val)[0]
			_eigenvectors_sorted[counter] = _eigenvectors[index]
		eigenvalues = np.concatenate((eigenvalues, _eigenvalues_sorted[None, :]), axis=0)
		eigenvectors = np.concatenate((eigenvectors, _eigenvectors_sorted.ravel()[None, :]), axis=0)

		_triaxiality = (_eigenvalues_sorted[0] - _eigenvalues_sorted[1]) / (_eigenvalues_sorted[0] - _eigenvalues_sorted[2])
		triaxiality = np.append(triaxiality, _triaxiality)

		_sphericity = np.sqrt(_eigenvalues_sorted[2]) / np.sqrt(_eigenvalues_sorted[0])
		sphericity = np.append(sphericity, _sphericity)

		_elongation = np.sqrt(_eigenvalues_sorted[1]) / np.sqrt(_eigenvalues_sorted[0])
		elongation = np.append(elongation, _elongation)

		del mass
		del coords
		del _inertia_tensor
		del _eigenvalues
		del _eigenvalues_sorted
		del _eigenvectors_sorted
		del _triaxiality
		del _sphericity
		del _elongation

		morphology_dict = {
				'inertia_tensor': inertia_tensor[::-1],
				'eigenvalues'   : eigenvalues[::-1],
				'eigenvectors'  : eigenvectors[::-1],
				'triaxiality'   : triaxiality[::-1],
				'sphericity'    : sphericity[::-1],
				'elongation'    : elongation[::-1],
		}
		return morphology_dict

