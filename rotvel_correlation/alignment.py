import sys
import os
import warnings
import numpy as np
from typing import Dict, List
from copy import deepcopy
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.cluster import Cluster
from save import dict2hdf as write
angle = deepcopy(Cluster.angle_between_vectors)

def group_alignment(groupreport: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
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
	x_axis, y_axis, z_axis = np.identity(3, dtype=np.int)
	peculiar_velocity = groupreport['zero_momentum_frame']
	angular_velocity = groupreport['angular_velocity']
	angular_momentum = groupreport['angular_momentum']
	eigenvectors = groupreport['eigenvectors'].reshape((4,3,3))
	a_vec, b_vec, c_vec = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]
	del eigenvectors

	v_l = np.zeros((4,4), dtype=np.float)
	v_w = np.zeros_like(v_l)
	l_w = np.zeros_like(v_l)
	a_v = np.zeros_like(v_l)
	a_l = np.zeros_like(v_l)
	a_w = np.zeros_like(v_l)
	b_v = np.zeros_like(v_l)
	b_l = np.zeros_like(v_l)
	b_w = np.zeros_like(v_l)
	c_v = np.zeros_like(v_l)
	c_l = np.zeros_like(v_l)
	c_w = np.zeros_like(v_l)
	a_b = np.zeros_like(v_l)
	b_c = np.zeros_like(v_l)
	c_a = np.zeros_like(v_l)

	for i,j in product(list(range(4)), repeat=2):
		v_l[i, j] = angle(peculiar_velocity[i], angular_momentum[j])
		v_w[i, j] = angle(peculiar_velocity[i], angular_velocity[j])
		l_w[i, j] = angle(angular_momentum[i], angular_velocity[j])
		a_v[i, j] = angle(a_vec[i], peculiar_velocity[j])
		a_l[i, j] = angle(a_vec[i], angular_momentum[j])
		a_w[i, j] = angle(a_vec[i], angular_velocity[j])
		b_v[i, j] = angle(b_vec[i], peculiar_velocity[j])
		b_l[i, j] = angle(b_vec[i], angular_momentum[j])
		b_w[i, j] = angle(b_vec[i], angular_velocity[j])
		c_v[i, j] = angle(c_vec[i], peculiar_velocity[j])
		c_l[i, j] = angle(c_vec[i], angular_momentum[j])
		c_w[i, j] = angle(c_vec[i], angular_velocity[j])
		a_b[i, j] = angle(a_vec[i], b_vec[j])
		b_c[i, j] = angle(b_vec[i], c_vec[j])
		c_a[i, j] = angle(c_vec[i], a_vec[j])

	# Check eigenvector orthogonality
	for i in range(4):
		a_b[i, i] -= 90.
		b_c[i, i] -= 90.
		c_a[i, i] -= 90.
	if (
			int(np.trace(a_b)) is not 0 or
			int(np.trace(b_c)) is not 0 or
			int(np.trace(c_a))
	):
		warnings.warn('Detected non-orthogonal semiaxes. Check inertia tensor.')
		print(a_b, b_c, c_a, sep='\n')

	angle_dict = {
			'v_l' : v_l,
			'v_w' : v_w,
			'l_w' : l_w,
			'a_v' : a_v,
			'a_l' : a_l,
			'a_w' : a_w,
			'b_v' : b_v,
			'b_l' : b_l,
			'b_w' : b_w,
			'c_v' : c_v,
			'c_l' : c_l,
			'c_w' : c_w
	}
	return angle_dict


def save_report(clusterID: int, redshift: str, glob: List[np.ndarray] = None) -> None:
	data_required = {
			'partType0': ['groupnumber', 'subgroupnumber', 'mass', 'coordinates', 'velocity', 'temperature', 'sphdensity'],
			'partType1': ['groupnumber', 'subgroupnumber', 'mass', 'coordinates', 'velocity'],
			'partType4': ['groupnumber', 'subgroupnumber', 'mass', 'coordinates', 'velocity']
	}
	cluster = Cluster(simulation_name='bahamas',
	                  clusterID=clusterID,
	                  redshift=redshift,
	                  requires=data_required)

	setattr(cluster, 'pgn0', glob[0])
	setattr(cluster, 'pgn1', glob[1])
	setattr(cluster, 'pgn4', glob[2])
	cluster.import_requires()
	apertures = cluster.generate_apertures()
	master_dict = {}
	for i, r_a in enumerate(apertures):
		halo_output = {
				**cluster.group_fofinfo(aperture_radius=cluster.r200),
				**cluster.group_dynamics(aperture_radius=cluster.r200),
				**cluster.group_morphology(aperture_radius=cluster.r200)
		}
		alignment_dict = group_alignment(halo_output)
		halo_output = {
				**halo_output,
				**alignment_dict
		}
		master_dict[f'aperture{i:02d}'] = halo_output
		del halo_output, alignment_dict

	if not os.path.exists(os.path.join(cluster.pathSave, 'alignment_project')):
		os.makedirs(os.path.join(cluster.pathSave, 'alignment_project'))
	pathFile = os.path.join(cluster.pathSave, 'alignment_project', f"{cluster.redshift}")
	if not os.path.exists(pathFile):
		os.makedirs(pathFile)
	write.save_dict_to_hdf5(master_dict, os.path.join(pathFile, f"halo_{cluster.clusterID}"))
	del cluster