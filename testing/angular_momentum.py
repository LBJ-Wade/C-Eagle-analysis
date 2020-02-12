"""
------------------------------------------------------------------
FILE:   angular momentum.py
AUTHOR: Edo Altamura
DATE:   19-11-2019
------------------------------------------------------------------
This file is part of the 'testing' package.
-------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Cluster



def angle_between_vectors(v1, v2):
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # Return the result in degrees
    return angle*180/np.pi


def angular_momentum_PartType_alignment_matrix(cluster, specific_angular_momentum = False, aperture_radius = None):
    """
    angular_momentum_PartType_alignment_matrix() takes the angular momenta of each particle type
    in the cluster and computes the respective angles between each of them, in the form of a matrix.

    :param cluster: (cluster.Cluster)
    :param specific_angular_momentum: (bool)
        Rescales the angular momentum vector with each particle type's total mass, yielding
        the specific angular momentum for each component
    :return: (numpy.ndarray)
        Returns the alignment 2D matrix with entries corresponding to the angles in degrees
        between particle types.
    """
    # Compute the angular momentum of all high res particle types
    ang_momenta, sum_of_masses = cluster.group_angular_momentum(out_allPartTypes=True, aperture_radius = aperture_radius)

    # The specific angular momentum is given by j = J/M
    if specific_angular_momentum:
        ang_momenta = np.divide(ang_momenta, sum_of_masses.reshape((len(ang_momenta), 1)))

    # Compute the alignment matrix
    alignment_matrix = np.zeros((len(ang_momenta), len(ang_momenta)), dtype=float)
    for i in range(len(ang_momenta)):
        for j in range(len(ang_momenta)):
            if i == j:
                alignment_matrix[i][j] = 0.
            else:
                alignment_matrix[i][j] = angle_between_vectors(ang_momenta[i], ang_momenta[j])
    return alignment_matrix


def matrix_to_dataframe(matrix):
    df = pd.DataFrame(matrix)
    df.columns = ["Gas", "Dark matter", "Stars", "Black holes"]
    df.index = ["Gas", "Dark matter", "Stars", "Black holes"]
    print(df)

def alignment_DM_to_gas(matrix):    return matrix[0][1]
def alignment_DM_to_stars(matrix):  return matrix[1][2]
def alignment_stars_to_gas(matrix): return matrix[0][2]


def derotate(cluster, align : str = 'gas', aperture_radius = None, cluster_rest_frame = False):
    """ This method derotates the positions and velocity vectors of
    particles from the Gadget-centric coordinate basis to a custom basis.

    :parameter rotation_matrix : array_like, shape (3, 3)
        The rotation matrix acts on the coordinates and velocty of
        particles and rotates them to the newly defined coordinate basis.

    :return particle_coordinates, particle_velocities: tuple
        Returns the coordinates and velocities of the particle in the
        cluster in the new coordinate basis.
    """
    if aperture_radius is None:
        aperture_radius = cluster.group_r500()
        aperture_radius = cluster.comoving_length(aperture_radius)
        print('[ DEROTATE ]\t==>\tAperture radius set to default R500 true.')

    coords = cluster.particle_coordinates(align)
    coords = np.subtract(coords, cluster.group_centre_of_mass(out_allPartTypes=False, aperture_radius = aperture_radius)[0])
    vel = cluster.particle_velocity(align)

    if cluster_rest_frame:
        vel = np.subtract(vel, cluster.group_zero_momentum_frame(out_allPartTypes=False, aperture_radius = aperture_radius)[0])

    coords = cluster.comoving_length(coords)
    vel = cluster.comoving_velocity(vel)

    if align == 'off':
        return coords, vel

    ang_momenta, _ = cluster.group_angular_momentum(out_allPartTypes=True, aperture_radius=aperture_radius)

    if align == 'gas' or align == '0':
        ang_momentum = ang_momenta[0]
    elif align == 'dark_matter' or align == '1':
        ang_momentum = ang_momenta[1]
    elif align == 'stars' or align == '4':
        ang_momentum = ang_momenta[4]
    elif align == 'black_holes' or align == '5':
        ang_momentum = ang_momenta[5]

    z_axis_unit_vector = [0,0,1]
    rot_matrix = cluster.rotation_matrix_from_vectors(ang_momentum, z_axis_unit_vector)
    coords = cluster.apply_rotation_matrix(rot_matrix, coords)
    vel = cluster.apply_rotation_matrix(rot_matrix, vel)

    return coords, vel





if __name__ == "__main__":

    from rendering import plot_angularmomentum_vectors

    cluster = Cluster(clusterID = 0, redshift = 0.)
    angmom, masses = cluster.group_angular_momentum(out_allPartTypes=True)
    m = angular_momentum_PartType_alignment_matrix(cluster)
    print(m)
    plot_angularmomentum_vectors(angmom,
                                     axes = None,
                                     plot_unitSphere = False,
                                     normalise_length = False,
                                     make_all_unitary = False,
                                     )
    plt.show()