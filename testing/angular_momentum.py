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





if __name__ == "__main__":

    from rendering import plot_angularmomentum_vectors

    cluster = Cluster(clusterID = 15, redshift = 0.101)
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