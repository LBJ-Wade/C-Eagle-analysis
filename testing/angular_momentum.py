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
from matplotlib import pyplot as plt

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Cluster
from rendering import Map
import map_plot_parameters as plotpar
plotpar.set_defaults_plot()


def angle_between_vectors(v1, v2):
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # Return the result in degrees
    return angle*180/np.pi


def angular_momentum_PartType_alignment_matrix(cluster):


    # Compute the angular momentum of all high res particle types
    ang_momenta, sum_of_masses = cluster.group_angular_momentum(out_allPartTypes=True)

    # Compute the alignment matrix
    alignment_matrix = np.zeros((len(ang_momenta), len(ang_momenta)), dtype=float)
    for i in range(len(ang_momenta)):
        for j in range(len(ang_momenta)):
            if i == j:
                alignment_matrix[i][j] = 0.
            else:
                alignment_matrix[i][j] = angle_between_vectors(ang_momenta[i], ang_momenta[j])

    return alignment_matrix

cluster = Cluster(clusterID=4, redshift=0.101)
m = angular_momentum_PartType_alignment_matrix(cluster)
print(m)