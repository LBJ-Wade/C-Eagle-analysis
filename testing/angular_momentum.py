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
from itertools import cycle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


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


###############################################################
#           PLOT METHODS AND CLASSES                          #
###############################################################

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)



def plot_angularmomentum_vectors(vectors,
                                 axes = None,
                                 plot_unitSphere = False,
                                 normalise_length = True,
                                 make_all_unitary = False
                                 ):
    """
    Function that uses the Arrow3D class to plot vectors in 3D space.

    :param vectors: (np.ndarray)
                    1D np.array for single vector or 2D np.array for more than 1 vector
    :param axes: (matplotlib.pyplot.Axes.axes)
                    environment in which to plot the vectors
                    If None one will be automatically created
    :param plot_unitSphere: (bool)
                    Default = False. Plots a wire-framed unitary sphere.
    :param normalise_length: (bool)
                    Default = True. Normalises the vectors to that with the largest magnitude.
    :param make_all_unitary: (bool)
                    Default = False. Normalises each vector by its magnitude, making them all unitary.
    :return: No returns
    """
    if type(vectors) is not np.ndarray:
        vectors = np.array(vectors)

    if axes is None:
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        axes.set_aspect("equal")

    if plot_unitSphere:
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        axes.plot_wireframe(x, y, z, color="lime", alpha = 0.2)

    # draw a point at origin
    axes.scatter([0], [0], [0], color="k", s=100)

    # Manipulate vectors
    if vectors.ndim == 1:
        # If there's only one vector, then the np.array will be 1D
        print('[ PLOT 3D VECTOR ]\t==>\tOnly one vector detected.')
        if normalise_length or make_all_unitary:
            vectors = np.divide(vectors, np.linalg.norm(vectors))

        a = Arrow3D([0, vectors[0]], [0, vectors[1]], [0, vectors[2]], mutation_scale=20,
                    lw=1, arrowstyle="-|>", color="k")
        axes.add_artist(a)
        axes.set_xlim([-np.max(vectors), np.max(vectors)])
        axes.set_ylim([-np.max(vectors), np.max(vectors)])
        axes.set_zlim([-np.max(vectors), np.max(vectors)])

    else:
        # If there's more than 1 vector, then the np.array will be 2D
        vectors_magnitudes = np.linalg.norm(vectors, axis = 1)
        if normalise_length:
            # Normalise all vectors to the largest in magnitude
            vectors = np.divide(vectors, np.max(vectors_magnitudes))

        # Automate colors and labels
        cycol = cycle('bgrcmk')
        legend_labels = [r'$\mathrm{Gas}$',
                         r'$\mathrm{Highres DM}$',
                         r'$\mathrm{Stars}$',
                         r'$\mathrm{Black holes}$']

        for vector, magnitude, label in zip(vectors, vectors_magnitudes, legend_labels):
            if make_all_unitary:
                vector = np.divide(vector, magnitude)

            arrow_color = next(cycol)
            a = Arrow3D([0, vector[0]], [0, vector[1]], [0, vector[2]], mutation_scale=20,
                        lw=1, arrowstyle="-|>", color = arrow_color)
            axes.scatter([], [], c=arrow_color, marker=r"$\longrightarrow$", s = 70, label = label )
            axes.add_artist(a)
            print('[ PLOT 3D VECTOR ]\t==>\tDrawing vector {}'.format(legend_labels.index(label)))

        if make_all_unitary:
            axes.set_xlim([-1.5, 1.5])
            axes.set_ylim([-1.5, 1.5])
            axes.set_zlim([-1.5, 1.5])
        else:
            axes.set_xlim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])
            axes.set_ylim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])
            axes.set_zlim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])

        axes.legend(loc="best", markerscale=3)

    axes.set_xlabel(r'$x$')
    axes.set_ylabel(r'$y$')
    axes.set_zlabel(r'$z$')


if __name__ == "__main__":
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