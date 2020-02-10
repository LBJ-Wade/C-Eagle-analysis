"""
------------------------------------------------------------------
FILE:   rendering.py
AUTHOR: Edo Altamura
DATE:   18-11-2019
------------------------------------------------------------------
This file contains methods and classes for rendering data:
    - maps
    - plots
    - diagrams
-------------------------------------------------------------------
"""

from matplotlib import pyplot as plt
import numpy as np
from itertools import cycle

from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import map_plot_parameters as plotpar
plotpar.set_defaults_plot()


import matplotlib as mpl
import scipy as sp


#################################
#                               #
#	          M A P             #
# 			C L A S S           #
#							    #
#################################

class Colorscheme:
    def __init__(self):
        pass

    def balanced(self):   return ['#E27D60', '#85DCB',  '#E8A87C', '#C38D9E', '#41B3A3']
    def natural(self):    return ['#8D8741', '#659DBD', '#DAAD86', '#BC986A', '#FBEEC1']
    def futuristic(self): return ['#2C3531', '#116466', '#D9B08C', '#FFCB9A', '#D1E8E2']


class Map():
    def __init__(self, cluster = None):
        self.cluster = cluster

    @staticmethod
    def get_centers_from_bins(bins):
        """ return centers from bin sequence """
        return (bins[:-1] + bins[1:]) / 2

    @staticmethod
    def bins_meshify(x, y, x_bins, y_bins):
        """

        """
        _, xbins, ybins = np.histogram2d(x, y, bins=(x_bins, y_bins), weights=None)
        lt = Map.get_centers_from_bins(xbins)
        lm = Map.get_centers_from_bins(ybins)
        cX_v, cY_v = np.meshgrid(lt, lm)
        return cX_v, cY_v

    @staticmethod
    def bins_evaluate(x, y, x_bins, y_bins, weights=None):
        """

        """
        H, _, _ = np.histogram2d(x, y, bins=(x_bins, y_bins), weights=weights)
        return H.T

    @staticmethod
    def plot_circle(axes, x, y, r, **kwargs):
        """
        This static method is used to overplot circles on plots.
        Used e.g. for marking r200 or r500.

        :param axes: The mpl.axes class object the circle will be associated with.
        :param x: The x-coord of the centre of the circle.
        :param y: The y-coord of the centre of the circle.
        :param r: The radius of the circle.
        :param kwargs: Rendering options, including label and colour etc.
        :return: None.
        """
        axes.add_artist(Circle((x, y), radius = r, **kwargs))
        txt = r'$R_{500}$'
        # Add text around it
        axes.annotate(txt, (x, y + 1.1 * r), size=15)




    def xyz_projections(self, *args,
                            xyzdata = None,
                            weights = None,
                            plot_limit = None,
                            nbins = None,
                            circle_pars = None,
                            special_markers = None,
                            special_markers_labels = None,
                            **kwargs):
        """

        :param args:
        :param xyzdata: (numpy 2D array)
        :param weights:
        :param kwargs:
        :return:
        """
        data_are_parsed = xyzdata is not None or \
                          weights is not None or \
                          nbins is not None


        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

        cmap =      [plt.get_cmap('Greens_r'), plt.get_cmap('Greens_r'), plt.get_cmap('Greens_r')]
        xlabel =    [r'$x\mathrm{/Mpc}$', r'$y\mathrm{/Mpc}$', r'$x\mathrm{/Mpc}$']
        ylabel =    [r'$y\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$']
        thirdAX =   [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
        cbarlabel = [r'$\sum_{i} m_i\ [\mathrm{M_\odot}]$',
                     r'$\sum_{i} m_i\ [\mathrm{M_\odot}]$',
                     r'$\sum_{i} m_i\ [\mathrm{M_\odot}]$']

        for i in [0, 1, 2]:
            # Handle data
            if i == 0:
                if data_are_parsed:
                    x_Data = xyzdata[:, 0]
                    y_Data = xyzdata[:, 1]

                x_specialMarkers = special_markers[:, 0]
                y_specialMarkers = special_markers[:, 1]
            elif i == 1:
                if data_are_parsed:
                    x_Data = xyzdata[:, 1]
                    y_Data = xyzdata[:, 2]

                x_specialMarkers = special_markers[:, 1]
                y_specialMarkers = special_markers[:, 2]
            elif i == 2:
                if data_are_parsed:
                    x_Data = xyzdata[:, 0]
                    y_Data = xyzdata[:, 2]

                x_specialMarkers = special_markers[:, 0]
                y_specialMarkers = special_markers[:, 2]

            if data_are_parsed:
                x_bins = np.linspace(-plot_limit, plot_limit, nbins)
                y_bins = np.linspace(-plot_limit, plot_limit, nbins)
                Cx, Cy = Map.bins_meshify(x_Data, y_Data, x_bins, y_bins)
                count = Map.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights=weights)

                norm = colors.LogNorm(vmin=10**8, vmax=np.max(count))
                img = axes[i].pcolor(Cx, Cy, count, cmap=cmap[i], norm= norm)

            # Render elements in plots
            axes[i].set_aspect('equal')

            # Plot circles
            Map.plot_circle(axes[i], *circle_pars, color='black', fill=False, linestyle='--', label=r'$R_{200}$')

            # Plot the special markers
            for x, y, txt in zip(x_specialMarkers, y_specialMarkers, special_markers_labels):
                axes[i].scatter(x, y, color='red', linestyle='--')
                axes[i].annotate(txt, (x, y), size = 15)

            axes[i].set_xlim(-plot_limit, plot_limit)
            axes[i].set_ylim(-plot_limit, plot_limit)
            axes[i].set_xlabel(xlabel[i])
            axes[i].set_ylabel(ylabel[i])
            axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size=15)

            if data_are_parsed:
                ax2_divider = make_axes_locatable(axes[i])
                cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
                cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
                cbar.set_label(cbarlabel[i], labelpad=-70)
                # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
                cax2.xaxis.set_ticks_position("top")
            print("[MAP PANEL]\t==> completed:", i)



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
                                 labels = None,
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
        vectors = np.asarray(vectors)

    colors = Colorscheme().natural()

    if axes is None:
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        axes.set_aspect("equal")

        # Edit the pane layout
        axes.grid(False)

        axes.xaxis.pane.fill = colors[-1]
        axes.yaxis.pane.fill = colors[-1]
        axes.zaxis.pane.fill = colors[-1]

        axes.xaxis.pane.set_edgecolor(colors[-1])
        axes.yaxis.pane.set_edgecolor(colors[-1])
        axes.zaxis.pane.set_edgecolor(colors[-1])

        axes.set_xlabel(r'$x$')
        axes.set_ylabel(r'$y$')
        axes.set_zlabel(r'$z$')

        # draw a point at origin
        axes.scatter([0], [0], [0], color="k", s=80)

    if plot_unitSphere:
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        axes.plot_wireframe(x, y, z, color='#AFD275', alpha = 0.2)

        # Draw line of sight observer
        LineOfSight_color = '#EB3F11'
        LineOfSight = Arrow3D([0, 0], [-2, -1], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color=LineOfSight_color)
        axes.scatter([], [], c=LineOfSight_color, marker=r"$\longrightarrow$", s=70, label=r'Line of sight')
        axes.add_artist(LineOfSight)
        print('[ PLOT 3D VECTOR ]\t==>\tDrawing observer_LineOfSight.')

        # Draw reference rotation vector
        # Reference_Ang_Momentum_color = '#E59813'
        # Reference_Ang_Momentum = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=3, arrowstyle="-|>", color=Reference_Ang_Momentum_color)
        # axes.scatter([], [], c=Reference_Ang_Momentum_color, marker=r"$\longrightarrow$", s=70, label=r'Reference angular momentum')
        # axes.add_artist(Reference_Ang_Momentum)
        # print('[ PLOT 3D VECTOR ]\t==>\tDrawing Reference_Ang_Momentum.')



    # Manipulate vectors
    if vectors.ndim == 1:
        # If there's only one vector, then the np.array will be 1D
        print('[ PLOT 3D VECTOR ]\t==>\tOnly one vector detected.')
        if normalise_length or make_all_unitary:
            vectors = np.divide(vectors, np.linalg.norm(vectors))

        a = Arrow3D([0, vectors[0]], [0, vectors[1]], [0, vectors[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        axes.add_artist(a)
        axes.set_xlim([-np.max(vectors), np.max(vectors)])
        axes.set_ylim([-np.max(vectors), np.max(vectors)])
        axes.set_zlim([-np.max(vectors), np.max(vectors)])

    else:
        # If there's more than 1 vector, then the np.array will be 2D
        vectors_magnitudes = np.linalg.norm(vectors, axis = 1)

        if normalise_length:
            vectors = np.divide(vectors, np.max(vectors_magnitudes))

        if labels is not None:
            assert labels.__len__() == vectors_magnitudes.__len__()
        else:
            labels = ['']*vectors_magnitudes.__len__()

        for vector, magnitude, label, color in zip(vectors, vectors_magnitudes, labels, colors):

            if make_all_unitary:
                vector = np.divide(vector, magnitude)

            a = Arrow3D([0, vector[0]], [0, vector[1]], [0, vector[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color = color)
            axes.scatter([], [], c=color, marker=r"$\longrightarrow$", s = 70, label = label )
            axes.add_artist(a)
            print('[ PLOT 3D VECTOR ]\t==>\tDrawing vector {}'.format(labels.index(label)))

        if make_all_unitary:
            axes.set_xlim([-1.5, 1.5])
            axes.set_ylim([-1.5, 1.5])
            axes.set_zlim([-1.5, 1.5])
        else:
            axes.set_xlim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])
            axes.set_ylim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])
            axes.set_zlim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])

    axes.legend(loc="best", markerscale=3)





"""
legend_labels = [r'$\mathrm{Gas}$',
                         r'$\mathrm{Highres DM}$',
                         r'$\mathrm{Stars}$',
                         r'$\mathrm{Black holes}$']
"""


def derotate_field():
    import cluster

    cluster = cluster.Cluster(clusterID=0, redshift=0.)

    vector_reference = [0, 0, 1]
    vector_to_rotate = [[0, 1, 1], [1, 1, 1], [0, 3, 1]]
    # rotation = cluster.rotation_matrix_from_vectors(vector_reference, vector_to_rotate); print(rotation)
    rotation = cluster.rotation_matrix_from_vectors(vector_to_rotate[0], vector_reference); print(rotation)

    rotated_vector = cluster.apply_rotation_matrix(rotation, vector_to_rotate); print(rotated_vector)
    print(np.vstack((vector_reference, vector_to_rotate, rotated_vector)))

    legend_labels = [r'vector_reference',
                     r'vector_to_rotate_1',
                     r'vector_to_rotate_2',
                     r'vector_to_rotate_3',
                     r'rotated_vector_1',
                     r'rotated_vector_2',
                     r'rotated_vector_3']

    plot_angularmomentum_vectors(np.vstack((vector_reference, vector_to_rotate, rotated_vector)),
                                 labels = legend_labels,
                                 axes=None,
                                 plot_unitSphere=True,
                                 normalise_length=False,
                                 make_all_unitary=False,
                                 )



if __name__ == "__main__":
    derotate_field()
    plt.show()

