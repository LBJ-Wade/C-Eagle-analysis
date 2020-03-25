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
import os
import sys
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.cluster import Cluster
from visualisation.rendering import Colorscheme

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class LosGeometry(Axes):

    # Inherit static methods from cluster.Cluster
    rotation_matrix_about_axis = staticmethod(Cluster.rotation_matrix_about_axis)
    apply_rotation_matrix = staticmethod(Cluster.apply_rotation_matrix)

    rotation_matrix_about_x = Cluster.rotation_matrix_about_x
    rotation_matrix_about_y = Cluster.rotation_matrix_about_y
    rotation_matrix_about_z = Cluster.rotation_matrix_about_z

    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.inset_axes = None
        self.los_vector = [[0, -2, 0], [0, -1, 0]]
        self.los_label = [0, -2.2, -0.2]
        self.observer_rotation_matrix = None

    # Reading methods
    def get_figure(self):
        return self.figure

    def get_inset_axes(self):
        return self.inset_axes

    def get_los_vector(self):
        return self.los_vector

    def get_los_label(self):
        return self.los_label

    def get_observer_rotation_matrix(self):
        return self.observer_rotation_matrix

    # Writing methods
    def set_figure(self, new_figure: Figure) -> None:
        """
        Set a new `figure` attribute to the class.

        :param new_axes:  expect a matplotlib.figure.Figure object
            The new matplotlib.figure.Figure environment to build the diagram in.

        :return: None
        """
        self.figure = new_figure

    def set_inset_axes(self, new_inset_axes: Axes) -> None:
        """
        Set a new `axes` attribute to the class.

        :param new_inset_axes:  expect a matplotlib.axes.Axes object
            The new *INSET* matplotlib.axes.Axes environment to build the 3d diagram in.

        :return: None
        """
        self.inset_axes = new_inset_axes

    def set_inset_geometry(self, left, bottom, width, height) -> None:
        """
        Generates an `inset_axes` within the `axes` environment, according to the geometry
        specified by the four positional arguments:

        :param left: expect float between (0,1).
            Specifies the left-hand side boundary of the `inset_axes`, as a fraction of the
            `axes` dimensions.

        :param bottom: expect float between (0,1).
            Specifies the bottom boundary of the `inset_axes`, as a fraction of the
            `axes` dimensions.

        :param width: expect float between (0,1).
            Specifies the width of the `inset_axes`, as a fraction of the `axes` dimensions.

        :param height: expect float between (0,1).
            Specifies the height of the `inset_axes`, as a fraction of the `axes` dimensions.

        :return: None
        """
        inset_axis = self.figure.add_axes([left, bottom, width, height], projection='3d')
        if self.inset_axes is None:
            self.set_inset_axes(inset_axis)
            self.inset_axes.patch.set_alpha(0) # Transparent background


    def set_observer(self, rot_x: float = None, rot_y: float = None, rot_z: float = None) -> None:
        """
        Derotates the observer's viewpoint around the 3 axes of the cluster's frame. Note that the
        whole cluster might already have been derotated and aligned to the particle's angular momentum
        vector.
        This function sets the observer's orientation to the new position and sets also a new attribute
        to the LosGeometry class with the observer's rotation matrix, useful ofr computing the
        scalar kSZ map along a particular line-of-sight.

        :param rot_x: expected float within (0, 360)
            The angle in degrees by which the observer's viewpoint is derotated about the x axis.

        :param rot_y: expected float within (0, 360)
            The angle in degrees by which the observer's viewpoint is derotated about the y axis.

        :param rot_z: expected float within (0, 360)
            The angle in degrees by which the observer's viewpoint is derotated about the y axis.

        :return: None
        """

        # Start from default always
        self.los_vector = [[0, -2, 0], [0, -1, 0]]
        self.los_label = [0, -2.2, -0.2]

        rot_x = 0. if rot_x is None else rot_x
        rot_y = 0. if rot_y is None else rot_y
        rot_z = 0. if rot_z is None else rot_z

        rotation_matrix_x = self.rotation_matrix_about_x(rot_x * np.pi / 180)
        rotation_matrix_y = self.rotation_matrix_about_y(rot_y * np.pi / 180)
        rotation_matrix_z = self.rotation_matrix_about_z(rot_z * np.pi / 180)

        combined_matrix = np.asmatrix(rotation_matrix_x).dot(np.asmatrix(rotation_matrix_y))
        combined_matrix = np.asmatrix(combined_matrix).dot(np.asmatrix(rotation_matrix_z))
        self.observer_rotation_matrix = combined_matrix

        new_los_vector = self.apply_rotation_matrix(combined_matrix, self.los_vector)
        self.los_vector = new_los_vector
        new_los_label = self.apply_rotation_matrix(combined_matrix, self.los_label)
        self.los_label = new_los_label


    def draw_observer(self):

        los_vector_reshaped = np.asarray(self.los_vector).T.reshape((3,2)).tolist()
        LineOfSight_color = '#EB3F11'
        LineOfSight = Arrow3D(los_vector_reshaped[0], los_vector_reshaped[1], los_vector_reshaped[2],
                              mutation_scale=20, lw=3, arrowstyle="-|>", color=LineOfSight_color)
        self.inset_axes.scatter([], [], c=LineOfSight_color, marker=r"$\mathbf{\longrightarrow}$", s=70,
                                label=r'$\mathrm{Line~of~sight}$')
        self.inset_axes.text(self.los_label[0], self.los_label[1], self.los_label[2], r'$\mathcal{O}$', color = LineOfSight_color)
        self.inset_axes.add_artist(LineOfSight)
        print('[ PLOT 3D VECTOR ]\t==>\tDrawing observer_LineOfSight.')


    def get_legend(self, axes):
        h, l = self.inset_axes.get_legend_handles_labels()
        return h, l

    def draw_legend(self, axes):
        h, l = self.get_legend(axes)
        axes.legend(h, l, loc="upper right", markerscale=3, fancybox=True, framealpha=0.6)


    def plot_angularmomentum_vectors(self,
                                     vectors,
                                     labels = None,
                                     plot_unitSphere = False,
                                     normalise_length = True,
                                     make_all_unitary = False):
        """
        Function that uses the Arrow3D class to plot vectors in 3D space.

        :param vectors: (np.ndarray)
                        1D np.array for single vector or 2D np.array for more than 1 vector

        :param plot_unitSphere: (bool)
                        Default = False. Plots a wire-framed unitary sphere.

        :param normalise_length: (bool)
                        Default = True. Normalises the vectors to that with the largest magnitude.

        :param make_all_unitary: (bool)
                        Default = False. Normalises each vector by its magnitude, making them all unitary.

        :return: No returns
        """
        plt.rcParams.update({'font.size': 17})
        if type(vectors) is not np.ndarray:
            vectors = np.asarray(vectors)

        colors = Colorscheme().natural()


        self.inset_axes.set_aspect("equal")

        # Edit the pane layout
        self.inset_axes.grid(False)

        self.inset_axes.xaxis.pane.fill = colors[-1]
        self.inset_axes.yaxis.pane.fill = colors[-1]
        self.inset_axes.zaxis.pane.fill = colors[-1]

        self.inset_axes.xaxis.pane.set_edgecolor(colors[-1])
        self.inset_axes.yaxis.pane.set_edgecolor(colors[-1])
        self.inset_axes.zaxis.pane.set_edgecolor(colors[-1])

        self.inset_axes.set_xlabel(r'$x$', labelpad=-15)
        self.inset_axes.set_ylabel(r'$y$', labelpad=-15)
        self.inset_axes.set_zlabel(r'$z$', labelpad=-15)

        # draw a point at origin
        self.inset_axes.scatter([0], [0], [0], color="k", s=80)

        if plot_unitSphere:
            # draw sphere
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            self.inset_axes.plot_wireframe(x, y, z, color='#AFD275', alpha = 0.2)

            # Draw line of sight observer
            self.draw_observer()

            # Draw reference rotation vector
            Reference_Ang_Momentum_color = '#E59813'
            Reference_Ang_Momentum = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=3, arrowstyle="-|>", color=Reference_Ang_Momentum_color)
            self.inset_axes.scatter([], [], c=Reference_Ang_Momentum_color, marker=r"$\mathbf{\longrightarrow}$", s=70,
                                    label=r'$\mathrm{Reference~} \mathbf{L}$')
            self.inset_axes.add_artist(Reference_Ang_Momentum)
            print('[ PLOT 3D VECTOR ]\t==>\tDrawing Reference_Ang_Momentum.')



        # Manipulate vectors
        if vectors.ndim == 1:
            # If there's only one vector, then the np.array will be 1D
            print('[ PLOT 3D VECTOR ]\t==>\tOnly one vector detected.')
            if normalise_length or make_all_unitary:
                vectors = np.divide(vectors, np.linalg.norm(vectors))

            a = Arrow3D([0, vectors[0]], [0, vectors[1]], [0, vectors[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
            self.inset_axes.add_artist(a)
            self.inset_axes.set_xlim([-np.max(vectors), np.max(vectors)])
            self.inset_axes.set_ylim([-np.max(vectors), np.max(vectors)])
            self.inset_axes.set_zlim([-np.max(vectors), np.max(vectors)])

        else:
            # If there's more than 1 vector, then the np.array will be 2D
            vectors_magnitudes = np.linalg.norm(vectors, axis = 1)

            if normalise_length:
                vectors = np.divide(vectors, np.max(vectors_magnitudes))

            if labels is None:
                labels = [''] * vectors_magnitudes.__len__()

            assert labels.__len__() == vectors_magnitudes.__len__()

            for vector, magnitude, label, color in zip(vectors, vectors_magnitudes, labels, colors):

                if make_all_unitary:
                    vector = np.divide(vector, magnitude)

                a = Arrow3D([0, vector[0]], [0, vector[1]], [0, vector[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color = color)
                self.inset_axes.scatter([], [], c=color, marker=r"$\longrightarrow$", s = 70, label = label )
                self.inset_axes.add_artist(a)
                print('[ PLOT 3D VECTOR ]\t==>\tDrawing vector {}'.format(labels.index(label)))

            if make_all_unitary:
                self.inset_axes.set_xlim([-1.5, 1.5])
                self.inset_axes.set_ylim([-1.5, 1.5])
                self.inset_axes.set_zlim([-1.5, 1.5])
                plt.setp(self.inset_axes.get_xticklabels(), visible=False)
                plt.setp(self.inset_axes.get_yticklabels(), visible=False)
                plt.setp(self.inset_axes.get_zticklabels(), visible=False)
                self.inset_axes.xaxis.set_major_locator(plt.NullLocator())
                self.inset_axes.yaxis.set_major_locator(plt.NullLocator())
                self.inset_axes.zaxis.set_major_locator(plt.NullLocator())

            else:
                self.inset_axes.set_xlim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])
                self.inset_axes.set_ylim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])
                self.inset_axes.set_zlim([-np.max(vectors_magnitudes), np.max(vectors_magnitudes)])

        # self.inset_axes.legend(loc="upper right", markerscale=3, fancybox=True, framealpha=0.6)



class TestSuite(LosGeometry):

    def __init__(self):
        print('Using TestSuite mode.')

    def _TEST_basic_LoS(self):

        fig = plt.figure(figsize=(15, 15))
        axes = fig.add_axes()

        diagram = LosGeometry(fig, axes)
        diagram.set_inset_geometry(0.6, 0.0, 0.4, 0.4)
        diagram.set_observer(rot_x=0, rot_y=0, rot_z=90)
        vectors = [
            [0,1,1],
            [2,5,6],
            [-3,-2,0]
        ]
        diagram.plot_angularmomentum_vectors(vectors,
                                             labels = None,
                                             plot_unitSphere = True,
                                             normalise_length = False,
                                             make_all_unitary = True)




    def _TEST_derotate_field(self):
        from import_toolkit.cluster import Cluster
        from testing import angular_momentum

        cluster = Cluster(simulation_name='celr_e', clusterID = 0, redshift = 'z000p000')
        mass = cluster.particle_masses('gas')

        coords, vel = angular_momentum.derotate(cluster, align='gas', cluster_rest_frame=True, derotate_block=True)
        angular_momentum_vector_GADGET, _ = cluster.angular_momentum(mass, vel, coords)

        coords, vel = angular_momentum.derotate(cluster, align='gas', cluster_rest_frame=True, derotate_block=False)
        angular_momentum_vector_DEROT, _ = cluster.angular_momentum(mass, vel, coords)

        self.plot_angularmomentum_vectors(np.vstack((angular_momentum_vector_GADGET, angular_momentum_vector_DEROT)),
                                     labels = None,
                                     axes=None,
                                     plot_unitSphere=True,
                                     normalise_length=False,
                                     make_all_unitary=True,
                                     )

if __name__ == "__main__":

    exec(open('visualisation/light_mode.py').read())
    TestSuite()._TEST_basic_LoS()
    plt.show()