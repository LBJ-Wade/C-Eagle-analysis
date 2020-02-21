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

import numpy as np

from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from visualisation import map_plot_parameters as plotpar

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


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))


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


    def xyz_projections(self,
                        xyzdata = None,
                        weights = None,
                        weights_labels = None,
                        plot_limit = None,
                        nbins = None,
                        colorbar_type = None,
                        circle_pars = None,
                        circle_labels = None,
                        special_markers_pars = None,
                        special_markers_labels = None):

        """
        Inside the function scope, the `x` and `y` refer to the vertical and horizontal axes in the plt.figure
        object. E.g. :
            - x_specialMarkers : the horizontal coordinate of the markers displayed in the panel
            - y_specialMarkers : the vertical coordinate of the markers displayed in the panel

        :param xyzdata:
        :param weights:
        :param plot_limit:
        :param nbins:

        :param circle_pars: format is (N,4) shape np.ndarray
            [[x_centre, y_centre, z_centre, radius], [...], ...]

        :param special_markers: format is (N,3) shape np.ndarray
            [[x_marker, y_marker, z_marker], [...], ...]

        :param special_markers_labels:
        :return: None
        """
        xyzdata_OK = True if xyzdata is not None else False
        weights_OK = True if weights is not None else False
        weights_labels_OK = True if weights_labels is not None else False
        plot_limit_OK = True if plot_limit is not None else False
        nbins_OK = True if nbins is not None else False
        colorbar_type_OK = True if colorbar_type is not None else False
        circle_pars_OK = True if circle_pars is not None else False
        circle_labels_OK = True if circle_labels is not None else False
        special_markers_pars_OK = True if special_markers_pars is not None else False
        special_markers_labels_OK = True if special_markers_labels is not None else False

        data_are_parsed = xyzdata_OK and nbins_OK and plot_limit_OK

        xyzdata = np.asarray(xyzdata)
        weights = np.asarray(weights)
        circle_pars = np.asarray(circle_pars)
        circle_labels = np.asarray(circle_labels)
        special_markers_pars = np.asarray(special_markers_pars)
        special_markers_labels = np.asarray(special_markers_labels)

        if circle_pars_OK and circle_pars.shape == (4,):
            circle_pars = circle_pars.reshape((1, 4))

        if circle_labels_OK and circle_labels.shape == ():
            circle_labels = circle_labels.reshape((1))

        if special_markers_pars_OK and special_markers_pars.shape == (3,):
            special_markers_pars = special_markers_pars.reshape((1, 3))

        if special_markers_labels_OK and special_markers_labels.shape == ():
            special_markers_labels = special_markers_labels.reshape((1))

        if circle_labels_OK:
            assert len(circle_labels) == len(circle_pars), ("Expected equal numbers of circle labels and circle "
                                                            "parameters, "
                                                            "got {} circle labels and {} circle parameters.".format(
                                                            len(circle_labels), len(circle_pars)))

        if special_markers_labels_OK:
            assert len(special_markers_labels) == len(special_markers_pars), ("Expected equal numbers of "
                                                                              "special_markers labels and "
                                                                              "special_markers parameters, "
                                                            "got {} special_markers labels and {} special_markers parameters.".format(
                                                            len(special_markers_labels), len(special_markers_pars)))

        if weights_labels_OK:
            assert len(weights_labels) == 3, "Expected 3 labels for colorbars. If you wish to enter the same label " \
                                             "for all of them, use `['my_colorbar_label']*3` in `weights_labels`."
        else:
            weights_labels = ['']*3

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

        cmap =      [plt.get_cmap('seismic'), plt.get_cmap('seismic'), plt.get_cmap('seismic_r')]
        xlabel =    [r'$x\mathrm{/Mpc}$', r'$y\mathrm{/Mpc}$', r'$x\mathrm{/Mpc}$']
        ylabel =    [r'$y\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$']

        thirdAX =   [r'$\bigodot z$', r'$\bigodot x$', r'$\bigotimes y$']
        line_of_sight = [r'$\ \uparrow \mathcal{O}$', r'$\rightarrow \mathcal{O}$', r'$\bigotimes \mathcal{O}$']

        axes_panes =   {'xy' : [0, 1, 2],
                        'yz' : [1, 2, 0],
                        'xz' : [0, 2, 1]}

        for pane_iterator, (axes_pane_name, axes_pane_indices) in enumerate(zip(axes_panes.keys(), axes_panes.values())):

            axes[pane_iterator].set_aspect('equal')
            axes[pane_iterator].set_xlabel(xlabel[pane_iterator])
            axes[pane_iterator].set_ylabel(ylabel[pane_iterator])
            axes[pane_iterator].annotate(thirdAX[pane_iterator], (0.03, 0.03), textcoords='axes fraction', size=15)
            axes[pane_iterator].annotate(line_of_sight[pane_iterator], (0.03, 0.1), textcoords='axes fraction',
                                         size=15)

            axes[pane_iterator].set_xlim(-plot_limit, plot_limit)
            axes[pane_iterator].set_ylim(-plot_limit, plot_limit)

            if data_are_parsed:
                x_Data = xyzdata[:, axes_pane_indices[0]]
                y_Data = xyzdata[:, axes_pane_indices[1]]

                if weights_OK and weights.ndim == 2:
                    weights = weights[:, axes_pane_indices[2]]

                x_bins = np.linspace(-plot_limit, plot_limit, nbins)
                y_bins = np.linspace(-plot_limit, plot_limit, nbins)
                Cx, Cy = Map.bins_meshify(x_Data, y_Data, x_bins, y_bins)
                count = Map.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights=weights)

                assert colorbar_type_OK, "`colorbar_type` cannot be None."
                if colorbar_type == 'log':
                    norm = colors.LogNorm(vmin=10**8, vmax=np.max(count))
                elif colorbar_type == 'midpointLinear':
                    norm = MidpointNormalize(vmin=count.min(), vmax=count.max(), midpoint=0)
                elif colorbar_type[0] == 'symlog':
                    norm = colors.SymLogNorm(linthresh=colorbar_type[1], linscale=0.5, vmin=-np.abs(count).max(), vmax=np.abs(count).max())
                else:
                    raise(ValueError("`colorbar_type` must be`log` or`midpointLinear` or `symlog`." ))

                img = axes[pane_iterator].pcolor(Cx, Cy, count, cmap=cmap[pane_iterator], norm = norm)

                ax2_divider = make_axes_locatable(axes[pane_iterator])
                cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
                cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
                cbar.set_label(weights_labels[pane_iterator], labelpad=-70)
                # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
                cax2.xaxis.set_ticks_position("top")

            # Plot the special markers
            if special_markers_pars_OK:

                x_specialMarkers = special_markers_pars[:, axes_pane_indices[0]]
                y_specialMarkers = special_markers_pars[:, axes_pane_indices[1]]

                for x, y in zip(x_specialMarkers, y_specialMarkers):
                    axes[pane_iterator].scatter(x, y, color='red', linestyle='--')

                if special_markers_labels_OK:
                    for x, y, txt in zip(x_specialMarkers, y_specialMarkers, special_markers_labels):
                        axes[pane_iterator].annotate(txt, (x, y), size=15)

            # Plot the circles
            if circle_pars_OK:

                x_circleCentres = circle_pars[:, axes_pane_indices[0]]
                y_circleCentres = circle_pars[:, axes_pane_indices[1]]

                for x, y, r in zip(x_circleCentres, y_circleCentres, circle_pars[:, 3]):
                    axes[pane_iterator].add_artist(Circle((x, y), radius=r, color='black', fill=False, linestyle='--'))

                if circle_labels_OK:
                    for x, y, r, txt in zip(x_circleCentres, y_circleCentres, circle_pars[:, 3], circle_labels):
                        axes[pane_iterator].annotate(txt, (x, y + 1.1 * r), size=15)

            print("[ MAP ]\t==> Panel {} completed.".format(axes_pane_name))



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
        Reference_Ang_Momentum_color = '#E59813'
        Reference_Ang_Momentum = Arrow3D([0, 0], [0, 0], [0, 1], mutation_scale=20, lw=3, arrowstyle="-|>", color=Reference_Ang_Momentum_color)
        axes.scatter([], [], c=Reference_Ang_Momentum_color, marker=r"$\longrightarrow$", s=70, label=r'Reference angular momentum')
        axes.add_artist(Reference_Ang_Momentum)
        print('[ PLOT 3D VECTOR ]\t==>\tDrawing Reference_Ang_Momentum.')



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

        if labels is None:
            labels = [''] * vectors_magnitudes.__len__()

        assert labels.__len__() == vectors_magnitudes.__len__()

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

class TestSuite(Map):

    def __init__(self):
        print('Using TestSuite mode.')


    def _TEST_velocity_map(self):

        self.xyz_projections(xyzdata=None,
                             weights=None,
                             plot_limit=10,
                             nbins=None,
                             circle_pars=[[0,0,0,1], [2,2,2,1]],
                             circle_labels=[r'label1', r'label2'],
                             special_markers_pars=[[0,0,0], [2,2,2]],
                             special_markers_labels=[r'marker1', r'marker2'])

    def _TEST_CELR_velocity_field(self):

        from cluster import Cluster
        from testing import angular_momentum

        cluster = Cluster(simulation_name='celr_b', clusterID = 0, redshift = 'z000p000')
        r500 = cluster.group_r500()
        mass = cluster.particle_masses('gas')

        coords, vel = angular_momentum.derotate(cluster, align='gas', aperture_radius=r500, cluster_rest_frame=True)
        momentum_lineOfSight = (vel.T * mass).T

        cbarlabel = [r'$\sum_{i} m_i v_{z, i} / \sum_{i} m_i \ [\mathrm{km\ s^{-1}}]$',
                     r'$\sum_{i} m_i v_{x, i} / \sum_{i} m_i \ [\mathrm{km\ s^{-1}}]$',
                     r'$\sum_{i} m_i v_{y, i} / \sum_{i} m_i \ [\mathrm{km\ s^{-1}}]$']

        self.xyz_projections(xyzdata=coords,
                             weights= momentum_lineOfSight,
                             plot_limit=3*r500,
                             weights_labels=cbarlabel,
                             nbins=100,
                             colorbar_type='midpointLinear',
                             circle_pars=[[0, 0, 0, r500], [0, 0, 0, 5*r500]],
                             circle_labels=[r'$R_{500}$', r'$5\times R_{500}$'],
                             special_markers_pars=[0, 0, 0],
                             special_markers_labels=r'CoM')


    def _TEST_CELR_yrkSZ_field(self):

        from cluster import Cluster
        from testing import angular_momentum

        cluster = Cluster(simulation_name='celr_e', clusterID = 0, redshift = 'z000p000')
        r500 = cluster.group_r500()
        mass = cluster.particle_masses('gas')

        coords, vel = angular_momentum.derotate(cluster, align='gas', aperture_radius=r500, cluster_rest_frame=True)

        from unyt import hydrogen_mass, speed_of_light, thompson_cross_section
        plot_limit = 3*r500
        nbins = 100
        bins = np.linspace(-plot_limit, plot_limit, nbins)
        pixel_area = (bins[1] - bins[0])**2
        kSZ = np.multiply((vel.T * mass).T, (-1) * thompson_cross_section / (pixel_area * speed_of_light * hydrogen_mass * 1.16))

        self.xyz_projections(xyzdata=coords,
                             weights= kSZ,
                             weights_labels=[r'$y_{kSZ} / \mathrm{Mpc}^2$']*3,
                             plot_limit=plot_limit,
                             nbins=nbins,
                             colorbar_type=('symlog', 1e-6),
                             circle_pars=[[0, 0, 0, r500], [0, 0, 0, 5*r500]],
                             circle_labels=[r'$R_{500}$', r'$5\times R_{500}$'],
                             special_markers_pars=[0, 0, 0],
                             special_markers_labels=r'CoM')

    def _TEST_derotate_field(self):
        from cluster import Cluster
        from testing import angular_momentum

        cluster = Cluster(simulation_name='celr_e', clusterID = 0, redshift = 'z000p000')
        mass = cluster.particle_masses('gas')

        coords, vel = angular_momentum.derotate(cluster, align='gas', cluster_rest_frame=True, derotate_block=True)
        angular_momentum_vector_GADGET, _ = cluster.angular_momentum(mass, vel, coords)

        coords, vel = angular_momentum.derotate(cluster, align='gas', cluster_rest_frame=True, derotate_block=False)
        angular_momentum_vector_DEROT, _ = cluster.angular_momentum(mass, vel, coords)

        plot_angularmomentum_vectors(np.vstack((angular_momentum_vector_GADGET, angular_momentum_vector_DEROT)),
                                     labels = None,
                                     axes=None,
                                     plot_unitSphere=True,
                                     normalise_length=False,
                                     make_all_unitary=True,
                                     )

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    TestSuite()._TEST_derotate_field()
    plt.show()


