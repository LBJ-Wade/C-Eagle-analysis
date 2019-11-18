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
import matplotlib as mpl
import numpy as np
import scipy as sp
from matplotlib.patches import Circle
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#################################
#                               #
#	          M A P             #
# 			C L A S S           #
#							    #
#################################

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


    def xyz_projections(self, *args, xyzdata = None, weights = None,
                            plot_limit = None, nbins = None,
                            circle_pars = None, **kwargs):
        """

        :param args:
        :param xyzdata: (numpy 2D array)
        :param weights:
        :param kwargs:
        :return:
        """

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 9))

        cmap =      [plt.get_cmap('Greens'), plt.get_cmap('Greens'), plt.get_cmap('Greens')]
        xlabel =    [r'$x\mathrm{/arcmin}$', r'$y\mathrm{/arcmin}$', r'$x\mathrm{/arcmin}$']
        ylabel =    [r'$y\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$', r'$z\mathrm{/arcmin}$']
        thirdAX =   [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
        cbarlabel = [r'$\sum_{i} m_i v_{z, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$',
                     r'$\sum_{i} m_i v_{x, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$',
                     r'$\sum_{i} m_i v_{y, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$']

        for i in [0, 1, 2]:
            # Handle data
            if i == 0:
                x_Data = xyzdata[:, 0]
                y_Data = xyzdata[:, 1]
            elif i == 1:
                x_Data = xyzdata[:, 1]
                y_Data = xyzdata[:, 2]
            elif i == 2:
                x_Data = xyzdata[:, 0]
                y_Data = xyzdata[:, 2]


            x_bins = np.linspace(-plot_limit, plot_limit, nbins)
            y_bins = np.linspace(-plot_limit, plot_limit, nbins)
            Cx, Cy = Map.bins_meshify(x_Data, y_Data, x_bins, y_bins)
            count = Map.bins_evaluate(x_Data, y_Data, x_bins, y_bins, weights=weights)

            norm = colors.LogNorm(vmin=10 ** -4, vmax=10 ** 3)
            img = axes[i].pcolor(Cx, Cy, count, cmap=cmap[i], norm= norm)

            # Render elements in plots
            axes[i].set_aspect('equal')
            Map.plot_circle(axes[i], *circle_pars, color='black', fill=False, linestyle='--', label=r'$R_{200}$')

            axes[i].set_xlim(-plot_limit, plot_limit)
            axes[i].set_ylim(-plot_limit, plot_limit)
            axes[i].set_xlabel(xlabel[i])
            axes[i].set_ylabel(ylabel[i])
            axes[i].annotate(thirdAX[i], (0.03, 0.03), textcoords='axes fraction', size=15)
            # if title:
            #    axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))
            # Colorbar adjustments
            ax2_divider = make_axes_locatable(axes[i])
            cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
            cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
            cbar.set_label(cbarlabel[i], labelpad=-70)
            # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
            cax2.xaxis.set_ticks_position("top")
            print("run completed:", i)
