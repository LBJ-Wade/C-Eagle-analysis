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

        # Add text around it
        curve = [-np.cos(np.linspace(0, 2 * np.pi, 200)),
                  np.sin(np.linspace(0, 2 * np.pi, 200))]

        text = r'R_{500}'
        curved_text = CurvedText(
            x=curve[0],
            y=curve[1],
            text=text,  # 'this this is a very, very long text',
            va='bottom',
            axes=axes,  ##calls ax.add_artist in __init__
        )


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

        cmap =      [plt.get_cmap('Greens'), plt.get_cmap('Greens'), plt.get_cmap('Greens')]
        xlabel =    [r'$x\mathrm{/Mpc}$', r'$y\mathrm{/Mpc}$', r'$x\mathrm{/Mpc}$']
        ylabel =    [r'$y\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$', r'$z\mathrm{/Mpc}$']
        thirdAX =   [r'$\bigotimes z$', r'$\bigotimes x$', r'$\bigodot y$']
        cbarlabel = [r'$\sum_{i} m_i v_{z, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$',
                     r'$\sum_{i} m_i v_{x, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$',
                     r'$\sum_{i} m_i v_{y, i}\ [\mathrm{M_\odot\ km\ s^{-1}}]$']

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

                norm = colors.LogNorm(vmin=10 ** -3, vmax=10 ** 3)
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
                # if title:
                #    axes[i].set_title(r'$\mathrm{MACSIS\ halo\ } %3d \qquad z = %8.3f$' % (num_halo, redshift))
                # Colorbar adjustments
            if data_are_parsed:
                ax2_divider = make_axes_locatable(axes[i])
                cax2 = ax2_divider.append_axes("top", size="5%", pad="2%")
                cbar = plt.colorbar(img, cax=cax2, orientation='horizontal')
                cbar.set_label(cbarlabel[i], labelpad=-70)
                # cax2.xaxis.set_tick_labels(['0',' ','0.5',' ','1',' ', '1.5',' ','2'])
                cax2.xaxis.set_ticks_position("top")
            print("[MAP PANEL]\t==> completed:", i)


from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used



"""
IMPLEMENTATION OF THE WARPED TEXT CLASS


if __name__ == '__main__':
    Figure, Axes = plt.subplots(2,2, figsize=(7,7), dpi=100)


    N = 100

    curves = [
        [
            np.linspace(0,1,N),
            np.linspace(0,1,N),
        ],
        [
            np.linspace(0,2*np.pi,N),
            np.sin(np.linspace(0,2*np.pi,N)),
        ],
        [
            -np.cos(np.linspace(0,2*np.pi,N)),
            np.sin(np.linspace(0,2*np.pi,N)),
        ],
        [
            np.cos(np.linspace(0,2*np.pi,N)),
            np.sin(np.linspace(0,2*np.pi,N)),
        ],
    ]

    texts = [
        'straight lines work the same as rotated text',
        'wavy curves work well on the convex side',
        'you even can annotate parametric curves',
        'changing the plotting direction also changes text orientation',
    ]

    for ax, curve, text in zip(Axes.reshape(-1), curves, texts):
        #plotting the curve
        ax.plot(*curve, color='b')

        #adjusting plot limits
        stretch = 0.2
        xlim = ax.get_xlim()
        w = xlim[1] - xlim[0]
        ax.set_xlim([xlim[0]-stretch*w, xlim[1]+stretch*w])
        ylim = ax.get_ylim()
        h = ylim[1] - ylim[0]
        ax.set_ylim([ylim[0]-stretch*h, ylim[1]+stretch*h])

        #adding the text
        text = CurvedText(
            x = curve[0],
            y = curve[1],
            text=text,#'this this is a very, very long text',
            va = 'bottom',
            axes = ax, ##calls ax.add_artist in __init__
        )

    plt.show()
"""
