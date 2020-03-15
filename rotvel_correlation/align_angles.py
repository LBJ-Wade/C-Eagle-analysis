"""
------------------------------------------------------------------
FILE:   rotvel_correlation/align_angles.py
AUTHOR: Edo Altamura
DATE:   15-03-2020
------------------------------------------------------------------
The save python package generates partial raw data from the simulation
data. The pull.py file accesses and reads these data and returns
then in a format that can be used for plotting, further processing
etc.
-------------------------------------------------------------------
"""

from mpi4py import MPI
import itertools
import numpy as np
import sys
import os.path
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster
from read import pull

class CorrelationMatrix(pull.FOFRead):

    def __init__(self, cluster: Cluster):

        super().__init__(cluster)
        self.aperture_index = None

    def fix_aperture(self, aperture: np.int):
        self.aperture_index = aperture

    def plot_matrix(self):
        apertures = self.pull_apertures()
        angle_matrix = self.pull_rot_vel_alignment_angles()

        if self.aperture_index is not None:
            apertures = apertures[self.aperture_index]
            angle_matrix = angle_matrix[self.aperture_index]

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        norm = colors.LogNorm(vmin=1e-6, vmax=180)
        im = ax.imshow(angle_matrix, cmap='RdBu', norm=norm, origin='lower')
        # Manipulate the colorbar on the side
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.minorticks_off()
        cbar.ax.set_ylabel(r'$\Delta \theta$ \quad [degrees]', rotation=270, size=25, labelpad=30)

        x_labels = [
            r'$\mathbf{\widehat{v_{pec}}}$',
            r'$\mathbf{\widehat{L}}$',
            r'$\mathbf{\widehat{v_{pec}^{(gas)}}}$',
            r'$\mathbf{\widehat{v_{pec}^{(DM)}}}$',
            r'$\mathbf{\widehat{v_{pec}^{(stars)}}}$',
            r'$\mathbf{\widehat{v_{pec}^{(BH)}}}$',
            r'$\mathbf{\widehat{L^{(gas)}}}$',
            r'$\mathbf{\widehat{L^{(DM)}}}$',
            r'$\mathbf{\widehat{L^{(stars)}}}$',
            r'$\mathbf{\widehat{L^{(BH)}}}$'
        ]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(x_labels)

        # Loop over data dimensions and create text annotations.
        for i in range(len(x_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, r"{:.2f}".format(angle_matrix[i, j]),
                               ha="center", va="center", color="k", size = 15)

        ax.set_title(r"Aperture = {:.2f} Mpc".format(apertures), size = 20)
        fig.tight_layout()
        plt.show()





if __name__ == '__main__':
    exec(open('visualisation/light_mode.py').read())
    cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
    matrix = CorrelationMatrix(cluster)
    matrix.fix_aperture(10)
    matrix.plot_matrix()

