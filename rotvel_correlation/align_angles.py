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

import numpy as np
import sys
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster
from import_toolkit.simulation import Simulation
from read import pull

class CorrelationMatrix(pull.FOFRead):

    def __init__(self, cluster: Cluster):

        super().__init__(cluster)

    def get_data(self):
        return self.pull_rot_vel_alignment_angles()

    def get_apertures(self):
        return self.pull_apertures()

    @staticmethod
    def plot_matrix(angle_matrix, std0_matrix, std1_matrix, apertures):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        im = ax.imshow(angle_matrix, cmap='Set3_r', vmin=0, vmax=180, origin='lower')
        # Manipulate the colorbar on the side
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, ticks=np.linspace(0, 180, 13))
        cbar.ax.minorticks_off()
        cbar.ax.set_ylabel(r'$\Delta \theta$ \quad [degrees]', rotation=270, size=25, labelpad=30)

        x_labels = [
            r'$\mathbf{\widehat{v_{pec}}}$',
            r'$\mathbf{\widehat{L}}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(gas)}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(DM)}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(stars)}$',
            r'$\mathbf{\widehat{v_{pec}}}^{(BH)}$',
            r'$\mathbf{\widehat{L}}^{(gas)}$',
            r'$\mathbf{\widehat{L}}^{(DM)}$',
            r'$\mathbf{\widehat{L}}^{(stars)}$',
            r'$\mathbf{\widehat{L}}^{(BH)}$'
        ]

        ticks_major = np.arange(0, len(x_labels), 1)
        ticks_minor = np.arange(-0.5, len(x_labels), 1)
        ax.set_xticks(ticks_major, minor=False)
        ax.set_yticks(ticks_major, minor=False)
        ax.set_xticks(ticks_minor, minor=True)
        ax.set_yticks(ticks_minor, minor=True)
        ax.set_xticklabels(x_labels, rotation = 90)
        ax.set_yticklabels(x_labels)


        # Loop over data dimensions and create text annotations.
        for i in range(len(x_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, r"${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                                angle_matrix[i, j],
                                std1_matrix[i, j] - angle_matrix[i, j],
                                angle_matrix[i, j] - std0_matrix[i, j]),
                                ha="center", va="center", color="k", size = 15)

        ax.set_title(r"Aperture = {:.2f}".format(apertures) + r"\ $R_{500\ true}$", size = 8)
        ax.grid(b = True, which='minor',color='w', linestyle='-', linewidth=10, alpha = 1)

        ax.tick_params(which='minor', length=0, color='w')
        ax.tick_params(which='major', length=0, color='w')
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor('w')

        fig.tight_layout()
        # plt.show()





if __name__ == '__main__':
    exec(open(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())

    def test():
        cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
        matrix = CorrelationMatrix(cluster)
        data = matrix.get_data()
        aperture = matrix.get_apertures()

        matrix.plot_matrix(data[15], aperture[15])

    def all_clusters(apertureidx):
        simulation = Simulation(simulation_name='celr_e')
        matrix_list = []
        aperture_list = []
        aperture_self_similar = []
        for id in simulation.clusterIDAllowed:
            cluster = Cluster(simulation_name='celr_e', clusterID=id, redshift='z000p000')
            print(f'Analysing cluster {cluster.clusterID}')
            matrix = CorrelationMatrix(cluster)
            data = matrix.get_data()[apertureidx]
            aperture = matrix.get_apertures()[apertureidx]
            matrix_list.append(data)
            aperture_list.append(aperture)
            aperture_self_similar.append(aperture/cluster.r500)

        if not os.path.exists(os.path.join(simulation.pathSave, simulation.simulation_name, 'rotvel_correlation')):
            os.makedirs(os.path.join(simulation.pathSave, simulation.simulation_name, 'rotvel_correlation'))

        average_aperture = np.mean(aperture_self_similar, axis=0)
        average_matrix = np.percentile(matrix_list, 50, axis=0)
        std0_matrix = np.percentile(matrix_list, 15.9, axis=0)
        std1_matrix = np.percentile(matrix_list, 84.1, axis=0)

        matrix.plot_matrix(average_matrix, std0_matrix, std1_matrix, average_aperture)
        print(f"Saving matrix | aperture {apertureidx} | redshift {cluster.redshift}")
        plt.savefig(os.path.join(simulation.pathSave, simulation.simulation_name, 'rotvel_correlation',
                                 f'meanPMstd_{cluster.redshift}_aperture_{apertureidx}.png'))



    for i in range(20):
        all_clusters(i)




