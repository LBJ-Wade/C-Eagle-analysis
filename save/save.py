"""
------------------------------------------------------------------
FILE:   save.py
AUTHOR: Edo Altamura
DATE:   20-11-2019
------------------------------------------------------------------
In order to make the data post-processing more manageable, instead
of calculating quantities from the simulations every time, just
compute them once and store them in a hdf5 file.
This process of data reduction level condenses data down to a few KB
or MB and it is possible to transfer it locally for further analysis.
-------------------------------------------------------------------
"""

import sys
import os
import itertools
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.simulation import Simulation
from import_toolkit._cluster_retriever import redshift_str2num


class SimulationOutput(Simulation):

    # How the directory is structured
    directory_levels = ['simulation_name',
                        'cluster_ID',
                        'redshift',
                        'data.hdf5 && other_outputs.*']

    def __init__(self, simulation_name: str = None):

        super().__init__(simulation_name=simulation_name)
        self.check_data_structure()

    def check_data_structure(self):
        print('[ SimulationOutput ]\t==> Checking output data structure...')

        if not os.path.exists(os.path.join(self.pathSave, self.simulation_name)):
            os.makedirs(os.path.join(self.pathSave, self.simulation_name))
        print(f'\t{self.directory_levels[0]} checked.')

        for cluster_number, cluster_redshift in itertools.product(self.clusterIDAllowed, self.redshiftAllowed):

            out_path = os.path.join(self.pathSave,
                                    self.simulation_name,
                                    f'halo{self.halo_Num(cluster_number)}',
                                    f'halo{self.halo_Num(cluster_number)}_{cluster_redshift}')
            if not os.path.exists(out_path):
                os.makedirs(out_path)

        print(f'\t{self.directory_levels[1]} checked.')
        print(f'\t{self.directory_levels[2]} checked.')


    @staticmethod
    def list_files(startpath):
        for root, dirs, files in os.walk(startpath):
            dirs.sort()
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            files.sort()
            for f in files:
                print('{}{}'.format(subindent, f))

    def print_directory_tree(self):
        print(self.pathSave)
        self.list_files(os.path.join(self.pathSave, self.simulation_name))

    def dir_tree_to_dict(self, path_):
        file_token = ''
        for root, dirs, files in os.walk(path_):
            tree = {d: self.dir_tree_to_dict(os.path.join(root, d)) for d in dirs}
            tree.update({f: file_token for f in files})
            return tree  # note we discontinue iteration trough os.walk

    @staticmethod
    def draw_pie(dist,
                 xpos,
                 ypos,
                 size,
                 ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 12))

        # for incremental pie slices
        cumsum = np.cumsum(dist)
        cumsum = cumsum / cumsum[-1]
        pie = [0] + cumsum.tolist()
        colors = ['lime', 'red']
        for r1, r2, c in zip(pie[:-1], pie[1:], colors):
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            ax.scatter([xpos], [ypos], marker=xy, s=size, facecolor=c, alpha = 0.5)

        return ax

    def status_plot(self):

        timestr = time.strftime("%d%m%Y-%H%M%S")


        fig = plt.figure(figsize=(11, 20))
        ax = fig.add_subplot(111)

        ax.set_title('{:s}    Output status record    {:s}'.format(self.simulation, timestr))
        ax.set_xlabel('redshift')
        ax.set_ylabel('ClusterID')

        for cluster_number, cluster_redshift in itertools.product(self.clusterIDAllowed, self.redshiftAllowed):

            out_path = os.path.join(self.pathSave,
                                    self.simulation_name,
                                    f'halo{self.halo_Num(cluster_number)}',
                                    f'halo{self.halo_Num(cluster_number)}_{cluster_redshift}')


            num_of_files_expected = 6
            num_of_files = len([name for name in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, name))])
            self.draw_pie([num_of_files, num_of_files_expected - num_of_files],
                          redshift_str2num(cluster_redshift), cluster_number, 100, ax=ax)

        plt.tight_layout()

        plt.savefig(os.path.join(self.pathSave,
                                 self.simulation_name,
                                 f"{self.simulation_name}_OutputStatusReport_{timestr}.png"), dpi=300)

if __name__ == '__main__':
    exec(open(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.path.pardir, 'visualisation', 'light_mode.py'))).read())
    for sim in ['ceagle', 'celr_e', 'celr_b', 'macsis']:
        out = SimulationOutput(simulation_name = sim)
        out.status_plot()
