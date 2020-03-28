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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from import_toolkit.simulation import Simulation


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



if __name__ == '__main__':
    for sim in ['ceagle', 'celr_e', 'celr_b', 'macsis']:
        SimulationOutput(simulation_name = sim)
