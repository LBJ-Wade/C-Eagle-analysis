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

import h5py
import sys
import os
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cluster import Simulation, Cluster
import progressbar



class SimulationOutput(Simulation):

    # How the directory is structured
    directory_levels = ['simulation_name',
                        'cluster_ID',
                        'redshift',
                        'data.hdf5 && other_outputs.*']

    def __init__(self, simulation_name: str = None):

        super().__init__(simulation_name=simulation_name)
        self.check_data_structure()

    @progressbar.ProgressBar()
    def check_data_structure(self):
        print('[ SimulationOutput ]\t==> Checking output data structure...')

        if not os.path.exists(os.path.join(self.pathSave, self.simulation_name)):
            os.makedirs(os.path.join(self.pathSave, self.simulation_name))
        print(f'\t{self.directory_levels[0]} checked.')

        counter = 0
        length_operation = len(self.clusterIDAllowed)*len(self.redshiftAllowed)

        for cluster_number, cluster_redshift in itertools.product(self.clusterIDAllowed, self.redshiftAllowed):

            out_path = os.path.join(self.pathSave,
                                    self.simulation_name,
                                    f'halo{self.halo_Num(cluster_number)}',
                                    f'halo{self.halo_Num(cluster_number)}_{cluster_redshift}')
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            yield ((counter + 1) / length_operation)  # Give control back to decorator
            counter += 1

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



def create_files_set(simulation_name = None):
    """
    Create a hdf5 file to store simulation data.
    :param simulation: (string) Name of the simulation
    :return: No returns: create file with the following structure

    filename.hdf5
    |
    |----------CE_00
    |          |
    |          |-------z000p000
    |          |       |
    |          |       |----------dynamical merger index (with sttributes and description)
    |          |       |----------thermodynamic merger index (with sttributes and description)
    |          |       |----------alignment matrix (with sttributes and description)
    |          |       +---------- ...
    |          |
    |          |-------z000p001
    |          |       |
    .          .       .
    .          .       .
    .          .       .

    EXAMPLE IMPLEMENTATION
    ----------------------
    if __main__ == "__main__":
        create_file('C-EAGLE')
        create_file('CELR-eagle')

    """
    simulation_obj = Simulation(simulation_name = simulation_name)
    process_iterator = itertools.product(simulation_obj.clusterIDAllowed, simulation_obj.redshiftAllowed)

    if not os.path.exists(simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output'):
        os.makedirs(simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output')
        print('Making directory: ', simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output')

    if not os.path.exists(simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output/collective_output'):
        os.makedirs(simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output/collective_output')
        print('Making directory: ', simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output/collective_output')


    for halo_num, redshift in process_iterator:

        file_name = simulation_obj.cluster_prefix + simulation_obj.halo_Num(halo_num) + redshift
        fileCompletePath = simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output/collective_output/' + file_name + '.hdf5'

        with h5py.File(fileCompletePath, "w") as file:

            # folder_name = simulation_obj.cluster_prefix + simulation_obj.halo_Num(halo_num)
            # halo_folder = file.create_group(folder_name)            #
            # redshift_folder = halo_folder.create_group(redshift)

            file.create_group('FOF')
            file.create_group('Particles')
            file.create_group('Subgroups')

    # Check that all files have been correctly created: do not want the cosma system to block some of them
    num_files_created = list(process_iterator).__len__()
    num_files_in_directory = os.listdir(simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output/collective_output' ).__len__()
    assert num_files_created == num_files_in_directory, 'Not all files have been created.'



def create_dataset(fileCompletePath,
                   subfolder = None,
                   dataset_name = None,
                   input_data = None,
                   attributes = None):
    """
    Append dataset to the specific cluster at specific redshift.
    :param simulation: (Simulation object)
    :param cluster: (Cluster object)
    :param dataset_name:
    :param input_data:
    :param attributes:
    :return:
    """

    with h5py.File(fileCompletePath, "r+") as file:
    # subfolder_name = simulation.cluster_prefix + simulation_obj.halo_Num(cluster.clusterID) + '/' +
    # redshift_num2str(cluster.redshift)
        file_halo_redshift = file[subfolder]

        if dataset_name is not None and input_data is not None:
            try:
                del file[subfolder + '/' + dataset_name]
                print('[  SAVE  ] ===> Deleting old dataset: {}'.format(dataset_name))

            except:
                pass

            finally:
                print('[  SAVE  ] ===> Creating new dataset: {} on file {}'.format(dataset_name, fileCompletePath))
                dataset = file_halo_redshift.create_dataset(dataset_name, data = input_data)

        if attributes is not None:
            dataset.attrs['Description'] = attributes
