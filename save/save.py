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
import os.path
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster
from _cluster_retriever import halo_Num, redshift_str2num, redshift_num2str

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

        file_name = simulation_obj.cluster_prefix + halo_Num(halo_num) + redshift
        fileCompletePath = simulation_obj.pathSave + '/' + simulation_obj.simulation + '_output/collective_output/' + file_name + '.hdf5'

        with h5py.File(fileCompletePath, "w") as file:

            # folder_name = simulation_obj.cluster_prefix + halo_Num(halo_num)
            # halo_folder = file.create_group(folder_name)            #
            # redshift_folder = halo_folder.create_group(redshift)

            file.create_group('FOF')
            file.create_group('Particles')
            file.create_group('Subgroups')


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
    # subfolder_name = simulation.cluster_prefix + halo_Num(cluster.clusterID) + '/' + redshift_num2str(cluster.redshift)
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
