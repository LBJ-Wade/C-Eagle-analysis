"""
------------------------------------------------------------------
FILE:   save.py
AUTHOR: Edo Altamura
DATE:   20-11-2019
------------------------------------------------------------------

-------------------------------------------------------------------
"""

import h5py
import cluster
from _cluster_retriever import halo_Num, redshift_str2num, redshift_num2str

def create_file(simulation):
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

    """
    simulation = cluster.Simulation(simulation_name = simulation)
    fileCompletePath = simulation.pathSave + '/' + simulation.simulation + '__processed_data.hdf5'
    with h5py.File(fileCompletePath, "w") as file:
        for halo_num in simulation.clusterIDAllowed:
            folder_name = simulation.cluster_prefix + halo_Num(halo_num)
            halo_folder = file.create_group(folder_name)
            for redshift in simulation.redshiftAllowed:
                halo_folder.create_group(redshift)


def create_dataset(simulation,
                   cluster,
                   dataset_name = None,
                   input_data = None,
                   attributes = None,
                   **kwargs):
    """
    Append dataset to the specific cluster at specific redshift.
    :param simulation:
    :param cluster:
    :param dataset_name:
    :param input_data:
    :param attributes:
    :param kwargs:
    :return:
    """

    simulation = cluster.Simulation(simulation_name=simulation)
    fileCompletePath = simulation.pathSave + '/' + simulation.simulation + '__processed_data.hdf5'
    with h5py.File(fileCompletePath, "r+") as file:
        subfolder_name = simulation.cluster_prefix + str(cluster.clusterID) + '/' + redshift_num2str(cluster.redshift)
        file_halo_redshift = file[subfolder_name]
        if dataset_name is not None and input_data is not None:
            dataset = file_halo_redshift.create_dataset(dataset_name, data = input_data)
        if attributes is not None:
            dataset.attrs['Description'] = attributes

# create_file('C-EAGLE')
# create_file('CELR-eagle')
def print_structure(hdf5_file, out_attributes = False):
    pass


