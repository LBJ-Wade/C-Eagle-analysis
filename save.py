"""
------------------------------------------------------------------
FILE:   save.py
AUTHOR: Edo Altamura
DATE:   20-11-2019
------------------------------------------------------------------

-------------------------------------------------------------------
"""
import os
import numpy as np
import h5py
import cluster

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
            folder_name = simulation.cluster_prefix + str(halo_num)
            halo_folder = file.create_group(folder_name)
            for redshift in simulation.redshiftAllowed:
                halo_folder.create_group(redshift)


create_file('C-EAGLE')
create_file('CELR-eagle')