"""
------------------------------------------------------------------
FILE:   read/pull.py
AUTHOR: Edo Altamura
DATE:   11-03-2020
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from cluster import Simulation, Cluster


class FOFRead(Simulation):

    def __init__(self, cluster: Cluster):

        super(Simulation, self).__init__(simulation_name=cluster.simulation_name)
        self.cluster = cluster
        self.FOFDirectory = os.path.join(cluster.pathSave,
                                         cluster.simulation_name,
                                         f'halo{self.halo_Num(cluster.clusterID)}',
                                         f'halo{self.halo_Num(cluster.clusterID)}_{cluster.redshift}')

    def get_cluster(self):
        return self.cluster

    def get_directory(self):
        return self.FOFDirectory

    def pull_angmom_alignment_angles(self):
        """
        Compute the angular momentum alignment angles between different particle types, based on the results from the angular
        momentum output. This function therefore requires the angular momentum file to exist.
        :return: None
        """
        assert os.path.isfile(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5')), ("Angular momentum data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have already been computed for this cluster and this redshift.")

        # Read angular momentum data
        input_file = h5py.File(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5'), 'r')

        Total_angmom = np.array(input_file.get('Total_angmom'))
        ParType0_angmom = np.array(input_file.get('ParType0_angmom'))
        ParType1_angmom = np.array(input_file.get('ParType1_angmom'))
        ParType4_angmom = np.array(input_file.get('ParType4_angmom'))
        ParType5_angmom = np.array(input_file.get('ParType5_angmom'))

        input_file.close()