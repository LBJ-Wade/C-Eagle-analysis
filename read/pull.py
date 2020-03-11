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

        super().__init__(simulation_name=cluster.simulation_name)
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

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have already been computed for this cluster and this redshift.")

        # Read aperture data
        with h5py.File(os.path.join(self.FOFDirectory, 'apertures.hdf5'), 'r') as input_file:
            apertures = np.array(input_file.get('Apertures'))

        # Read angular momentum data
        with h5py.File(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5'), 'r') as input_file:
            Total_angmom    = np.array(input_file.get('Total_angmom'))
            ParType0_angmom = np.array(input_file.get('ParType0_angmom'))
            ParType1_angmom = np.array(input_file.get('ParType1_angmom'))
            ParType4_angmom = np.array(input_file.get('ParType4_angmom'))
            ParType5_angmom = np.array(input_file.get('ParType5_angmom'))

        # One 5x5 matrix for each aperture
        alignment_matrix = np.zeros((len(apertures), 5, 5), dtype=np.float)

        for m, r in np.ndenumerate(apertures):

            # Gather ang momenta for a single aperture
            ang_momenta = np.zeros((5,3))
            ang_momenta[0] = Total_angmom[m]
            ang_momenta[1] = ParType0_angmom[m]
            ang_momenta[2] = ParType1_angmom[m]
            ang_momenta[3] = ParType4_angmom[m]
            ang_momenta[4] = ParType5_angmom[m]

            for i, j in itertools.product(range(5), range(5)):
                if i == j:
                    alignment_matrix[m][i][j] = 0.
                else:
                    alignment_matrix[m][i][j] = self.cluster.angle_between_vectors(ang_momenta[i], ang_momenta[j])

        return alignment_matrix


if __name__ == '__main__':

    cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
    read = FOFRead(cluster)

    print(read.pull_angmom_alignment_angles())

