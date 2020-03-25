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

import itertools
import numpy as np
import sys
import os.path
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Simulation, Cluster


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

    def pull_apertures(self):

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                   f"found in {self.FOFDirectory}."
                                                                                   "Check that they have "
                                                                                   "already been computed for this "
                                                                                   "cluster and this redshift.")

        # Read aperture data
        with h5py.File(os.path.join(self.FOFDirectory, 'apertures.hdf5'), 'r') as input_file:
            apertures = np.array(input_file.get('Apertures'))

        return apertures

    def pull_angmom_alignment_angles(self):
        """
        Compute the angular momentum alignment angles between different particle types, based on the results from the angular
        momentum output. This function therefore requires the angular momentum file to exist.
        :return: None
        """
        assert os.path.isfile(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5')), ("Angular momentum data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have "
                                                                                          "already been computed for "
                                                                                          "this cluster and this redshift.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                  f"found in {self.FOFDirectory}."
                                                                                  "Check that they have "
                                                                                  "already been computed for this "
                                                                                  "cluster and this redshift.")

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
    
    def pull_peculiar_velocity_alignment_angles(self):
        """
        Compute the angular momentum alignment angles between different particle types, based on the results from the angular
        momentum output. This function therefore requires the angular momentum file to exist.
        :return: None
        """
        assert os.path.isfile(os.path.join(self.FOFDirectory, 'peculiar_velocity.hdf5')), ("Angular momentum data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have "
                                                                                           "already been computed "
                                                                                           "for this cluster and "
                                                                                           "this redshift.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                    f"found in {self.FOFDirectory}."
                                                                                    "Check that they have "
                                                                                   "already been computed for this "
                                                                                   "cluster and this redshift.")

        # Read aperture data
        with h5py.File(os.path.join(self.FOFDirectory, 'apertures.hdf5'), 'r') as input_file:
            apertures = np.array(input_file.get('Apertures'))

        # Read peculiar velocity data
        with h5py.File(os.path.join(self.FOFDirectory, 'peculiar_velocity.hdf5'), 'r') as input_file:
            Total_ZMF    = np.array(input_file.get('Total_ZMF'))
            ParType0_ZMF = np.array(input_file.get('ParType0_ZMF'))
            ParType1_ZMF = np.array(input_file.get('ParType1_ZMF'))
            ParType4_ZMF = np.array(input_file.get('ParType4_ZMF'))
            ParType5_ZMF = np.array(input_file.get('ParType5_ZMF'))

        # One 5x5 matrix for each aperture
        alignment_matrix = np.zeros((len(apertures), 5, 5), dtype=np.float)

        for m, r in np.ndenumerate(apertures):

            # Gather peculiar velocities for a single aperture
            ang_momenta = np.zeros((5,3))
            ang_momenta[0] = Total_ZMF[m]
            ang_momenta[1] = ParType0_ZMF[m]
            ang_momenta[2] = ParType1_ZMF[m]
            ang_momenta[3] = ParType4_ZMF[m]
            ang_momenta[4] = ParType5_ZMF[m]

            for i, j in itertools.product(range(5), range(5)):
                if i == j:
                    alignment_matrix[m][i][j] = 0.
                else:
                    alignment_matrix[m][i][j] = self.cluster.angle_between_vectors(ang_momenta[i], ang_momenta[j])

        return alignment_matrix
    
    def pull_rot_vel_alignment_angles(self):

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'peculiar_velocity.hdf5')), ("Angular momentum data not "
                                                                                           f"found in {self.FOFDirectory}."
                                                                                           "Check that they have "
                                                                                           "already been computed "
                                                                                           "for this cluster and "
                                                                                           "this redshift.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                   f"found in {self.FOFDirectory}."
                                                                                   "Check that they have "
                                                                                   "already been computed for this "
                                                                                   "cluster and this redshift.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5')), ("Angular momentum data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have "
                                                                                          "already been computed for "
                                                                                          "this cluster and this redshift.")


        # Read aperture data
        with h5py.File(os.path.join(self.FOFDirectory, 'apertures.hdf5'), 'r') as input_file:
            apertures = np.array(input_file.get('Apertures'))

        # Read angular momentum data
        with h5py.File(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5'), 'r') as input_file:
            Total_angmom = np.array(input_file.get('Total_angmom'))
            ParType0_angmom = np.array(input_file.get('ParType0_angmom'))
            ParType1_angmom = np.array(input_file.get('ParType1_angmom'))
            ParType4_angmom = np.array(input_file.get('ParType4_angmom'))
            ParType5_angmom = np.array(input_file.get('ParType5_angmom'))

        # Read peculiar valocity data
        with h5py.File(os.path.join(self.FOFDirectory, 'peculiar_velocity.hdf5'), 'r') as input_file:
            Total_ZMF = np.array(input_file.get('Total_ZMF'))
            ParType0_ZMF = np.array(input_file.get('ParType0_ZMF'))
            ParType1_ZMF = np.array(input_file.get('ParType1_ZMF'))
            ParType4_ZMF = np.array(input_file.get('ParType4_ZMF'))
            ParType5_ZMF = np.array(input_file.get('ParType5_ZMF'))

        # One 5x5 matrix for each aperture
        alignment_matrix = np.zeros((len(apertures), 10, 10), dtype=np.float)

        for m, r in np.ndenumerate(apertures):

            # Gather corr entry for a single aperture
            corr_entry = np.zeros((10, 3))
            corr_entry[0] = Total_ZMF[m]
            corr_entry[1] = Total_angmom[m]
            corr_entry[2] = ParType0_ZMF[m]
            corr_entry[3] = ParType1_ZMF[m]
            corr_entry[4] = ParType4_ZMF[m]
            corr_entry[5] = ParType5_ZMF[m]
            corr_entry[6] = ParType0_angmom[m]
            corr_entry[7] = ParType1_angmom[m]
            corr_entry[8] = ParType4_angmom[m]
            corr_entry[9] = ParType5_angmom[m]
            

            for i, j in itertools.product(range(10), range(10)):
                if i == j:
                    alignment_matrix[m][i][j] = 0.
                else:
                    alignment_matrix[m][i][j] = self.cluster.angle_between_vectors(corr_entry[i], corr_entry[j])

        return alignment_matrix


    def pull_rot_vel_magnitudes_vectors(self):

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'peculiar_velocity.hdf5')), ("Angular momentum data not "
                                                                                           f"found in {self.FOFDirectory}."
                                                                                           "Check that they have "
                                                                                           "already been computed "
                                                                                           "for this cluster and "
                                                                                           "this redshift.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'apertures.hdf5')), ("Apertures data not "
                                                                                   f"found in {self.FOFDirectory}."
                                                                                   "Check that they have "
                                                                                   "already been computed for this "
                                                                                   "cluster and this redshift.")

        assert os.path.isfile(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5')), ("Angular momentum data not "
                                                                                          f"found in {self.FOFDirectory}."
                                                                                          "Check that they have "
                                                                                          "already been computed for "
                                                                                          "this cluster and this redshift.")

        # Read aperture data
        with h5py.File(os.path.join(self.FOFDirectory, 'apertures.hdf5'), 'r') as input_file:
            apertures = np.array(input_file.get('Apertures'))

        # Read angular momentum data
        with h5py.File(os.path.join(self.FOFDirectory, 'angular_momentum.hdf5'), 'r') as input_file:
            Total_angmom = np.array(input_file.get('Total_angmom'))


        # Read peculiar valocity data
        with h5py.File(os.path.join(self.FOFDirectory, 'peculiar_velocity.hdf5'), 'r') as input_file:
            Total_ZMF = np.array(input_file.get('Total_ZMF'))






if __name__ == '__main__':

    cluster = Cluster(simulation_name = 'celr_e', clusterID = 0, redshift = 'z000p000')
    read = FOFRead(cluster)

    print(read.pull_apertures())
    print(read.pull_peculiar_velocity_alignment_angles())

