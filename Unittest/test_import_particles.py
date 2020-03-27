import os
import sys
import unittest
import numpy as np
import h5py as h5

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from import_toolkit.cluster import Cluster

np.set_printoptions(suppress=True)

class TestMixin(unittest.TestCase):

    def test_particle_group_number(self):
        # Read in celr_e | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/particledata_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_particles_029_z000p000.0.hdf5'), 'r') as f:
            hd5set = f['/PartType1/GroupNumber']
            group_number = hd5set[...]
            print(f"\n{' celr_e | halo 0 | z=0 ':-^60}")
            print(f"Particles with group number < 0: {len(np.where(group_number<0)[0])} particles found.")
            print(f"Particles with group number = 0: {len(np.where(group_number==0)[0])} particles found.")
            print(f"Particles with group number = 1: {len(np.where(group_number==1)[0])} particles found.")

        # Read in celr_b | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/bahamas/halo_00/data/particledata_029'
        with h5.File(os.path.join(path, 'eagle_subfind_particles_029.0.hdf5'), 'r') as f:
            hd5set = f['/PartType1/GroupNumber']
            group_number = hd5set[...]
            print(f"\n{' celr_b | halo 0 | z=0 ':-^60}")
            print(f"Particles with group number < 0: {len(np.where(group_number < 0)[0])} particles found.")
            print(f"Particles with group number = 0: {len(np.where(group_number == 0)[0])} particles found.")
            print(f"Particles with group number = 1: {len(np.where(group_number == 1)[0])} particles found.")

        # Read in macsis | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-hens1/macsis/macsis_gas/halo_0000/data/particledata_022'
        with h5.File(os.path.join(path, 'eagle_subfind_particles_022.0.hdf5'), 'r') as f:
            hd5set = f['/PartType1/GroupNumber']
            group_number = hd5set[...]
            print(f"\n{' macsis | halo 0 | z=0 ':-^60}")
            print(f"Particles with group number < 0: {len(np.where(group_number < 0)[0])} particles found.")
            print(f"Particles with group number = 0: {len(np.where(group_number == 0)[0])} particles found.")
            print(f"Particles with group number = 1: {len(np.where(group_number == 1)[0])} particles found.")

        # Read in ceagle | halo 0 | z=0
        path = '/cosma5/data/dp004/C-EAGLE/Complete_Sample/CE_00/data/particledata_029_z000p000'
        group_number = np.zeros(0, dtype=np.int)
        file_index = 0

        while file_index > -1:
            try:
                with h5.File(os.path.join(path, f'eagle_subfind_particles_029_z000p000.{str(file_index)}.hdf5'),
                             'r') as f:
                    hd5set = f['/PartType1/GroupNumber']
                    group_number = np.concatenate((group_number, hd5set[...]), axis=0)
                    file_index += 1
            except:
                file_index = -1

        print(f"\n{' ceagle | halo 0 | z=0 ':-^60}")
        print(f"Particles with group number < 0: {len(np.where(group_number < 0)[0])} particles found.")
        print(f"Particles with group number = 0: {len(np.where(group_number == 0)[0])} particles found.")
        print(f"Particles with group number = 1: {len(np.where(group_number == 1)[0])} particles found.")

    def test_filenames(self):
        # Read in celr_e | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/eagle'
        exists_dir = os.path.isdir(os.path.join(path, 'halo_00/data/particledata_029_z000p000'))
        exists_file = os.path.isfile(os.path.join(path, 'halo_00/data/particledata_029_z000p000',
                                                  'eagle_subfind_particles_029_z000p000.0.hdf5'))

        print(f"\n{' celr_e | halo 0 | z=0 ':-^60}")
        print(f"Data directory exists: {exists_dir}.")
        print(f"Data file exists: {exists_file}.")

        # Read in celr_b | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/bahamas'
        exists_dir = os.path.isdir(os.path.join(path, 'halo_00/data/particledata_029'))
        exists_file = os.path.isfile(os.path.join(path, 'halo_00/data/particledata_029',
                                                  'eagle_subfind_particles_029.0.hdf5'))

        print(f"\n{' celr_b | halo 0 | z=0 ':-^60}")
        print(f"Data directory exists: {exists_dir}.")
        print(f"Data file exists: {exists_file}.")

        # Read in macsis | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-hens1/macsis/macsis_gas'
        exists_dir = os.path.isdir(os.path.join(path, 'halo_0000/data/particledata_022'))
        exists_file = os.path.isfile(os.path.join(path, 'halo_0000/data/particledata_022',
                                                  'eagle_subfind_particles_022.0.hdf5'))

        print(f"\n{' macsis | halo 0 | z=0 ':-^60}")
        print(f"Data directory exists: {exists_dir}.")
        print(f"Data file exists: {exists_file}.")

        # Read in ceagle | halo 0 | z=0
        path = '/cosma5/data/dp004/C-EAGLE/Complete_Sample'
        exists_dir = os.path.isdir(os.path.join(path, 'CE_00/data/particledata_029_z000p000'))
        print(f"\n{' ceagle | halo 0 | z=0 ':-^60}")
        print(f"Data directory exists: {exists_dir}.")
        collection_exists_file = []
        file_index = 0
        exists_file = True

        while exists_file:
            exists_file = os.path.isfile(os.path.join(path, f'CE_00/data/particledata_029_z000p000',
                                                      f'eagle_subfind_particles_029_z000p000.{str(file_index)}.hdf5'))
            collection_exists_file.append(exists_file)
            print(f"Data file {file_index:03d} exists: {exists_file}.")
            file_index += 1

        print(f"{' SOFTWARE TEST ':=^60}")
        for sim in ['celr_e', 'celr_b', 'macsis']:
            cluster = Cluster(simulation_name=sim, clusterID=0, redshift='z000p000')
            print(f"\n {sim}{' | halo 0 | z=0 ':-^60}")
            print("cluster.groups_filePaths", cluster.groups_filePaths(), sep='\n')
            print("cluster.partdata_filePaths", cluster.partdata_filePaths(), sep='\n')





if __name__ == '__main__':
    unittest.main()


