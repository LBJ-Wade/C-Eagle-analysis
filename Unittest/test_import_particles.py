import unittest
import h5py as h5
import numpy as np
import os
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

if __name__ == '__main__':
    unittest.main()


