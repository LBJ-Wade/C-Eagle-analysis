import unittest
import h5py as h5
import numpy as np
import os

np.set_printoptions(suppress=True)

class TestRotVel(unittest.TestCase):
    def test_rotvel_alignment_GAS(self):

        # Read in celr_e | halo 0 | z=0
        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/groups_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_tab_029_z000p000.0.hdf5'), 'r') as f:

            hd5set = f['/FOF/Group_R_Crit500']
            r500 = hd5set[...][0]

            hd5set = f['/FOF/GroupCentreOfPotential']
            CoP = hd5set[...][0]

        path = '/cosma5/data/dp004/dc-pear3/data/eagle/halo_00/data/particledata_029_z000p000'
        with h5.File(os.path.join(path, 'eagle_subfind_particles_029_z000p000.0.hdf5'), 'r') as f:

            hd5set = f['/PartType0/GroupNumber']
            group_number = hd5set[...]

            hd5set = f['/PartType0/Coordinates']
            coordinates = hd5set[...]

            hd5set = f['/PartType0/Velocity']
            velocity = hd5set[...]

            hd5set = f['/PartType0/Mass']
            mass = hd5set[...]

            hd5set = f['/PartType0/Temperature']
            temperature = hd5set[...]

        # Convert in cluster-centric radial coordinates (`_r` == `rescaled`)

        coordinates_r = np.sqrt(
            (coordinates[:, 0] - CoP[0]) ** 2 +
            (coordinates[:, 1] - CoP[1]) ** 2 +
            (coordinates[:, 2] - CoP[2]) ** 2
        )

        # Apply masks:
        #   aperture radius = R500_true
        #   no equation of state
        #   central FOF group
        mask = np.where(
            (coordinates_r < r500) &
            (temperature < 10**5) &
            (group_number == 1)
        )[0]

        group_number = group_number[mask]
        coordinates = coordinates[mask]
        coordinates_r = coordinates_r[mask]
        velocity = velocity[mask]
        mass = mass[mask]
        temperature = temperature[mask]

        # Compute peculiar velocity
        pec_velocity = np.sum(velocity*mass[:, None], axis = 0)/np.sum(mass)

        velocity_r = np.sqrt(
            (velocity[:, 0] - pec_velocity[0]) ** 2 +
            (velocity[:, 1] - pec_velocity[1]) ** 2 +
            (velocity[:, 2] - pec_velocity[2]) ** 2
        )

        # Compute angular momentum as r [cross] v
        linear_momentum_r = velocity_r*mass[:, None]
        ang_momentum = np.sum(np.cross(coordinates_r, linear_momentum_r, axis = 0), axis = 0)/np.sum(mass)

        # Compute angle between pec_velocity and ang_momentum
        delta_theta = np.arccos(np.dot(pec_velocity, ang_momentum) / (np.linalg.norm(pec_velocity) * np.linalg.norm(
            ang_momentum))) * 180/np.pi

        # Display results
        print(f"Shape of r500: {r500.shape:<10}")
        print(f"Shape of CoP: {CoP.shape:<10}")
        print(f"Shape of group_number: {group_number.shape:<10}")
        print(f"Shape of coordinates: {coordinates.shape:<10}")
        print(f"Shape of velocity: {velocity.shape:<10}")
        print(f"Shape of mass: {mass.shape:<10}")
        print(f"Shape of temperature: {temperature.shape:<10}")
        print('\n')
        print(f"Shape of mask: {mask.shape:<10}")
        print(f"Shape of coordinates_r: {coordinates_r.shape:<10}")
        print(f"Shape of velocity_r: {velocity_r.shape:<10}")
        print(f"Shape of pec_velocity: {pec_velocity.shape:<10}")
        print(f"Shape of ang_momentum: {ang_momentum.shape:<10}")
        print('\n\n')
        print(f"pec_velocity = {pec_velocity}")
        print(f"ang_momentum = {ang_momentum}")
        print(f"delta_theta = {delta_theta}")



if __name__ == '__main__':
    unittest.main()


